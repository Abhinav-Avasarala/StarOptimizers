"""
KV260 DPU Pose Estimation — sp_net (stable final version)
==========================================================
Everything in the main loop is INLINE — no function calls.
This is required on aarch64/KV260 to avoid DMA memory conflicts
between VART and OpenCV when using function call stack frames.
"""

import cv2
import numpy as np
import xir
import vart
import time
from flask import Flask, request, jsonify, Response
import threading
import socket
import sys

# ── Exercise selection (CLI argument) ─────────────────────────────────────────
_valid = ["bicep_curl", "squat", "lateral_raise"]
if len(sys.argv) < 2 or sys.argv[1] not in _valid:
    print("Usage: python3 dpu_pose.py [bicep_curl|squat|lateral_raise]")
    sys.exit(1)
exercise = sys.argv[1]
print(f"[INFO] Exercise selected: {exercise}")
exercise = {"bicep_curl": 1, "squat": 2, "lateral_raise": 3}[exercise]

# ── Config ────────────────────────────────────────────────────────────────────
XMODEL_PATH = "/home/ubuntu/sp_net/sp_net.xmodel"
CAMERA_INDEX = 0
DISPLAY_W    = 640
DISPLAY_H    = 480
INPUT_W      = 128
INPUT_H      = 224

SKELETON = [
    (0,1),(1,2),(3,4),(4,5),(0,3),
    (6,7),(7,8),(9,10),(10,11),(6,9),
    (13,0),(13,3),(13,12),(6,13),(9,13),
]
NAMES = [
    "R_shl","R_elb","R_wri","L_shl","L_elb","L_wri",
    "R_hip","R_kne","R_ank","L_hip","L_kne","L_ank",
    "Head","Neck"
]
MEAN = np.array([104., 117., 123.], dtype=np.float32)
EXERCISE_NAMES = {1: "Bicep Curl", 2: "Squat", 3: "Lateral Raise"}

# ── Load runners ──────────────────────────────────────────────────────────────
graph = xir.Graph.deserialize(XMODEL_PATH)
root  = graph.get_root_subgraph()
conv_sg = fc_sg = None
for s in root.get_children():
    if not s.has_attr("device") or s.get_attr("device") != "DPU": continue
    name = s.get_name()
    if "conv" in name or "7x7" in name:        conv_sg = s
    elif "fc" in name or "coordinate" in name: fc_sg   = s
assert conv_sg and fc_sg

conv_r = vart.Runner.create_runner(conv_sg, "run")
fc_r   = vart.Runner.create_runner(fc_sg,   "run")
print(f"[INFO] Conv: {conv_sg.get_name()}")
print(f"[INFO] FC:   {fc_sg.get_name()}")

# ── Pre-allocate ALL buffers once ─────────────────────────────────────────────
in_data   = np.ascontiguousarray(np.zeros([1, 224, 128, 3],  dtype=np.int8))
out_data  = np.ascontiguousarray(np.zeros([1, 7,   4,  184], dtype=np.int8))
pool_int8 = np.ascontiguousarray(np.zeros([1, 1,   1,  184], dtype=np.int8))
out_data2 = np.ascontiguousarray(np.zeros([1, 28],           dtype=np.int8))
display   = np.ascontiguousarray(np.zeros([DISPLAY_H, DISPLAY_W, 3], dtype=np.uint8))
print("[INFO] Buffers pre-allocated")

# ── Shared Flask state ────────────────────────────────────────────────────────
latest_imu = {
    "accel": {"x": 0.0, "y": 0.0, "z": 0.0},
    "gyro":  {"x": 0.0, "y": 0.0, "z": 0.0},
    "temp":  0.0,
    "humidity": 0.0,
    "timestamp": 0
}
vibrate_command = {"vibrate": False, "pattern": "none", "reason": "", "severity": "none"}
imu_lock = threading.Lock()
vib_lock = threading.Lock()
alerts       = []   # pre-init so /status works before main loop runs
output_frame = None
frame_lock   = threading.Lock()
live_state      = {"exercise": "none", "rep_count": 0, "alerts": [], "instructions": [], "phase": "idle", "velocity": 0.0,
                   "last_rep_feedback": "", "rep_buffer_count": 0, "total_reps": 0, "feedback_age": 9999.0}
live_lock       = threading.Lock()
exercise_change = {"pending": False, "value": 0}
exercise_lock   = threading.Lock()

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
assert cap.isOpened(), "Camera not found"
print(f"[INFO] Camera: {int(cap.get(3))}x{int(cap.get(4))}")
print(f"[INFO] Exercise: {EXERCISE_NAMES[exercise]}")
print("[INFO] Running — press 'q' to quit, 'r' to reset reps\n")

fps   = 0.0
fc    = 0
t_fps = time.time()

# Rep counter state (pre-allocated outside loop)
rep_count = 0
rep_state = "up" if exercise == 2 else "down"
_ema_metric = 0.0   # EMA of angle (curl) or relative elbow-shoulder gap (raise)
_last_rep_t = 0.0   # cooldown: min 0.5s between reps
frame_count = 0
_ax = _gz = _tp = _imu_ay = 0.0   # IMU locals, pre-init before first frame
_imu_ts = 0

# Phase tracking (pre-allocated outside loop)
ACTIVE_THRESHOLD = 8.0   # px/frame — full form checks
SLOW_THRESHOLD   = 3.0   # px/frame — slow controlled movement still gets checks
ALERT_COOLDOWN   = 3.0

ALERT_SEVERITY = {
    # Critical
    "Push right knee outward":         "critical",
    "Push left knee outward":          "critical",
    "Keep chest up - back rounding":   "critical",
    "Pin your elbow to your body":     "critical",
    # Moderate
    "Raise higher - not at shoulder level yet": "moderate",
    "Too high - stop at shoulder level":        "moderate",
    "Raise both arms evenly":                   "moderate",
    "Curl more - squeeze the bicep":            "moderate",
    "Go deeper - break parallel":               "moderate",
    "High body temp - rest":                    "moderate",
    # Minor
    "Slow down - control the movement": "minor",
    "Slow down - control the squat":    "minor",
    "Slow down - control the curl":     "minor",
    "Excessive forward lean":           "minor",
    "Moving too fast":                  "minor",
}

CRITICAL_IMMEDIATE = {
    "Push right knee outward",
    "Push left knee outward",
    "Keep chest up - back rounding",
    "Pin your elbow to your body",
    "Excessive forward lean",
    "High body temp - rest",
}

rep_phase          = "idle"
phase_frame_count  = 0
last_alert_time    = {}
smoothed_kps       = [(0, 0)] * 14
kps_history        = [[(0, 0)] * 14 for _ in range(5)]
history_idx        = 0
prev_kps           = [(0, 0)] * 14
joint_velocities   = [0.0] * 14
max_velocity       = 0.0
instructions           = []
curl_started           = False
squat_min_angle        = 180.0
rep_alerts_buffer      = []
last_rep_feedback      = ""
rep_start_time         = 0.0
squat_angles_this_rep  = []
curl_angles_this_rep   = []
raise_heights_this_rep = []
total_reps              = 0
feedback_display_timer  = 0.0
active_frames_this_rep  = 0
FEEDBACK_DISPLAY_DURATION = 4.0

# ── Flask sensor server ───────────────────────────────────────────────────────
app = Flask(__name__)

@app.route('/imu', methods=['POST'])
def post_imu():
    data = request.get_json(silent=True) or {}
    with imu_lock:
        latest_imu["accel"]     = data.get("accel",     latest_imu["accel"])
        latest_imu["gyro"]      = data.get("gyro",      latest_imu["gyro"])
        latest_imu["temp"]      = float(data.get("temp",     latest_imu["temp"]))
        latest_imu["humidity"]  = float(data.get("humidity", latest_imu["humidity"]))
        latest_imu["timestamp"] = data.get("timestamp", latest_imu["timestamp"])
    return jsonify({"status": "ok"})

@app.route('/vibrate', methods=['GET'])
def get_vibrate():
    with vib_lock:
        resp = dict(vibrate_command)
        vibrate_command["vibrate"] = False
    return jsonify(resp)

@app.route('/status', methods=['GET'])
def get_status():
    with imu_lock:
        imu_snap = {
            "accel":     dict(latest_imu["accel"]),
            "gyro":      dict(latest_imu["gyro"]),
            "temp":      latest_imu["temp"],
            "humidity":  latest_imu["humidity"],
            "timestamp": latest_imu["timestamp"],
        }
    with vib_lock:
        vib_snap = dict(vibrate_command)
    with live_lock:
        state_snap = dict(live_state)
    return jsonify({
        "imu":              imu_snap,
        "vibrate":          vib_snap,
        "exercise":         state_snap["exercise"],
        "reps":             state_snap["rep_count"],
        "alerts":           state_snap["alerts"],
        "instructions":     state_snap.get("instructions", []),
        "phase":            state_snap.get("phase", "idle"),
        "velocity":         state_snap.get("velocity", 0.0),
        "last_rep_feedback":state_snap.get("last_rep_feedback", ""),
        "rep_buffer_count": state_snap.get("rep_buffer_count", 0),
        "total_reps":       state_snap.get("total_reps", 0),
        "feedback_age":     state_snap.get("feedback_age", 9999.0),
    })

def generate_frames():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>KV260 Fitness Monitor</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: #0d0d0d;
      color: #e0e0e0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      padding: 12px 20px;
      background: #111;
      border-bottom: 1px solid #222;
      font-size: 13px;
      color: #555;
      letter-spacing: 1px;
      text-transform: uppercase;
    }
    header span { color: #0af; }

    .main {
      flex: 1;
      display: flex;
      flex-direction: row;
      overflow: hidden;
    }

    /* ── Video panel (left 70%) ─────────────────────── */
    .video-panel {
      flex: 7;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 0;
    }
    .video-panel img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    /* ── Side panel (right 30%) ─────────────────────── */
    .side-panel {
      flex: 3;
      background: #111;
      border-left: 1px solid #1e1e1e;
      display: flex;
      flex-direction: column;
      padding: 20px 16px;
      gap: 16px;
      overflow-y: auto;
      min-width: 220px;
    }

    .card {
      background: #181818;
      border-radius: 10px;
      padding: 14px 16px;
      border: 1px solid #222;
    }
    .card-label {
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 1.5px;
      color: #555;
      margin-bottom: 6px;
    }

    #exercise-name {
      font-size: 26px;
      font-weight: 700;
      letter-spacing: 0.5px;
    }
    .ex-squat        { color: #4fc3f7; }
    .ex-bicep_curl   { color: #ffb74d; }
    .ex-lateral_raise{ color: #81c784; }
    .ex-none         { color: #555; }

    #rep-count {
      font-size: 72px;
      font-weight: 800;
      color: #fff;
      line-height: 1;
      text-align: center;
    }

    #alerts-list { display: flex; flex-direction: column; gap: 6px; min-height: 20px; }
    .alert-card {
      background: #3a0d0d;
      border: 1px solid #c62828;
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 12px;
      color: #ff8a80;
    }
    .no-alert { font-size: 12px; color: #2e7d32; }

    #instructions-list { display: flex; flex-direction: column; gap: 5px; }
    .instruction-card {
      background: #1a1a1a;
      border: 1px solid #2a2a2a;
      border-radius: 6px;
      padding: 7px 10px;
      font-size: 12px;
      color: #888;
    }

    .phase-row { display: flex; align-items: center; gap: 8px; font-size: 13px; }
    .phase-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
    .phase-active      { background: #4caf50; box-shadow: 0 0 6px #4caf50; }
    .phase-slow_active { background: #4fc3f7; box-shadow: 0 0 6px #4fc3f7; }
    .phase-returning   { background: #ffb300; box-shadow: 0 0 6px #ffb300; }
    .phase-idle        { background: #444; }
    #phase-label    { font-weight: 600; }
    #velocity-label { font-size: 11px; color: #555; margin-left: auto; }

    .band-row {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
    }
    .dot {
      width: 10px; height: 10px;
      border-radius: 50%;
      flex-shrink: 0;
    }
    .dot-on  { background: #4caf50; box-shadow: 0 0 6px #4caf50; }
    .dot-off { background: #444; }

    .imu-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px 10px;
      font-size: 11px;
      color: #888;
    }
    .imu-grid span { color: #ccc; }

    .stats-row {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      color: #666;
    }
    .stats-row span { color: #aaa; }

    /* ── Exercise buttons ───────────────────────────── */
    .btn-row {
      display: flex;
      gap: 8px;
      padding: 12px 16px;
      background: #0d0d0d;
      border-top: 1px solid #1e1e1e;
      flex-wrap: wrap;
    }
    .ex-btn {
      flex: 1;
      min-width: 100px;
      padding: 12px 8px;
      border: none;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.15s, transform 0.1s;
      letter-spacing: 0.3px;
    }
    .ex-btn:active { transform: scale(0.96); }
    .btn-squat  { background: #0d3a4f; color: #4fc3f7; border: 1px solid #4fc3f7; }
    .btn-curl   { background: #3e2a00; color: #ffb74d; border: 1px solid #ffb74d; }
    .btn-raise  { background: #1b3620; color: #81c784; border: 1px solid #81c784; }
    .ex-btn:hover { opacity: 0.8; }

    /* ── Mobile responsive ──────────────────────────── */
    @media (max-width: 700px) {
      .main { flex-direction: column; }
      .video-panel { flex: none; height: 55vw; }
      .side-panel  { flex: none; border-left: none; border-top: 1px solid #1e1e1e; }
      #rep-count   { font-size: 52px; }
    }
  </style>
</head>
<body>
  <header>KV260 &nbsp;/&nbsp; <span>DPU Pose Monitor</span> &nbsp;/&nbsp; FPGA Inference</header>

  <div class="main">
    <!-- Live video -->
    <div class="video-panel">
      <img src="/video" alt="Live pose stream"/>
    </div>

    <!-- Side panel -->
    <div class="side-panel">

      <div class="card">
        <div class="card-label">Exercise</div>
        <div id="exercise-name" class="ex-none">—</div>
      </div>

      <div class="card">
        <div class="card-label">Reps</div>
        <div id="rep-count" style="font-size:72px;font-weight:800;color:#00ff00;line-height:1;text-align:center">0</div>
      </div>

      <div class="card">
        <div class="card-label">Phase</div>
        <div class="phase-row">
          <div class="phase-dot phase-idle" id="phase-dot"></div>
          <span id="phase-label">Idle</span>
          <span id="velocity-label">0 px/f</span>
        </div>
      </div>

      <div class="card">
        <div class="card-label">Instructions</div>
        <div id="instructions-list"><span class="instruction-card">—</span></div>
      </div>

      <div class="card" id="last-rep-card">
        <div class="card-label">Last Rep</div>
        <div id="last-rep-feedback" style="font-size:13px;color:#aaa;min-height:20px;line-height:1.4">
          Complete a rep for feedback
        </div>
      </div>

      <div class="card">
        <div class="card-label">This Rep</div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
          <div id="rec-dot" style="width:9px;height:9px;border-radius:50%;background:#333;flex-shrink:0;transition:background 0.2s"></div>
          <span id="rec-label" style="font-size:12px;color:#555">Idle</span>
          <span id="rec-issues" style="font-size:11px;color:#666;margin-left:auto"></span>
        </div>
        <div id="urgent-alerts" style="display:flex;flex-direction:column;gap:4px"></div>
      </div>

      <div class="card">
        <div class="card-label">Wearable Band</div>
        <div class="band-row">
          <div class="dot dot-off" id="band-dot"></div>
          <span id="band-label">Not connected</span>
        </div>
      </div>

      <div class="card">
        <div class="card-label">IMU Readings</div>
        <div class="imu-grid">
          <div>Accel X</div><div><span id="ax">—</span></div>
          <div>Accel Y</div><div><span id="ay">—</span></div>
          <div>Accel Z</div><div><span id="az">—</span></div>
          <div>Gyro X</div><div><span id="gx">—</span></div>
          <div>Gyro Y</div><div><span id="gy">—</span></div>
          <div>Gyro Z</div><div><span id="gz">—</span></div>
          <div>Temp</div><div><span id="temp">—</span></div>
          <div>Humidity</div><div><span id="hum">—</span></div>
        </div>
      </div>

      <div class="card">
        <div class="card-label">Performance</div>
        <div class="stats-row">
          <div>FPS</div><div><span id="fps">—</span></div>
        </div>
        <div class="stats-row" style="margin-top:4px">
          <div>Latency</div><div><span id="latency">—</span></div>
        </div>
      </div>

    </div><!-- /side-panel -->
  </div><!-- /main -->

  <div class="btn-row">
    <button class="ex-btn btn-squat" onclick="setExercise('squat')">&#9675; Squat</button>
    <button class="ex-btn btn-curl"  onclick="setExercise('bicep_curl')">&#9675; Bicep Curl</button>
    <button class="ex-btn btn-raise" onclick="setExercise('lateral_raise')">&#9675; Lateral Raise</button>
  </div>

  <script>
    const EXERCISE_LABELS = {
      squat: "Squat",
      bicep_curl: "Bicep Curl",
      lateral_raise: "Lateral Raise",
      none: "—"
    };
    const EX_CLASS = {
      squat: "ex-squat",
      bicep_curl: "ex-bicep_curl",
      lateral_raise: "ex-lateral_raise",
      none: "ex-none"
    };

    async function poll() {
      try {
        const r = await fetch("/status");
        const d = await r.json();

        // Exercise name
        const exKey = String(d.exercise);
        const nameEl = document.getElementById("exercise-name");
        nameEl.textContent = EXERCISE_LABELS[exKey] || exKey;
        nameEl.className = EX_CLASS[exKey] || "ex-none";

        // Rep count
        document.getElementById("rep-count").textContent = d.total_reps ?? d.reps ?? 0;

        // Phase indicator
        const phase = d.phase || "idle";
        const PHASE_LABELS = { idle: "Idle", active: "Active", slow_active: "Slow Active", returning: "Returning" };
        const dotEl = document.getElementById("phase-dot");
        dotEl.className = "phase-dot phase-" + phase;
        document.getElementById("phase-label").textContent = PHASE_LABELS[phase] || phase;
        document.getElementById("velocity-label").textContent =
          (d.velocity ?? 0).toFixed(1) + " px/f";

        // Instructions
        const instrEl = document.getElementById("instructions-list");
        if (d.instructions && d.instructions.length) {
          instrEl.innerHTML = d.instructions
            .map(t => `<div class="instruction-card">&#8227; ${t}</div>`)
            .join("");
        } else {
          instrEl.innerHTML = '<span class="instruction-card">—</span>';
        }

        // Last rep feedback card — respect 4s display timer
        const feedbackAge = d.feedback_age ?? 9999;
        const feedbackActive = feedbackAge < 4.0;
        const feedback = feedbackActive ? (d.last_rep_feedback || "") : "";
        document.getElementById("last-rep-feedback").textContent = feedback || "Complete a rep for feedback";
        document.getElementById("last-rep-feedback").style.color = feedback ? "#ccc" : "#444";

        // This rep — recording state + urgent alerts
        const isRecording = phase === "active" || phase === "slow_active";
        const recDot = document.getElementById("rec-dot");
        recDot.style.background = isRecording ? "#e53935" : "#333";
        recDot.style.boxShadow  = isRecording ? "0 0 6px #e53935" : "none";
        document.getElementById("rec-label").textContent = isRecording ? "Recording..." : "Idle";
        document.getElementById("rec-label").style.color  = isRecording ? "#e57373" : "#555";
        const bufCount = d.rep_buffer_count || 0;
        document.getElementById("rec-issues").textContent = isRecording ? (bufCount === 0 ? "0 issues" : bufCount + " issue" + (bufCount > 1 ? "s" : "") + " detected") : "";
        const urgentEl = document.getElementById("urgent-alerts");
        if (d.alerts && d.alerts.length) {
          urgentEl.innerHTML = d.alerts.map(a => `<div style="background:#3a0000;border:1px solid #c62828;border-radius:5px;padding:6px 9px;font-size:11px;color:#ff8a80">&#9888; ${a}</div>`).join("");
        } else { urgentEl.innerHTML = ""; }

        // Band status
        const connected = d.imu && d.imu.timestamp > 0;
        document.getElementById("band-dot").className = "dot " + (connected ? "dot-on" : "dot-off");
        document.getElementById("band-label").textContent = connected ? "Connected" : "Not connected";

        // IMU
        if (d.imu) {
          const a = d.imu.accel || {}, g = d.imu.gyro || {};
          document.getElementById("ax").textContent   = (a.x ?? 0).toFixed(3) + " g";
          document.getElementById("ay").textContent   = (a.y ?? 0).toFixed(3) + " g";
          document.getElementById("az").textContent   = (a.z ?? 0).toFixed(3) + " g";
          document.getElementById("gx").textContent   = (g.x ?? 0).toFixed(1) + " °/s";
          document.getElementById("gy").textContent   = (g.y ?? 0).toFixed(1) + " °/s";
          document.getElementById("gz").textContent   = (g.z ?? 0).toFixed(1) + " °/s";
          document.getElementById("temp").textContent = (d.imu.temp ?? 0).toFixed(1) + " °C";
          document.getElementById("hum").textContent  = (d.imu.humidity ?? 0).toFixed(1) + " %";
        }

        // Stats — served from status if present, otherwise blank
        if (d.fps != null)     document.getElementById("fps").textContent     = d.fps.toFixed(1);
        if (d.latency != null) document.getElementById("latency").textContent = d.latency.toFixed(1) + " ms";

      } catch(e) { /* board unreachable — keep last values */ }
    }

    async function setExercise(name) {
      await fetch("/exercise", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ exercise: name })
      });
    }

    poll();
    setInterval(poll, 500);
  </script>
</body>
</html>'''

@app.route('/exercise', methods=['POST'])
def post_exercise():
    data    = request.get_json(silent=True) or {}
    mapping = {"bicep_curl": 1, "squat": 2, "lateral_raise": 3}
    name    = data.get("exercise", "")
    if name in mapping:
        with exercise_lock:
            exercise_change["pending"] = True
            exercise_change["value"]   = mapping[name]
    with live_lock:
        cur = live_state["exercise"]
    return jsonify({"status": "ok", "exercise": cur})

flask_thread = threading.Thread(
    target=lambda: app.run(host='0.0.0.0', port=5000, use_reloader=False, debug=False),
    daemon=True
)
flask_thread.start()
_hostname = socket.gethostname()
_local_ip = socket.gethostbyname(_hostname)
print(f"[INFO] Video stream: http://{_local_ip}:5000/")
print(f"[INFO] Or try:       http://kria.local:5000/")

# ── Main loop (everything inline) ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret: continue

    frame_count += 1

    # Apply pending exercise change from /exercise endpoint
    with exercise_lock:
        if exercise_change["pending"]:
            exercise          = exercise_change["value"]
            rep_count         = 0
            rep_state         = "up" if exercise == 2 else "down"
            _ema_metric       = 0.0
            _last_rep_t       = 0.0
            rep_phase         = "idle"
            phase_frame_count = 0
            last_alert_time   = {}
            history_idx       = 0
            prev_kps          = [(0, 0)] * 14
            for _fi in range(5):
                kps_history[_fi] = [(0, 0)] * 14
            curl_started           = False
            squat_min_angle        = 180.0
            rep_alerts_buffer      = []
            last_rep_feedback      = ""
            rep_start_time         = 0.0
            squat_angles_this_rep  = []
            curl_angles_this_rep   = []
            raise_heights_this_rep = []
            total_reps             = 0
            rep_count              = 0
            feedback_display_timer = 0.0
            active_frames_this_rep = 0
            exercise_change["pending"] = False

    ti = time.time()

    # Stage 1: preprocess + conv DPU
    img = cv2.resize(frame, (INPUT_W, INPUT_H)).astype(np.float32)
    img -= MEAN
    np.copyto(in_data, np.clip(img * 0.5, -128, 127).astype(np.int8)[np.newaxis])
    job = conv_r.execute_async([in_data], [out_data])
    conv_r.wait(job)

    # Stage 2: CPU avgpool
    pool = out_data.astype(np.float32).mean(axis=(1, 2), keepdims=True)
    np.copyto(pool_int8, np.clip(pool * 8.0, -128, 127).astype(np.int8))

    # Stage 3: fc DPU
    job2 = fc_r.execute_async([pool_int8], [out_data2])
    fc_r.wait(job2)
    coords = out_data2.astype(np.float32).flatten() * 4.0

    ms = (time.time() - ti) * 1000

    # Decode keypoints
    kps = []
    for i in range(14):
        x = int(np.clip(coords[2*i]   * DISPLAY_W / INPUT_W, 0, DISPLAY_W - 1))
        y = int(np.clip(coords[2*i+1] * DISPLAY_H / INPUT_H, 0, DISPLAY_H - 1))
        kps.append((x, y))

    # Temporal smoothing — 5-frame history buffer
    kps_history[history_idx] = kps[:]
    history_idx = (history_idx + 1) % 5
    smoothed_kps = []
    for _k in range(14):
        _avg_x = int(sum(kps_history[_f][_k][0] for _f in range(5)) / 5)
        _avg_y = int(sum(kps_history[_f][_k][1] for _f in range(5)) / 5)
        smoothed_kps.append((_avg_x, _avg_y))
    for _i in range(14):
        joint_velocities[_i] = float(np.sqrt(
            (smoothed_kps[_i][0] - prev_kps[_i][0])**2 +
            (smoothed_kps[_i][1] - prev_kps[_i][1])**2))
    max_velocity = max(joint_velocities)
    prev_kps = smoothed_kps[:]
    kps = smoothed_kps

    # Phase detection — two-tier
    if max_velocity > ACTIVE_THRESHOLD:
        if rep_phase != "active":
            phase_frame_count = 0
        rep_phase = "active"
        phase_frame_count += 1
    elif max_velocity > SLOW_THRESHOLD:
        if rep_phase == "idle":
            phase_frame_count = 0
        rep_phase = "slow_active"
        phase_frame_count += 1
    else:
        if rep_phase in ("active", "slow_active") and phase_frame_count > 5:
            rep_phase = "returning"
        elif rep_phase == "returning":
            rep_phase = "idle"
            phase_frame_count = 0
        else:
            rep_phase = "idle"

    # FPS
    fc += 1
    elapsed = time.time() - t_fps
    if elapsed > 0.5:
        fps   = fc / elapsed
        fc    = 0
        t_fps = time.time()

    # ── Posture + rep counting (silent collection, per-rep feedback) ─────────────
    alerts       = []   # urgent IMU safety only
    instructions = []
    now = time.time()
    if len(kps) == 14:
        shoulder_width = abs(kps[0][0] - kps[3][0])
        _edge_kps = [k for k in [kps[0],kps[1],kps[2],kps[3],kps[4],kps[5]]
                     if k[0] < 5 or k[0] > DISPLAY_W-5 or k[1] < 5 or k[1] > DISPLAY_H-5]
        pose_reliable = shoulder_width > 80 and len(_edge_kps) == 0

        if not pose_reliable:
            instructions = ["Move closer / face camera directly"]

        elif exercise == 1:  # Bicep Curl ─────────────────────────────────────
            _ba = np.array(kps[0]) - np.array(kps[1])
            _bc = np.array(kps[2]) - np.array(kps[1])
            _ang = float(np.degrees(np.arccos(np.clip(
                np.dot(_ba, _bc) / (np.linalg.norm(_ba) * np.linalg.norm(_bc) + 1e-6), -1, 1))))
            if _ang < 120:
                curl_started = True
            instructions = ["Keep elbow pinned to side", "Full range of motion"]
            if rep_phase in ("active", "slow_active"):
                active_frames_this_rep += 1
                curl_angles_this_rep.append(_ang)
                if kps[1][0] > kps[0][0] + 40 and "Pin your elbow to your body" not in rep_alerts_buffer:
                    rep_alerts_buffer.append("Pin your elbow to your body")
                    _at = "Pin your elbow to your body"
                    _now_t = time.time()
                    if _now_t - last_alert_time.get("VIB_" + _at, 0) > 2.0:
                        with vib_lock:
                            vibrate_command["vibrate"] = True; vibrate_command["pattern"] = "long"
                            vibrate_command["reason"] = _at; vibrate_command["severity"] = "critical"
                        last_alert_time["VIB_" + _at] = _now_t
                if curl_started and _ang > 150 and "Curl more - squeeze the bicep" not in rep_alerts_buffer:
                    rep_alerts_buffer.append("Curl more - squeeze the bicep")
                if rep_phase == "active" and max_velocity > 30 and "Slow down - control the curl" not in rep_alerts_buffer:
                    rep_alerts_buffer.append("Slow down - control the curl")
            _ema_metric = 0.35 * _ang + 0.65 * _ema_metric
            if rep_state == "down" and _ema_metric < 100:
                rep_state = "up"
            elif rep_state == "up" and _ema_metric > 130 and (time.time() - _last_rep_t) > 0.5:
                rep_state = "down"; total_reps += 1; rep_count = total_reps; _last_rep_t = time.time()
                curl_started = False
                _srank = {"critical": 3, "moderate": 2, "minor": 1}
                if rep_alerts_buffer:
                    _worst = max(rep_alerts_buffer, key=lambda a: _srank.get(ALERT_SEVERITY.get(a, "minor"), 1))
                    last_rep_feedback = f"Rep {rep_count} - {_worst}"
                else:
                    last_rep_feedback = f"Rep {rep_count} - great form!"
                if curl_angles_this_rep:
                    last_rep_feedback += f" (peak: {min(curl_angles_this_rep):.0f}deg)"
                feedback_display_timer = time.time()
                _non_crit = [a for a in rep_alerts_buffer if a not in CRITICAL_IMMEDIATE]
                with vib_lock:
                    if _non_crit:
                        vibrate_command["vibrate"] = True; vibrate_command["pattern"] = "short"
                        vibrate_command["reason"] = last_rep_feedback; vibrate_command["severity"] = "moderate"
                    else:
                        vibrate_command["vibrate"] = False; vibrate_command["pattern"] = "none"
                        vibrate_command["severity"] = "none"
                rep_alerts_buffer = []; curl_angles_this_rep = []; rep_start_time = time.time()
                active_frames_this_rep = 0

        elif exercise == 2:  # Squat ───────────────────────────────────────────
            _ba = np.array(kps[6]) - np.array(kps[7])
            _bc = np.array(kps[8]) - np.array(kps[7])
            _ang = float(np.degrees(np.arccos(np.clip(
                np.dot(_ba, _bc) / (np.linalg.norm(_ba) * np.linalg.norm(_bc) + 1e-6), -1, 1))))
            if rep_phase in ("active", "slow_active"):
                active_frames_this_rep += 1
                squat_angles_this_rep.append(_ang)
            instructions = ["Feet shoulder width apart", "Keep chest up"]
            if rep_phase in ("active", "slow_active"):
                if kps[7][0] < kps[6][0] - 25 and "Push right knee outward" not in rep_alerts_buffer:
                    rep_alerts_buffer.append("Push right knee outward")
                    _at = "Push right knee outward"
                    _now_t = time.time()
                    if _now_t - last_alert_time.get("VIB_" + _at, 0) > 2.0:
                        with vib_lock:
                            vibrate_command["vibrate"] = True; vibrate_command["pattern"] = "long"
                            vibrate_command["reason"] = _at; vibrate_command["severity"] = "critical"
                        last_alert_time["VIB_" + _at] = _now_t
                if kps[10][0] > kps[9][0] + 25 and "Push left knee outward" not in rep_alerts_buffer:
                    rep_alerts_buffer.append("Push left knee outward")
                    _at = "Push left knee outward"
                    _now_t = time.time()
                    if _now_t - last_alert_time.get("VIB_" + _at, 0) > 2.0:
                        with vib_lock:
                            vibrate_command["vibrate"] = True; vibrate_command["pattern"] = "long"
                            vibrate_command["reason"] = _at; vibrate_command["severity"] = "critical"
                        last_alert_time["VIB_" + _at] = _now_t
                if kps[13][1] > kps[6][1] + 20 and "Keep chest up - back rounding" not in rep_alerts_buffer:
                    rep_alerts_buffer.append("Keep chest up - back rounding")
                    _at = "Keep chest up - back rounding"
                    _now_t = time.time()
                    if _now_t - last_alert_time.get("VIB_" + _at, 0) > 2.0:
                        with vib_lock:
                            vibrate_command["vibrate"] = True; vibrate_command["pattern"] = "long"
                            vibrate_command["reason"] = _at; vibrate_command["severity"] = "critical"
                        last_alert_time["VIB_" + _at] = _now_t
                if rep_phase == "active" and max_velocity > 30 and "Slow down - control the squat" not in rep_alerts_buffer:
                    rep_alerts_buffer.append("Slow down - control the squat")
            _ema_metric = 0.35 * _ang + 0.65 * _ema_metric
            if rep_state == "up" and _ema_metric < 130:
                rep_state = "down"
            elif rep_state == "down" and _ema_metric > 150 and (time.time() - _last_rep_t) > 0.5:
                if squat_angles_this_rep and min(squat_angles_this_rep) > 120 and "Go deeper - break parallel" not in rep_alerts_buffer:
                    rep_alerts_buffer.append("Go deeper - break parallel")
                rep_state = "up"; total_reps += 1; rep_count = total_reps; _last_rep_t = time.time()
                _srank = {"critical": 3, "moderate": 2, "minor": 1}
                if rep_alerts_buffer:
                    _worst = max(rep_alerts_buffer, key=lambda a: _srank.get(ALERT_SEVERITY.get(a, "minor"), 1))
                    last_rep_feedback = f"Rep {rep_count} - {_worst}"
                else:
                    last_rep_feedback = f"Rep {rep_count} - great form!"
                if squat_angles_this_rep:
                    last_rep_feedback += f" (depth: {min(squat_angles_this_rep):.0f}deg)"
                feedback_display_timer = time.time()
                _non_crit = [a for a in rep_alerts_buffer if a not in CRITICAL_IMMEDIATE]
                with vib_lock:
                    if _non_crit:
                        vibrate_command["vibrate"] = True; vibrate_command["pattern"] = "short"
                        vibrate_command["reason"] = last_rep_feedback; vibrate_command["severity"] = "moderate"
                    else:
                        vibrate_command["vibrate"] = False; vibrate_command["pattern"] = "none"
                        vibrate_command["severity"] = "none"
                rep_alerts_buffer = []; squat_angles_this_rep = []; rep_start_time = time.time()
                active_frames_this_rep = 0

        elif exercise == 3:  # Lateral Raise ──────────────────────────────────
            instructions = ["Raise arms to shoulder height", "Slight bend in elbow"]
            if rep_phase in ("active", "slow_active"):
                active_frames_this_rep += 1
                raise_heights_this_rep.append(kps[1][1] - kps[0][1])
                if kps[1][1] > kps[0][1] + 40 and "Raise higher - not at shoulder level yet" not in rep_alerts_buffer:
                    rep_alerts_buffer.append("Raise higher - not at shoulder level yet")
                if kps[1][1] < kps[0][1] - 50 and "Too high - stop at shoulder level" not in rep_alerts_buffer:
                    rep_alerts_buffer.append("Too high - stop at shoulder level")
                _left_raise  = kps[0][1] - kps[4][1]
                _right_raise = kps[0][1] - kps[1][1]
                if abs(_left_raise - _right_raise) > 40 and "Raise both arms evenly" not in rep_alerts_buffer:
                    rep_alerts_buffer.append("Raise both arms evenly")
                if rep_phase == "active" and max_velocity > 35 and "Slow down - control the movement" not in rep_alerts_buffer:
                    rep_alerts_buffer.append("Slow down - control the movement")
            _ema_metric = 0.35 * (kps[1][1] - kps[0][1]) + 0.65 * _ema_metric
            if rep_state == "down" and _ema_metric < 30:
                rep_state = "up"
            elif rep_state == "up" and _ema_metric > 50 and (time.time() - _last_rep_t) > 0.5:
                rep_state = "down"; total_reps += 1; rep_count = total_reps; _last_rep_t = time.time()
                _srank = {"critical": 3, "moderate": 2, "minor": 1}
                if rep_alerts_buffer:
                    _worst = max(rep_alerts_buffer, key=lambda a: _srank.get(ALERT_SEVERITY.get(a, "minor"), 1))
                    last_rep_feedback = f"Rep {rep_count} - {_worst}"
                else:
                    last_rep_feedback = f"Rep {rep_count} - great form!"
                if raise_heights_this_rep:
                    last_rep_feedback += f" (height: {'good' if min(raise_heights_this_rep) < 20 else 'low'})"
                feedback_display_timer = time.time()
                _non_crit = [a for a in rep_alerts_buffer if a not in CRITICAL_IMMEDIATE]
                with vib_lock:
                    if _non_crit:
                        vibrate_command["vibrate"] = True; vibrate_command["pattern"] = "short"
                        vibrate_command["reason"] = last_rep_feedback; vibrate_command["severity"] = "moderate"
                    else:
                        vibrate_command["vibrate"] = False; vibrate_command["pattern"] = "none"
                        vibrate_command["severity"] = "none"
                rep_alerts_buffer = []; raise_heights_this_rep = []; rep_start_time = time.time()
                active_frames_this_rep = 0

    # IMU fusion — urgent safety alerts only (fire immediately, not per-rep)
    with imu_lock:
        _ax     = latest_imu["accel"]["x"]
        _imu_ay = latest_imu["accel"]["y"]
        _gz     = latest_imu["gyro"]["z"]
        _tp     = latest_imu["temp"]
        _imu_ts = latest_imu["timestamp"]
    if _imu_ts > 0:
        if _tp > 38.5:
            _at = "High body temp - rest"
            if now - last_alert_time.get(_at, 0) > ALERT_COOLDOWN:
                alerts.append(_at); last_alert_time[_at] = now
        if abs(_gz) > 200:
            _at = "Moving too fast"
            if now - last_alert_time.get(_at, 0) > ALERT_COOLDOWN:
                alerts.append(_at); last_alert_time[_at] = now

    # Publish live state for /status endpoint
    with live_lock:
        live_state["exercise"]          = exercise
        live_state["rep_count"]         = rep_count
        live_state["alerts"]            = alerts
        live_state["instructions"]      = instructions
        live_state["phase"]             = rep_phase
        live_state["velocity"]          = round(max_velocity, 1)
        live_state["last_rep_feedback"] = last_rep_feedback
        live_state["rep_buffer_count"]  = len(rep_alerts_buffer)
        live_state["total_reps"]        = total_reps
        live_state["feedback_age"]      = round(time.time() - feedback_display_timer, 1)

    # ── Draw (inline, reuse display buffer) ───────────────────────────────────
    cv2.resize(frame, (DISPLAY_W, DISPLAY_H), dst=display)
    for a, b in SKELETON:
        cv2.line(display, kps[a], kps[b], (0, 255, 0), 2)
    for idx, (x, y) in enumerate(kps):
        cv2.circle(display, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(display, NAMES[idx], (x+4, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

    # Top bar (90px): exercise name left, rep counts right
    cv2.rectangle(display, (0, 0), (DISPLAY_W, 90), (0, 0, 0), -1)
    cv2.putText(display, EXERCISE_NAMES[exercise],
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(display, str(total_reps),
                (DISPLAY_W - 120, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 4)

    # Info line (just below top bar)
    _band_str = "Band: CONNECTED"     if _imu_ts > 0 else "Band: NOT CONNECTED"
    _band_col = (0, 220, 0)           if _imu_ts > 0 else (0, 220, 220)
    cv2.putText(display, f"FPS:{fps:.1f} {ms:.1f}ms  ACC:{_ax:.1f},{_imu_ay:.1f}  GYRO:{_gz:.0f}  T:{_tp:.1f}C",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(display, _band_str,
                (430, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _band_col, 1)

    # Phase + velocity indicator
    cv2.putText(display, f"Phase: {rep_phase}  Vel: {max_velocity:.0f}px/f",
                (10, DISPLAY_H - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

    # Instructions bar — always visible, subtle grey
    cv2.rectangle(display, (0, DISPLAY_H - 90), (DISPLAY_W, DISPLAY_H - 55), (40, 40, 40), -1)
    if instructions:
        cv2.putText(display, " | ".join(instructions), (10, DISPLAY_H - 63),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    # Feedback bar — timed, resets after FEEDBACK_DISPLAY_DURATION seconds
    _feedback_active = (time.time() - feedback_display_timer) < FEEDBACK_DISPLAY_DURATION
    if alerts:
        cv2.rectangle(display, (0, DISPLAY_H - 55), (DISPLAY_W, DISPLAY_H), (0, 0, 200), -1)
        cv2.putText(display, f"! {alerts[0]}", (10, DISPLAY_H - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif _feedback_active and last_rep_feedback:
        cv2.rectangle(display, (0, DISPLAY_H - 55), (DISPLAY_W, DISPLAY_H), (20, 60, 20), -1)
        cv2.putText(display, last_rep_feedback, (10, DISPLAY_H - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.rectangle(display, (0, DISPLAY_H - 55), (DISPLAY_W, DISPLAY_H), (30, 30, 30), -1)
        cv2.putText(display, "Complete a rep for feedback", (10, DISPLAY_H - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)
    # Recording dot — active phase indicator
    if rep_phase in ("active", "slow_active"):
        cv2.circle(display, (DISPLAY_W - 20, DISPLAY_H - 70), 8, (0, 0, 255), -1)
        cv2.putText(display, "recording", (DISPLAY_W - 90, DISPLAY_H - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 100, 100), 1)

    # Push annotated frame to MJPEG stream
    with frame_lock:
        output_frame = display.copy()

    if frame_count % 100 == 0:
        print(f"[SUMMARY] Exercise:{EXERCISE_NAMES[exercise]} Reps:{total_reps} LastFeedback:{last_rep_feedback[:50]} Band:{'connected' if _imu_ts > 0 else 'disconnected'} Latency:{ms:.1f}ms FPS:{fps:.1f}")

    _key = cv2.waitKey(1) & 0xFF
    if _key == ord('q'):
        break
    elif _key == ord('r'):
        rep_count = 0; total_reps = 0
        rep_state = "up" if exercise == 2 else "down"
        _ema_metric = 0.0; _last_rep_t = 0.0
        rep_alerts_buffer = []; curl_angles_this_rep = []
        squat_angles_this_rep = []; raise_heights_this_rep = []
        last_rep_feedback = ""
        feedback_display_timer = 0.0; active_frames_this_rep = 0

cap.release()
print(f"[INFO] Done. Final reps: {rep_count}")
