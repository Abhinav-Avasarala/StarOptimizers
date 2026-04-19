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
from flask import Flask, request, jsonify
import threading

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
vibrate_command = {"vibrate": False, "pattern": "none", "reason": ""}
imu_lock = threading.Lock()
vib_lock = threading.Lock()
alerts   = []   # pre-init so /status works before main loop runs
live_state      = {"exercise": "none", "rep_count": 0, "alerts": []}
live_lock       = threading.Lock()
exercise_change = {"pending": False, "value": 0}
exercise_lock   = threading.Lock()

# ── Exercise selection screen ─────────────────────────────────────────────────
_sel = np.zeros((320, 520, 3), dtype=np.uint8)
_sel[:] = (30, 30, 30)
cv2.putText(_sel, "Select Exercise",      ( 75,  60), cv2.FONT_HERSHEY_SIMPLEX, 1.1,  (0, 255, 255), 2)
cv2.putText(_sel, "1  -  Bicep Curl",    (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9,  (255, 255, 255), 2)
cv2.putText(_sel, "2  -  Squat",         (100, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.9,  (255, 255, 255), 2)
cv2.putText(_sel, "3  -  Lateral Raise", (100, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9,  (255, 255, 255), 2)
cv2.putText(_sel, "Press key to begin",  ( 90, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 100),  2)
cv2.imshow("Select Exercise", _sel)

exercise = 0
while exercise == 0:
    _k = cv2.waitKey(100) & 0xFF
    if   _k == ord('1'): exercise = 1
    elif _k == ord('2'): exercise = 2
    elif _k == ord('3'): exercise = 3
cv2.destroyWindow("Select Exercise")

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
        "imu":      imu_snap,
        "vibrate":  vib_snap,
        "exercise": state_snap["exercise"],
        "reps":     state_snap["rep_count"],
        "alerts":   state_snap["alerts"],
    })

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
print("[INFO] Sensor server running on port 5000")

# ── Main loop (everything inline) ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret: continue

    frame_count += 1

    # Apply pending exercise change from /exercise endpoint
    with exercise_lock:
        if exercise_change["pending"]:
            exercise    = exercise_change["value"]
            rep_count   = 0
            rep_state   = "up" if exercise == 2 else "down"
            _ema_metric = 0.0
            _last_rep_t = 0.0
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

    # FPS
    fc += 1
    elapsed = time.time() - t_fps
    if elapsed > 0.5:
        fps   = fc / elapsed
        fc    = 0
        t_fps = time.time()

    # ── Posture + rep counting (fully inline, no function calls) ──────────────
    alerts = []
    if len(kps) == 14:

        if exercise == 1:  # Bicep Curl ───────────────────────────────────────
            _ba = np.array(kps[0]) - np.array(kps[1])
            _bc = np.array(kps[2]) - np.array(kps[1])
            _ang = float(np.degrees(np.arccos(np.clip(
                np.dot(_ba, _bc) / (np.linalg.norm(_ba) * np.linalg.norm(_bc) + 1e-6), -1, 1))))
            if abs(kps[1][0] - kps[0][0]) > 40:
                alerts.append("Keep elbow pinned to side")
            if _ang > 150:
                alerts.append("Full range of motion")
            # Rep: angle-based with EMA smoothing + cooldown (position-independent)
            _ema_metric = 0.35 * _ang + 0.65 * _ema_metric
            if rep_state == "down" and _ema_metric < 70:
                rep_state = "up"
            elif rep_state == "up" and _ema_metric > 140 and (time.time() - _last_rep_t) > 0.5:
                rep_state = "down"
                rep_count += 1
                _last_rep_t = time.time()

        elif exercise == 2:  # Squat ──────────────────────────────────────────
            _ba = np.array(kps[6]) - np.array(kps[7])
            _bc = np.array(kps[8]) - np.array(kps[7])
            _ang = float(np.degrees(np.arccos(np.clip(
                np.dot(_ba, _bc) / (np.linalg.norm(_ba) * np.linalg.norm(_bc) + 1e-6), -1, 1))))
            if _ang > 120:
                alerts.append("Squat deeper")
            if kps[7][0] < kps[6][0] - 25:
                alerts.append("R knee caving in")
            if kps[10][0] > kps[9][0] + 25:
                alerts.append("L knee caving in")
            if kps[13][1] > kps[6][1] + 20:
                alerts.append("Keep chest up")
            # Rep: knee angle-based with EMA smoothing + cooldown (position-independent)
            _ema_metric = 0.35 * _ang + 0.65 * _ema_metric
            if rep_state == "up" and _ema_metric < 110:
                rep_state = "down"
            elif rep_state == "down" and _ema_metric > 155 and (time.time() - _last_rep_t) > 0.5:
                rep_state = "up"
                rep_count += 1
                _last_rep_t = time.time()

        elif exercise == 3:  # Lateral Raise ───────────────────────────────────
            _ba = np.array(kps[0]) - np.array(kps[1])
            _bc = np.array(kps[2]) - np.array(kps[1])
            _ang = float(np.degrees(np.arccos(np.clip(
                np.dot(_ba, _bc) / (np.linalg.norm(_ba) * np.linalg.norm(_bc) + 1e-6), -1, 1))))
            if kps[1][1] > kps[0][1] + 30:
                alerts.append("Raise to shoulder height")
            if kps[1][1] < kps[0][1] - 40:
                alerts.append("Don't raise above shoulder")
            if _ang > 170:
                alerts.append("Keep slight bend in elbow")
            # Rep: elbow-shoulder y gap, EMA smoothed + cooldown (position-independent)
            # positive = elbow below shoulder, ~0 or negative = elbow at/above shoulder
            _ema_metric = 0.35 * (kps[1][1] - kps[0][1]) + 0.65 * _ema_metric
            if rep_state == "down" and _ema_metric < 15:
                rep_state = "up"
            elif rep_state == "up" and _ema_metric > 55 and (time.time() - _last_rep_t) > 0.5:
                rep_state = "down"
                rep_count += 1
                _last_rep_t = time.time()

    # IMU fusion — append sensor-derived alerts
    with imu_lock:
        _ax     = latest_imu["accel"]["x"]
        _imu_ay = latest_imu["accel"]["y"]
        _gz     = latest_imu["gyro"]["z"]
        _tp     = latest_imu["temp"]
        _imu_ts = latest_imu["timestamp"]
    if _ax > 0.4:
        alerts.append("Excessive forward lean")
    if abs(_gz) > 150:
        alerts.append("Moving too fast")
    if _tp > 38.5:
        alerts.append("High body temp — rest")

    # Vibration — fires after pose + IMU alerts are both collected
    if alerts:
        with vib_lock:
            vibrate_command["vibrate"] = True
            vibrate_command["pattern"] = "long" if len(alerts) > 1 else "short"
            vibrate_command["reason"]  = alerts[0]

    # Publish live state for /status endpoint
    with live_lock:
        live_state["exercise"]  = exercise
        live_state["rep_count"] = rep_count
        live_state["alerts"]    = alerts

    # ── Draw (inline, reuse display buffer) ───────────────────────────────────
    cv2.resize(frame, (DISPLAY_W, DISPLAY_H), dst=display)
    for a, b in SKELETON:
        cv2.line(display, kps[a], kps[b], (0, 255, 0), 2)
    for idx, (x, y) in enumerate(kps):
        cv2.circle(display, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(display, NAMES[idx], (x+4, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

    # Top bar (70px): exercise name left, rep count large green right
    cv2.rectangle(display, (0, 0), (DISPLAY_W, 70), (0, 0, 0), -1)
    cv2.putText(display, EXERCISE_NAMES[exercise],
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(display, "REPS",
                (DISPLAY_W - 85, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1)
    cv2.putText(display, str(rep_count),
                (DISPLAY_W - 80, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 220, 0), 3)

    # Info line: FPS / latency / IMU values + band connection status
    _band_str = "Band: CONNECTED"     if _imu_ts > 0 else "Band: NOT CONNECTED"
    _band_col = (0, 220, 0)           if _imu_ts > 0 else (0, 220, 220)
    cv2.putText(display, f"FPS:{fps:.1f} {ms:.1f}ms  ACC:{_ax:.1f},{_imu_ay:.1f}  GYRO:{_gz:.0f}  T:{_tp:.1f}C",
                (10, DISPLAY_H - 67), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(display, _band_str,
                (430, DISPLAY_H - 67), cv2.FONT_HERSHEY_SIMPLEX, 0.35, _band_col, 1)

    # Bottom bar (60px): green = good form, red = first alert
    if alerts:
        cv2.rectangle(display, (0, DISPLAY_H - 60), (DISPLAY_W, DISPLAY_H), (0, 0, 180), -1)
        cv2.putText(display, alerts[0],
                    (10, DISPLAY_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.rectangle(display, (0, DISPLAY_H - 60), (DISPLAY_W, DISPLAY_H), (0, 140, 0), -1)
        cv2.putText(display, "GOOD FORM",
                    (10, DISPLAY_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("KV260 DPU Pose", display)

    if frame_count % 100 == 0:
        print(f"[SUMMARY] Exercise:{EXERCISE_NAMES[exercise]} Reps:{rep_count} Alerts:{alerts} Band:{'connected' if _imu_ts > 0 else 'disconnected'} Latency:{ms:.1f}ms FPS:{fps:.1f}")

    _key = cv2.waitKey(1) & 0xFF
    if _key == ord('q'):
        break
    elif _key == ord('r'):
        rep_count = 0
        rep_state = "up" if exercise == 2 else "down"
        _ema_metric = 0.0
        _last_rep_t = 0.0

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Done. Final reps: {rep_count}")
