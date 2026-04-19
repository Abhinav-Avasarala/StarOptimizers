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

# ── Main loop (everything inline) ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret: continue

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
            # Rep: avg hip y > 300 = bottom, < 220 = standing
            _hy = (kps[6][1] + kps[9][1]) / 2.0
            if rep_state == "up" and _hy > 300:
                rep_state = "down"
            elif rep_state == "down" and _hy < 220:
                rep_state = "up"
                rep_count += 1

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

    # ── Draw (inline, reuse display buffer) ───────────────────────────────────
    cv2.resize(frame, (DISPLAY_W, DISPLAY_H), dst=display)
    for a, b in SKELETON:
        cv2.line(display, kps[a], kps[b], (0, 255, 0), 2)
    for idx, (x, y) in enumerate(kps):
        cv2.circle(display, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(display, NAMES[idx], (x+4, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

    # Top bar: exercise name + rep count
    cv2.rectangle(display, (0, 0), (DISPLAY_W, 48), (0, 0, 0), -1)
    cv2.putText(display, f"{EXERCISE_NAMES[exercise]}   Reps: {rep_count}",
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # FPS / latency line
    cv2.putText(display, f"FPS:{fps:.1f}  {ms:.1f}ms  DPUCZDX8G-B4096",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Bottom bar: form status
    if alerts:
        cv2.rectangle(display, (0, DISPLAY_H - 52), (DISPLAY_W, DISPLAY_H), (0, 0, 180), -1)
        cv2.putText(display, alerts[0],
                    (10, DISPLAY_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    else:
        cv2.rectangle(display, (0, DISPLAY_H - 52), (DISPLAY_W, DISPLAY_H), (0, 140, 0), -1)
        cv2.putText(display, "GOOD FORM",
                    (10, DISPLAY_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("KV260 DPU Pose", display)

    if fc % 30 == 1:
        print(f"[PERF] {ms:.1f}ms | FPS:{fps:.1f} | reps:{rep_count} | alerts:{alerts}")

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
