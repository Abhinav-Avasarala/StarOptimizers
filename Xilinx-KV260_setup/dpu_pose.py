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

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
assert cap.isOpened(), "Camera not found"
print(f"[INFO] Camera: {int(cap.get(3))}x{int(cap.get(4))}")
print("[INFO] Running — press 'q' to quit\n")

fps    = 0.0
fc     = 0
t0     = time.time()
t_fps  = time.time()

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
    for k in range(14):
        x = int(np.clip(coords[2*k]   * DISPLAY_W / INPUT_W, 0, DISPLAY_W - 1))
        y = int(np.clip(coords[2*k+1] * DISPLAY_H / INPUT_H, 0, DISPLAY_H - 1))
        kps.append((x, y))

    # FPS
    fc += 1
    elapsed = time.time() - t_fps
    if elapsed > 0.5:
        fps   = fc / elapsed
        fc    = 0
        t_fps = time.time()

    # Posture analysis (inline)
    alerts = []
    if len(kps) == 14:
        def _ang(a, b, c):
            ba  = np.array(a) - np.array(b)
            bc  = np.array(c) - np.array(b)
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        if _ang(kps[6], kps[7],  kps[8])  > 175: alerts.append("WARNING: R knee hyperextension")
        if _ang(kps[9], kps[10], kps[11]) > 175: alerts.append("WARNING: L knee hyperextension")
        if _ang(kps[6], kps[7],  kps[8])  <  70: alerts.append("WARNING: Deep squat - check alignment")
        if _ang(kps[0], kps[1],  kps[2])  > 170: alerts.append("WARNING: R elbow locked out")
        if _ang(kps[3], kps[4],  kps[5])  > 170: alerts.append("WARNING: L elbow locked out")
        if abs(kps[0][1] - kps[3][1])     >  30: alerts.append("WARNING: Uneven shoulders")
        if abs(kps[6][1] - kps[9][1])     >  30: alerts.append("WARNING: Uneven hips")

    # Draw (inline, reuse display buffer)
    cv2.resize(frame, (DISPLAY_W, DISPLAY_H), dst=display)
    for a, b in SKELETON:
        cv2.line(display, kps[a], kps[b], (0, 255, 0), 2)
    for idx, (x, y) in enumerate(kps):
        cv2.circle(display, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(display, NAMES[idx], (x+4, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
    cv2.putText(display, f"FPS:{fps:.1f}  {ms:.1f}ms  DPUCZDX8G-B4096",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    for i, a in enumerate(alerts):
        cv2.putText(display, a, (10, DISPLAY_H - 15 - i*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 50, 255), 2)

    cv2.imshow("KV260 DPU Pose", display)

    if fc % 30 == 1:
        print(f"[PERF] {ms:.1f}ms | FPS:{fps:.1f} | alerts:{alerts}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Done.")
