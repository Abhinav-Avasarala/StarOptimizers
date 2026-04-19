# 🏋️ Workout Injury Prevention System — AMD KV260 FPGA

Real-time human pose estimation running **entirely on FPGA fabric** using the AMD KV260's DPU. Detects 14 body joint keypoints from a live camera feed and provides instant posture analysis and injury prevention alerts — with **~20ms latency** vs 1000ms+ cloud inference.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Hardware & Software Stack](#hardware--software-stack)
- [Architecture](#architecture)
- [Model Details](#model-details)
- [Setup Guide](#setup-guide)
- [Running the Pipeline](#running-the-pipeline)
- [Keypoint Map](#keypoint-map)
- [Injury Detection Logic](#injury-detection-logic)
- [Performance](#performance)
- [Known Issues & Debugging Notes](#known-issues--debugging-notes)
- [What's Next](#whats-next)

---

## Project Overview

### Before (Cloud Pipeline)
```
KV260 Camera → JPEG encode → HTTP POST to DigitalOcean GPU →
Flask + MediaPipe → JSON keypoints → Draw skeleton  ← ~1000ms latency
```

### After (FPGA Pipeline)
```
KV260 Camera → resize 128×224 → DPU Conv subgraph →
CPU avgpool → DPU FC subgraph → decode coords → Draw skeleton  ← ~20ms latency
```

**50× latency improvement. Zero cloud dependency. Entire inference on FPGA fabric.**

---

## Hardware & Software Stack

### Hardware
- **Board:** AMD KV260 Kria Vision AI Starter Kit
- **DPU:** DPUCZDX8G, ISA1, B4096 config
- **DPU Fingerprint:** `0x101000016010407`
- **Camera:** Logitech webcam (1920×1080 native, USB)

### Software
| Component | Version |
|---|---|
| OS | Ubuntu 22.04 (aarch64) |
| XRT (Xilinx Runtime) | 2.13.479 |
| VART | 2.5.0 |
| XIR | 2022-07-20 build |
| OpenCV | System (via apt) |
| Python | 3.10 |
| FPGA Overlay | `kv260-benchmark-b4096` |

### Installed Packages
```bash
sudo apt install vitis-ai-runtime libxir1 libunilog1
sudo apt install xlnx-firmware-kv260-benchmark-b4096
```

---

## Architecture

### Two-Stage DPU Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        KV260 FPGA                               │
│                                                                 │
│  Camera Frame (1920×1080)                                       │
│       │                                                         │
│       ▼                                                         │
│  cv2.resize → 128×224                                           │
│  Mean subtract [104, 117, 123]                                  │
│  Quantize ×0.5 → INT8          (fix_point = -1)                │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────┐                           │
│  │  DPU: subgraph_conv1/7x7_s2    │  ← FPGA Fabric            │
│  │  Inception/GoogLeNet backbone   │                           │
│  │  IN:  [1, 224, 128, 3]  INT8   │                           │
│  │  OUT: [1, 7,   4,  184] INT8   │                           │
│  └─────────────────────────────────┘                           │
│       │                                                         │
│       ▼                                                         │
│  CPU: Global Average Pool                                       │
│  mean(axis=(H,W)) → [1, 1, 1, 184]                            │
│  Quantize ×8 → INT8            (fix_point = 3)                │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────┐                           │
│  │  DPU: subgraph_fc_coordinate   │  ← FPGA Fabric            │
│  │  Fully-connected regression     │                           │
│  │  IN:  [1, 1, 1, 184]  INT8    │                           │
│  │  OUT: [1, 28]          INT8   │                           │
│  └─────────────────────────────────┘                           │
│       │                                                         │
│       ▼                                                         │
│  Dequantize ×4 → float          (fix_point = -2)              │
│  Decode 28 values → 14 (x,y) keypoints                        │
│  Scale to display resolution (640×480)                         │
│       │                                                         │
│       ▼                                                         │
│  Posture analysis → injury alerts                              │
│  Draw skeleton overlay → cv2.imshow                            │
└─────────────────────────────────────────────────────────────────┘
```

### Quantization Scale Reference
| Stage | fix_point | Scale formula | Direction |
|---|---|---|---|
| Conv input | -1 | float × 2^(-1) = ×0.5 | float→INT8 |
| Conv output | 0 | INT8 × 1.0 | INT8→float |
| FC input | 3 | float × 2^3 = ×8 | float→INT8 |
| FC output | -2 | INT8 × 2^2 = ×4 | INT8→float |

---

## Model Details

**Model:** `sp_net` from Vitis AI Model Zoo r2.5.0
**Target:** `zcu102 / zcu104 / kv260`
**Type:** CNN — Inception/GoogLeNet-style backbone with FC regression head
**Task:** Single-person 2D body pose estimation via direct coordinate regression (not heatmap-based)

### Why sp_net
- Pre-compiled `.xmodel` with fingerprint exactly matching KV260 B4096 DPU
- Direct coordinate regression — no heatmap post-processing needed
- Smallest full-body model in the Vitis AI zoo
- INT8 quantized — runs natively on DPU without CPU fallback

### Model Limitations
- 14 joints only (no fingers, no face landmarks)
- No per-joint confidence scores — occluded joints still output coordinates
- Trained on general pose data, not gym-specific footage
- Single person only

---

## Setup Guide

### 1. Load the DPU Overlay

The KV260 auto-loads `k26-starter-kits` on boot which occupies the only PL slot. You must unload it first:

```bash
# Check current state
sudo xmutil listapps

# Unload whatever is active (k26-starter-kits loads on boot)
sudo xmutil unloadapp

# Load the DPU overlay
sudo xmutil loadapp kv260-benchmark-b4096

# Verify DRM render node appeared
ls /dev/dri/renderD128
```

> ⚠️ **Important:** Always run `sudo xmutil unloadapp` before `loadapp`. Loading without unloading first returns `load Error: -1`.

### 2. Verify DPU is alive

```bash
# Check DPU fingerprint
sudo show_dpu
# Expected: fingerprint = 0x101000016010407, DPUCZDX8G:DPUCZDX8G_1

# Verify Python bindings
python3 -c "import vart; import xir; print('VART OK')"
```

### 3. Download sp_net model

```bash
wget "https://www.xilinx.com/bin/public/openDownload?filename=sp_net-zcu102_zcu104_kv260-r2.5.0.tar.gz" \
     -O sp_net.tar.gz
tar -xzf sp_net.tar.gz
# Model will be at: ~/sp_net/sp_net.xmodel
```

### 4. Verify model fingerprint matches board

```bash
python3 -c "
import xir
graph = xir.Graph.deserialize('/home/ubuntu/sp_net/sp_net.xmodel')
for s in graph.get_root_subgraph().get_children():
    if s.has_attr('device') and s.get_attr('device') == 'DPU':
        print(s.get_name(), hex(s.get_attr('dpu_fingerprint')))
"
# Both DPU subgraphs should show: 0x101000016010407
```

---

## Running the Pipeline

```bash
# Make sure overlay is loaded first (see Setup step 1)
python3 dpu_pose.py
```

Press `q` to quit.

### Expected terminal output
```
[INFO] Conv: subgraph_conv1/7x7_s2
[INFO] FC:   subgraph_fc_coordinate_bias_new
[INFO] Buffers pre-allocated
[INFO] Camera: 1920x1080
[INFO] Running — press 'q' to quit

[PERF] 18.4ms | FPS:52.1 | alerts:[]
[PERF] 19.1ms | FPS:51.8 | alerts:['WARNING: Uneven shoulders']
```

---

## Keypoint Map

```
Index │ Joint          │ Index │ Joint
──────┼────────────────┼───────┼──────────────
  0   │ Right Shoulder │   7   │ Right Knee
  1   │ Right Elbow    │   8   │ Right Ankle
  2   │ Right Wrist    │   9   │ Left Hip
  3   │ Left Shoulder  │  10   │ Left Knee
  4   │ Left Elbow     │  11   │ Left Ankle
  5   │ Left Wrist     │  12   │ Head
  6   │ Right Hip      │  13   │ Neck
```

---

## Injury Detection Logic

Joint angles are computed using the dot-product formula at each joint. Current checks:

| Check | Condition | Alert |
|---|---|---|
| Knee hyperextension | angle > 175° | WARNING: R/L knee hyperextension |
| Deep squat alignment | knee angle < 70° | WARNING: Deep squat - check alignment |
| Elbow lockout | angle > 170° | WARNING: R/L elbow locked out |
| Shoulder imbalance | height diff > 30px | WARNING: Uneven shoulders |
| Hip imbalance | height diff > 30px | WARNING: Uneven hips |

---

## Performance

| Metric | Cloud (before) | FPGA (after) |
|---|---|---|
| Latency | ~1000ms | ~18–25ms |
| Network dependency | Yes | None |
| Inference location | DigitalOcean GPU | KV260 FPGA fabric |
| Power | Cloud server | ~5W board |

---

## Known Issues & Debugging Notes

These are hard-won lessons from getting this working — documented so you don't repeat them.

### 1. `/dev/dpu*` never appears — use `/dev/dri/renderD128` instead
Modern VART uses the DRM subsystem. The device node is `/dev/dri/renderD128`, not `/dev/dpu0`. If `renderD128` appears after `loadapp`, the DPU is working correctly.

### 2. `kv260-smartcam` does NOT have a DPU
The smartcam overlay is a video ISP pipeline only. Use `kv260-benchmark-b4096` for DPU inference.

### 3. `k26-starter-kits` auto-loads on boot and blocks slot 0
Always run `sudo xmutil unloadapp` before loading any overlay. Attempting to load without unloading returns `load Error: -1`.

### 4. Segfault / Bus error with VART + OpenCV on aarch64
**Root cause:** VART's DMA engine allocates physically contiguous memory. Allocating new numpy arrays each frame moves them to different physical addresses, corrupting DMA pointers when OpenCV memory operations run concurrently.

**Fix — two rules that must both be followed:**
```python
# RULE 1: Pre-allocate ALL buffers once outside the loop
in_data   = np.ascontiguousarray(np.zeros([1,224,128,3], dtype=np.int8))
out_data  = np.ascontiguousarray(np.zeros([1,7,4,184],  dtype=np.int8))
pool_int8 = np.ascontiguousarray(np.zeros([1,1,1,184],  dtype=np.int8))
out_data2 = np.ascontiguousarray(np.zeros([1,28],       dtype=np.int8))
display   = np.ascontiguousarray(np.zeros([480,640,3],  dtype=np.uint8))

# RULE 2: Update buffers in-place with copyto, never reassign
np.copyto(in_data, new_data)
cv2.resize(frame, (640,480), dst=display)  # dst= reuses buffer
```

**Rule 3: Keep everything inline in the main loop — no function calls.**
Calling helper functions (`draw()`, `infer()`, `analyse()`) inside the loop causes stack frame interactions with the DMA memory region on aarch64, leading to intermittent crashes. The entire pipeline must be written inline.

### 5. sp_net is a two-subgraph model — not one
The xmodel contains two DPU subgraphs plus CPU and USER subgraphs. You must load and run both DPU subgraphs separately with a CPU avgpool between them. Attempting to run only the largest subgraph gives wrong output.

```
subgraph_conv1/7x7_s2          → DPU (run this first)
subgraph_fc_coordinate_bias_new → DPU (run this second)
subgraph_inception_5b/...       → CPU (handled by avgpool)
subgraph_demo                   → USER (do not create runner for this — crashes)
```

### 6. Camera native resolution is 1920×1080
Do not set `CAP_PROP_FRAME_WIDTH/HEIGHT` to 640×480 — it fights the GStreamer backend and causes warnings. Let the camera run at native resolution and resize in software before both model input and display.

---

## What's Next

- [ ] Tune posture angle thresholds with real gym footage
- [ ] Add rep counter (squat, deadlift, bench) using joint trajectory tracking  
- [ ] Try `kv260-aibox-reid` overlay which includes camera ISP pipeline
- [ ] Evaluate `openpose_pruned` xmodel for better accuracy
- [ ] Add mobile UI to receive posture alerts over WebSocket
- [ ] Fine-tune sp_net on gym-specific pose dataset for better accuracy on exercise positions
