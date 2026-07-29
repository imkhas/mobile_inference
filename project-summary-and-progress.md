# Road Defect Detection — Mobile Edge AI
## Project Summary & Progress Report

---

## 📋 Project Overview

A fully on-device Android application that detects road defects in real time using the phone's camera, running a quantized YOLO11n model via TensorFlow Lite — no cloud dependency, no internet required.

The project's core question: **can a regular smartphone replace dedicated edge hardware (Raspberry Pi + Hailo, Jetson Orin Nano) for road defect detection in the field?**

| | Smartphone (this project) | Raspberry Pi + Hailo | Jetson Orin Nano |
|---|---|---|---|
| Cost | Low — hardware already owned | Low–Medium | Higher |
| Offline operation | ✅ | ✅ | ✅ |
| Portability | Excellent | Requires mount + power | Requires mount + power |
| NPU acceleration | Device-dependent (see findings) | Dedicated Hailo accelerator | Dedicated GPU + TensorRT |

---

## 🎯 Detection Classes

Model trained on Dataset V10.0 — **6 classes** (evolved from 4 classes in V9.8, where a single `cracks` class was split into three subtypes):

| Index | Short name | Full name |
|---|---|---|
| 0 | `ac` | Alligator Cracking |
| 1 | `potholes` | Pothole |
| 2 | `raveling` | Raveling |
| 3 | `sw` | Stagnant Water |
| 4 | `tc` | Transverse Cracking |
| 5 | `lc` | Longitudinal Cracking |

---

## 🏗️ System Architecture

**Model:** YOLO11n → exported to TFLite (INT8 quantized), input 640×640, output `(1, 10, 8400)` (4 box coords + 6 class scores × 8400 candidate boxes)

**Pipeline:**
```
Camera (CameraX, 30 FPS, landscape locked)
        ↓
Frame Sampling (every 4th frame — configurable via AppConfig)
        ↓
Letterbox/Pad (640×480 raw → 640×640 square, black-padded top/bottom)
        ↓
Normalize to Float32 (pixel values / 255.0)
        ↓
TFLite Inference (NNAPI → GPU → CPU fallback chain)
        ↓
Decode Output (confidence filter + NMS)
        ↓
Bounding Box Overlay (all detections drawn on live camera feed)
        ↓
Red Alert Banner (top-confidence detection, class name + %)
```

**Hardware acceleration fallback chain:**
```
Tier 1: NNAPI delegate  →  Tier 2: GPU delegate  →  Tier 3: CPU (default)
```

**Tech stack:** Kotlin, CameraX, TFLite, Gradle CLI + VS Code (no Android Studio GUI — RAM constraint workaround), tested on Realme 11 5G (Dimensity 6100+)

---

## 📁 Code Structure

```
app/src/main/java/com/roaddefect/demo/
├── AppConfig.kt           — single source of truth for all constants + log toggle
├── AppLog.kt              — log wrapper (one flag to silence all debug logs)
├── Detection.kt           — data class for one decoded detection
├── DetectionProcessor.kt  — pure decode/NMS/IoU logic (testable, no Android dependency)
├── ImageUtils.kt          — camera frame → Bitmap conversion + letterbox
├── MainActivity.kt        — camera lifecycle, frame routing, UI updates, error handling
├── ModelInterpreter.kt    — TFLite wrapper, delegate fallback, inference + circuit breaker
└── BoundingBoxOverlay.kt  — custom View drawing boxes on top of camera preview

app/src/test/java/com/roaddefect/demo/
└── DetectionProcessorTest.kt  — 9 unit tests (IoU, NMS, decode logic) — JVM only, no device needed
```

---

## 📊 Current Status

| Feature / Stage | Status |
|---|---|
| Camera permission + live preview | ✅ Working |
| Frame extraction (`ImageAnalysis`) | ✅ Working |
| TFLite model loading + inference | ✅ Working |
| Output decoding (confidence filter + NMS) | ✅ Working |
| Red alert banner on detection | ✅ Working |
| Bounding box overlay on detection | ✅ Working |
| Landscape orientation lock | ✅ Done (AndroidManifest) |
| NNAPI delegate | ✅ Implemented — initializes but no NPU on test device |
| GPU delegate | ✅ Implemented — fails on this model/device, auto-falls back to CPU |
| CPU-only battery/thermal baseline (18 min) | ✅ Measured |
| Soak test (70 min) | ✅ Measured — see Performance Findings |
| Unit tests (9 tests) | ✅ Passing |
| Centralized config (`AppConfig`) | ✅ Done |
| Log toggle (`AppLog`) | ✅ Done |
| Error handling (model load, camera, circuit breaker) | ✅ Done |
| Per-class threshold tuning | ⏳ Not yet done |
| Multi-device validation | ⏳ Only tested on Realme 11 5G |
| Crash reporting (Crashlytics) | ⏳ Not yet done |
| Long-duration lifecycle testing (phone call, screen lock) | ⏳ Not yet done |

---

## 📈 Performance Findings

### Test device
**Realme 11 5G** — MediaTek Dimensity 6100+ (2× Cortex-A76 + 6× Cortex-A55, Mali-G57 MC2 GPU, **no dedicated NPU**)

### Hardware acceleration comparison

| Delegate | Result |
|---|---|
| NNAPI | Initializes successfully — but no NPU means it routes to CPU internally anyway |
| GPU | Fails to initialize (`"batch size mismatch, expected 1 but got 400"`) — auto-falls back to CPU |
| CPU | Working baseline — all three paths currently converge here on this device |

### Inference timing (CPU-equivalent, all delegate paths)

| Metric | 18-min baseline | 70-min soak test |
|---|---|---|
| Average inference | 291.3 ms/frame | TBD (log analysis pending) |
| Min | 236 ms | TBD |
| Max | 1013 ms | TBD |
| Sustained FPS | ~3.4 FPS | TBD |
| Throttling observed | None within test window | TBD |

### Battery & thermal

| Metric | 18-min baseline | 70-min soak test |
|---|---|---|
| Battery drain | 10% / hour | **28.3% / hour** |
| Temperature rise | +3.3°C | +1.7°C |
| End temperature | 32.7°C | 31.0°C |

**Notable finding on the soak test:** battery drain rate (28.3%/hour) was significantly higher than the 18-minute baseline (10%/hour) — **2.8× difference**. At 28.3%/hour, a full charge lasts ~3.5 hours continuous operation. The temperature rise was actually *lower* in the longer test (1.7°C vs 3.3°C), suggesting the phone's thermal management stabilized over time. Root cause of the battery discrepancy needs further investigation — possible factors include: screen brightness, background processes, wireless ADB vs USB connection during testing, or the model simply drawing more sustained power than the short test captured.

> ⚠️ **Soak test inference timing analysis still pending** — paste the `grep/awk` output from `soak_test_log.txt` to complete this section

---

## 🔧 Engineering Decisions & Resolved Issues

### Deliberate design choices
- **Frame sampling (every 4th frame):** prevents thermal overload from continuous inference at 30 FPS — controlled via `AppConfig.FRAME_SKIP_INTERVAL`
- **Letterbox padding (not stretch):** preserves true proportions when mapping 640×480 camera frames to 640×640 model input — black bars added top/bottom
- **Float32 input despite INT8 model:** this specific TFLite export retains a float32 input tensor — model handles float→int8 conversion internally (confirmed via byte-size mismatch debugging)
- **Banner + bounding boxes:** both kept — boxes show precise location, banner gives a clear glanceable summary
- **Landscape lock:** fixes the portrait detection accuracy issue entirely by preventing the app from running in portrait orientation
- **No Android Studio GUI:** replaced with Gradle CLI + VS Code due to RAM constraints (6.7GB RAM — Android Studio crashed repeatedly during Gradle sync)

### Key bugs found and fixed
| Bug | Symptom | Fix |
|---|---|---|
| Byte-size mismatch | Inference crashed every frame (`4915200` vs `1228800` bytes expected) | Rewrote `bitmapToByteBuffer()` to allocate float32 (`×4`) and normalize `/255.0f` |
| Bounding box scale | Boxes were sub-pixel (~0.7px wide, invisible) | Detection coordinates are normalized (0.0–1.0), not pixel values — multiply by `sourceWidth/Height` first |
| Letterbox padding | Boxes rendered disproportionately tall | Overlay must subtract 80px padding and scale against real content height (480px), not full padded square (640px) |
| GPU dependency crash | `ClassNotFoundException: GpuDelegateFactory$Options` | Needed both `tensorflow-lite-gpu` AND `tensorflow-lite-gpu-api` dependencies — native `.so` and Java wrapper are in separate artifacts |
| Test file in wrong source set | `Unresolved reference: junit` in `assembleDebug` | `DetectionProcessorTest.kt` belonged in `src/test/`, not `src/main/` |
| `.gitignore` misnamed | Build artifacts not excluded from git | File was named `gitignore` (missing leading dot) — renamed to `.gitignore` |

---

## ⚠️ Open Items & Known Limitations

### Active blockers / gaps

| # | Item | Priority |
|---|---|---|
| 1 | **Multi-device validation** — all findings scoped to Realme 11 5G only | High |
| 2 | **Per-class threshold tuning** — using untuned defaults (0.25 confidence, 0.45 IoU) | Medium |
| 3 | **Soak test timing analysis** — log captured but not yet analyzed | Medium |
| 4 | **Battery drain discrepancy** — 28.3%/hour (soak) vs 10%/hour (baseline) needs explanation | Medium |
| 5 | **Lifecycle edge cases** — phone call, screen lock, backgrounding mid-session untested | Medium |
| 6 | **Crash reporting** — no visibility into failures without `adb logcat` | Low |
| 7 | **Frame-skip interval tuning** — currently placeholder, not empirically derived | Low |

### Explicit scope boundaries (deliberate, not oversights)
- NNAPI/GPU delegates are in place in code but provide no acceleration on this device (no NPU; GPU incompatible with this model's structure)
- Performance numbers are one-device measurements — may not generalize to other budget Android hardware
- No cloud sync, no data logging to server, no map visualization — fully on-device only

---

## 🚀 Next Steps

1. **Analyze soak test timing log** — compare inference speed at start vs end of 70-min session to confirm or rule out throttling
2. **Investigate battery drain discrepancy** — repeat the baseline test under identical conditions to the soak test to isolate the variable
3. **Lifecycle testing** — simulate phone call interrupt, screen lock, backgrounding during a session
4. **Obtain additional test device(s)** — single-device validation is the biggest remaining credibility gap
5. **Crash reporting** — integrate Firebase Crashlytics for production-level visibility
6. **Per-class threshold tuning** — gather real-world detections across all 6 classes, tune empirically
7. **Thread count optimization** — test `setNumThreads()` explicitly, cheap to try, may give modest CPU speedup
8. **Consider reduced resolution (640→320)** — potential ~4× inference speedup at accuracy cost to thin classes (needs empirical validation)

---

*Full technical build/debug history: `docs/dev-guide.md`*
*Raw test logs: `test_logs/` (gitignored, local only)*
