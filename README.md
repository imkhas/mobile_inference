# Road Defect Demo — Mobile Edge AI

On-device road defect detection on Android, using a YOLO11n model converted to TFLite (INT8 quantized), running live on a smartphone camera feed.

This started as an exploration of whether a regular smartphone could match the accuracy and reliability of dedicated edge AI hardware (e.g. Raspberry Pi + AI accelerator, Jetson Orin Nano) for real-time road defect detection using hardware most people already carry instead of dedicated boards.

The idea: mount a phone on a bike, ride around, and have it automatically flag road defects in real time, fully offline, no cloud dependency.

---

## What it detects

6 road defect classes:

| Class |
|---|
| Alligator Cracking |
| Pothole |
| Raveling |
| Stagnant Water |
| Transverse Cracking |
| Longitudinal Cracking |

> Earlier in the project, cracking was tracked as a single combined class. It was later split into three subtypes (alligator/transverse/longitudinal) for more specific detection.

---

## Current status

| Stage | Status |
|---|---|
| Camera permission + live preview | ✅ Working |
| Frame extraction (`ImageAnalysis`) | ✅ Working |
| TFLite model loading + inference | ✅ Working |
| Output decoding (confidence filter + NMS) | ✅ Working |
| Red alert banner on detection | ✅ Working |
| Per-class confidence threshold tuning | ⏳ Not yet tuned (using default `0.25f`) |
| Inference delegate confirmation (NPU/GPU/CPU) | ⏳ Not yet confirmed |

See `docs/dev-guide.md` for full build history, debugging notes, and open items.

---

## Tech stack

- **Language:** Kotlin
- **Camera:** CameraX (`Preview` + `ImageAnalysis` use cases)
- **Model:** YOLO11n → TFLite (INT8), trained via Ultralytics
- **Inference runtime:** TensorFlow Lite Interpreter
- **Build tool:** Gradle CLI (no Android Studio GUI dependency — see dev guide for why)
- **Min SDK:** 24 (Android 7.0) — supports older/lower-spec devices
- **Target/Compile SDK:** 36

---

## Project structure

```
RoadDefectDemo/
├── build.gradle.kts
├── settings.gradle.kts
├── gradle.properties
├── gradlew / gradlew.bat
├── gradle/wrapper/
│   ├── gradle-wrapper.jar
│   └── gradle-wrapper.properties
├── .gitignore
└── app/
    ├── build.gradle.kts
    └── src/main/
        ├── AndroidManifest.xml
        ├── java/com/roaddefect/demo/
        │   └── MainActivity.kt
        ├── assets/
        │   └── best_int8.tflite
        └── res/
            ├── layout/
            │   └── activity_main.xml
            └── values/
```

---

## Setup (cloning this repo fresh)

The trained model file (`best_int8.tflite`) is **not included in this repo** (see `.gitignore`) — it's too large and version-specific to track in git.

1. Clone the repo
2. Train your own model (see `docs/dev-guide.md` for the training/export pipeline used here), or supply your own `.tflite` model trained on similar classes
3. Place the model at:
   ```
   app/src/main/assets/best_int8.tflite
   ```
4. Confirm your Android SDK environment is set up (`ANDROID_HOME`, `adb` working) — see `docs/dev-guide.md` Section 2 for full details if starting from scratch on a new machine
5. Build and install:
   ```bash
   ./gradlew installDebug && adb shell am start -n com.roaddefect.demo/.MainActivity
   ```

---

## Known limitations / open items

- Confidence and IoU thresholds are using untuned defaults
- Which hardware delegate (NPU/GPU/CPU) actually executes inference hasn't been confirmed/logged yet
- No bounding boxes drawn on screen — alert banner only shows the top-confidence detection's class + confidence (deliberate scope decision for the initial build)
- Not yet tested on older/lower-spec phones, only on the primary dev device so far

Full details, debugging history, and notes: see `docs/dev-guide.md`.

---

## Why this exists

This was built to answer a practical question: can a phone running a quantized model on-device hold up well enough in both accuracy and thermal/battery behavior over a real session and to be a viable, low-cost alternative to dedicated edge hardware for field-based defect detection.
