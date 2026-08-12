package com.roaddefect.demo

/**
 * Single source of truth for all tunable constants used across the app.
 * Change a value here instead of hunting through MainActivity/ModelInterpreter/etc.
 */
object AppConfig {

    // --- Model / inference ---
    const val MODEL_FILE_NAME = "best_int8(640).tflite"
    const val MODEL_INPUT_SIZE = 640
    const val CONFIDENCE_THRESHOLD = 0.40f
    const val IOU_THRESHOLD = 0.45f

    // --- Camera / frame handling ---
    const val FRAME_SKIP_INTERVAL = 4

    // --- Letterbox padding (camera frame 640x480 padded into 640x640 square) ---
    // Recalculated automatically below — do not hardcode 80f separately elsewhere.
    const val CAMERA_FRAME_WIDTH = 640
    const val CAMERA_FRAME_HEIGHT = 480
    val LETTERBOX_PADDING: Float
        get() = (MODEL_INPUT_SIZE - (CAMERA_FRAME_HEIGHT * MODEL_INPUT_SIZE / CAMERA_FRAME_WIDTH)) / 2f

    // --- Class names — must match training order exactly (data.yml / model.names in Colab) ---
    val CLASS_NAMES = arrayOf(
        "cracks",        // 0: cracks
        "potholes",      // 1: potholes
        "raveling",      // 2: raveling
        "sw",            // 3: sw (stagnant water)
        "manhole_drain"  // 4: manhole_drain
    )

    // --- Logging ---
    // Flip this one flag to silence all AppLog.d() calls app-wide (e.g. for release builds)
    // without deleting/commenting individual Log.d lines throughout the codebase.
    const val DEBUG_LOGGING_ENABLED = true
}