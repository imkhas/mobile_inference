package com.roaddefect.demo

/**
 * Represents one decoded detection from the model's raw output.
 * x, y, width, height are all in 640x640 model-input-space coordinates.
 */
data class Detection(
    val classId: Int,
    val className: String,
    val confidence: Float,
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float
)