package com.roaddefect.demo

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class BoundingBoxOverlay(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {

    private var detections: List<Detection> = emptyList()
    private var sourceWidth = 1
    private var sourceHeight = 1

    private val boxPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 36f
        style = Paint.Style.FILL
    }

    private val textBackgroundPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
    }

    /**
     * Call this whenever new detections arrive. sourceWidth/sourceHeight should be
     * the model's input size (640x640) — used to scale coordinates to the view's
     * actual pixel size.
     */
    fun updateDetections(newDetections: List<Detection>, modelInputSize: Int) {
        detections = newDetections
        sourceWidth = modelInputSize
        sourceHeight = modelInputSize
        AppLog.d("BoundingBoxOverlay", "Received ${newDetections.size} detections, invalidating view")
        invalidate() // triggers onDraw()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // Letterbox padding: camera frame is 640x480, padded into 640x640 square.
        // Since width was the larger dimension, padding is on top/bottom only.
        // Value now comes from AppConfig instead of being hardcoded here —
        // recalculates automatically if CAMERA_FRAME_WIDTH/HEIGHT ever change.
        val padding = AppConfig.LETTERBOX_PADDING
        val contentHeight = sourceHeight - (padding * 2)

        val scaleX = width.toFloat() / sourceWidth
        val scaleY = height.toFloat() / contentHeight // scale based on REAL content height, not full 640

        for (detection in detections) {
            val pixelX = detection.x * sourceWidth
            val pixelY = (detection.y * sourceHeight) - padding // subtract padding to get true position within real content
            val pixelWidth = detection.width * sourceWidth
            val pixelHeight = detection.height * sourceHeight

            val left = (pixelX - pixelWidth / 2) * scaleX
            val top = (pixelY - pixelHeight / 2) * scaleY
            val right = (pixelX + pixelWidth / 2) * scaleX
            val bottom = (pixelY + pixelHeight / 2) * scaleY

            canvas.drawRect(left, top, right, bottom, boxPaint)
            val label = "${detection.className} ${(detection.confidence * 100).toInt()}%"
            val textWidth = textPaint.measureText(label)
            canvas.drawRect(left, top - 40f, left + textWidth + 16f, top, textBackgroundPaint)
            canvas.drawText(label, left + 8f, top - 10f, textPaint)
        }
    }
}