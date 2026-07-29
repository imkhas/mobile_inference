package com.roaddefect.demo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

/**
 * Converts a CameraX ImageProxy (YUV_420_888 format) into an ARGB Bitmap.
 */
fun ImageProxy.toBitmap(): Bitmap {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
    val imageBytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}

/**
 * Flag #1 fix: pads a rectangular camera frame into a square, preserving aspect ratio.
 * Scales proportionally to fit, then adds black borders to fill the remaining space —
 * matches the letterbox preprocessing used during model training, so inference input
 * matches training input shape exactly (no stretching/distortion).
 */
fun letterboxToSquare(imageProxy: ImageProxy, targetSize: Int): Bitmap {
    val bitmap = imageProxy.toBitmap()

    android.util.Log.d("LetterboxDebug", "Original frame: ${bitmap.width}x${bitmap.height}")

    val originalWidth = bitmap.width
    val originalHeight = bitmap.height

    val scale = targetSize.toFloat() / maxOf(originalWidth, originalHeight)
    val scaledWidth = (originalWidth * scale).toInt()
    val scaledHeight = (originalHeight * scale).toInt()

    val scaledBitmap = Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)

    val squareBitmap = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(squareBitmap)
    canvas.drawColor(Color.BLACK)

    val left = (targetSize - scaledWidth) / 2f
    val top = (targetSize - scaledHeight) / 2f
    canvas.drawBitmap(scaledBitmap, left, top, null)

    return squareBitmap
}