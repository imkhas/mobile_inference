package com.roaddefect.demo

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Wraps the TFLite model: loading, preprocessing, running inference, and
 * decoding raw output into usable Detection objects. Keeping this separate
 * from MainActivity means model-specific logic can be tested/debugged
 * independently of camera/UI concerns.
 */
class ModelInterpreter(context: Context, modelFileName: String = "best_int8.tflite") {

    private val tflite: Interpreter
    val modelInputSize = 640

    val classNames = arrayOf(
        "Alligator Cracking",   // 0: ac
        "Pothole",               // 1: potholes
        "Raveling",              // 2: raveling
        "Stagnant Water",        // 3: sw
        "Transverse Cracking",   // 4: tc
        "Longitudinal Cracking"  // 5: lc
    )

    init {
        tflite = Interpreter(loadModelFile(context, modelFileName))
        Log.d("TFLiteInit", "Model loaded successfully")
        logModelShapes()
    }

    /**
     * Loads a .tflite file from app/src/main/assets/ into a memory-mapped ByteBuffer.
     */
    private fun loadModelFile(context: Context, modelFileName: String): ByteBuffer {
        val assetManager = context.assets
        val fileDescriptor = assetManager.openFd(modelFileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Logs the model's expected input/output tensor shapes — useful for confirming
     * the model matches expectations (640x640 input, 6-class output) before relying on it.
     */
    private fun logModelShapes() {
        val inputShape = tflite.getInputTensor(0).shape()
        val outputShape = tflite.getOutputTensor(0).shape()
        Log.d("TFLiteInit", "Input shape: ${inputShape.joinToString()}")
        Log.d("TFLiteInit", "Output shape: ${outputShape.joinToString()}")
    }

    /**
     * Converts a Bitmap into the normalized float32 ByteBuffer the model expects.
     * Confirmed via byte-size mismatch debugging: model expects 4,915,200 bytes
     * (640*640*3*4), meaning float32 input, normalized 0.0-1.0 — matching how
     * Ultralytics trains (pixel values / 255.0), regardless of internal INT8 quantization.
     */
    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(1 * modelInputSize * modelInputSize * 3 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(modelInputSize * modelInputSize)
        bitmap.getPixels(pixels, 0, modelInputSize, 0, 0, modelInputSize, modelInputSize)

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
        }

        byteBuffer.rewind()
        return byteBuffer
    }

    /**
     * Runs inference on a letterboxed 640x640 Bitmap and returns decoded detections.
     * Returns null if inference throws (caller logs/handles as needed).
     */
    fun runInference(bitmap: Bitmap, frameNum: Int): List<Detection>? {
        val inputBuffer = bitmapToByteBuffer(bitmap)
        val outputShape = tflite.getOutputTensor(0).shape()
        val output = Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }

        return try {
            val startTime = System.nanoTime()
            tflite.run(inputBuffer, output)
            val inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000
            Log.d("InferenceTiming", "Frame #$frameNum: ${inferenceTimeMs}ms")

            decodeOutput(output)
        } catch (e: Exception) {
            Log.e("Inference", "Inference failed on frame #$frameNum: ${e.message}")
            null
        }
    }

    /**
     * Decodes raw model output (1, 10, 8400) into a list of real detections.
     * Output layout: rows 0-3 = box (x_center, y_center, w, h),
     * rows 4-9 = confidence scores for each of the 6 classes.
     * Applies a confidence threshold, then NMS to remove duplicate overlapping boxes.
     */
    private fun decodeOutput(
        output: Array<Array<FloatArray>>,
        confidenceThreshold: Float = 0.25f,
        iouThreshold: Float = 0.45f
    ): List<Detection> {
        val rawDetections = mutableListOf<Detection>()
        val numCandidates = output[0][0].size // 8400

        for (i in 0 until numCandidates) {
            var bestClassId = -1
            var bestScore = 0f
            for (classIndex in classNames.indices) {
                val score = output[0][4 + classIndex][i]
                if (score > bestScore) {
                    bestScore = score
                    bestClassId = classIndex
                }
            }

            if (bestScore >= confidenceThreshold && bestClassId != -1) {
                rawDetections.add(
                    Detection(
                        classId = bestClassId,
                        className = classNames[bestClassId],
                        confidence = bestScore,
                        x = output[0][0][i],
                        y = output[0][1][i],
                        width = output[0][2][i],
                        height = output[0][3][i]
                    )
                )
            }
        }

        return nonMaxSuppression(rawDetections, iouThreshold)
    }

    /**
     * Removes overlapping duplicate boxes, keeping only the highest-confidence box
     * per cluster of overlapping detections (standard NMS algorithm).
     */
    private fun nonMaxSuppression(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val selected = mutableListOf<Detection>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            selected.add(best)
            sorted.removeAll { calculateIoU(best, it) > iouThreshold }
        }

        return selected
    }

    /**
     * Intersection-over-Union between two boxes — measures how much they overlap.
     */
    private fun calculateIoU(a: Detection, b: Detection): Float {
        val aLeft = a.x - a.width / 2
        val aRight = a.x + a.width / 2
        val aTop = a.y - a.height / 2
        val aBottom = a.y + a.height / 2

        val bLeft = b.x - b.width / 2
        val bRight = b.x + b.width / 2
        val bTop = b.y - b.height / 2
        val bBottom = b.y + b.height / 2

        val intersectLeft = maxOf(aLeft, bLeft)
        val intersectTop = maxOf(aTop, bTop)
        val intersectRight = minOf(aRight, bRight)
        val intersectBottom = minOf(aBottom, bBottom)

        val intersectArea = maxOf(0f, intersectRight - intersectLeft) * maxOf(0f, intersectBottom - intersectTop)
        val aArea = a.width * a.height
        val bArea = b.width * b.height
        val unionArea = aArea + bArea - intersectArea

        return if (unionArea <= 0f) 0f else intersectArea / unionArea
    }

    fun close() {
        tflite.close()
    }
}