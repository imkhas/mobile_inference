package com.roaddefect.demo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.widget.TextView

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var alertBanner: TextView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var tflite: Interpreter

    private val frameSkipInterval = 4
    private var frameCount = 0
    private val modelInputSize = 640

    data class Detection(
    val classId: Int,
    val className: String,
    val confidence: Float,
    val x: Float, // center x, in 640x640 space
    val y: Float, // center y
    val width: Float,
    val height: Float
    )   

    // Class names — must match the order the model was trained with (model.names in Colab).
    // PLACEHOLDER: replace with your actual 6 class names once confirmed from `print(model.names)`
    private val classNames = arrayOf(
        "Alligator Cracking",   // 0: ac
        "Pothole",               // 1: potholes
        "Raveling",              // 2: raveling
        "Stagnant Water",        // 3: sw
        "Transverse Cracking",   // 4: tc
        "Longitudinal Cracking"  // 5: lc
    )

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCamera()
            } else {
                android.widget.Toast.makeText(
                    this, "Camera permission is required", android.widget.Toast.LENGTH_LONG
                ).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        alertBanner = findViewById(R.id.alertBanner)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Load the TFLite model once, at startup — NOT on every frame (loading is expensive)
        tflite = Interpreter(loadModelFile())
        Log.d("TFLiteInit", "Model loaded successfully")
        logModelShapes()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    /**
     * Loads best_int8.tflite from app/src/main/assets/ into a memory-mapped ByteBuffer.
     */
    private fun loadModelFile(): ByteBuffer {
        val assetManager = assets
        val fileDescriptor = assetManager.openFd("best_int8.tflite")
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

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy: ImageProxy ->
                        frameCount++

                        if (frameCount % frameSkipInterval == 0) {
                            val squareBitmap = letterboxToSquare(imageProxy, modelInputSize)
                            runInference(squareBitmap, frameCount)
                        } else {
                            Log.d("FrameAnalysis", "Skipping frame #$frameCount")
                        }

                        imageProxy.close() // Flag #3 — must run regardless of branch taken
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalysis
                )
            } catch (exc: Exception) {
                exc.printStackTrace()
            }

        }, ContextCompat.getMainExecutor(this))
    }

    /**
 * Decodes raw model output (1, 10, 8400) into a list of real detections.
 * Output layout per the model: rows 0-3 = box (x_center, y_center, w, h),
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
        // Find the class with the highest score for this candidate
        var bestClassId = -1
        var bestScore = 0f
        for (classIndex in 0 until classNames.size) {
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

    /**
     * Converts the letterboxed Bitmap into the input format the model expects,
     * runs inference, and logs raw output for now.
     *
     * NOTE: This is intentionally minimal — just proving the model runs and
     * produces output. Confidence filtering, NMS, and box-decoding (Section 5.5
     * of the system design doc) are NOT implemented yet. That's the next step
     * once we confirm inference itself runs without crashing.
     */
    private fun runInference(bitmap: Bitmap, frameNum: Int) {
        val inputBuffer = bitmapToByteBuffer(bitmap)

        // Output shape from Colab export: (1, 10, 8400) for 4 classes,
        // but yours is 6 classes -> shape is (1, 10, 8400) where 10 = 4 box coords + 6 classes
        // Confirm this matches logModelShapes() output before trusting these dimensions.
        val outputShape = tflite.getOutputTensor(0).shape()
        val output = Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }

    try {
        tflite.run(inputBuffer, output)

    val detections = decodeOutput(output)
    if (detections.isNotEmpty()) {
        val top = detections.maxByOrNull { it.confidence }!!
        Log.d("Detection", "Frame #$frameNum: ${top.className} (${(top.confidence * 100).toInt()}%)")

        runOnUiThread {
            alertBanner.text = "⚠ ${top.className} detected (${(top.confidence * 100).toInt()}%)"
            alertBanner.visibility = android.view.View.VISIBLE
        }
    } else {
        Log.d("Detection", "Frame #$frameNum: no detections above threshold")

        runOnUiThread {
            alertBanner.visibility = android.view.View.GONE
        }
    }
    } catch (e: Exception) {
        Log.e("Inference", "Inference failed on frame #$frameNum: ${e.message}")
    }
    }

    /**
    * Converts a Bitmap into the normalized float32 ByteBuffer the model expects.
    * Confirmed via the byte-size mismatch error: model expects 4,915,200 bytes
    * (640*640*3*4), meaning float32 input, normalized 0.0-1.0 — matching how
    * Ultralytics trains (pixel values / 255.0), regardless of internal INT8 quantization.
    */
    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(1 * modelInputSize * modelInputSize * 3 * 4) // ×4 for float32
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

    private fun ImageProxy.toBitmap(): Bitmap {
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

    private fun letterboxToSquare(imageProxy: ImageProxy, targetSize: Int): Bitmap {
        val bitmap = imageProxy.toBitmap()

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

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        tflite.close()
    }
}