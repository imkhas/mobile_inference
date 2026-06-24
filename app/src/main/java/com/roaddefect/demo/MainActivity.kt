package com.roaddefect.demo

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var alertBanner: TextView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var modelInterpreter: ModelInterpreter

    private val frameSkipInterval = 4
    private var frameCount = 0

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

        // All model loading/setup now lives in ModelInterpreter — see ModelInterpreter.kt
        modelInterpreter = ModelInterpreter(this)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
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
                            // letterboxToSquare() now lives in ImageUtils.kt
                            val squareBitmap = letterboxToSquare(imageProxy, modelInterpreter.modelInputSize)
                            handleInference(squareBitmap, frameCount)
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
     * Delegates to ModelInterpreter for the actual inference + decoding,
     * then updates the UI based on the result. This is the only place
     * MainActivity touches detection logic directly.
     */
    private fun handleInference(bitmap: android.graphics.Bitmap, frameNum: Int) {
        val detections = modelInterpreter.runInference(bitmap, frameNum)

        if (detections != null && detections.isNotEmpty()) {
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
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        modelInterpreter.close()
    }
}
