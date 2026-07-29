package com.roaddefect.demo

import android.Manifest
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.view.Gravity
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var viewFinder: PreviewView
    private lateinit var alertBanner: TextView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var modelInterpreter: ModelInterpreter
    private lateinit var boundingBoxOverlay: BoundingBoxOverlay

    private var frameCount = 0

    // ── GPS ───────────────────────────────────────────────────────────────────
    private var locationManager: LocationManager? = null
    private var lastLocation: Location? = null
    private val LOCATION_PERMISSION_REQUEST = 1001

    private val locationListener = LocationListener { location ->
        lastLocation = location
        AppLog.d("GPS", "Location updated: ${location.latitude}, ${location.longitude}")
    }

    // ── Camera permission launcher ────────────────────────────────────────────
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

        viewFinder    = findViewById(R.id.viewFinder)
        alertBanner   = findViewById(R.id.alertBanner)
        boundingBoxOverlay = findViewById(R.id.boundingBoxOverlay)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // initialise detection logger — creates CSV file with header
        DetectionLogger.init(this)
        AppLog.d("MainActivity", "Log file: ${DetectionLogger.getLogFilePath()}")

        // load model
        modelInterpreter = try {
            ModelInterpreter(this)
        } catch (e: Exception) {
            AppLog.e("MainActivity", "Failed to load model: ${e.message}")
            showFatalError("Could not load the detection model.\nPlease reinstall the app or contact support.")
            return
        }

        // request location permission
        setupLocation()

        // request camera permission or start camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    // ── Location setup ────────────────────────────────────────────────────────

    private fun setupLocation() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                ),
                LOCATION_PERMISSION_REQUEST
            )
        } else {
            startLocationUpdates()
        }
    }

    private fun startLocationUpdates() {
        try {
            locationManager = getSystemService(LOCATION_SERVICE) as LocationManager
            locationManager?.requestLocationUpdates(
                LocationManager.GPS_PROVIDER,
                1000L,  // update every 1 second
                1f,     // or every 1 metre — whichever comes first
                locationListener
            )
            AppLog.d("GPS", "Location updates started")
        } catch (e: Exception) {
            AppLog.e("GPS", "Location setup failed: ${e.message}")
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == LOCATION_PERMISSION_REQUEST &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            startLocationUpdates()
            AppLog.d("GPS", "Location permission granted")
        } else {
            AppLog.d("GPS", "Location permission denied — detections will log with 0.0, 0.0")
        }
    }

    // ── Error screen ──────────────────────────────────────────────────────────

    private fun showFatalError(message: String) {
        setContentView(TextView(this).apply {
            text = message
            textSize = 18f
            gravity = Gravity.CENTER
            setPadding(48, 48, 48, 48)
        })
    }

    // ── Camera ────────────────────────────────────────────────────────────────

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

                        if (frameCount % AppConfig.FRAME_SKIP_INTERVAL == 0) {

                            // ── Stage 1: Preprocessing ────────────────────
                            val t0 = System.nanoTime()
                            val squareBitmap = letterboxToSquare(
                                imageProxy,
                                modelInterpreter.modelInputSize
                            )
                            val tPreprocess = (System.nanoTime() - t0) / 1_000_000

                            // ── Stage 2: Inference ────────────────────────
                            val t1 = System.nanoTime()
                            val detections = modelInterpreter.runInference(squareBitmap, frameCount)
                            val tInference = (System.nanoTime() - t1) / 1_000_000

                            // ── Stage 3: UI update + GPS log ──────────────
                            val t2 = System.nanoTime()
                            handleDetectionUI(detections, frameCount)
                            val tDraw = (System.nanoTime() - t2) / 1_000_000

                            // ── Total pipeline ────────────────────────────
                            val tTotal = tPreprocess + tInference + tDraw
                            AppLog.d(
                                "PipelineLatency",
                                "Frame #$frameCount | " +
                                "preprocess=${tPreprocess}ms | " +
                                "inference=${tInference}ms | " +
                                "draw=${tDraw}ms | " +
                                "total=${tTotal}ms"
                            )

                        } else {
                            AppLog.d("FrameAnalysis", "Skipping frame #$frameCount")
                        }

                        imageProxy.close()
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalysis
                )
            } catch (exc: Exception) {
                AppLog.e("MainActivity", "Camera binding failed: ${exc.message}")
                runOnUiThread {
                    showFatalError("Could not start the camera.\nIt may be in use by another app, or unavailable on this device.")
                }
            }

        }, ContextCompat.getMainExecutor(this))
    }

    // ── Detection UI + GPS logging ────────────────────────────────────────────

    /**
     * Updates bounding box overlay and alert banner.
     * Logs the top detection with GPS coordinates to CSV.
     */
    private fun handleDetectionUI(detections: List<Detection>?, frameNum: Int) {
        runOnUiThread {
            boundingBoxOverlay.updateDetections(
                detections ?: emptyList(),
                modelInterpreter.modelInputSize
            )
        }

        if (detections != null && detections.isNotEmpty()) {
            val top = detections.maxByOrNull { it.confidence }!!

            // ── GPS log — write detection + location to CSV ───────────────
            DetectionLogger.log(top, lastLocation, frameNum)

            AppLog.d(
                "Detection",
                "Frame #$frameNum: ${top.className} " +
                "(${(top.confidence * 100).toInt()}%) " +
                "@ lat=${lastLocation?.latitude ?: "no GPS"} " +
                "lng=${lastLocation?.longitude ?: "no GPS"}"
            )

            runOnUiThread {
                alertBanner.text =
                    "⚠ ${top.className} detected (${(top.confidence * 100).toInt()}%)"
                alertBanner.visibility = android.view.View.VISIBLE
            }
        } else {
            AppLog.d("Detection", "Frame #$frameNum: no detections above threshold")
            runOnUiThread {
                alertBanner.visibility = android.view.View.GONE
            }
        }
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────

    override fun onDestroy() {
        super.onDestroy()
        locationManager?.removeUpdates(locationListener)
        cameraExecutor.shutdown()
        if (::modelInterpreter.isInitialized) {
            modelInterpreter.close()
        }
    }
}