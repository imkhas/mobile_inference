package com.roaddefect.demo

import android.content.Context
import android.location.Location
import android.os.Environment
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object DetectionLogger {

    private const val TAG = "DetectionLogger"
    private var logFile: File? = null
    private val dateFormat     = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
    private val fileNameFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())

    fun init(context: Context) {
        try {
            // saves to /sdcard/Android/data/com.roaddefect.demo/files/test_logs/
            // mirrors your local test_logs/ folder convention
            val dir = File(
                context.getExternalFilesDir(null),
                "test_logs"
            )
            if (!dir.exists()) dir.mkdirs()

            val fileName = "gps_detections_${fileNameFormat.format(Date())}.txt"
            logFile = File(dir, fileName)

            // write header
            FileWriter(logFile, true).use { writer ->
                writer.append("# Road Defect Detection — GPS Log\n")
                writer.append("# Generated: ${dateFormat.format(Date())}\n")
                writer.append("# Format: timestamp | lat | lng | class | confidence | frame\n")
                writer.append("#\n")
            }

            AppLog.d(TAG, "Log file created: ${logFile?.absolutePath}")

        } catch (e: Exception) {
            AppLog.e(TAG, "Failed to create log file: ${e.message}")
        }
    }

    fun log(detection: Detection, location: Location?, frameNum: Int) {
        try {
            val timestamp = dateFormat.format(Date())
            val lat       = location?.latitude  ?: 0.0
            val lng       = location?.longitude ?: 0.0
            val hasGps    = if (location != null) "GPS_OK" else "NO_GPS"

            val line = "$timestamp | $lat | $lng | ${detection.className} | " +
                       "${"%.2f".format(detection.confidence * 100)}% | " +
                       "frame#$frameNum | $hasGps\n"

            FileWriter(logFile, true).use { writer ->
                writer.append(line)
            }

            AppLog.d(TAG, "Logged: ${detection.className} @ $lat, $lng")

        } catch (e: Exception) {
            AppLog.e(TAG, "Failed to log detection: ${e.message}")
        }
    }

    fun getLogFilePath(): String = logFile?.absolutePath ?: "not initialized"
}