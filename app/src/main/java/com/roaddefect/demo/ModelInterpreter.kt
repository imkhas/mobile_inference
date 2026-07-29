package com.roaddefect.demo

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Wraps the TFLite model: loading, delegate fallback (NNAPI -> GPU -> XNNPACK CPU),
 * preprocessing, and running inference. Decode/NMS logic lives in
 * DetectionProcessor instead, so it can be unit tested independently.
 *
 * Delegate chain:
 *   Tier 1: NNAPI     — routes to NPU on supported devices (no NPU on Dimensity 6100+)
 *   Tier 2: GPU       — fixed batch size mismatch via CompatibilityList + dummy test
 *   Tier 3: XNNPACK   — optimised CPU, faster than plain CPU fallback
 */
class ModelInterpreter(
    context: Context,
    modelFileName: String = AppConfig.MODEL_FILE_NAME
) {

    private val tflite: Interpreter
    private var nnApiDelegate: NnApiDelegate? = null
    private var gpuDelegate: GpuDelegate? = null

    val modelInputSize = AppConfig.MODEL_INPUT_SIZE
    val classNames = AppConfig.CLASS_NAMES

    // Circuit breaker: if inference fails this many times in a row, stop attempting
    // it on every frame (avoids burning CPU/battery retrying a broken state forever).
    // Resets to 0 the moment a single inference succeeds again.
    private var consecutiveFailures = 0
    private val maxConsecutiveFailures = 10

    init {
        tflite = buildInterpreter(context, modelFileName)
        AppLog.d("TFLiteInit", "Model loaded successfully")
        AppLog.d("TFLiteInit", "Active delegate — NNAPI: ${nnApiDelegate != null}, GPU: ${gpuDelegate != null}")
        logModelShapes()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Delegate fallback chain
    // ─────────────────────────────────────────────────────────────────────────

    private fun buildInterpreter(context: Context, modelFileName: String): Interpreter {
        val modelBuffer = loadModelFile(context, modelFileName)

        // Tier 1: NNAPI
        tryNnapi(modelBuffer)?.let {
            AppLog.d("TFLiteInit", "Using NNAPI delegate")
            return it
        }

        // Tier 2: GPU — uses CompatibilityList to fix batch size mismatch
        tryGpu(modelBuffer)?.let {
            AppLog.d("TFLiteInit", "Using GPU delegate")
            return it
        }

        // Tier 3: XNNPACK CPU — meaningfully faster than plain CPU on Cortex-A55/A76
        AppLog.d("TFLiteInit", "Using XNNPACK CPU fallback")
        return Interpreter(modelBuffer, Interpreter.Options().apply {
            setUseXNNPACK(true)
            setNumThreads(4)
        })
    }

    private fun tryNnapi(modelBuffer: ByteBuffer): Interpreter? {
        return try {
            val delegate = NnApiDelegate()
            nnApiDelegate = delegate
            val interpreter = Interpreter(
                modelBuffer,
                Interpreter.Options().addDelegate(delegate)
            )
            AppLog.d("TFLiteInit", "NNAPI delegate initialized successfully")
            interpreter
        } catch (e: Exception) {
            AppLog.e("TFLiteInit", "NNAPI delegate failed: ${e.message}")
            nnApiDelegate?.close()
            nnApiDelegate = null
            null
        }
    }

    private fun tryGpu(modelBuffer: ByteBuffer): Interpreter? {
        // CompatibilityList picks the correct GPU options for this device
        // and avoids the "batch size mismatch, expected 1 but got 400" crash
        // caused by the dynamic output tensor shape in YOLO TFLite models.
        val compatList = CompatibilityList()
        if (!compatList.isDelegateSupportedOnThisDevice) {
            AppLog.d("TFLiteInit", "GPU delegate not supported on this device")
            return null
        }

        return try {
            val delegate = GpuDelegate(
                compatList.bestOptionsForThisDevice.apply {
                    isPrecisionLossAllowed = true   // allows FP16 on GPU — faster
                    inferencePreference =
                        GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED
                }
            )
            gpuDelegate = delegate

            val interpreter = Interpreter(
                modelBuffer,
                Interpreter.Options().addDelegate(delegate)
            )

            // Run a dummy forward pass to catch the batch size mismatch
            // before committing to this delegate mid-session.
            // If this throws, we fall through to XNNPACK cleanly.
            val inputShape = interpreter.getInputTensor(0).shape()
            val dummyInput = ByteBuffer.allocateDirect(
                inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3] * 4
            ).apply { order(ByteOrder.nativeOrder()) }

            val outputShape = interpreter.getOutputTensor(0).shape()
            val dummyOutput = Array(outputShape[0]) {
                Array(outputShape[1]) { FloatArray(outputShape[2]) }
            }

            interpreter.run(dummyInput, dummyOutput)
            AppLog.d("TFLiteInit", "GPU delegate test pass — batch size compatible")
            interpreter

        } catch (e: Exception) {
            AppLog.e("TFLiteInit", "GPU delegate failed: ${e.message}")
            gpuDelegate?.close()
            gpuDelegate = null
            null
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Model loading
    // ─────────────────────────────────────────────────────────────────────────

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

    private fun logModelShapes() {
        val inputShape = tflite.getInputTensor(0).shape()
        val outputShape = tflite.getOutputTensor(0).shape()
        AppLog.d("TFLiteInit", "Input shape: ${inputShape.joinToString()}")
        AppLog.d("TFLiteInit", "Output shape: ${outputShape.joinToString()}")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Preprocessing
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Converts a Bitmap into the normalized float32 ByteBuffer the model expects.
     * Model expects float32 input (4 bytes/channel), normalized 0.0-1.0,
     * regardless of internal INT8 quantization — confirmed via byte-size mismatch debugging.
     */
    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(
            1 * modelInputSize * modelInputSize * 3 * 4
        )
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

    // ─────────────────────────────────────────────────────────────────────────
    // Inference
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Runs inference on a letterboxed 640x640 Bitmap and returns decoded detections.
     * Decode/NMS logic delegated to DetectionProcessor (testable independently).
     * Returns null if inference throws, or if the circuit breaker has tripped
     * after too many consecutive failures (caller treats both cases the same way).
     */
    fun runInference(bitmap: Bitmap, frameNum: Int): List<Detection>? {
        if (consecutiveFailures >= maxConsecutiveFailures) {
            AppLog.e(
                "Inference",
                "Skipping frame #$frameNum — too many consecutive failures ($consecutiveFailures)"
            )
            return null
        }

        val inputBuffer = bitmapToByteBuffer(bitmap)
        val outputShape = tflite.getOutputTensor(0).shape()
        val output = Array(outputShape[0]) {
            Array(outputShape[1]) { FloatArray(outputShape[2]) }
        }

        return try {
            val startTime = System.nanoTime()
            tflite.run(inputBuffer, output)
            val inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000
            AppLog.d("InferenceTiming", "Frame #$frameNum: ${inferenceTimeMs}ms")

            consecutiveFailures = 0
            DetectionProcessor.decodeOutput(output, classNames)
        } catch (e: Exception) {
            consecutiveFailures++
            AppLog.e(
                "Inference",
                "Inference failed on frame #$frameNum (failure $consecutiveFailures): ${e.message}"
            )
            null
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Cleanup
    // ─────────────────────────────────────────────────────────────────────────

    fun close() {
        tflite.close()
        nnApiDelegate?.close()
        gpuDelegate?.close()
    }
}