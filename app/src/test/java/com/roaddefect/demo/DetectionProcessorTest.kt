package com.roaddefect.demo

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for DetectionProcessor — pure logic, no TFLite/Android dependency.
 * Run with: ./gradlew test (no device/emulator needed)
 */
class DetectionProcessorTest {

    private fun makeDetection(
        classId: Int = 0,
        confidence: Float = 0.9f,
        x: Float = 100f,
        y: Float = 100f,
        width: Float = 50f,
        height: Float = 50f
    ) = Detection(
        classId = classId,
        className = "TestClass$classId",
        confidence = confidence,
        x = x,
        y = y,
        width = width,
        height = height
    )

    // --- calculateIoU tests ---

    @Test
    fun `identical boxes have IoU of 1`() {
        val a = makeDetection(x = 100f, y = 100f, width = 50f, height = 50f)
        val b = makeDetection(x = 100f, y = 100f, width = 50f, height = 50f)

        val iou = DetectionProcessor.calculateIoU(a, b)

        assertEquals(1.0f, iou, 0.001f)
    }

    @Test
    fun `completely separate boxes have IoU of 0`() {
        val a = makeDetection(x = 0f, y = 0f, width = 10f, height = 10f)
        val b = makeDetection(x = 1000f, y = 1000f, width = 10f, height = 10f)

        val iou = DetectionProcessor.calculateIoU(a, b)

        assertEquals(0f, iou, 0.001f)
    }

    @Test
    fun `partially overlapping boxes have IoU between 0 and 1`() {
        // Box A: centered at (100,100), 50x50 -> spans (75,75) to (125,125)
        // Box B: centered at (120,120), 50x50 -> spans (95,95) to (145,145)
        // These overlap partially
        val a = makeDetection(x = 100f, y = 100f, width = 50f, height = 50f)
        val b = makeDetection(x = 120f, y = 120f, width = 50f, height = 50f)

        val iou = DetectionProcessor.calculateIoU(a, b)

        assertTrue("IoU should be > 0 for overlapping boxes", iou > 0f)
        assertTrue("IoU should be < 1 for non-identical boxes", iou < 1f)
    }

    // --- nonMaxSuppression tests ---

    @Test
    fun `NMS keeps both boxes when they do not overlap`() {
        val a = makeDetection(confidence = 0.9f, x = 0f, y = 0f, width = 10f, height = 10f)
        val b = makeDetection(confidence = 0.8f, x = 1000f, y = 1000f, width = 10f, height = 10f)

        val result = DetectionProcessor.nonMaxSuppression(listOf(a, b), iouThreshold = 0.45f)

        assertEquals(2, result.size)
    }

    @Test
    fun `NMS keeps only the highest-confidence box among heavily overlapping boxes`() {
        val highConfidence = makeDetection(confidence = 0.9f, x = 100f, y = 100f, width = 50f, height = 50f)
        val lowConfidence = makeDetection(confidence = 0.5f, x = 102f, y = 102f, width = 50f, height = 50f) // nearly identical position

        val result = DetectionProcessor.nonMaxSuppression(listOf(lowConfidence, highConfidence), iouThreshold = 0.45f)

        assertEquals(1, result.size)
        assertEquals(0.9f, result[0].confidence, 0.001f)
    }

    @Test
    fun `NMS on empty list returns empty list`() {
        val result = DetectionProcessor.nonMaxSuppression(emptyList(), iouThreshold = 0.45f)

        assertTrue(result.isEmpty())
    }

    @Test
    fun `NMS on single detection returns that detection unchanged`() {
        val single = makeDetection(confidence = 0.7f)

        val result = DetectionProcessor.nonMaxSuppression(listOf(single), iouThreshold = 0.45f)

        assertEquals(1, result.size)
        assertEquals(single, result[0])
    }

    // --- decodeOutput tests ---

    /**
     * Builds a minimal fake model output tensor (1, 10, N) for testing decodeOutput
     * without needing a real TFLite model. classScores.size determines N (candidate count).
     */
    private fun buildFakeOutput(
        boxes: List<FloatArray>, // each entry: [x, y, w, h]
        classScores: List<FloatArray> // each entry: scores for all 6 classes, for one candidate
    ): Array<Array<FloatArray>> {
        val numCandidates = boxes.size
        val output = Array(1) { Array(10) { FloatArray(numCandidates) } }

        for (i in 0 until numCandidates) {
            output[0][0][i] = boxes[i][0] // x
            output[0][1][i] = boxes[i][1] // y
            output[0][2][i] = boxes[i][2] // w
            output[0][3][i] = boxes[i][3] // h
            for (classIndex in 0 until 6) {
                output[0][4 + classIndex][i] = classScores[i][classIndex]
            }
        }
        return output
    }

    private val testClassNames = arrayOf("A", "B", "C", "D", "E", "F")

    @Test
    fun `decodeOutput filters out candidates below confidence threshold`() {
        val output = buildFakeOutput(
            boxes = listOf(floatArrayOf(100f, 100f, 50f, 50f)),
            classScores = listOf(floatArrayOf(0.1f, 0f, 0f, 0f, 0f, 0f)) // below default 0.25 threshold
        )

        val detections = DetectionProcessor.decodeOutput(output, testClassNames, confidenceThreshold = 0.25f)

        assertTrue(detections.isEmpty())
    }

    @Test
    fun `decodeOutput keeps candidates above confidence threshold`() {
        val output = buildFakeOutput(
            boxes = listOf(floatArrayOf(100f, 100f, 50f, 50f)),
            classScores = listOf(floatArrayOf(0.8f, 0f, 0f, 0f, 0f, 0f))
        )

        val detections = DetectionProcessor.decodeOutput(output, testClassNames, confidenceThreshold = 0.25f)

        assertEquals(1, detections.size)
        assertEquals("A", detections[0].className)
        assertEquals(0.8f, detections[0].confidence, 0.001f)
    }

    @Test
    fun `decodeOutput assigns the class with the highest score, not just the first`() {
        val output = buildFakeOutput(
            boxes = listOf(floatArrayOf(100f, 100f, 50f, 50f)),
            classScores = listOf(floatArrayOf(0.3f, 0.9f, 0.1f, 0f, 0f, 0f)) // class index 1 ("B") is highest
        )

        val detections = DetectionProcessor.decodeOutput(output, testClassNames, confidenceThreshold = 0.25f)

        assertEquals(1, detections.size)
        assertEquals("B", detections[0].className)
        assertEquals(1, detections[0].classId)
    }

    @Test
    fun `decodeOutput on all-zero output returns no detections`() {
        val output = buildFakeOutput(
            boxes = listOf(floatArrayOf(0f, 0f, 0f, 0f)),
            classScores = listOf(floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f))
        )

        val detections = DetectionProcessor.decodeOutput(output, testClassNames)

        assertTrue(detections.isEmpty())
    }
}