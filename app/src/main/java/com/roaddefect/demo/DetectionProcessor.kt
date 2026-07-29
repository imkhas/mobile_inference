package com.roaddefect.demo

/**
 * Pure logic for turning raw model output into final detections.
 * Deliberately has zero dependency on TFLite, Android, or Context —
 * this means it can be unit tested directly on the JVM (./gradlew test),
 * no emulator or physical device required.
 */
object DetectionProcessor {

    /**
     * Decodes raw model output (1, 10, 8400) into a list of real detections.
     * Output layout: rows 0-3 = box (x_center, y_center, w, h),
     * rows 4-9 = confidence scores for each of the 6 classes.
     * Applies a confidence threshold, then NMS to remove duplicate overlapping boxes.
     */
    fun decodeOutput(
        output: Array<Array<FloatArray>>,
        classNames: Array<String>,
        confidenceThreshold: Float = AppConfig.CONFIDENCE_THRESHOLD,
        iouThreshold: Float = AppConfig.IOU_THRESHOLD
    ): List<Detection> {
        val rawDetections = mutableListOf<Detection>()
        val numCandidates = output[0][0].size

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
    fun nonMaxSuppression(detections: List<Detection>, iouThreshold: Float): List<Detection> {
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
     * Public so it can be tested directly with simple hand-constructed Detection pairs.
     */
    fun calculateIoU(a: Detection, b: Detection): Float {
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
}