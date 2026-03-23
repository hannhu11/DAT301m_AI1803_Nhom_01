package com.codex.vietocr.ml

import kotlin.math.max
import kotlin.math.roundToInt

object LineSegmenter {

    data class IntRect(
        val left: Int,
        val top: Int,
        val right: Int,
        val bottom: Int
    ) {
        val width: Int = right - left
        val height: Int = bottom - top
    }

    fun findLineRects(
        binary: FloatArray,
        width: Int,
        height: Int,
        minRowActivityRatio: Float,
        minColumnActivityRatio: Float
    ): List<IntRect> {
        if (binary.isEmpty() || width <= 0 || height <= 0) {
            return emptyList()
        }

        val rowThreshold = max(1, (width * minRowActivityRatio).roundToInt())
        val gapTolerance = max(1, height / 80)
        val minLineHeight = max(4, height / 40)

        val bands = mutableListOf<Pair<Int, Int>>()
        var currentStart = -1
        var lastActiveRow = -1
        var blankRun = 0

        for (row in 0 until height) {
            val ink = rowInk(binary, width, row)
            if (ink >= rowThreshold) {
                if (currentStart == -1) {
                    currentStart = row
                }
                lastActiveRow = row
                blankRun = 0
            } else if (currentStart != -1) {
                blankRun++
                if (blankRun > gapTolerance) {
                    val bottom = lastActiveRow + 1
                    if (bottom - currentStart >= minLineHeight) {
                        bands += currentStart to bottom
                    }
                    currentStart = -1
                    lastActiveRow = -1
                    blankRun = 0
                }
            }
        }

        if (currentStart != -1) {
            val bottom = lastActiveRow + 1
            if (bottom - currentStart >= minLineHeight) {
                bands += currentStart to bottom
            }
        }

        return bands.mapNotNull { (top, bottom) ->
            val rect = findHorizontalBounds(
                binary = binary,
                width = width,
                top = top,
                bottom = bottom,
                minColumnActivityRatio = minColumnActivityRatio
            )
            if (rect.width <= 0 || rect.height <= 0) {
                null
            } else {
                rect
            }
        }
    }

    private fun rowInk(binary: FloatArray, width: Int, row: Int): Int {
        var count = 0
        val offset = row * width
        for (column in 0 until width) {
            if (binary[offset + column] > 0.5f) {
                count++
            }
        }
        return count
    }

    private fun findHorizontalBounds(
        binary: FloatArray,
        width: Int,
        top: Int,
        bottom: Int,
        minColumnActivityRatio: Float
    ): IntRect {
        val lineHeight = bottom - top
        val columnThreshold = max(1, (lineHeight * minColumnActivityRatio).roundToInt())
        var left = width
        var right = -1

        for (column in 0 until width) {
            var ink = 0
            for (row in top until bottom) {
                if (binary[row * width + column] > 0.5f) {
                    ink++
                }
            }
            if (ink >= columnThreshold) {
                if (left == width) {
                    left = column
                }
                right = column
            }
        }

        if (right < left) {
            return IntRect(0, top, width, bottom)
        }

        return IntRect(left, top, right + 1, bottom)
    }
}
