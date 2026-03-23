package com.codex.vietocr

import com.codex.vietocr.ml.LineSegmenter
import org.junit.Assert.assertEquals
import org.junit.Test

class LineSegmenterTest {

    @Test
    fun findLineRectsSplitsThreeBands() {
        val width = 12
        val height = 24
        val binary = FloatArray(width * height)

        fillBand(binary, width, 2, 6, 2, 9)
        fillBand(binary, width, 10, 15, 1, 10)
        fillBand(binary, width, 17, 22, 3, 8)

        val rects = LineSegmenter.findLineRects(
            binary = binary,
            width = width,
            height = height,
            minRowActivityRatio = 0.1f,
            minColumnActivityRatio = 0.1f
        )

        assertEquals(3, rects.size)
        assertEquals(2, rects[0].top)
        assertEquals(6, rects[0].bottom)
        assertEquals(10, rects[1].top)
        assertEquals(15, rects[1].bottom)
        assertEquals(17, rects[2].top)
        assertEquals(22, rects[2].bottom)
    }

    private fun fillBand(
        binary: FloatArray,
        width: Int,
        top: Int,
        bottomExclusive: Int,
        left: Int,
        rightExclusive: Int
    ) {
        for (row in top until bottomExclusive) {
            for (column in left until rightExclusive) {
                binary[row * width + column] = 1f
            }
        }
    }
}
