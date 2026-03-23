package com.codex.vietocr

import com.codex.vietocr.ml.TextPostProcessor
import com.codex.vietocr.ml.VietnameseAutoCorrector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class TextPostProcessorTest {

    @Test
    fun decodeCollapsesRepeatsAndBlankToken() {
        val charset = listOf('a', 'b', 'c')
        val blankIndex = 3
        val sequence = arrayOf(
            floatArrayOf(0.1f, 0.9f, 0.0f, 0.0f),
            floatArrayOf(0.1f, 0.8f, 0.1f, 0.0f),
            floatArrayOf(0.0f, 0.0f, 0.0f, 1.0f),
            floatArrayOf(0.9f, 0.1f, 0.0f, 0.0f)
        )

        val decoded = TextPostProcessor.decode(sequence, charset, blankIndex)

        assertEquals("ba", decoded.text)
        assertEquals(90f, decoded.confidence, 0.001f)
    }

    @Test
    fun autocorrectRestoresVietnameseAccents() {
        val autoCorrector = VietnameseAutoCorrector(listOf("xin", "chào", "cảm", "ơn"))

        val suggestion = autoCorrector.suggest("xin chao")

        assertNotNull(suggestion)
        assertEquals("xin chào", suggestion)
    }

    @Test
    fun levenshteinDistanceMatchesExpectedEdits() {
        assertEquals(1, VietnameseAutoCorrector.levenshtein("viet", "viết"))
    }
}

