package com.codex.vietocr.ml

import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

class OcrAnalyzer(
    private val engine: OnnxOcrEngine,
    private val onResult: (OcrResult) -> Unit,
    private val onFailure: (Throwable) -> Unit,
    private val shouldAnalyze: () -> Boolean = { true }
) : ImageAnalysis.Analyzer {

    private var lastFrameStartMs: Long = 0L
    private var smoothedFps: Float = 0f
    private var lastStableKey: String = ""
    private var stableCount: Int = 0

    override fun analyze(image: ImageProxy) {
        if (!shouldAnalyze()) {
            image.close()
            return
        }
        val start = SystemClock.elapsedRealtime()
        var fullBitmap: android.graphics.Bitmap? = null
        var roiBitmap: android.graphics.Bitmap? = null

        try {
            updateFps(start)
            fullBitmap = ImagePreprocessor.imageProxyToBitmap(image)
            roiBitmap = ImagePreprocessor.cropToRoi(fullBitmap, OcrModelConfig.ROI_FRACTION)
            val rawResult = engine.recognize(roiBitmap).copy(
                frameLatencyMs = SystemClock.elapsedRealtime() - start,
                fps = smoothedFps
            )
            val result = stabilizeResult(rawResult)
            onResult(result)
        } catch (throwable: Throwable) {
            onFailure(throwable)
        } finally {
            roiBitmap?.recycle()
            fullBitmap?.recycle()
            image.close()
        }
    }

    private fun updateFps(frameStartMs: Long) {
        if (lastFrameStartMs > 0L) {
            val deltaMs = (frameStartMs - lastFrameStartMs).coerceAtLeast(1L)
            val instantFps = 1000f / deltaMs.toFloat()
            smoothedFps = if (smoothedFps <= 0f) {
                instantFps
            } else {
                (smoothedFps * 0.85f) + (instantFps * 0.15f)
            }
        }
        lastFrameStartMs = frameStartMs
    }

    private fun stabilizeResult(result: OcrResult): OcrResult {
        val normalized = result.recognizedText
            .lowercase()
            .replace(Regex("\\s+"), " ")
            .trim()

        if (normalized.isBlank()) {
            lastStableKey = ""
            stableCount = 0
            return result
        }

        if (normalized == lastStableKey) {
            stableCount += 1
        } else {
            lastStableKey = normalized
            stableCount = 1
        }

        if (result.confidence >= OcrModelConfig.DIRECT_SHOW_CONFIDENCE || stableCount >= OcrModelConfig.REQUIRED_STABLE_FRAMES) {
            return result
        }

        return result.copy(
            recognizedText = "",
            confidence = 0f,
            suggestedCorrection = null,
            modelInputWidth = 0,
            lineCount = 0,
            recognizedLines = emptyList(),
            correctedLines = emptyList()
        )
    }
}
