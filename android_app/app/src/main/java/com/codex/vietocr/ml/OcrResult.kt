package com.codex.vietocr.ml

data class OcrResult(
    val recognizedText: String,
    val confidence: Float,
    val suggestedCorrection: String?,
    val modelInputWidth: Int,
    val lineCount: Int = 1,
    val recognizedLines: List<String> = emptyList(),
    val correctedLines: List<String> = emptyList(),
    val frameLatencyMs: Long = 0L,
    val fps: Float = 0f
)
