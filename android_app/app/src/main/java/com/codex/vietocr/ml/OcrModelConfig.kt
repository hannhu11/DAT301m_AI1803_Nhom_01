package com.codex.vietocr.ml

import android.graphics.RectF

enum class OcrMode(val displayName: String, val modelAssetPath: String) {
    CLEAN("Giấy sạch", "models/crnn_clean.onnx"),
    ROBUST("Điều kiện xấu", "models/crnn_robust.onnx")
}

object OcrModelConfig {
    const val CHARSET_ASSET_PATH = "ocr_charset.txt"
    const val DICTIONARY_ASSET_PATH = "vietnamese_dictionary.txt"

    const val INPUT_HEIGHT = 96
    const val MIN_INPUT_WIDTH = 48
    const val MAX_INPUT_WIDTH = 2304
    const val SEGMENTATION_TARGET_HEIGHT = 256
    const val MAX_PARAGRAPH_LINES = 8

    const val ANALYSIS_WIDTH = 1280
    const val ANALYSIS_HEIGHT = 720

    const val BACKGROUND_WINDOW_RADIUS = 15
    const val ADAPTIVE_BLOCK_SIZE = 31
    const val ADAPTIVE_C = 10f
    const val MIN_ROW_ACTIVITY_RATIO = 0.012f
    const val MIN_COLUMN_ACTIVITY_RATIO = 0.025f
    const val HORIZONTAL_PADDING_RATIO = 0.02f
    const val VERTICAL_PADDING_RATIO = 0.10f
    const val MIN_CONFIDENCE_CLEAN = 72f
    const val MIN_CONFIDENCE_ROBUST = 60f
    const val DIRECT_SHOW_CONFIDENCE = 86f
    const val REQUIRED_STABLE_FRAMES = 2

    val ROI_FRACTION = RectF(0.06f, 0.18f, 0.94f, 0.66f)

    val DEFAULT_MODE = OcrMode.CLEAN
}
