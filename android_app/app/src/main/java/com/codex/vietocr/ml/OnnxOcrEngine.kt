package com.codex.vietocr.ml

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.Locale

class OnnxOcrEngine(context: Context, mode: OcrMode = OcrModelConfig.DEFAULT_MODE) : AutoCloseable {

    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val charset: List<Char> = loadCharset(context)
    private val session: OrtSession = createSession(context, mode)
    private val inputName: String = session.inputInfo.keys.first()
    val currentMode: OcrMode = mode

    @Synchronized
    fun recognize(bitmap: Bitmap): OcrResult {
        data class LineRecognition(
            val rawText: String,
            val correctedText: String?,
            val confidence: Float,
            val inputWidth: Int
        )

        val lineBitmaps = ImagePreprocessor.segmentLineBitmaps(bitmap)
        val lineRecognitions = mutableListOf<LineRecognition>()

        for (lineBitmap in lineBitmaps) {
            try {
                val prepared = ImagePreprocessor.preprocess(lineBitmap)
                val shape = longArrayOf(
                    1L,
                    OcrModelConfig.INPUT_HEIGHT.toLong(),
                    prepared.width.toLong(),
                    1L
                )

                OnnxTensor.createTensor(environment, FloatBuffer.wrap(prepared.tensorData), shape).use { tensor ->
                    session.run(mapOf(inputName to tensor)).use { output ->
                        @Suppress("UNCHECKED_CAST")
                        val rawOutput = output.get(0).value as Array<Array<FloatArray>>
                        val decoded = TextPostProcessor.decode(
                            sequence = rawOutput.first(),
                            charset = charset,
                            blankIndex = charset.size
                        )
                        if (isAcceptedDecodedText(decoded.text, decoded.confidence)) {
                            lineRecognitions += LineRecognition(
                                rawText = decoded.text,
                                correctedText = null,
                                confidence = decoded.confidence,
                                inputWidth = prepared.width
                            )
                        }
                    }
                }
            } finally {
                lineBitmap.recycle()
            }
        }

        if (lineRecognitions.isEmpty()) {
            return OcrResult(
                recognizedText = "",
                confidence = 0f,
                suggestedCorrection = null,
                modelInputWidth = 0,
                lineCount = 0,
                recognizedLines = emptyList(),
                correctedLines = emptyList()
            )
        }

        val recognizedLines = lineRecognitions.map { it.rawText }
        val paragraph = recognizedLines.joinToString(separator = "\n")

        return OcrResult(
            recognizedText = paragraph,
            confidence = lineRecognitions.map { it.confidence }.average().toFloat(),
            suggestedCorrection = null,
            modelInputWidth = lineRecognitions.maxOf { it.inputWidth },
            lineCount = recognizedLines.size,
            recognizedLines = recognizedLines,
            correctedLines = recognizedLines
        )
    }

    override fun close() {
        session.close()
    }

    private fun createSession(context: Context, mode: OcrMode): OrtSession {
        val modelBytes = context.assets.open(mode.modelAssetPath).use { it.readBytes() }
        val options = OrtSession.SessionOptions()
        return environment.createSession(modelBytes, options)
    }

    private fun loadCharset(context: Context): List<Char> {
        val raw = context.assets.open(OcrModelConfig.CHARSET_ASSET_PATH).use {
            String(it.readBytes(), Charsets.UTF_8)
        }.trimEnd('\n', '\r')
        return raw.toList()
    }

    private fun isAcceptedDecodedText(text: String, confidence: Float): Boolean {
        val normalized = text.trim().replace(Regex("\\s+"), " ")
        if (normalized.isBlank()) {
            return false
        }

        val minConfidence = when (currentMode) {
            OcrMode.CLEAN -> OcrModelConfig.MIN_CONFIDENCE_CLEAN
            OcrMode.ROBUST -> OcrModelConfig.MIN_CONFIDENCE_ROBUST
        }
        if (confidence < minConfidence) {
            return false
        }

        val compact = normalized.replace(" ", "")
        if (compact.length < 2) {
            return false
        }

        val meaningfulChars = compact.count { it.isLetterOrDigit() }
        if (meaningfulChars < 2) {
            return false
        }

        val tokenList = normalized.split(' ').filter { it.isNotBlank() }
        val longestToken = tokenList.maxOfOrNull { token ->
            token.count { it.isLetterOrDigit() }
        } ?: 0
        if (longestToken < 2) {
            return false
        }

        val nonSpaceChars = normalized.count { !it.isWhitespace() }.coerceAtLeast(1)
        val signalRatio = meaningfulChars.toFloat() / nonSpaceChars.toFloat()
        if (signalRatio < 0.55f) {
            return false
        }

        val lower = normalized.lowercase(Locale.getDefault())
        val repeatedNoise = Regex("""(.)\1{4,}""").containsMatchIn(lower)
        if (repeatedNoise) {
            return false
        }

        return true
    }
}
