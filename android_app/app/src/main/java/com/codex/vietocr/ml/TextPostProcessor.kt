package com.codex.vietocr.ml

import kotlin.math.exp

object TextPostProcessor {

    data class DecodedText(
        val text: String,
        val confidence: Float
    )

    fun decode(
        sequence: Array<FloatArray>,
        charset: List<Char>,
        blankIndex: Int
    ): DecodedText {
        if (sequence.isEmpty()) {
            return DecodedText(text = "", confidence = 0f)
        }

        val builder = StringBuilder()
        val confidences = mutableListOf<Float>()
        var previousIndex = -1

        for (timeStep in sequence) {
            val probabilities = ensureProbabilities(timeStep)
            var bestIndex = 0
            var bestProbability = probabilities[0]
            for (index in 1 until probabilities.size) {
                if (probabilities[index] > bestProbability) {
                    bestIndex = index
                    bestProbability = probabilities[index]
                }
            }

            if (bestIndex != previousIndex && bestIndex != blankIndex && bestIndex in charset.indices) {
                builder.append(charset[bestIndex])
                confidences += bestProbability
            }
            previousIndex = bestIndex
        }

        val decodedText = builder.toString().trim()
        val averageConfidence = if (confidences.isEmpty()) {
            0f
        } else {
            confidences.average().toFloat() * 100f
        }

        return DecodedText(
            text = decodedText,
            confidence = averageConfidence
        )
    }

    fun softmax(values: FloatArray): FloatArray {
        val maxValue = values.maxOrNull() ?: 0f
        val exps = FloatArray(values.size)
        var sum = 0.0
        for (index in values.indices) {
            val value = exp((values[index] - maxValue).toDouble()).toFloat()
            exps[index] = value
            sum += value
        }
        if (sum == 0.0) {
            return FloatArray(values.size)
        }
        return FloatArray(values.size) { index -> (exps[index] / sum).toFloat() }
    }

    private fun ensureProbabilities(values: FloatArray): FloatArray {
        var sum = 0f
        var allWithinRange = true
        for (value in values) {
            sum += value
            if (value < 0f || value > 1f) {
                allWithinRange = false
            }
        }

        val looksNormalized = allWithinRange && kotlin.math.abs(sum - 1f) < 0.05f
        return if (looksNormalized) values else softmax(values)
    }
}
