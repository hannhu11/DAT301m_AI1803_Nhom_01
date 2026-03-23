package com.codex.vietocr.ml

import java.text.Normalizer
import java.util.Locale
import kotlin.math.abs
import kotlin.math.min

class VietnameseAutoCorrector(words: Collection<String>) {

    private data class DictionaryEntry(
        val original: String,
        val folded: String
    )

    private val dictionary: List<String> = words
        .map { it.trim() }
        .filter { it.isNotEmpty() }
        .distinct()

    private val entries: List<DictionaryEntry> = dictionary.map { word ->
        DictionaryEntry(
            original = word,
            folded = fold(word)
        )
    }

    private val foldedIndex: Map<String, String> = entries.associate { entry ->
        entry.folded to entry.original
    }

    private val entriesByFirst: Map<Char?, List<DictionaryEntry>> = entries.groupBy { entry ->
        entry.folded.firstOrNull()
    }

    fun suggest(text: String): String? {
        if (text.isBlank()) {
            return null
        }

        var changed = false
        val corrected = tokenize(text).joinToString(separator = "") { token ->
            if (!token.any(Char::isLetter)) {
                token
            } else {
                val suggestion = correctWord(token)
                if (suggestion != token) {
                    changed = true
                }
                suggestion
            }
        }

        return corrected.takeIf { changed && it != text }
    }

    private fun correctWord(word: String): String {
        if (word.length < 2) {
            return word
        }

        val foldedWord = fold(word)
        foldedIndex[foldedWord]?.let { exactCandidate ->
            return if (exactCandidate == word) {
                word
            } else {
                applyCasePattern(word, exactCandidate)
            }
        }

        val first = foldedWord.firstOrNull()
        val threshold = if (foldedWord.length <= 4) 1 else 2

        var bestCandidate: String? = null
        var bestDistance = Int.MAX_VALUE

        val candidates = entriesByFirst[first] ?: entries

        for (candidate in candidates) {
            val foldedCandidate = candidate.folded
            if (first != null && foldedCandidate.firstOrNull() != first) {
                continue
            }
            if (abs(foldedCandidate.length - foldedWord.length) > threshold) {
                continue
            }

            val distance = levenshtein(foldedWord, foldedCandidate)
            if (distance < bestDistance) {
                bestDistance = distance
                bestCandidate = candidate.original
            }
        }

        val resolved = if (bestDistance <= threshold) bestCandidate else null
        return resolved?.let { applyCasePattern(word, it) } ?: word
    }

    private fun tokenize(text: String): List<String> {
        val tokens = mutableListOf<String>()
        val buffer = StringBuilder()
        var currentIsWord: Boolean? = null

        for (char in text) {
            val isWord = char.isLetter()
            if (currentIsWord == null || currentIsWord == isWord) {
                buffer.append(char)
                currentIsWord = isWord
            } else {
                tokens += buffer.toString()
                buffer.clear()
                buffer.append(char)
                currentIsWord = isWord
            }
        }

        if (buffer.isNotEmpty()) {
            tokens += buffer.toString()
        }
        return tokens
    }

    private fun applyCasePattern(original: String, corrected: String): String {
        return when {
            original.all { !it.isLetter() || it.isUpperCase() } -> corrected.uppercase(Locale.getDefault())
            original.firstOrNull()?.isUpperCase() == true -> corrected.replaceFirstChar {
                if (it.isLowerCase()) it.titlecase(Locale.getDefault()) else it.toString()
            }
            else -> corrected
        }
    }

    private fun fold(value: String): String {
        val normalized = Normalizer.normalize(value.lowercase(Locale.getDefault()), Normalizer.Form.NFD)
            .replace("đ", "d")
        return COMBINING_MARKS.replace(normalized, "")
    }

    companion object {
        private val COMBINING_MARKS = "\\p{InCombiningDiacriticalMarks}+".toRegex()

        fun levenshtein(left: String, right: String): Int {
            if (left == right) {
                return 0
            }
            if (left.isEmpty()) {
                return right.length
            }
            if (right.isEmpty()) {
                return left.length
            }

            val previous = IntArray(right.length + 1) { it }
            val current = IntArray(right.length + 1)

            for (i in 1..left.length) {
                current[0] = i
                for (j in 1..right.length) {
                    val cost = if (left[i - 1] == right[j - 1]) 0 else 1
                    current[j] = min(
                        min(current[j - 1] + 1, previous[j] + 1),
                        previous[j - 1] + cost
                    )
                }
                for (j in previous.indices) {
                    previous[j] = current[j]
                }
            }
            return previous[right.length]
        }
    }
}

