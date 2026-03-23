package com.codex.vietocr.ml

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

object ImagePreprocessor {

    data class PreparedImage(
        val tensorData: FloatArray,
        val width: Int
    )

    private data class BinaryImage(
        val binaryData: FloatArray,
        val width: Int,
        val height: Int
    )

    fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val nv21 = yuv420888ToNv21(image)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val stream = ByteArrayOutputStream()
        check(yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 92, stream)) {
            "Could not encode camera frame"
        }
        val jpegBytes = stream.toByteArray()
        val decoded = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
            ?: error("Could not decode camera frame")
        return rotateBitmap(decoded, image.imageInfo.rotationDegrees)
    }

    fun cropToRoi(bitmap: Bitmap, roiFraction: RectF): Bitmap {
        val left = (bitmap.width * roiFraction.left).roundToInt().coerceIn(0, bitmap.width - 1)
        val top = (bitmap.height * roiFraction.top).roundToInt().coerceIn(0, bitmap.height - 1)
        val right = (bitmap.width * roiFraction.right).roundToInt().coerceIn(left + 1, bitmap.width)
        val bottom = (bitmap.height * roiFraction.bottom).roundToInt().coerceIn(top + 1, bitmap.height)
        return Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)
    }

    fun preprocess(bitmap: Bitmap): PreparedImage {
        val scale = OcrModelConfig.INPUT_HEIGHT.toFloat() / bitmap.height.toFloat()
        val targetWidth = (bitmap.width * scale)
            .roundToInt()
            .coerceIn(OcrModelConfig.MIN_INPUT_WIDTH, OcrModelConfig.MAX_INPUT_WIDTH)

        val scaled = Bitmap.createScaledBitmap(bitmap, targetWidth, OcrModelConfig.INPUT_HEIGHT, true)
        val binaryImage = buildBinaryImage(scaled, targetWidth, OcrModelConfig.INPUT_HEIGHT)
        if (scaled !== bitmap) {
            scaled.recycle()
        }

        return PreparedImage(tensorData = binaryImage.binaryData, width = targetWidth)
    }

    fun segmentLineBitmaps(bitmap: Bitmap): List<Bitmap> {
        if (bitmap.width <= 1 || bitmap.height <= 1) {
            return listOf(bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false))
        }

        val analysisHeight = bitmap.height
            .coerceAtLeast(OcrModelConfig.INPUT_HEIGHT)
            .coerceAtMost(OcrModelConfig.SEGMENTATION_TARGET_HEIGHT)
        val scale = analysisHeight.toFloat() / bitmap.height.toFloat()
        val analysisWidth = (bitmap.width * scale).roundToInt().coerceAtLeast(1)

        val analysisBitmap = if (analysisWidth == bitmap.width && analysisHeight == bitmap.height) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, analysisWidth, analysisHeight, true)
        }

        val binaryImage = buildBinaryImage(analysisBitmap, analysisWidth, analysisHeight)
        val rects = LineSegmenter.findLineRects(
            binary = binaryImage.binaryData,
            width = binaryImage.width,
            height = binaryImage.height,
            minRowActivityRatio = OcrModelConfig.MIN_ROW_ACTIVITY_RATIO,
            minColumnActivityRatio = OcrModelConfig.MIN_COLUMN_ACTIVITY_RATIO
        ).take(OcrModelConfig.MAX_PARAGRAPH_LINES)

        if (analysisBitmap !== bitmap) {
            analysisBitmap.recycle()
        }

        if (rects.isEmpty()) {
            return emptyList()
        }

        val horizontalPadding = (bitmap.width * OcrModelConfig.HORIZONTAL_PADDING_RATIO).roundToInt()
        val verticalPadding = (bitmap.height * OcrModelConfig.VERTICAL_PADDING_RATIO).roundToInt()

        return rects.map { rect ->
            val mapped = mapRectToOriginal(
                rect = rect,
                analysisWidth = binaryImage.width,
                analysisHeight = binaryImage.height,
                originalWidth = bitmap.width,
                originalHeight = bitmap.height,
                horizontalPadding = horizontalPadding,
                verticalPadding = verticalPadding
            )
            Bitmap.createBitmap(bitmap, mapped.left, mapped.top, mapped.width, mapped.height)
        }
    }

    private fun mapRectToOriginal(
        rect: LineSegmenter.IntRect,
        analysisWidth: Int,
        analysisHeight: Int,
        originalWidth: Int,
        originalHeight: Int,
        horizontalPadding: Int,
        verticalPadding: Int
    ): LineSegmenter.IntRect {
        val left = (((rect.left.toFloat() / analysisWidth) * originalWidth).roundToInt() - horizontalPadding)
            .coerceAtLeast(0)
        val top = (((rect.top.toFloat() / analysisHeight) * originalHeight).roundToInt() - verticalPadding)
            .coerceAtLeast(0)
        val right = (((rect.right.toFloat() / analysisWidth) * originalWidth).roundToInt() + horizontalPadding)
            .coerceAtMost(originalWidth)
        val bottom = (((rect.bottom.toFloat() / analysisHeight) * originalHeight).roundToInt() + verticalPadding)
            .coerceAtMost(originalHeight)
        return LineSegmenter.IntRect(
            left = left,
            top = top,
            right = max(left + 1, right),
            bottom = max(top + 1, bottom)
        )
    }

    private fun buildBinaryImage(bitmap: Bitmap, width: Int, height: Int): BinaryImage {
        val grayscale = toGrayscale(bitmap)
        val normalized = normalizeByBackground(grayscale, width, height)
        clampBrightValues(normalized, 0.9f)
        val equalized = histogramEqualize(normalized)
        clampBrightValues(equalized, 0.75f)
        val binary = adaptiveThresholdInverted(
            equalized,
            width,
            height,
            OcrModelConfig.ADAPTIVE_BLOCK_SIZE,
            OcrModelConfig.ADAPTIVE_C
        )
        val cleaned = removeIsolated(binary, width, height)
        return BinaryImage(
            binaryData = cleaned,
            width = width,
            height = height
        )
    }

    private fun toGrayscale(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        return FloatArray(pixels.size) { index ->
            val pixel = pixels[index]
            val red = (pixel shr 16) and 0xFF
            val green = (pixel shr 8) and 0xFF
            val blue = pixel and 0xFF
            (0.299f * red) + (0.587f * green) + (0.114f * blue)
        }
    }

    private fun normalizeByBackground(source: FloatArray, width: Int, height: Int): FloatArray {
        val integral = buildIntegralImage(source, width, height)
        val radius = OcrModelConfig.BACKGROUND_WINDOW_RADIUS
        return FloatArray(source.size) { index ->
            val x = index % width
            val y = index / width
            val left = max(0, x - radius)
            val top = max(0, y - radius)
            val right = min(width - 1, x + radius)
            val bottom = min(height - 1, y + radius)
            val area = ((right - left + 1) * (bottom - top + 1)).toFloat()
            val mean = sumRegion(integral, width, left, top, right, bottom) / area
            ((source[index] * 255f) / max(mean, 1f)).coerceIn(0f, 255f)
        }
    }

    private fun clampBrightValues(source: FloatArray, fraction: Float) {
        val maxValue = source.maxOrNull() ?: return
        val threshold = maxValue * fraction
        for (index in source.indices) {
            if (source[index] >= threshold) {
                source[index] = 255f
            }
        }
    }

    private fun histogramEqualize(source: FloatArray): FloatArray {
        val histogram = IntArray(256)
        for (value in source) {
            histogram[value.roundToInt().coerceIn(0, 255)]++
        }

        var cdfMin = 0
        var cumulative = 0
        for (count in histogram) {
            cumulative += count
            if (count > 0) {
                cdfMin = cumulative
                break
            }
        }

        if (cdfMin == source.size) {
            return source.copyOf()
        }

        cumulative = 0
        val lookup = FloatArray(256)
        for (index in histogram.indices) {
            cumulative += histogram[index]
            lookup[index] = (((cumulative - cdfMin).toFloat() / (source.size - cdfMin).coerceAtLeast(1)) * 255f)
                .coerceIn(0f, 255f)
        }

        return FloatArray(source.size) { index ->
            lookup[source[index].roundToInt().coerceIn(0, 255)]
        }
    }

    private fun adaptiveThresholdInverted(
        source: FloatArray,
        width: Int,
        height: Int,
        blockSize: Int,
        c: Float
    ): FloatArray {
        val integral = buildIntegralImage(source, width, height)
        val radius = blockSize / 2
        return FloatArray(source.size) { index ->
            val x = index % width
            val y = index / width
            val left = max(0, x - radius)
            val top = max(0, y - radius)
            val right = min(width - 1, x + radius)
            val bottom = min(height - 1, y + radius)
            val area = ((right - left + 1) * (bottom - top + 1)).toFloat()
            val mean = sumRegion(integral, width, left, top, right, bottom) / area
            if (source[index] < mean - c) 1f else 0f
        }
    }

    private fun removeIsolated(source: FloatArray, width: Int, height: Int): FloatArray {
        val output = source.copyOf()
        for (y in 0 until height) {
            for (x in 0 until width) {
                val index = y * width + x
                if (source[index] < 0.5f) {
                    continue
                }

                var neighbors = 0
                for (dy in -1..1) {
                    for (dx in -1..1) {
                        val nx = x + dx
                        val ny = y + dy
                        if (nx in 0 until width && ny in 0 until height) {
                            if (source[ny * width + nx] > 0.5f) {
                                neighbors++
                            }
                        }
                    }
                }

                if (neighbors <= 2) {
                    output[index] = 0f
                }
            }
        }
        return output
    }

    private fun buildIntegralImage(source: FloatArray, width: Int, height: Int): FloatArray {
        val integral = FloatArray((width + 1) * (height + 1))
        for (y in 1..height) {
            var rowSum = 0f
            for (x in 1..width) {
                rowSum += source[(y - 1) * width + (x - 1)]
                integral[y * (width + 1) + x] = integral[(y - 1) * (width + 1) + x] + rowSum
            }
        }
        return integral
    }

    private fun sumRegion(
        integral: FloatArray,
        width: Int,
        left: Int,
        top: Int,
        right: Int,
        bottom: Int
    ): Float {
        val stride = width + 1
        val x1 = left
        val y1 = top
        val x2 = right + 1
        val y2 = bottom + 1
        return integral[y2 * stride + x2] - integral[y1 * stride + x2] -
            integral[y2 * stride + x1] + integral[y1 * stride + x1]
    }

    private fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) {
            return bitmap
        }
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        bitmap.recycle()
        return rotated
    }

    private fun yuv420888ToNv21(image: ImageProxy): ByteArray {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val uvSize = width * height / 2
        val nv21 = ByteArray(ySize + uvSize)

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        var position = 0
        for (row in 0 until height) {
            yBuffer.position(row * yPlane.rowStride)
            yBuffer.get(nv21, position, width)
            position += width
        }

        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val chromaHeight = height / 2
        val chromaWidth = width / 2

        for (row in 0 until chromaHeight) {
            val uRowStart = row * uPlane.rowStride
            val vRowStart = row * vPlane.rowStride
            for (col in 0 until chromaWidth) {
                nv21[position++] = vBuffer.get(vRowStart + col * vPlane.pixelStride)
                nv21[position++] = uBuffer.get(uRowStart + col * uPlane.pixelStride)
            }
        }

        return nv21
    }
}
