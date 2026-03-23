package com.codex.vietocr.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.codex.vietocr.R
import com.codex.vietocr.ml.OcrModelConfig

class RoiOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val scrimPaint = Paint().apply {
        color = 0x7A081711
    }

    private val framePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(2f)
        color = ContextCompat.getColor(context, R.color.green_soft)
    }

    private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeWidth = dp(6f)
        color = ContextCompat.getColor(context, R.color.orange_primary)
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.white)
        textSize = dp(14f)
        isFakeBoldText = true
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val roi = roiRect()
        val radius = dp(18f)

        canvas.drawRect(0f, 0f, width.toFloat(), roi.top, scrimPaint)
        canvas.drawRect(0f, roi.top, roi.left, roi.bottom, scrimPaint)
        canvas.drawRect(roi.right, roi.top, width.toFloat(), roi.bottom, scrimPaint)
        canvas.drawRect(0f, roi.bottom, width.toFloat(), height.toFloat(), scrimPaint)

        canvas.drawRoundRect(roi, radius, radius, framePaint)
        drawCorners(canvas, roi)
        canvas.drawText("Đặt 1-8 dòng chữ viết tay vào khung", roi.left, roi.top - dp(12f), labelPaint)
    }

    private fun drawCorners(canvas: Canvas, rect: RectF) {
        val length = dp(26f)
        val radius = dp(18f)

        canvas.drawLine(rect.left, rect.top + radius, rect.left, rect.top + radius + length, cornerPaint)
        canvas.drawLine(rect.left + radius, rect.top, rect.left + radius + length, rect.top, cornerPaint)

        canvas.drawLine(rect.right, rect.top + radius, rect.right, rect.top + radius + length, cornerPaint)
        canvas.drawLine(rect.right - radius, rect.top, rect.right - radius - length, rect.top, cornerPaint)

        canvas.drawLine(rect.left, rect.bottom - radius, rect.left, rect.bottom - radius - length, cornerPaint)
        canvas.drawLine(rect.left + radius, rect.bottom, rect.left + radius + length, rect.bottom, cornerPaint)

        canvas.drawLine(rect.right, rect.bottom - radius, rect.right, rect.bottom - radius - length, cornerPaint)
        canvas.drawLine(rect.right - radius, rect.bottom, rect.right - radius - length, rect.bottom, cornerPaint)
    }

    private fun roiRect(): RectF {
        val fraction = OcrModelConfig.ROI_FRACTION
        return RectF(
            width * fraction.left,
            height * fraction.top,
            width * fraction.right,
            height * fraction.bottom
        )
    }

    private fun dp(value: Float): Float {
        return value * resources.displayMetrics.density
    }
}
