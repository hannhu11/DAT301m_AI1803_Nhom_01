package com.codex.vietocr

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Size
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.codex.vietocr.databinding.ActivityMainBinding
import com.codex.vietocr.ml.OcrAnalyzer
import com.codex.vietocr.ml.OcrMode
import com.codex.vietocr.ml.OcrModelConfig
import com.codex.vietocr.ml.OcrResult
import com.codex.vietocr.ml.OnnxOcrEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    private var cameraProvider: ProcessCameraProvider? = null
    private var ocrEngine: OnnxOcrEngine? = null
    private var lastResult: OcrResult? = null
    private var currentMode: OcrMode = OcrModelConfig.DEFAULT_MODE
    @Volatile
    private var isResultLocked: Boolean = false
    @Volatile
    private var captureRequested: Boolean = false

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                bindCameraUseCases()
            } else {
                binding.statusValue.text = getString(R.string.status_camera_permission_required)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        binding.copyButton.isEnabled = false
        binding.shareButton.isEnabled = false

        binding.lockButton.setOnClickListener { toggleResultLock() }
        binding.captureButton.setOnClickListener { requestCaptureFrame() }
        binding.copyButton.setOnClickListener { copyBestText() }
        binding.shareButton.setOnClickListener { shareResult() }

        // Mode toggle listener
        binding.modeToggleGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (isChecked) {
                val newMode = when (checkedId) {
                    R.id.btnModeClean -> OcrMode.CLEAN
                    R.id.btnModeRobust -> OcrMode.ROBUST
                    else -> return@addOnButtonCheckedListener
                }
                if (newMode != currentMode) {
                    currentMode = newMode
                    switchMode(newMode)
                }
            }
        }

        loadModelAndStart(currentMode)
        updateActionButtons()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraProvider?.unbindAll()
        cameraExecutor.shutdown()
        ocrEngine?.close()
    }

    private fun switchMode(mode: OcrMode) {
        // Stop camera analysis while loading new model
        cameraProvider?.unbindAll()
        ocrEngine?.close()
        ocrEngine = null
        lastResult = null
        isResultLocked = false
        captureRequested = false

        binding.recognizedTextValue.text = getString(R.string.placeholder_text)
        binding.confidenceValue.text = getString(R.string.placeholder_confidence)
        binding.metaValue.text = getString(R.string.meta_initial)
        binding.copyButton.isEnabled = false
        binding.shareButton.isEnabled = false
        updateActionButtons()

        loadModelAndStart(mode)
    }

    private fun loadModelAndStart(mode: OcrMode) {
        lifecycleScope.launch {
            binding.statusValue.text = "Đang tải model ${mode.displayName}..."
            val modelResult = withContext(Dispatchers.Default) {
                runCatching { OnnxOcrEngine(applicationContext, mode) }
            }

            modelResult
                .onSuccess {
                    ocrEngine = it
                    binding.statusValue.text = "✓ ${mode.displayName}"
                    updateActionButtons()
                    ensureCameraPermission()
                }
                .onFailure {
                    binding.statusValue.text = getString(R.string.status_model_error)
                    binding.metaValue.text = it.message ?: getString(R.string.status_model_error)
                }
        }
    }

    private fun ensureCameraPermission() {
        val granted = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        if (granted) {
            bindCameraUseCases()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun bindCameraUseCases() {
        val engine = ocrEngine ?: return
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(
            {
                runCatching {
                    val provider = cameraProviderFuture.get()
                    cameraProvider = provider

                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(binding.previewView.surfaceProvider)
                    }

                    val analysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setTargetResolution(
                            Size(
                                OcrModelConfig.ANALYSIS_WIDTH,
                                OcrModelConfig.ANALYSIS_HEIGHT
                            )
                        )
                        .build()
                        .also {
                            it.setAnalyzer(
                                cameraExecutor,
                                OcrAnalyzer(
                                    engine = engine,
                                    onResult = ::renderResult,
                                    onFailure = { throwable ->
                                        runOnUiThread {
                                            binding.statusValue.text = throwable.message
                                                ?: getString(R.string.status_camera_error)
                                        }
                                    },
                                    shouldAnalyze = { !isResultLocked || captureRequested }
                                )
                            )
                        }

                    provider.unbindAll()
                    provider.bindToLifecycle(
                        this,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        analysis
                    )
                }.onFailure {
                    binding.statusValue.text = getString(R.string.status_camera_error)
                    binding.metaValue.text = it.message ?: getString(R.string.status_camera_error)
                }
            },
            ContextCompat.getMainExecutor(this)
        )
    }

    private fun renderResult(result: OcrResult) {
        runOnUiThread {
            if (captureRequested) {
                if (result.recognizedText.isBlank()) {
                    return@runOnUiThread
                }
                captureRequested = false
                isResultLocked = true
                binding.statusValue.text = getString(R.string.status_result_locked)
                applyResult(result)
                updateActionButtons()
                return@runOnUiThread
            }

            if (isResultLocked) {
                return@runOnUiThread
            }

            applyResult(result)
            updateActionButtons()
        }
    }

    private fun applyResult(result: OcrResult) {
        lastResult = result
        binding.recognizedTextValue.text = if (result.recognizedText.isBlank()) {
            getString(R.string.placeholder_text)
        } else {
            result.recognizedText
        }
        binding.confidenceValue.text = String.format(
            Locale.getDefault(),
            "%.1f%%",
            result.confidence
        )
        binding.metaValue.text = buildString {
            append(currentMode.displayName)
            append(" | ")
            append(result.lineCount)
            append(" dòng | ")
            append(result.frameLatencyMs)
            append(" ms")
            if (result.modelInputWidth > 0) {
                append(" | w=")
                append(result.modelInputWidth)
            }
        }
        binding.copyButton.isEnabled = result.recognizedText.isNotBlank()
        binding.shareButton.isEnabled = result.recognizedText.isNotBlank()
    }

    private fun toggleResultLock() {
        if (!isResultLocked) {
            val result = lastResult
            if (result == null || result.recognizedText.isBlank()) {
                Toast.makeText(this, getString(R.string.no_result_to_lock), Toast.LENGTH_SHORT).show()
                return
            }
            isResultLocked = true
            captureRequested = false
            binding.statusValue.text = getString(R.string.status_result_locked)
        } else {
            isResultLocked = false
            binding.statusValue.text = "✓ ${currentMode.displayName}"
        }
        updateActionButtons()
    }

    private fun requestCaptureFrame() {
        captureRequested = true
        isResultLocked = false
        binding.statusValue.text = getString(R.string.status_capture_pending)
        updateActionButtons()
    }

    private fun updateActionButtons() {
        binding.lockButton.text = if (isResultLocked) {
            getString(R.string.unlock_button)
        } else {
            getString(R.string.lock_button)
        }
        binding.lockButton.isEnabled = ocrEngine != null
        binding.captureButton.isEnabled = ocrEngine != null && !captureRequested
        if (captureRequested) {
            binding.captureButton.text = getString(R.string.capture_pending_button)
        } else {
            binding.captureButton.text = getString(R.string.capture_button)
        }
    }

    private fun copyBestText() {
        val result = lastResult ?: return
        val textToCopy = result.recognizedText
        if (textToCopy.isBlank()) return

        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("ocr_result", textToCopy))
        Toast.makeText(this, textToCopy, Toast.LENGTH_SHORT).show()
    }

    private fun shareResult() {
        val result = lastResult ?: return
        if (result.recognizedText.isBlank()) return

        val shareBody = buildString {
            append("Nhận diện (${currentMode.displayName}): ")
            append(result.recognizedText)
            append("\nĐộ tin cậy: ")
            append(String.format(Locale.getDefault(), "%.1f%%", result.confidence))
        }

        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_SUBJECT, getString(R.string.share_title))
            putExtra(Intent.EXTRA_TEXT, shareBody)
        }

        startActivity(Intent.createChooser(intent, getString(R.string.share_title)))
    }
}
