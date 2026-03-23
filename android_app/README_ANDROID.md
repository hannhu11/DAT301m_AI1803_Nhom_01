# Viet Handwriting OCR Android

## Model contract recovered from notebooks
- Clean ONNX asset: `app/src/main/assets/models/crnn_clean.onnx`
- Robust ONNX asset: `app/src/main/assets/models/crnn_robust.onnx`
- Input tensor: `[1, 96, dynamicWidth, 1]`
- Output tensor: `[1, timeSteps, 221]`
- Character count: `220`
- Blank index: `220`
- Character set asset: `app/src/main/assets/ocr_charset.txt`

## What the app does
- Uses CameraX preview + `STRATEGY_KEEP_ONLY_LATEST` analysis.
- Crops a fixed center ROI for handwritten text.
- Lets the user switch between `Giấy sạch` and `Điều kiện xấu`.
- Runs ONNX Runtime inference on the ROI.
- Applies greedy CTC decoding and reports confidence.
- Suggests Vietnamese corrections using Levenshtein distance and a starter dictionary.
- Lets the user copy or share the OCR result.

## Important files
- `app/src/main/java/com/codex/vietocr/MainActivity.kt`
- `app/src/main/java/com/codex/vietocr/ml/OcrAnalyzer.kt`
- `app/src/main/java/com/codex/vietocr/ml/OnnxOcrEngine.kt`
- `app/src/main/java/com/codex/vietocr/ml/ImagePreprocessor.kt`
- `app/src/main/java/com/codex/vietocr/ml/VietnameseAutoCorrector.kt`
- `app/src/main/res/layout/activity_main.xml`

## Notes
- The preprocessing pipeline is implemented in Kotlin to stay Android-native. It approximates the notebook flow with grayscale conversion, background normalization, histogram equalization, adaptive thresholding, and isolated-pixel cleanup.
- Use a larger OCR dictionary asset for materially better correction quality under live camera input.
- This workspace includes a Gradle wrapper and can be built with Android Studio JBR or a compatible local JDK.
