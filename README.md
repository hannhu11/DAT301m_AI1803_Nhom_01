# DAT301m_AI1803_Nhom_01

Submission repository for the DAT301m group project on Vietnamese handwriting OCR under adverse conditions.

## Repository Structure

- `report/`
  - Final report PDF: `DAT301m_AI1803_Nhom_01.pdf`
  - LaTeX source: `DAT301m_AI1803_Nhom_01.tex`
  - Figures used in the report
- `overleaf/`
  - `main.tex`
  - report figures
  - upload-ready Overleaf package zip
- `dat301m-ocr-train-8.ipynb`
  - profile-search notebook used for robust, clean, and compromise checkpoint analysis
- `dat301m-ocr-train-15.ipynb`
  - clean specialist with selective token correction
- `dat301m-ocr-train-17.ipynb`
  - runtime and mobile-oriented OCR notebook
- `notebooks/`
  - additional conversion and testing notebooks/scripts
- `outputs/`
  - exported outputs for train 8, train 15, and train 17
- `models/onnx/`
  - Android-safe ONNX models
- `android_app/`
  - Android Studio project source
- `apk/`
  - demo APK for mobile OCR testing

## Report-to-Code Mapping

- The one-checkpoint compromise result discussed in the report is derived from `dat301m-ocr-train-8.ipynb`.
- The clean specialist ablation discussed in the report is derived from `dat301m-ocr-train-15.ipynb`.
- The runtime ablation and page-segmentation prototype discussed in the report are derived from `dat301m-ocr-train-17.ipynb`.
- The Android demo uses exported ONNX models from the same CNN-BiLSTM-CTC model family.

## Main Deliverables

- Final report PDF: `report/DAT301m_AI1803_Nhom_01.pdf`
- Overleaf package: `overleaf/DAT301m_AI1803_Nhom_01_overleaf.zip`
- Android APK: `apk/DAT301m_AI1803_Nhom_01.apk`
- ONNX models:
  - `models/onnx/crnn_clean.onnx`
  - `models/onnx/crnn_robust.onnx`

## Notes

- Large binary artifacts are tracked with Git LFS.
- The repository keeps the original notebook names for traceability against the experimental outputs.
