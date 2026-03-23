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

## Main Deliverables

- Final report PDF: `report/DAT301m_AI1803_Nhom_01.pdf`
- Overleaf package: `overleaf/DAT301m_AI1803_Nhom_01_overleaf.zip`
- Android APK: `apk/DAT301m_AI1803_Nhom_01.apk`
- ONNX models:
  - `models/onnx/crnn_clean.onnx`
  - `models/onnx/crnn_robust.onnx`


