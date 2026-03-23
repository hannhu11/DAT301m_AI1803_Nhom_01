"""
=== CONVERT KERAS MODEL SANG ONNX (FIX LAMBDA LAYER) ===

Notebook này xây dựng lại kiến trúc model bằng code (không dùng Lambda layer),
sau đó load weights từ file .keras và convert sang ONNX.

Lý do: Model .keras được save bằng Keras 3 nhưng chứa Lambda layer.
Keras 3 không thể tự suy ra output shape của Lambda khi load lại.
→ Giải pháp: Rebuild model bằng code, dùng Reshape thay Lambda.
"""

# ============================================================
# CELL 1: Cài đặt thư viện
# ============================================================
import os, subprocess, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# IMPORTANT: Android ONNX Runtime cannot execute TensorFlow CuDNN-exported RNN ops
# such as CudnnRNNV3. Force CPU-only export from the very beginning.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

install("tf2onnx")
install("onnx")

print("✅ Đã cài đặt xong tf2onnx và onnx")

# ============================================================
# CELL 2: Import thư viện
# ============================================================
import keras
keras.config.enable_unsafe_deserialization()

import tensorflow as tf
import tf2onnx
import numpy as np
import zipfile
import tempfile
import onnx

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

print(f"TensorFlow: {tf.__version__}, Keras: {keras.__version__}, tf2onnx: {tf2onnx.__version__}")
print("✅ Import thành công")

# ============================================================
# CELL 3: Cấu hình đường dẫn (CHỈNH SỬA CHO PHÙ HỢP)
# ============================================================
TRAIN8_KERAS = "/kaggle/input/datasets/trinhminhkhoak18hcm/ocr-inference-train-8/ocr_inference.keras"
TRAIN8_WEIGHTS = "/kaggle/input/datasets/trinhminhkhoak18hcm/profile-robust-balanced-train-8/profile_robust_balanced.weights.h5"

TRAIN15_KERAS = "/kaggle/input/datasets/trinhminhkhoak18hcm/ocr-sentence-export-train-15/ocr_sentence_export.keras"
TRAIN15_WEIGHTS = "/kaggle/input/datasets/trinhminhkhoak18hcm/selected-sentence-weights-train-15/selected_sentence.weights.h5"

OUTPUT_DIR = "/kaggle/working"
print("✅ Cấu hình đường dẫn xong")

# ============================================================
# CELL 4: Xây dựng kiến trúc model bằng code
# ============================================================
def build_crnn_model(num_classes=221):
    """
    Xây dựng lại kiến trúc CRNN-CTC model.
    Kiến trúc được suy ra từ config JSON của model gốc:
    - CNN: 5 Conv2D blocks (64→128→256→256→512) với BatchNorm + ReLU + MaxPool
    - Reshape: (batch, 1, width, 512) → (batch, width, 512) [thay Lambda bằng Reshape]
    - RNN: 2x Bidirectional LSTM (256 units, dropout=0.25)
    - Output: Dense 221 (softmax) cho CTC decoding
    """
    from keras import layers, Model

    inp = layers.Input(shape=(96, None, 1), name='input_image')

    # Block 1: Conv2D(64) + BN + ReLU + MaxPool(3,2)
    x = layers.Conv2D(64, (3,3), padding='same', name='conv2d')(inp)
    x = layers.BatchNormalization(name='batch_normalization')(x)
    x = layers.Activation('relu', name='activation')(x)
    x = layers.MaxPooling2D(pool_size=(3,2), strides=(3,2), name='max_pooling2d')(x)
    # → (batch, 32, W/2, 64)

    # Block 2: Conv2D(128) + BN + ReLU + MaxPool(2,2)
    x = layers.Conv2D(128, (3,3), padding='same', name='conv2d_1')(x)
    x = layers.BatchNormalization(name='batch_normalization_1')(x)
    x = layers.Activation('relu', name='activation_1')(x)
    x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='max_pooling2d_1')(x)
    # → (batch, 16, W/4, 128)

    # Block 3: Conv2D(256) + BN + ReLU + Dropout + MaxPool(2,1)
    x = layers.Conv2D(256, (3,3), padding='same', name='conv2d_2')(x)
    x = layers.BatchNormalization(name='batch_normalization_2')(x)
    x = layers.Activation('relu', name='activation_2')(x)
    x = layers.Dropout(0.2, name='dropout')(x)
    x = layers.MaxPooling2D(pool_size=(2,1), strides=(2,1), name='max_pooling2d_2')(x)
    # → (batch, 8, W/4, 256)

    # Block 4: Conv2D(256) + BN + ReLU + Dropout + MaxPool(2,1)
    x = layers.Conv2D(256, (3,3), padding='same', name='conv2d_3')(x)
    x = layers.BatchNormalization(name='batch_normalization_3')(x)
    x = layers.Activation('relu', name='activation_3')(x)
    x = layers.Dropout(0.2, name='dropout_1')(x)
    x = layers.MaxPooling2D(pool_size=(2,1), strides=(2,1), name='max_pooling2d_3')(x)
    # → (batch, 4, W/4, 256)

    # Block 5: Conv2D(512) + BN + ReLU + MaxPool(3,1)
    x = layers.Conv2D(512, (3,3), padding='same', name='conv2d_4')(x)
    x = layers.BatchNormalization(name='batch_normalization_4')(x)
    x = layers.Activation('relu', name='activation_4')(x)
    x = layers.MaxPooling2D(pool_size=(3,1), strides=(3,1), name='max_pooling2d_4')(x)
    # → (batch, 1, W/4, 512)

    # Reshape: (batch, 1, W, 512) → (batch, W, 512)
    # Avoid Lambda here so export stays simpler and more portable.
    x = layers.Reshape(target_shape=(-1, 512), name='reshape_seq')(x)
    # → (batch, W/4, 512)

    # Bidirectional LSTM x2
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.25, name='forward_lstm'),
        backward_layer=layers.LSTM(256, return_sequences=True, dropout=0.25, go_backwards=True, name='backward_lstm'),
        name='bidirectional'
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.25, name='forward_lstm_1'),
        backward_layer=layers.LSTM(256, return_sequences=True, dropout=0.25, go_backwards=True, name='backward_lstm_1'),
        name='bidirectional_1'
    )(x)

    # Output layer
    out = layers.Dense(num_classes, activation='softmax', name='dense')(x)

    model = Model(inputs=inp, outputs=out, name='ocr_inference_model')
    return model

print("✅ Hàm build model đã sẵn sàng")

# ============================================================
# CELL 5: Hàm load weights từ file .keras (zip archive)
# ============================================================
def load_weights_from_keras_file(model, keras_path):
    """
    File .keras là 1 zip chứa config.json và model.weights.h5.
    Extract weights và load vào model đã build.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(keras_path, 'r') as z:
            z.extractall(tmpdir)

        weights_file = os.path.join(tmpdir, 'model.weights.h5')
        if os.path.exists(weights_file):
            model.load_weights(weights_file)
            print(f"✅ Đã load weights từ {keras_path}")
        else:
            # Thử tìm file weights khác trong zip
            for f in os.listdir(tmpdir):
                if f.endswith('.h5'):
                    model.load_weights(os.path.join(tmpdir, f))
                    print(f"✅ Đã load weights từ {f}")
                    return
            print(f"❌ Không tìm thấy file weights trong {keras_path}")

# ============================================================
# CELL 6: Hàm convert model
# ============================================================
def convert_to_onnx(keras_path, weights_path, output_path, model_name, num_classes=221):
    """Build model → load weights → convert ONNX"""
    print(f"\n{'='*60}")
    print(f"🔄 Đang convert: {model_name}")
    print(f"{'='*60}")

    if not os.path.exists(keras_path):
        print(f"❌ Không tìm thấy: {keras_path}")
        return False

    try:
        with tf.device("/CPU:0"):
            # 1. Build model architecture
            print("🔨 Đang build model architecture trên CPU...")
            model = build_crnn_model(num_classes=num_classes)
            model.summary()

            # 2. Load weights từ file .keras (zip)
            print(f"📂 Đang load weights từ {keras_path}...")
            load_weights_from_keras_file(model, keras_path)

            # 3. Nếu có file weights riêng (ví dụ: profile_robust_balanced.weights.h5)
            if weights_path and os.path.exists(weights_path):
                print(f"📂 Đang load weights bổ sung từ {weights_path}...")
                model.load_weights(weights_path)

            # Build once on CPU before conversion.
            _ = model(tf.zeros((1, 96, 256, 1), dtype=tf.float32), training=False)

            # 4. Convert sang ONNX
            print("🔄 Đang convert sang ONNX (CPU-safe export)...")
            input_spec = (tf.TensorSpec((1, 96, None, 1), tf.float32, name="input"),)

            tf2onnx.convert.from_keras(
                model,
                input_signature=input_spec,
                opset=13,
                output_path=output_path
            )

        if not os.path.exists(output_path):
            print("❌ File ONNX không được tạo")
            return False

        exported = onnx.load(output_path)
        op_types = sorted({node.op_type for node in exported.graph.node})
        print("🔍 Ops trong ONNX:", op_types)

        banned_ops = {"CudnnRNNV3"}
        hit_banned = sorted(banned_ops.intersection(op_types))
        if hit_banned:
            print(f"❌ ONNX vẫn chứa op không tương thích Android: {hit_banned}")
            return False

        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"✅ THÀNH CÔNG! → {output_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"❌ LỖI: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# CELL 7: Convert Train-8 Robust → crnn_robust.onnx
# ============================================================
robust_ok = convert_to_onnx(
    TRAIN8_KERAS,
    TRAIN8_WEIGHTS,
    os.path.join(OUTPUT_DIR, "crnn_robust.onnx"),
    "🛡️ Robust OCR (Train-8)"
)

# ============================================================
# CELL 8: Convert Train-15 Clean → crnn_clean.onnx
# ============================================================
# Lưu ý: train-15 có thể dùng kiến trúc khác (ocr_sentence_export.keras)
# Nếu lỗi, thử thay num_classes hoặc kiến trúc
clean_ok = convert_to_onnx(
    TRAIN15_KERAS,
    TRAIN15_WEIGHTS,
    os.path.join(OUTPUT_DIR, "crnn_clean.onnx"),
    "📄 Clean OCR (Train-15)"
)

# ============================================================
# CELL 9: Kết quả
# ============================================================
print(f"\n{'='*60}")
print(f"📊 KẾT QUẢ:")
print(f"{'='*60}")
print(f"  Robust: {'✅' if robust_ok else '❌'}")
print(f"  Clean : {'✅' if clean_ok else '❌'}")
print(f"\n📥 Download file ONNX từ tab 'Output' → copy vào app/src/main/assets/models/")
print("⚠️ Nếu output vẫn chứa CudnnRNNV3 thì KHÔNG dùng cho Android.")
