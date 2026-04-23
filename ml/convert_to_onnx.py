"""Convert Keras mammogram model to ONNX and optionally quantize it.

Usage:
    python ml/convert_to_onnx.py --model artifacts/best_model.keras --out web/model
    python ml/convert_to_onnx.py --model artifacts/best_model.keras --out web/model --quantize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _patch_keras_compat():
    """Monkey-patch Keras layers so old checkpoints load on newer TF."""
    import tensorflow as tf

    orig_bn = tf.keras.layers.BatchNormalization.__init__
    orig_dense = tf.keras.layers.Dense.__init__

    def _bn_init(self, *a, **kw):
        kw.pop("renorm", None)
        kw.pop("renorm_clipping", None)
        kw.pop("renorm_momentum", None)
        return orig_bn(self, *a, **kw)

    def _dense_init(self, *a, **kw):
        kw.pop("quantization_config", None)
        return orig_dense(self, *a, **kw)

    tf.keras.layers.BatchNormalization.__init__ = _bn_init
    tf.keras.layers.Dense.__init__ = _dense_init


def convert_keras_to_onnx(model_path: Path, onnx_path: Path) -> None:
    """Convert a Keras .keras model to ONNX format."""
    _patch_keras_compat()

    import tensorflow as tf
    import tf2onnx

    print(f"Loading Keras model from {model_path} ...")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    print(f"  Input shape : {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    input_sig = [tf.TensorSpec(shape=(1, 224, 224, 3), dtype=tf.float32, name="input")]
    print("Converting to ONNX ...")
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_sig, opset=13)

    import onnx
    onnx.save(onnx_model, str(onnx_path))
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Saved ONNX model: {onnx_path}  ({size_mb:.1f} MB)")
    return onnx_model


def quantize_onnx(onnx_path: Path, quantized_path: Path) -> None:
    """Apply dynamic quantization (INT8 weights) to reduce model size."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print("Quantizing ONNX model (dynamic INT8) ...")
    quantize_dynamic(
        str(onnx_path),
        str(quantized_path),
        weight_type=QuantType.QUInt8,
    )
    size_mb = quantized_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Quantized model: {quantized_path}  ({size_mb:.1f} MB)")


def validate_onnx(onnx_path: Path, model_path: Path, n_samples: int = 5) -> None:
    """Run a quick validation comparing ONNX vs original Keras outputs."""
    _patch_keras_compat()

    import tensorflow as tf
    import onnxruntime as ort

    print(f"\nValidating ONNX model against Keras ({n_samples} random inputs) ...")
    keras_model = tf.keras.models.load_model(str(model_path), compile=False)

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name

    max_diff = 0.0
    for i in range(n_samples):
        dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
        keras_out = keras_model.predict(dummy, verbose=0)
        onnx_out = sess.run(None, {input_name: dummy})[0]
        diff = float(np.max(np.abs(keras_out - onnx_out)))
        max_diff = max(max_diff, diff)
        print(f"  Sample {i+1}: Keras={keras_out[0]}, ONNX={onnx_out[0]}, MaxDiff={diff:.6f}")

    print(f"\n  Max absolute difference: {max_diff:.6f}")
    if max_diff < 0.01:
        print("  [OK] Models match closely!")
    elif max_diff < 0.05:
        print("  [WARN] Minor differences detected, but acceptable for quantized model")
    else:
        print("  [ERROR] Large differences detected — review model conversion")


def main():
    parser = argparse.ArgumentParser(description="Convert Keras model to ONNX")
    parser.add_argument("--model", type=Path, default=Path("artifacts/best_model.keras"),
                        help="Path to Keras .keras model")
    parser.add_argument("--out", type=Path, default=Path("web/model"),
                        help="Output directory for ONNX model")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply dynamic INT8 quantization")
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Validate ONNX vs Keras outputs")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    onnx_path = args.out / "mammogram_classifier.onnx"
    convert_keras_to_onnx(args.model, onnx_path)

    final_path = onnx_path
    if args.quantize:
        quantized_path = args.out / "mammogram_classifier_q.onnx"
        quantize_onnx(onnx_path, quantized_path)
        final_path = quantized_path

    if args.validate:
        validate_onnx(final_path, args.model)

    # Write a small metadata file for the web app
    meta = {
        "model_file": final_path.name,
        "input_shape": [1, 224, 224, 3],
        "class_names": ["normal", "benign", "malignant"],
        "quantized": args.quantize,
        "size_bytes": final_path.stat().st_size,
    }
    meta_path = args.out / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[OK] Metadata written to {meta_path}")
    print(f"[OK] Final model ready for deployment: {final_path}")


if __name__ == "__main__":
    main()
