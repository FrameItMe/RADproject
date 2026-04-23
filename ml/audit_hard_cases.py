"""Audit hard benign/malignant cases for targeted retraining.

Scans splits in dataset/, runs the current model, and writes hardest cases
(misclassified and low true-class confidence) to CSV and JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


IMG_SIZE = 224
CLASS_NAMES = ["normal", "benign", "malignant"]
TARGET_CLASSES = {"benign", "malignant"}


def preprocess_image(path: Path) -> np.ndarray:
    img = keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb")
    arr = keras.utils.img_to_array(img).astype("float32")
    arr = tf.image.rgb_to_grayscale(arr)
    arr = tf.image.grayscale_to_rgb(arr)
    arr = arr.numpy() / 255.0
    return arr


def gather_samples(data_dir: Path, split: str):
    samples = []
    for class_name in CLASS_NAMES:
        class_dir = data_dir / split / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.glob("*.png")):
            samples.append((path, class_name))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("experiments/audit/hard_cases"))
    parser.add_argument("--top_k", type=int, default=60)
    args = parser.parse_args()

    model = keras.models.load_model(args.model_path, compile=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for split in ("train", "val", "test"):
        split_rows = []
        samples = gather_samples(args.data_dir, split)
        for path, true_name in samples:
            if true_name not in TARGET_CLASSES:
                continue

            x = preprocess_image(path)
            probs = model.predict(np.expand_dims(x, axis=0), verbose=0)[0]

            true_idx = CLASS_NAMES.index(true_name)
            pred_idx = int(np.argmax(probs))
            pred_name = CLASS_NAMES[pred_idx]

            true_prob = float(probs[true_idx])
            pred_prob = float(probs[pred_idx])
            is_error = int(pred_idx != true_idx)

            # Prioritize wrong predictions, then low true confidence.
            difficulty = (10.0 if is_error else 0.0) + (1.0 - true_prob)

            row = {
                "split": split,
                "file": str(path),
                "true_class": true_name,
                "pred_class": pred_name,
                "true_prob": true_prob,
                "pred_prob": pred_prob,
                "p_normal": float(probs[0]),
                "p_benign": float(probs[1]),
                "p_malignant": float(probs[2]),
                "is_error": is_error,
                "difficulty": difficulty,
            }
            split_rows.append(row)
            all_rows.append(row)

        split_rows.sort(key=lambda r: (r["is_error"], r["difficulty"]), reverse=True)
        top_rows = split_rows[: args.top_k]

        csv_path = args.out_dir / f"{split}_hard_cases.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split",
                    "file",
                    "true_class",
                    "pred_class",
                    "true_prob",
                    "pred_prob",
                    "p_normal",
                    "p_benign",
                    "p_malignant",
                    "is_error",
                    "difficulty",
                ],
            )
            writer.writeheader()
            writer.writerows(top_rows)

    all_rows.sort(key=lambda r: (r["is_error"], r["difficulty"]), reverse=True)
    summary = {
        "model_path": str(args.model_path),
        "data_dir": str(args.data_dir),
        "top_k": args.top_k,
        "total_target_samples": len(all_rows),
        "total_errors": int(sum(r["is_error"] for r in all_rows)),
        "error_rate": float(sum(r["is_error"] for r in all_rows) / max(1, len(all_rows))),
        "top_global": all_rows[: args.top_k],
    }

    json_path = args.out_dir / "hard_case_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote audit to: {args.out_dir}")
    print(f"Total benign/malignant samples: {summary['total_target_samples']}")
    print(f"Total benign/malignant errors: {summary['total_errors']}")
    print(f"Error rate: {summary['error_rate']:.4f}")


if __name__ == "__main__":
    main()
