"""Train a 3-class mammogram classifier with transfer learning.

Expected dataset layout:

dataset/
  train/
    normal/
    benign/
    malignant/
  val/
    normal/
    benign/
    malignant/
  test/
    normal/
    benign/
    malignant/

This script fine-tunes a pretrained EfficientNet model and exports:
- a Keras model (.keras)
- a TensorFlow.js model directory for browser inference

Usage:
  python train_mammogram_model.py --data_dir dataset --out_dir artifacts
"""

from __future__ import annotations

import argparse
import json
import random
import math
from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


IMG_SIZE = 224
BATCH_SIZE = 16
SEED = 42
CLASS_NAMES = ["normal", "benign", "malignant"]


def count_split_samples(data_dir: Path, split: str) -> Counter:
    split_dir = data_dir / split
    counts = Counter()
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        counts[class_name] = len([p for p in class_dir.glob("*") if p.is_file()])
    return counts


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_class_weights(data_dir: Path):
    counts = count_split_samples(data_dir, "train")

    total = sum(counts.values())
    n_classes = len(CLASS_NAMES)
    weights = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        class_count = max(1, counts[class_name])
        weights[idx] = total / (n_classes * class_count)
    return weights


def apply_minority_boost(class_weights, boost: float):
    if not class_weights:
        return class_weights
    boosted = dict(class_weights)
    boosted[1] = float(boosted[1]) * boost
    boosted[2] = float(boosted[2]) * boost
    return boosted


def class_weight_vector(class_weights):
    if not class_weights:
        return None
    vec = np.array([float(class_weights[i]) for i in range(len(CLASS_NAMES))], dtype=np.float32)
    vec = vec / max(1e-6, float(vec.mean()))
    return vec


def build_datasets(data_dir: Path, balanced_sampling: bool = True):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    def preprocess(images, labels):
        images = tf.cast(images, tf.float32)
        # Mammography is grayscale by nature; force grayscale to reduce color noise.
        images = tf.image.rgb_to_grayscale(images)
        images = tf.image.grayscale_to_rgb(images)
        images = images / 255.0
        return images, labels

    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.map(preprocess, num_parallel_calls=autotune)
    val_ds = val_ds.map(preprocess, num_parallel_calls=autotune).cache().prefetch(autotune)
    test_ds = test_ds.map(preprocess, num_parallel_calls=autotune).cache().prefetch(autotune)

    if balanced_sampling:
        unbatched = train_ds.unbatch()
        per_class_streams = []
        for class_idx in range(len(CLASS_NAMES)):
            class_stream = unbatched.filter(
                lambda images, labels, class_idx=class_idx: tf.equal(
                    tf.argmax(labels, axis=-1), class_idx
                )
            )
            class_stream = class_stream.shuffle(512, reshuffle_each_iteration=True).repeat()
            per_class_streams.append(class_stream)

        train_ds = tf.data.Dataset.sample_from_datasets(
            per_class_streams,
            weights=[1.0 / len(CLASS_NAMES)] * len(CLASS_NAMES),
            seed=SEED,
        )
        train_ds = train_ds.batch(BATCH_SIZE).prefetch(autotune)
    else:
        train_ds = train_ds.cache().prefetch(autotune)

    return train_ds, val_ds, test_ds


def build_focal_loss(class_weight_alpha=None, gamma: float = 2.0):
    alpha = None
    if class_weight_alpha is not None:
        alpha = tf.constant(class_weight_alpha, dtype=tf.float32)

    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        modulating = tf.pow(1.0 - p_t, gamma)
        if alpha is not None:
            alpha_t = tf.reduce_sum(y_true * alpha, axis=-1)
        else:
            alpha_t = 1.0
        loss = -alpha_t * modulating * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    return focal_loss


def compute_macro_f1_and_balanced_accuracy(y_true_idx, y_pred_idx, class_names):
    conf = confusion_matrix_from_indices(y_true_idx, y_pred_idx, len(class_names))
    report = per_class_report(conf, class_names)
    recalls = [report[name]["recall"] for name in class_names]
    f1s = [report[name]["f1"] for name in class_names]
    return {
        "confusion_matrix": conf,
        "report": report,
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
    }


class ValMacroMetricsCallback(keras.callbacks.Callback):
    def __init__(self, val_ds, class_names):
        super().__init__()
        self.val_ds = val_ds
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_probs, val_true = collect_probs_and_labels(self.model, self.val_ds)
        y_true_idx = val_true.argmax(axis=1)
        y_pred_idx = val_probs.argmax(axis=1)
        summary = compute_macro_f1_and_balanced_accuracy(y_true_idx, y_pred_idx, self.class_names)
        logs["val_macro_f1"] = summary["macro_f1"]
        logs["val_balanced_accuracy"] = summary["balanced_accuracy"]
        print(
            f" - val_macro_f1: {summary['macro_f1']:.4f}"
            f" - val_balanced_accuracy: {summary['balanced_accuracy']:.4f}",
            end="",
        )


def build_model(num_classes: int = 3, class_weight_alpha=None) -> keras.Model:
    base = keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base._name = "backbone"
    base.trainable = False

    augment = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(height_factor=0.03, width_factor=0.03),
            layers.RandomZoom(height_factor=0.05, width_factor=0.05),
            layers.RandomContrast(0.08),
        ],
        name="augmentation",
    )

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)
    x = keras.applications.densenet.preprocess_input(x * 255.0)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(192, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=build_focal_loss(class_weight_alpha=class_weight_alpha, gamma=2.0),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def fine_tune_model(
    model: keras.Model,
    train_ds,
    val_ds,
    out_dir: Path,
    class_weights,
    steps_per_epoch=None,
    stage1_epochs: int = 16,
    stage2_epochs: int = 12,
    stage2_lr: float = 1e-5,
):
    best_path = out_dir / "best_model.keras"
    metric_callback = ValMacroMetricsCallback(val_ds, CLASS_NAMES)
    callbacks = [
        metric_callback,
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_macro_f1", mode="max"),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.35, monitor="val_macro_f1", mode="max"),
        keras.callbacks.ModelCheckpoint(str(best_path), save_best_only=True, monitor="val_macro_f1", mode="max"),
    ]

    stage1_kwargs = {
        "validation_data": val_ds,
        "epochs": stage1_epochs,
        "callbacks": callbacks,
    }
    if steps_per_epoch is not None:
        stage1_kwargs["steps_per_epoch"] = steps_per_epoch
    if class_weights:
        stage1_kwargs["class_weight"] = class_weights

    history_stage1 = model.fit(train_ds, **stage1_kwargs)

    base_candidates = [layer for layer in model.layers if isinstance(layer, keras.Model)]
    if not base_candidates:
        raise RuntimeError("Backbone model not found for fine-tuning")
    base_model = max(base_candidates, key=lambda layer: layer.count_params())
    base_model.trainable = True

    for layer in base_model.layers[:-80]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=stage2_lr),
        loss=build_focal_loss(class_weight_alpha=class_weight_vector(class_weights), gamma=1.5),
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    stage2_kwargs = {
        "validation_data": val_ds,
        "epochs": stage2_epochs,
        "callbacks": callbacks,
    }
    if steps_per_epoch is not None:
        stage2_kwargs["steps_per_epoch"] = steps_per_epoch
    if class_weights:
        stage2_kwargs["class_weight"] = class_weights

    history_stage2 = model.fit(train_ds, **stage2_kwargs)
    return history_stage1, history_stage2


def choose_calibrated_indices(probs: np.ndarray, class_names, thresholds):
    selected = []
    for row in probs:
        best_idx = int(np.argmax(row))
        best_score = -1.0
        for idx, class_name in enumerate(class_names):
            threshold = float(thresholds.get(class_name, 0.5))
            threshold = max(1e-6, threshold)
            score = float(row[idx]) / threshold
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
    return np.array(selected, dtype=np.int32)


def evaluate_model_with_calibration(test_probs, test_true, class_names, thresholds):
    y_true_idx = test_true.argmax(axis=1)
    y_pred_idx = choose_calibrated_indices(test_probs, class_names, thresholds)

    accuracy = float((y_true_idx == y_pred_idx).mean())
    ce_loss = keras.losses.CategoricalCrossentropy()(test_true, test_probs).numpy().item()

    auc_metric = keras.metrics.AUC(multi_label=True, num_labels=len(class_names), name="auc")
    auc_metric.update_state(test_true, test_probs)
    auc = float(auc_metric.result().numpy())

    summary = compute_macro_f1_and_balanced_accuracy(y_true_idx, y_pred_idx, class_names)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": summary["balanced_accuracy"],
        "macro_f1": summary["macro_f1"],
        "auc": auc,
        "loss": float(ce_loss),
        "report": summary["report"],
        "confusion_matrix": summary["confusion_matrix"],
    }


def collect_probs_and_labels(model: keras.Model, dataset):
    probs_all = []
    y_true_all = []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        probs_all.append(probs)
        y_true_all.append(labels.numpy())

    probs_np = tf.concat(probs_all, axis=0).numpy()
    y_true_np = tf.concat(y_true_all, axis=0).numpy()
    return probs_np, y_true_np


def confusion_matrix_from_indices(y_true_idx, y_pred_idx, n_classes: int):
    matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for t, p in zip(y_true_idx, y_pred_idx):
        matrix[int(t)][int(p)] += 1
    return matrix


def per_class_report(confusion_matrix, class_names):
    n_classes = len(class_names)
    total = sum(sum(row) for row in confusion_matrix)
    report = {}

    correct = 0
    for i in range(n_classes):
        tp = confusion_matrix[i][i]
        fp = sum(confusion_matrix[r][i] for r in range(n_classes) if r != i)
        fn = sum(confusion_matrix[i][c] for c in range(n_classes) if c != i)
        tn = total - tp - fp - fn
        correct += tp

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        support = sum(confusion_matrix[i])

        report[class_names[i]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    report["overall"] = {
        "accuracy": _safe_div(correct, total),
        "samples": int(total),
    }
    return report


def _f1_for_threshold(y_true_binary, y_score, threshold: float):
    y_pred = (y_score >= threshold).astype("int32")
    tp = int(((y_pred == 1) & (y_true_binary == 1)).sum())
    fp = int(((y_pred == 1) & (y_true_binary == 0)).sum())
    fn = int(((y_pred == 0) & (y_true_binary == 1)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return f1, precision, recall


def calibrate_thresholds(probs, y_true_onehot, class_names):
    y_true_idx = y_true_onehot.argmax(axis=1)
    grid = [x / 100 for x in range(25, 91, 5)]
    best = {
        "thresholds": {name: 0.5 for name in class_names},
        "macro_f1": -1.0,
        "balanced_accuracy": -1.0,
        "accuracy": -1.0,
        "report": None,
        "matrix": None,
    }

    for norm_t in grid:
        for ben_t in grid:
            for mal_t in grid:
                thresholds = {
                    "normal": float(norm_t),
                    "benign": float(ben_t),
                    "malignant": float(mal_t),
                }
                y_pred_idx = choose_calibrated_indices(probs, class_names, thresholds)
                summary = compute_macro_f1_and_balanced_accuracy(y_true_idx, y_pred_idx, class_names)
                accuracy = float((y_true_idx == y_pred_idx).mean())

                candidate = (
                    summary["macro_f1"],
                    summary["balanced_accuracy"],
                    accuracy,
                )
                current = (
                    best["macro_f1"],
                    best["balanced_accuracy"],
                    best["accuracy"],
                )
                if candidate > current:
                    best.update(
                        {
                            "thresholds": thresholds,
                            "macro_f1": summary["macro_f1"],
                            "balanced_accuracy": summary["balanced_accuracy"],
                            "accuracy": accuracy,
                            "report": summary["report"],
                            "matrix": summary["confusion_matrix"],
                        }
                    )

    details = {
        "optimized_for": "macro_f1_then_balanced_accuracy",
        "macro_f1": best["macro_f1"],
        "balanced_accuracy": best["balanced_accuracy"],
        "accuracy": best["accuracy"],
        "class_report": best["report"],
        "matrix": best["matrix"],
        "grid": {"start": 0.25, "end": 0.9, "step": 0.05},
    }
    return best["thresholds"], details


def export_tfjs(model_path: Path, out_dir: Path):
    try:
        import tensorflowjs as tfjs  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "tensorflowjs is not installed. Install it with: pip install tensorflowjs"
        ) from exc

    tfjs.converters.save_keras_model(
        tf.keras.models.load_model(model_path, compile=False),
        str(out_dir / "tfjs_model"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--no_balanced_sampling", action="store_true")
    parser.add_argument("--init_model", type=Path, default=None)
    parser.add_argument("--stage1_epochs", type=int, default=16)
    parser.add_argument("--stage2_epochs", type=int, default=12)
    parser.add_argument("--stage1_lr", type=float, default=1e-3)
    parser.add_argument("--stage2_lr", type=float, default=1e-5)
    parser.add_argument("--minority_boost", type=float, default=1.0)
    args = parser.parse_args()
    set_global_seed(SEED)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = build_datasets(
        args.data_dir,
        balanced_sampling=not args.no_balanced_sampling,
    )
    class_weights = compute_class_weights(args.data_dir) if not args.no_class_weights else None
    if class_weights and args.minority_boost > 1.0:
        class_weights = apply_minority_boost(class_weights, args.minority_boost)
    train_counts = count_split_samples(args.data_dir, "train")
    train_total = sum(train_counts.values())
    steps_per_epoch = max(1, math.ceil(train_total / BATCH_SIZE))
    if class_weights:
        print(f"Class weights: {class_weights}")
    else:
        print("Class weights: disabled")
    print(f"Balanced sampling: {'off' if args.no_balanced_sampling else 'on'}")
    print(f"Steps/epoch: {steps_per_epoch}")
    if args.init_model:
        print(f"Initializing from existing model: {args.init_model}")
        model = keras.models.load_model(args.init_model, compile=False)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.stage1_lr),
            loss=build_focal_loss(class_weight_alpha=class_weight_vector(class_weights), gamma=2.0),
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )
    else:
        model = build_model(
            num_classes=len(CLASS_NAMES),
            class_weight_alpha=class_weight_vector(class_weights),
        )
        # Keras optimizer LR can be a variable, tensor, or plain float depending on version.
        try:
            model.optimizer.learning_rate.assign(args.stage1_lr)
        except Exception:
            try:
                keras.backend.set_value(model.optimizer.learning_rate, args.stage1_lr)
            except Exception:
                model.optimizer.learning_rate = args.stage1_lr

    fine_tune_model(
        model,
        train_ds,
        val_ds,
        out_dir,
        class_weights,
        steps_per_epoch=steps_per_epoch,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage2_lr=args.stage2_lr,
    )

    best_path = out_dir / "best_model.keras"
    model = keras.models.load_model(best_path, compile=False)

    val_probs, val_true = collect_probs_and_labels(model, val_ds)
    thresholds, threshold_details = calibrate_thresholds(val_probs, val_true, CLASS_NAMES)

    val_true_idx = val_true.argmax(axis=1)
    val_pred_idx_argmax = val_probs.argmax(axis=1)
    val_argmax_summary = compute_macro_f1_and_balanced_accuracy(val_true_idx, val_pred_idx_argmax, CLASS_NAMES)

    val_pred_idx_cal = choose_calibrated_indices(val_probs, CLASS_NAMES, thresholds)
    val_cal_summary = compute_macro_f1_and_balanced_accuracy(val_true_idx, val_pred_idx_cal, CLASS_NAMES)

    test_probs, test_true = collect_probs_and_labels(model, test_ds)
    metrics = evaluate_model_with_calibration(test_probs, test_true, CLASS_NAMES, thresholds)

    model_path = out_dir / "mammogram_classifier.keras"
    model.save(model_path)

    class_map_path = out_dir / "class_map.json"
    class_map_path.write_text(json.dumps({str(i): name for i, name in enumerate(CLASS_NAMES)}, indent=2), encoding="utf-8")

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    confusion_path = out_dir / "val_confusion_matrix.json"
    confusion_path.write_text(
        json.dumps(
            {
                "class_names": CLASS_NAMES,
                "argmax": {
                    "matrix": val_argmax_summary["confusion_matrix"],
                    "report": val_argmax_summary["report"],
                    "macro_f1": val_argmax_summary["macro_f1"],
                    "balanced_accuracy": val_argmax_summary["balanced_accuracy"],
                },
                "calibrated": {
                    "matrix": val_cal_summary["confusion_matrix"],
                    "report": val_cal_summary["report"],
                    "macro_f1": val_cal_summary["macro_f1"],
                    "balanced_accuracy": val_cal_summary["balanced_accuracy"],
                    "thresholds": thresholds,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    calibration_path = out_dir / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "thresholds": thresholds,
                "details": threshold_details,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "model_path": str(model_path),
                "metrics": metrics,
                "class_map": str(class_map_path),
                "val_confusion_matrix": str(confusion_path),
                "calibration": str(calibration_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
