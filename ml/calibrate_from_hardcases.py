"""Calibrate class thresholds using hard-case CSV files.

This script reads hard-case CSV outputs (from audit_hard_cases.py), searches
thresholds for normal/benign/malignant, and optimizes for non-normal recall
(benign + malignant) first to reduce normal-overprediction in difficult cases.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CLASS_NAMES = ["normal", "benign", "malignant"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def load_rows(csv_paths: list[Path]):
    probs = []
    y_true = []

    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                true_name = (row.get("true_class") or "").strip().lower()
                if true_name not in CLASS_TO_IDX:
                    continue

                p_normal = float(row["p_normal"])
                p_benign = float(row["p_benign"])
                p_malignant = float(row["p_malignant"])

                probs.append([p_normal, p_benign, p_malignant])
                y_true.append(CLASS_TO_IDX[true_name])

    if not probs:
        raise SystemExit("No valid rows found in input CSV files.")

    return np.asarray(probs, dtype=np.float32), np.asarray(y_true, dtype=np.int32)


def choose_indices(probs: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    t = np.asarray(
        [
            max(1e-6, float(thresholds["normal"])),
            max(1e-6, float(thresholds["benign"])),
            max(1e-6, float(thresholds["malignant"])),
        ],
        dtype=np.float32,
    )
    scaled = probs / t
    return np.argmax(scaled, axis=1).astype(np.int32)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3):
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def per_class_metrics(cm: np.ndarray):
    n = cm.shape[0]
    total = int(cm.sum())
    report = {}

    for i, name in enumerate(CLASS_NAMES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(total - tp - fp - fn)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        report[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[i, :].sum()),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

    macro_f1 = float(np.mean([report[n]["f1"] for n in CLASS_NAMES]))
    balanced_acc = float(np.mean([report[n]["recall"] for n in CLASS_NAMES]))
    accuracy = safe_div(int(np.trace(cm)), total)

    benign_recall = report["benign"]["recall"]
    malignant_recall = report["malignant"]["recall"]
    bm_avg_recall = 0.5 * (benign_recall + malignant_recall)
    bm_min_recall = min(benign_recall, malignant_recall)

    normal_fp_into_non_normal = int(cm[0, 1] + cm[0, 2])

    summary = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "bm_avg_recall": bm_avg_recall,
        "bm_min_recall": bm_min_recall,
        "normal_fp_into_non_normal": normal_fp_into_non_normal,
        "report": report,
        "confusion_matrix": cm.tolist(),
    }
    return summary


def score_tuple(metrics: dict):
    # Lexicographic objective prioritizes catching benign/malignant first.
    return (
        float(metrics["bm_min_recall"]),
        float(metrics["bm_avg_recall"]),
        float(metrics["macro_f1"]),
        -float(metrics["normal_fp_into_non_normal"]),
        float(metrics["accuracy"]),
    )


def parse_thresholds_json(path: Path):
    if not path.exists():
        return {"normal": 0.5, "benign": 0.5, "malignant": 0.5}
    data = json.loads(path.read_text(encoding="utf-8"))
    t = data.get("thresholds", {})
    return {
        "normal": float(t.get("normal", 0.5)),
        "benign": float(t.get("benign", 0.5)),
        "malignant": float(t.get("malignant", 0.5)),
    }


def frange(start: float, stop: float, step: float):
    vals = []
    x = start
    while x <= stop + 1e-9:
        vals.append(round(x, 6))
        x += step
    return vals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, nargs="+", required=True)
    parser.add_argument("--base_calibration", type=Path, default=Path("artifacts/calibration.json"))
    parser.add_argument("--grid_start", type=float, default=0.2)
    parser.add_argument("--grid_end", type=float, default=0.9)
    parser.add_argument("--grid_step", type=float, default=0.02)
    parser.add_argument(
        "--out_json",
        type=Path,
        default=Path("experiments/audit/hard_cases/calibration_hardset_candidate.json"),
    )
    parser.add_argument(
        "--apply_to",
        type=Path,
        default=None,
        help="If set, overwrite this calibration file with candidate thresholds/details",
    )
    args = parser.parse_args()

    probs, y_true = load_rows(args.csv)

    base_thresholds = parse_thresholds_json(args.base_calibration)
    base_pred = choose_indices(probs, base_thresholds)
    base_metrics = per_class_metrics(confusion_matrix(y_true, base_pred))

    grid = frange(args.grid_start, args.grid_end, args.grid_step)
    best_thresholds = None
    best_metrics = None
    best_score = None

    for norm_t in grid:
        for ben_t in grid:
            for mal_t in grid:
                thresholds = {
                    "normal": float(norm_t),
                    "benign": float(ben_t),
                    "malignant": float(mal_t),
                }
                pred = choose_indices(probs, thresholds)
                metrics = per_class_metrics(confusion_matrix(y_true, pred))
                sc = score_tuple(metrics)
                if best_score is None or sc > best_score:
                    best_score = sc
                    best_thresholds = thresholds
                    best_metrics = metrics

    assert best_thresholds is not None and best_metrics is not None

    result = {
        "optimized_for": "bm_min_recall_then_bm_avg_recall_then_macro_f1",
        "source_csv": [str(p) for p in args.csv],
        "samples": int(len(y_true)),
        "grid": {
            "start": args.grid_start,
            "end": args.grid_end,
            "step": args.grid_step,
        },
        "baseline": {
            "thresholds": base_thresholds,
            **base_metrics,
        },
        "candidate": {
            "thresholds": best_thresholds,
            **best_metrics,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.apply_to is not None:
        payload = {
            "thresholds": best_thresholds,
            "details": result,
        }
        args.apply_to.parent.mkdir(parents=True, exist_ok=True)
        args.apply_to.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    if args.apply_to is not None:
        print(f"Applied candidate calibration to: {args.apply_to}")


if __name__ == "__main__":
    main()
