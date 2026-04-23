"""Prepare datateacher mammography dataset into train/val/test folders.

This version uses a stratified random split over all available images to reduce
distribution shift between train and test on tiny datasets.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
import csv

SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15
CLASS_MAP = {
    "normal": "normal",
    "benign": "benign",
    "benige": "benign",  # typo in provided dataset
    "malignant": "malignant",
}


def _normalize_class_name(name: str) -> str:
    key = name.strip().lower()
    if key not in CLASS_MAP:
        raise ValueError(f"Unknown class folder: {name}")
    return CLASS_MAP[key]


def _collect_pngs(folder: Path) -> list[Path]:
    return sorted([p for p in folder.glob("*.png") if p.is_file()])


def _copy_many(files: list[Path], dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in files:
        shutil.copy2(src, dst_dir / src.name)
        copied += 1
    return copied


def _canonical_labels_from_csv(src_root: Path) -> dict[str, str]:
    csv_path = src_root / "mias_derived_info.csv"
    labels: dict[str, str] = {}
    if not csv_path.exists():
        return labels

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = (row.get("REFNUM") or "").strip().lower()
            sev = (row.get("SEVERITY") or "").strip().lower()
            if not ref:
                continue
            if sev == "normal":
                labels[f"{ref}.png"] = "normal"
            elif sev == "benign":
                labels[f"{ref}.png"] = "benign"
            elif sev == "malignant":
                labels[f"{ref}.png"] = "malignant"
    return labels


def prepare_dataset(src_root: Path, out_root: Path) -> None:
    random.seed(SEED)

    if not src_root.exists():
        raise FileNotFoundError(f"Source not found: {src_root}")

    if out_root.exists():
        shutil.rmtree(out_root)

    for split in ("train", "val", "test"):
        for class_name in ("normal", "benign", "malignant"):
            (out_root / split / class_name).mkdir(parents=True, exist_ok=True)

    canonical = _canonical_labels_from_csv(src_root)

    # 1) Collect all images from root class folders and optional test folders.
    all_items: dict[str, tuple[Path, str]] = {}
    candidate_dirs = [p for p in src_root.iterdir() if p.is_dir()]
    for folder in candidate_dirs:
        if folder.name.lower() == "test":
            inner_dirs = [p for p in folder.iterdir() if p.is_dir()]
            for class_dir in inner_dirs:
                guessed_class = _normalize_class_name(class_dir.name)
                for img in _collect_pngs(class_dir):
                    label = canonical.get(img.name, guessed_class)
                    all_items[img.name] = (img, label)
            continue

        guessed_class = _normalize_class_name(folder.name)
        for img in _collect_pngs(folder):
            label = canonical.get(img.name, guessed_class)
            all_items[img.name] = (img, label)

    by_class: dict[str, list[Path]] = {"normal": [], "benign": [], "malignant": []}
    for _, (img_path, label) in all_items.items():
        by_class[label].append(img_path)

    # 2) Stratified split per class.
    for class_name, files in by_class.items():
        files = sorted(files)
        random.shuffle(files)

        n = len(files)
        test_count = max(1, int(n * TEST_RATIO))
        val_count = max(1, int(n * VAL_RATIO))

        test_files = files[:test_count]
        val_files = files[test_count:test_count + val_count]
        train_files = files[test_count + val_count:]

        _copy_many(train_files, out_root / "train" / class_name)
        _copy_many(val_files, out_root / "val" / class_name)
        _copy_many(test_files, out_root / "test" / class_name)

    # 3) Print summary
    print("Prepared dataset at:", out_root)
    for split in ("train", "val", "test"):
        counts = {}
        total = 0
        for class_name in ("normal", "benign", "malignant"):
            n = len(_collect_pngs(out_root / split / class_name))
            counts[class_name] = n
            total += n
        print(f"{split}: total={total} | {counts}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    source = project_root / "datateacher" / "extracted" / "classification"
    output = project_root / "dataset"
    prepare_dataset(source, output)
