"""Apply ROI-focused preprocessing for mammography training data.

- Crops black borders using a simple intensity mask
- For benign/malignant with known lesion coordinates, crops around lesion
- Enhances contrast with a gentle contrast stretch
- Can write processed images in-place or into a separate output directory
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps


SPLITS = ("train", "val", "test")
CLASSES = ("normal", "benign", "malignant")


def load_meta(csv_path: Path) -> dict[str, dict[str, float | str]]:
    meta: dict[str, dict[str, float | str]] = {}
    if not csv_path.exists():
        return meta

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = (row.get("REFNUM") or "").strip().lower()
            sev = (row.get("SEVERITY") or "").strip().lower()
            if not ref:
                continue

            def parse_float(v: str | None) -> Optional[float]:
                if v is None or not str(v).strip():
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            x = parse_float(row.get("X"))
            y = parse_float(row.get("Y"))
            r = parse_float(row.get("RADIUS"))
            meta[ref] = {"severity": sev, "x": x, "y": y, "r": r}
    return meta


def crop_black_borders(gray: Image.Image) -> Image.Image:
    arr = np.array(gray)
    mask = arr > 8
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return gray, (0, 0)
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    if right - left < 20 or bottom - top < 20:
        return gray, (0, 0)
    return gray.crop((left, top, right + 1, bottom + 1)), (left, top)


def crop_lesion(gray: Image.Image, x: float, y: float, r: float) -> Image.Image:
    w, h = gray.size
    side = max(160, int(r * 4.8))
    half = side // 2

    cx = int(round(x))
    cy = int(round(y))

    left = max(0, cx - half)
    top = max(0, cy - half)
    right = min(w, cx + half)
    bottom = min(h, cy + half)

    if right - left < 40 or bottom - top < 40:
        return gray
    return gray.crop((left, top, right, bottom))


def enhance_mammogram(gray: Image.Image, cls_name: str) -> Image.Image:
    del cls_name
    return ImageOps.autocontrast(gray, cutoff=0.5)


def process_image(
    img_path: Path,
    cls_name: str,
    meta: dict[str, dict[str, float | str]],
    output_path: Path,
    roi_output_path: Path | None = None,
) -> None:
    stem = img_path.stem.lower()
    img = Image.open(img_path).convert("L")

    work, (offset_x, offset_y) = crop_black_borders(img)

    info = meta.get(stem)
    if cls_name in {"benign", "malignant"} and info:
        x_orig = info.get("x")
        y_orig = info.get("y")
        r = info.get("r")
        
        if isinstance(x_orig, float) and isinstance(y_orig, float) and isinstance(r, float) and r > 0:
            # MIAS Y-coordinates are from bottom, but Pillow is from top!
            # MIAS images are originally 1024x1024, Y is from bottom.
            y_from_top = 1024.0 - y_orig
            
            # Offset the coordinates because 'work' has been cropped
            x_adj = x_orig - offset_x
            y_adj = y_from_top - offset_y
            
            roi_work = crop_lesion(work, x_adj, y_adj, r)
            if roi_output_path is not None:
                roi_work = enhance_mammogram(roi_work, cls_name)
                roi_output_path.parent.mkdir(parents=True, exist_ok=True)
                roi_work.save(roi_output_path, format="PNG", optimize=True)

    work = enhance_mammogram(work, cls_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work.save(output_path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    dataset = args.input_dir or (root / "dataset")
    output_dir = args.output_dir or dataset
    meta_csv = root / "datateacher" / "extracted" / "classification" / "mias_derived_info.csv"
    meta = load_meta(meta_csv)

    total = 0
    for split in SPLITS:
        for cls in CLASSES:
            folder = dataset / split / cls
            if not folder.exists():
                continue
            for img in folder.glob("*.png"):
                rel_path = img.relative_to(dataset)
                target_path = output_dir / rel_path
                roi_target_path = None
                if split == "train" and cls in {"benign", "malignant"}:
                    roi_target_path = target_path.with_name(f"{img.stem}_roi.png")
                process_image(img, cls, meta, target_path, roi_target_path)
                total += 1

    print(f"Processed images: {total}")
    print(f"Input dataset: {dataset}")
    print(f"Output dataset: {output_dir}")


if __name__ == "__main__":
    main()
