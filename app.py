from pathlib import Path
from io import BytesIO
import base64
import json
import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "5050"))

_allowed_ports = sorted({APP_PORT, 5000, 5050, 5500})
ALLOWED_ORIGINS = [
    origin
    for port in _allowed_ports
    for origin in (f"http://127.0.0.1:{port}", f"http://localhost:{port}")
]

MODEL_CANDIDATES = [
    ARTIFACTS_DIR / "best_model.keras",
    ARTIFACTS_DIR / "mammogram_classifier.keras",
]
CLASS_MAP_PATH = ARTIFACTS_DIR / "class_map.json"
CALIBRATION_PATH = ARTIFACTS_DIR / "calibration.json"


app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
CORS(
    app,
    resources={
        r"/classify": {"origins": ALLOWED_ORIGINS},
        r"/health": {"origins": ALLOWED_ORIGINS},
    },
)


class CompatBatchNormalization(tf.keras.layers.BatchNormalization):
    @classmethod
    def from_config(cls, config):
        cfg = dict(config)
        cfg.pop("renorm", None)
        cfg.pop("renorm_clipping", None)
        cfg.pop("renorm_momentum", None)
        return cls(**cfg)


_original_bn_init = tf.keras.layers.BatchNormalization.__init__
_original_dense_init = tf.keras.layers.Dense.__init__


def _compat_bn_init(self, *args, **kwargs):
    kwargs.pop("renorm", None)
    kwargs.pop("renorm_clipping", None)
    kwargs.pop("renorm_momentum", None)
    return _original_bn_init(self, *args, **kwargs)


def _compat_dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    return _original_dense_init(self, *args, **kwargs)


tf.keras.layers.BatchNormalization.__init__ = _compat_bn_init
tf.keras.layers.Dense.__init__ = _compat_dense_init


def load_model_and_mapping():
    model = None
    selected_model_path = None
    class_map = {0: "normal", 1: "benign", 2: "malignant"}
    thresholds = {"normal": 0.5, "benign": 0.5, "malignant": 0.5}

    for candidate in MODEL_CANDIDATES:
        if not candidate.exists():
            continue
        print(f"Loading model from {candidate}...")
        try:
            model = tf.keras.models.load_model(
                candidate,
                compile=False,
                custom_objects={"BatchNormalization": CompatBatchNormalization},
            )
            selected_model_path = candidate
            print("[OK] Model loaded successfully")
            break
        except Exception as exc:
            print(f"[ERROR] Error loading model {candidate}: {exc}")

    if model is None:
        print("[ERROR] No valid model artifact found in artifacts/")

    if CLASS_MAP_PATH.exists():
        with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
            class_map = json.load(f)
        print(f"[OK] Class map loaded: {class_map}")

    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
            calibration = json.load(f)
        thresholds.update(calibration.get("thresholds", {}))
        print(f"[OK] Calibration thresholds loaded: {thresholds}")
    else:
        print("[WARN] calibration.json not found, using default thresholds")

    return model, class_map, thresholds, selected_model_path


model, class_map, calibrated_thresholds, selected_model_path = load_model_and_mapping()
IMG_SIZE = 224


def preprocess_image(img):
    # Keep inference preprocessing aligned with training: full image, grayscale, resize.
    gray = img.convert("L")
    gray = gray.resize((IMG_SIZE, IMG_SIZE))

    arr = np.array(gray, dtype=np.float32) / 255.0
    arr = np.stack([arr] * 3, axis=-1)
    return arr


def center_crop_array(arr, ratio):
    h, w, _ = arr.shape
    crop_h = max(1, int(h * ratio))
    crop_w = max(1, int(w * ratio))
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)
    cropped = arr[top : top + crop_h, left : left + crop_w]
    cropped_img = Image.fromarray((cropped * 255.0).astype(np.uint8))
    resized = cropped_img.resize((IMG_SIZE, IMG_SIZE))
    resized_arr = np.array(resized, dtype=np.float32) / 255.0
    if resized_arr.ndim == 2:
        return np.stack([resized_arr] * 3, axis=-1)
    return resized_arr


def brightest_crop_array(arr, crop_size=112, stride=8):
    gray = arr[:, :, 0]
    h, w = gray.shape
    crop_size = min(crop_size, h, w)

    best_mean = None
    best_top = 0
    best_left = 0

    for top in range(0, h - crop_size + 1, max(1, stride)):
        for left in range(0, w - crop_size + 1, max(1, stride)):
            window_mean = float(gray[top : top + crop_size, left : left + crop_size].mean())
            if best_mean is None or window_mean > best_mean:
                best_mean = window_mean
                best_top = top
                best_left = left

    crop = arr[best_top : best_top + crop_size, best_left : best_left + crop_size]
    crop_img = Image.fromarray((crop * 255.0).astype(np.uint8))
    resized = crop_img.resize((IMG_SIZE, IMG_SIZE))
    resized_arr = np.array(resized, dtype=np.float32) / 255.0
    if resized_arr.ndim == 2:
        return np.stack([resized_arr] * 3, axis=-1)
    return resized_arr


def clamp01(value):
    return float(max(0.0, min(1.0, value)))


def extract_classification_features(image_arr):
    gray = np.asarray(image_arr[:, :, 0], dtype=np.float32) * 255.0
    h, w = gray.shape
    pixel_count = h * w

    flat = gray.reshape(-1)
    total_gray = float(np.sum(flat))
    total_sq_gray = float(np.sum(flat * flat))

    breast_mask = flat > 14.0
    breast_pixels = int(np.count_nonzero(breast_mask))
    breast_ratio = breast_pixels / pixel_count if pixel_count else 0.0

    if breast_pixels < pixel_count * 0.06:
        return {
            "breastRatio": breast_ratio,
            "breastMean": 0.0,
            "breastStdDev": 0.0,
            "lesionRatio": 0.0,
            "lesionMean": 0.0,
            "lesionContrast": 0.0,
            "lesionCircularity": 0.0,
            "lesionRoughness": 0.0,
            "lesionSpiculation": 0.0,
            "lesionCompactness": 0.0,
            "lesionEdgeSharpness": 0.0,
            "imageMean": total_gray / pixel_count if pixel_count else 0.0,
            "imageStdDev": 0.0,
            "lesionThreshold": 0.0,
            "normalScore": 1.0,
            "benignScore": 0.0,
            "malignantScore": 0.0,
        }

    breast_values = flat[breast_mask]
    breast_mean = float(np.mean(breast_values))
    breast_std = float(np.sqrt(max(0.0, np.mean(breast_values * breast_values) - breast_mean * breast_mean)))
    lesion_threshold = breast_mean + max(10.0, breast_std * 0.95)

    lesion_mask = (gray >= lesion_threshold) & (gray >= 14.0)
    lesion_pixels = int(np.count_nonzero(lesion_mask))
    lesion_ratio = lesion_pixels / breast_pixels if breast_pixels else 0.0

    if lesion_pixels < max(25, int(breast_pixels * 0.01)):
        image_mean = total_gray / pixel_count if pixel_count else 0.0
        image_std = float(np.sqrt(max(0.0, total_sq_gray / pixel_count - image_mean * image_mean))) if pixel_count else 0.0
        return {
            "breastRatio": breast_ratio,
            "breastMean": breast_mean,
            "breastStdDev": breast_std,
            "lesionRatio": lesion_ratio,
            "lesionMean": 0.0,
            "lesionContrast": 0.0,
            "lesionCircularity": 0.0,
            "lesionRoughness": 0.0,
            "lesionSpiculation": 0.0,
            "lesionCompactness": 0.0,
            "lesionEdgeSharpness": 0.0,
            "imageMean": image_mean,
            "imageStdDev": image_std,
            "lesionThreshold": float(lesion_threshold),
            "normalScore": 1.0,
            "benignScore": 0.0,
            "malignantScore": 0.0,
        }

    lesion_values = gray[lesion_mask]
    lesion_mean = float(np.mean(lesion_values))
    lesion_contrast = clamp01((lesion_mean - breast_mean) / (breast_std * 2.2 + 1.0))

    ys, xs = np.where(lesion_mask)
    lesion_min_x = int(xs.min())
    lesion_max_x = int(xs.max())
    lesion_min_y = int(ys.min())
    lesion_max_y = int(ys.max())
    bbox_width = max(1, lesion_max_x - lesion_min_x + 1)
    bbox_height = max(1, lesion_max_y - lesion_min_y + 1)
    bbox_area = float(bbox_width * bbox_height)
    compactness = lesion_pixels / bbox_area

    perimeter = 0
    edge_gradient_sum = 0.0
    edge_sample_count = 0
    sum_x = 0.0
    sum_y = 0.0

    lesion_yx = np.argwhere(lesion_mask)
    for y, x in lesion_yx:
        sum_x += float(x)
        sum_y += float(y)
        boundary = False
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                boundary = True
                continue
            if not lesion_mask[ny, nx]:
                boundary = True
                edge_gradient_sum += abs(float(gray[y, x] - gray[ny, nx]))
                edge_sample_count += 1
        if boundary:
            perimeter += 1

    centroid_x = sum_x / lesion_pixels
    centroid_y = sum_y / lesion_pixels
    radial_sum = 0.0
    radial_sq_sum = 0.0

    for y, x in lesion_yx:
        dx = float(x) - centroid_x
        dy = float(y) - centroid_y
        radius = float(np.sqrt(dx * dx + dy * dy))
        radial_sum += radius
        radial_sq_sum += radius * radius

    radial_mean = radial_sum / lesion_pixels
    radial_variance = max(0.0, radial_sq_sum / lesion_pixels - radial_mean * radial_mean)
    radial_std = float(np.sqrt(radial_variance))
    spiculation = clamp01(radial_std / radial_mean) if radial_mean > 0 else 0.0
    edge_sharpness = edge_gradient_sum / edge_sample_count if edge_sample_count > 0 else 0.0
    circularity = (4.0 * np.pi * lesion_pixels) / (perimeter * perimeter) if perimeter > 0 else 0.0
    roughness = perimeter / np.sqrt(max(1.0, float(lesion_pixels)))

    normal_score = (
        0.34 * clamp01(1 - lesion_ratio / 0.08)
        + 0.22 * clamp01(circularity / 1.1)
        + 0.16 * clamp01(1 - spiculation)
        + 0.14 * clamp01(1 - roughness / 8)
        + 0.14 * clamp01(1 - lesion_contrast)
    )

    benign_score = (
        0.22 * clamp01(1 - abs(lesion_ratio - 0.05) / 0.05)
        + 0.25 * clamp01(circularity / 1.1)
        + 0.18 * clamp01(1 - spiculation)
        + 0.17 * clamp01(compactness / 0.65)
        + 0.10 * lesion_contrast
        + 0.08 * clamp01(1 - roughness / 8)
    )

    malignant_score = (
        0.24 * clamp01((lesion_ratio - 0.03) / 0.14)
        + 0.24 * spiculation
        + 0.16 * clamp01(roughness / 8)
        + 0.12 * clamp01(1 - circularity / 1.1)
        + 0.08 * clamp01(edge_sharpness / 40)
        + 0.06 * clamp01((breast_std - 20) / 16)
        + 0.10 * clamp01(lesion_contrast)
    )

    image_mean = total_gray / pixel_count if pixel_count else 0.0
    image_std = float(np.sqrt(max(0.0, total_sq_gray / pixel_count - image_mean * image_mean))) if pixel_count else 0.0

    return {
        "breastRatio": breast_ratio,
        "breastMean": breast_mean,
        "breastStdDev": breast_std,
        "lesionRatio": lesion_ratio,
        "lesionMean": lesion_mean,
        "lesionContrast": lesion_contrast,
        "lesionCircularity": clamp01(circularity / 1.1),
        "lesionRoughness": clamp01(roughness / 8),
        "lesionSpiculation": clamp01(spiculation),
        "lesionCompactness": clamp01(compactness / 0.65),
        "lesionEdgeSharpness": clamp01(edge_sharpness / 40),
        "imageMean": image_mean,
        "imageStdDev": image_std,
        "lesionThreshold": float(lesion_threshold),
        "normalScore": normal_score,
        "benignScore": benign_score,
        "malignantScore": malignant_score,
    }


def predict_with_tta(image_arr):
    # Small TTA set helps stabilize predictions on tiny datasets.
    variants = [
        image_arr,
        np.fliplr(image_arr),
        np.clip(image_arr * 0.95, 0.0, 1.0),
        np.clip(image_arr * 1.05, 0.0, 1.0),
    ]

    batch = np.stack(variants, axis=0)
    preds = model.predict(batch, verbose=0)
    return np.mean(preds, axis=0)


def predict_with_ensemble(img):
    base_arr = preprocess_image(img)
    variants = [(1.0, base_arr)]

    combined = np.zeros(3, dtype=np.float32)
    for weight, variant in variants:
        probs = predict_with_tta(variant)
        combined += weight * probs

    total = float(np.sum(combined))
    if total > 0:
        combined /= total
    return combined


def hotspot_detector(image_arr):
    gray = np.asarray(image_arr[:, :, 0], dtype=np.float32) * 255.0
    h, w = gray.shape
    breast_mask = gray > 14.0
    if int(np.count_nonzero(breast_mask)) < max(1, int(h * w * 0.06)):
        return None

    breast_values = gray[breast_mask]
    bright_threshold = float(np.quantile(breast_values, 0.99))
    hotspot_mask = gray >= bright_threshold
    ys, xs = np.where(hotspot_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    bbox_left = int(xs.min())
    bbox_right = int(xs.max())
    bbox_top = int(ys.min())
    bbox_bottom = int(ys.max())
    bbox_width = bbox_right - bbox_left + 1
    bbox_height = bbox_bottom - bbox_top + 1
    bbox_area = max(1, bbox_width * bbox_height)
    density = float(len(xs)) / float(bbox_area)
    upper_half_ratio = float(np.mean(ys < h * 0.58))
    center_x = (bbox_left + bbox_right) / 2.0
    center_y = (bbox_top + bbox_bottom) / 2.0

    if (
        len(xs) >= 120
        and density >= 0.06
        and bbox_width <= int(w * 0.35)
        and bbox_height <= int(h * 0.42)
    ):
        return {
            "density": density,
            "count": int(len(xs)),
            "bbox": (bbox_left, bbox_top, bbox_right, bbox_bottom),
            "upperHalfRatio": upper_half_ratio,
            "centerX": center_x,
            "centerY": center_y,
        }

    return None


def choose_class_index(probs, thresholds):
    """Choose class from calibrated probabilities only (no extra guardrails)."""
    probs = np.asarray(probs, dtype=np.float32)

    norm_t = max(1e-6, float(thresholds.get("normal", 0.5)))
    ben_t = max(1e-6, float(thresholds.get("benign", 0.5)))
    mal_t = max(1e-6, float(thresholds.get("malignant", 0.5)))

    calibrated_scores = np.array(
        [
            probs[0] / norm_t,
            probs[1] / ben_t,
            probs[2] / mal_t,
        ],
        dtype=np.float32,
    )

    calibrated_best_idx = int(np.argmax(calibrated_scores))
    return calibrated_best_idx, "model-calibrated"


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/classify", methods=["POST", "OPTIONS"])
def classify():
    if request.method == "OPTIONS":
        return ("", 204)

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        image_data = base64.b64decode(data["image"])
        img = Image.open(BytesIO(image_data))
        base_arr = preprocess_image(img)
        probs = predict_with_ensemble(img)
        class_idx, decision_source = choose_class_index(probs, calibrated_thresholds)
        confidence = float(probs[class_idx])
        class_name = class_map.get(str(class_idx), class_map.get(class_idx, "unknown"))

        return jsonify(
            {
                "class": class_name,
                "confidence": confidence,
                    "decision_source": decision_source,
                "probabilities": {
                    "normal": float(probs[0]),
                    "benign": float(probs[1]),
                    "malignant": float(probs[2]),
                },
                "hotspot": None,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model is not None,
            "model_file": selected_model_path.name if selected_model_path else None,
        }
    )


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Mammogram Classification Server")
    print("=" * 50)
    print(f"Serving web app at http://{APP_HOST}:{APP_PORT}")
    print("Endpoint: POST /classify")
    print("=" * 50 + "\n")
    app.run(debug=False, host=APP_HOST, port=APP_PORT)