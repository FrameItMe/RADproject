/**
 * Inference helper — ONNX Runtime Web (primary) + rule-based fallback.
 *
 * This module runs entirely in the browser. No backend server needed.
 * Model: DenseNet121 quantized ONNX (~7 MB), loaded via onnxruntime-web WASM.
 */

/* ------------------------------------------------------------------ */
/* Global state                                                       */
/* ------------------------------------------------------------------ */
let _onnxSession = null;
let _modelStatus = 'not-loaded'; // 'not-loaded' | 'loading' | 'loaded' | 'error'
let _modelError = null;

const MODEL_PATH = 'model/mammogram_classifier_q.onnx';
const IMG_SIZE = 224;

// Calibration thresholds (from artifacts/calibration.json)
const CALIBRATION_THRESHOLDS = {
  normal: 0.45,
  benign: 0.30,
  malignant: 0.40,
};

const CLASS_NAMES = ['normal', 'benign', 'malignant'];

/* ------------------------------------------------------------------ */
/* Model loading                                                      */
/* ------------------------------------------------------------------ */

/**
 * Load the ONNX model. Call once on page load.
 * Returns a promise that resolves when ready or rejects on failure.
 */
async function loadOnnxModel() {
  if (_modelStatus === 'loaded' && _onnxSession) {
    return true;
  }
  if (_modelStatus === 'loading') {
    // Wait for the in-flight load to finish
    return new Promise((resolve) => {
      const check = setInterval(() => {
        if (_modelStatus !== 'loading') {
          clearInterval(check);
          resolve(_modelStatus === 'loaded');
        }
      }, 200);
    });
  }

  _modelStatus = 'loading';
  _modelError = null;
  updateModelStatusUI();

  try {
    // onnxruntime-web is loaded from CDN as a global `ort`
    if (typeof ort === 'undefined') {
      throw new Error('ONNX Runtime Web not loaded. Check CDN script in index.html.');
    }

    // Use WASM execution provider for broad compatibility
    ort.env.wasm.numThreads = 1; // Safe default; 4 can cause issues on some browsers
    ort.env.wasm.simd = true;

    console.log('[ONNX] Loading model from', MODEL_PATH, '...');
    const startTime = performance.now();

    _onnxSession = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
    console.log(`[ONNX] Model loaded in ${elapsed}s`);
    console.log('[ONNX] Input names:', _onnxSession.inputNames);
    console.log('[ONNX] Output names:', _onnxSession.outputNames);

    _modelStatus = 'loaded';
    updateModelStatusUI();
    return true;
  } catch (err) {
    console.error('[ONNX] Failed to load model:', err);
    _modelStatus = 'error';
    _modelError = err.message;
    updateModelStatusUI();
    return false;
  }
}

/**
 * Get current model status for UI display.
 */
function getModelStatus() {
  return { status: _modelStatus, error: _modelError };
}

/* ------------------------------------------------------------------ */
/* Preprocessing — must match training pipeline exactly               */
/* ------------------------------------------------------------------ */

/**
 * Convert ImageData to a Float32Array tensor [1, 224, 224, 3].
 * Pipeline: grayscale → resize → normalize → stack 3 channels.
 */
function preprocessForOnnx(imgData) {
  // Step 1: Convert to grayscale
  const w = imgData.width;
  const h = imgData.height;
  const src = imgData.data;
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const idx = i * 4;
    // Simple average (matches Python's .convert('L') for mammograms)
    gray[i] = (src[idx] * 0.299 + src[idx + 1] * 0.587 + src[idx + 2] * 0.114);
  }

  // Step 2: Resize to 224x224 using bilinear interpolation
  const resized = bilinearResize(gray, w, h, IMG_SIZE, IMG_SIZE);

  // Step 3: Normalize to [0, 1] and stack to 3 channels
  const tensorData = new Float32Array(1 * IMG_SIZE * IMG_SIZE * 3);
  for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
    const val = resized[i] / 255.0;
    // NHWC format: [batch, height, width, channel]
    tensorData[i * 3] = val;
    tensorData[i * 3 + 1] = val;
    tensorData[i * 3 + 2] = val;
  }

  return tensorData;
}

function bilinearResize(src, srcW, srcH, dstW, dstH) {
  const dst = new Float32Array(dstW * dstH);
  const xRatio = srcW / dstW;
  const yRatio = srcH / dstH;

  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const srcX = x * xRatio;
      const srcY = y * yRatio;
      const x0 = Math.floor(srcX);
      const y0 = Math.floor(srcY);
      const x1 = Math.min(x0 + 1, srcW - 1);
      const y1 = Math.min(y0 + 1, srcH - 1);
      const xFrac = srcX - x0;
      const yFrac = srcY - y0;

      const tl = src[y0 * srcW + x0];
      const tr = src[y0 * srcW + x1];
      const bl = src[y1 * srcW + x0];
      const br = src[y1 * srcW + x1];

      const top = tl + (tr - tl) * xFrac;
      const bottom = bl + (br - bl) * xFrac;
      dst[y * dstW + x] = top + (bottom - top) * yFrac;
    }
  }
  return dst;
}

/* ------------------------------------------------------------------ */
/* ONNX Inference with TTA                                            */
/* ------------------------------------------------------------------ */

/**
 * Run inference with test-time augmentation (TTA).
 * Matches Python predict_with_tta: original, flip, x0.95, x1.05
 */
async function runOnnxInference(imgData) {
  if (!_onnxSession) {
    throw new Error('ONNX model not loaded');
  }

  const tensorData = preprocessForOnnx(imgData);
  const inputName = _onnxSession.inputNames[0];

  // TTA variants
  const variants = [
    tensorData, // Original
    flipLR(tensorData, IMG_SIZE), // Horizontal flip
    scalePixels(tensorData, 0.95), // Slightly darker
    scalePixels(tensorData, 1.05), // Slightly brighter
  ];

  const allProbs = [];
  for (const variant of variants) {
    const tensor = new ort.Tensor('float32', variant, [1, IMG_SIZE, IMG_SIZE, 3]);
    const results = await _onnxSession.run({ [inputName]: tensor });
    const outputName = _onnxSession.outputNames[0];
    const output = results[outputName].data;
    allProbs.push(Array.from(output));
  }

  // Average predictions
  const avgProbs = [0, 0, 0];
  for (const probs of allProbs) {
    for (let i = 0; i < 3; i++) {
      avgProbs[i] += probs[i];
    }
  }
  for (let i = 0; i < 3; i++) {
    avgProbs[i] /= allProbs.length;
  }

  return avgProbs;
}

function flipLR(data, size) {
  const flipped = new Float32Array(data.length);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const srcIdx = (y * size + x) * 3;
      const dstIdx = (y * size + (size - 1 - x)) * 3;
      flipped[dstIdx] = data[srcIdx];
      flipped[dstIdx + 1] = data[srcIdx + 1];
      flipped[dstIdx + 2] = data[srcIdx + 2];
    }
  }
  return flipped;
}

function scalePixels(data, factor) {
  const scaled = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    scaled[i] = Math.min(1.0, Math.max(0.0, data[i] * factor));
  }
  return scaled;
}

/* ------------------------------------------------------------------ */
/* Calibrated Classification                                          */
/* ------------------------------------------------------------------ */

function chooseClassWithCalibration(probs) {
  const scores = [
    probs[0] / Math.max(1e-6, CALIBRATION_THRESHOLDS.normal),
    probs[1] / Math.max(1e-6, CALIBRATION_THRESHOLDS.benign),
    probs[2] / Math.max(1e-6, CALIBRATION_THRESHOLDS.malignant),
  ];

  let bestIdx = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[bestIdx]) {
      bestIdx = i;
    }
  }

  return {
    classIdx: bestIdx,
    className: CLASS_NAMES[bestIdx],
    confidence: probs[bestIdx],
  };
}

/* ------------------------------------------------------------------ */
/* Public API — classifyWithModel                                     */
/* ------------------------------------------------------------------ */

/**
 * Classify mammogram image using ONNX model (primary) or rule-based (fallback).
 * Returns: { label, confidence, probabilities, decision_source, modelUsed }
 */
async function classifyWithModel(imgData) {
  // Try ONNX model first
  if (_modelStatus === 'loaded' && _onnxSession) {
    try {
      const startTime = performance.now();
      const probs = await runOnnxInference(imgData);
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
      console.log(`[ONNX] Inference done in ${elapsed}s`, probs);

      const result = chooseClassWithCalibration(probs);
      return {
        label: result.className,
        confidence: Math.round(result.confidence * 100),
        probabilities: {
          normal: probs[0],
          benign: probs[1],
          malignant: probs[2],
        },
        decision_source: 'ONNX-model-calibrated',
        modelUsed: true,
        inferenceTime: elapsed,
      };
    } catch (err) {
      console.error('[ONNX] Inference failed, falling back to rule-based:', err);
    }
  }

  // Fallback: rule-based classification (from app.js extractClassificationFeatures)
  console.log('[Fallback] Using rule-based classification');
  return classifyRuleBased(imgData);
}

/**
 * Rule-based fallback classification.
 * Uses the extractClassificationFeatures + classifyImageDecisionTree from app.js.
 */
function classifyRuleBased(imgData) {
  // These functions are defined in app.js
  if (typeof classifyImageDecisionTree === 'function') {
    const result = classifyImageDecisionTree(imgData);
    return {
      label: result.label,
      confidence: result.confidence,
      probabilities: {
        normal: result.features.normalScore,
        benign: result.features.benignScore,
        malignant: result.features.malignantScore,
      },
      decision_source: 'rule-based-fallback',
      modelUsed: false,
      features: result.features,
    };
  }

  throw new Error('No classification method available. ONNX model not loaded and rule-based functions missing.');
}

/**
 * Check if model is available (replaces old server health check).
 */
async function checkModelServer() {
  return _modelStatus === 'loaded';
}

/* ------------------------------------------------------------------ */
/* UI helpers                                                         */
/* ------------------------------------------------------------------ */

function updateModelStatusUI() {
  const el = document.getElementById('modelStatus');
  if (!el) return;

  switch (_modelStatus) {
    case 'not-loaded':
      el.textContent = '⏳ Model not loaded yet';
      el.className = 'model-status status-waiting';
      break;
    case 'loading':
      el.textContent = '⏳ Loading ONNX model...';
      el.className = 'model-status status-loading';
      break;
    case 'loaded':
      el.textContent = '✅ ONNX model ready (in-browser)';
      el.className = 'model-status status-ready';
      break;
    case 'error':
      el.textContent = `⚠️ Model error: ${_modelError || 'unknown'} — using rule-based fallback`;
      el.className = 'model-status status-error';
      break;
  }
}
