const state = {
  originalImageData: null,
  page1Result: null,
  page2Base: null,
  page2Working: null,
  page2Result: null,
  page2History: [],
  page2HistoryCursor: -1,
  page2HistoryNextId: 1,
  morphInputImageData: null,
  morphResult: null,
  classifyInputImageData: null,
  noiseMode: "manual",
  noiseFilter: "mean"
};

const clamp = (value) => Math.max(0, Math.min(255, value));
const clamp01 = (value) => Math.max(0, Math.min(1, value));

const copyImageData = (imgData) =>
  new ImageData(new Uint8ClampedArray(imgData.data), imgData.width, imgData.height);

const drawImageData = (canvas, ctx, imgData) => {
  canvas.width = imgData.width;
  canvas.height = imgData.height;
  ctx.putImageData(imgData, 0, 0);
};

const clearCanvas = (canvas, ctx) => {
  canvas.width = 1;
  canvas.height = 1;
  ctx.clearRect(0, 0, 1, 1);
};

const loadImageFile = (file) =>
  new Promise((resolve, reject) => {
    if (!file) {
      reject(new Error("No file selected"));
      return;
    }

    const image = new Image();
    image.onload = () => {
      const temp = document.createElement("canvas");
      temp.width = image.width;
      temp.height = image.height;
      const tctx = temp.getContext("2d");
      tctx.drawImage(image, 0, 0);
      resolve(tctx.getImageData(0, 0, temp.width, temp.height));
      URL.revokeObjectURL(image.src);
    };
    image.onerror = () => reject(new Error("Image loading failed"));
    image.src = URL.createObjectURL(file);
  });

const applyNoiseReduction = (imgData, level) => {
  if (level === 0) {
    return copyImageData(imgData);
  }

  const w = imgData.width;
  const h = imgData.height;
  const src = imgData.data;
  const out = new Uint8ClampedArray(src.length);
  const normalized = clamp01(level / 20);
  const radius = Math.max(1, Math.min(4, Math.ceil(normalized * 4)));
  const blend = 0.18 + normalized * 0.77;

  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let r = 0;
      let g = 0;
      let b = 0;
      let count = 0;

      for (let ky = -radius; ky <= radius; ky += 1) {
        for (let kx = -radius; kx <= radius; kx += 1) {
          const px = x + kx;
          const py = y + ky;
          if (px < 0 || py < 0 || px >= w || py >= h) {
            continue;
          }
          const idx = (py * w + px) * 4;
          r += src[idx];
          g += src[idx + 1];
          b += src[idx + 2];
          count += 1;
        }
      }

      const idx = (y * w + x) * 4;
      out[idx] = clamp(src[idx] * (1 - blend) + (r / count) * blend);
      out[idx + 1] = clamp(src[idx + 1] * (1 - blend) + (g / count) * blend);
      out[idx + 2] = clamp(src[idx + 2] * (1 - blend) + (b / count) * blend);
      out[idx + 3] = src[idx + 3];
    }
  }

  return new ImageData(out, w, h);
};

const applyGaussianFilter = (imgData, passes = 1) => {
  const w = imgData.width;
  const h = imgData.height;
  const kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1];
  const offsets = [
    [-1, -1], [0, -1], [1, -1],
    [-1, 0], [0, 0], [1, 0],
    [-1, 1], [0, 1], [1, 1]
  ];

  let source = new Uint8ClampedArray(imgData.data);

  for (let pass = 0; pass < passes; pass += 1) {
    const out = new Uint8ClampedArray(source.length);

    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        let rs = 0;
        let gs = 0;
        let bs = 0;
        let weightSum = 0;

        for (let i = 0; i < offsets.length; i += 1) {
          const px = x + offsets[i][0];
          const py = y + offsets[i][1];
          if (px < 0 || py < 0 || px >= w || py >= h) {
            continue;
          }
          const idx = (py * w + px) * 4;
          const weight = kernel[i];
          rs += source[idx] * weight;
          gs += source[idx + 1] * weight;
          bs += source[idx + 2] * weight;
          weightSum += weight;
        }

        const outIdx = (y * w + x) * 4;
        out[outIdx] = clamp(rs / weightSum);
        out[outIdx + 1] = clamp(gs / weightSum);
        out[outIdx + 2] = clamp(bs / weightSum);
        out[outIdx + 3] = source[outIdx + 3];
      }
    }

    source = out;
  }

  return new ImageData(source, w, h);
};

const applyMedianFilter = (imgData, radius = 1) => {
  const w = imgData.width;
  const h = imgData.height;
  const src = imgData.data;
  const out = new Uint8ClampedArray(src.length);

  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const rv = [];
      const gv = [];
      const bv = [];

      for (let ky = -radius; ky <= radius; ky += 1) {
        for (let kx = -radius; kx <= radius; kx += 1) {
          const px = x + kx;
          const py = y + ky;
          if (px < 0 || py < 0 || px >= w || py >= h) {
            continue;
          }
          const idx = (py * w + px) * 4;
          rv.push(src[idx]);
          gv.push(src[idx + 1]);
          bv.push(src[idx + 2]);
        }
      }

      rv.sort((a, b) => a - b);
      gv.sort((a, b) => a - b);
      bv.sort((a, b) => a - b);
      const mid = Math.floor(rv.length / 2);
      const outIdx = (y * w + x) * 4;

      out[outIdx] = rv[mid];
      out[outIdx + 1] = gv[mid];
      out[outIdx + 2] = bv[mid];
      out[outIdx + 3] = src[outIdx + 3];
    }
  }

  return new ImageData(out, w, h);
};

const applyDetailRecovery = (baseImageData, filteredImageData, amount = 0.2) => {
  const out = new Uint8ClampedArray(filteredImageData.data.length);
  const base = baseImageData.data;
  const filtered = filteredImageData.data;

  for (let i = 0; i < filtered.length; i += 4) {
    out[i] = clamp(filtered[i] + amount * (base[i] - filtered[i]));
    out[i + 1] = clamp(filtered[i + 1] + amount * (base[i + 1] - filtered[i + 1]));
    out[i + 2] = clamp(filtered[i + 2] + amount * (base[i + 2] - filtered[i + 2]));
    out[i + 3] = filtered[i + 3];
  }

  return new ImageData(out, filteredImageData.width, filteredImageData.height);
};

const estimateNoiseVariance = (imgData) => {
  const w = imgData.width;
  const h = imgData.height;
  const d = imgData.data;
  let residualSum = 0;
  let residualSqSum = 0;
  let count = 0;

  for (let y = 1; y < h - 1; y += 1) {
    for (let x = 1; x < w - 1; x += 1) {
      let localMean = 0;
      let localCount = 0;

      for (let ky = -1; ky <= 1; ky += 1) {
        for (let kx = -1; kx <= 1; kx += 1) {
          const px = x + kx;
          const py = y + ky;
          const idx = (py * w + px) * 4;
          const gray = (d[idx] + d[idx + 1] + d[idx + 2]) / 3;
          localMean += gray;
          localCount += 1;
        }
      }

      localMean /= localCount;
      const centerIdx = (y * w + x) * 4;
      const centerGray = (d[centerIdx] + d[centerIdx + 1] + d[centerIdx + 2]) / 3;
      const residual = centerGray - localMean;

      residualSum += residual;
      residualSqSum += residual * residual;
      count += 1;
    }
  }

  if (count === 0) {
    return 0;
  }

  const meanResidual = residualSum / count;
  return Math.max(0, residualSqSum / count - meanResidual * meanResidual);
};

const suggestNoiseFilter = (variance) => {
  if (variance < 70) {
    return { filter: "mean", level: 6, label: "Mean Filter" };
  }
  if (variance < 190) {
    return { filter: "gaussian", level: 10, label: "Gaussian Filter" };
  }
  return { filter: "median", level: 14, label: "Median Filter" };
};

const applyNoiseFilter = (imgData, level, filterType) => {
  if (level === 0) {
    return copyImageData(imgData);
  }

  const normalized = clamp01(level / 20);

  if (filterType === "gaussian") {
    const passes = Math.max(1, Math.min(7, Math.ceil(level / 3)));
    const filtered = applyGaussianFilter(imgData, passes);
    return applyDetailRecovery(imgData, filtered, 0.3 - normalized * 0.15);
  }

  if (filterType === "median") {
    const radius = level <= 6 ? 1 : level <= 13 ? 2 : 3;
    const filtered = applyMedianFilter(imgData, radius);
    return applyDetailRecovery(imgData, filtered, 0.22 - normalized * 0.08);
  }

  const filtered = applyNoiseReduction(imgData, level);
  return applyDetailRecovery(imgData, filtered, 0.35 - normalized * 0.2);
};

const filterLabelMap = {
  mean: "Mean Filter",
  gaussian: "Gaussian Filter",
  median: "Median Filter"
};

const applyBrightness = (imgData, value) => {
  const out = new Uint8ClampedArray(imgData.data.length);
  for (let i = 0; i < imgData.data.length; i += 4) {
    out[i] = clamp(imgData.data[i] + value);
    out[i + 1] = clamp(imgData.data[i + 1] + value);
    out[i + 2] = clamp(imgData.data[i + 2] + value);
    out[i + 3] = imgData.data[i + 3];
  }
  return new ImageData(out, imgData.width, imgData.height);
};

const applyContrast = (imgData, value) => {
  const factor = (259 * (value + 255)) / (255 * (259 - value));
  const out = new Uint8ClampedArray(imgData.data.length);

  for (let i = 0; i < imgData.data.length; i += 4) {
    out[i] = clamp(factor * (imgData.data[i] - 128) + 128);
    out[i + 1] = clamp(factor * (imgData.data[i + 1] - 128) + 128);
    out[i + 2] = clamp(factor * (imgData.data[i + 2] - 128) + 128);
    out[i + 3] = imgData.data[i + 3];
  }

  return new ImageData(out, imgData.width, imgData.height);
};

const autoContrast = (imgData) => {
  let min = 255;
  let max = 0;
  const src = imgData.data;

  for (let i = 0; i < src.length; i += 4) {
    const gray = (src[i] + src[i + 1] + src[i + 2]) / 3;
    min = Math.min(min, gray);
    max = Math.max(max, gray);
  }

  if (max === min) {
    return copyImageData(imgData);
  }

  const scale = 255 / (max - min);
  const out = new Uint8ClampedArray(src.length);

  for (let i = 0; i < src.length; i += 4) {
    out[i] = clamp((src[i] - min) * scale);
    out[i + 1] = clamp((src[i + 1] - min) * scale);
    out[i + 2] = clamp((src[i + 2] - min) * scale);
    out[i + 3] = src[i + 3];
  }

  return new ImageData(out, imgData.width, imgData.height);
};

const buildClaheLut = (tilePixels, clipLimit) => {
  const hist = new Uint32Array(256);
  for (let i = 0; i < tilePixels.length; i += 1) {
    hist[tilePixels[i]] += 1;
  }

  const tileArea = tilePixels.length;
  const clipThreshold = Math.max(1, Math.floor((clipLimit * tileArea) / 256));

  let excess = 0;
  for (let i = 0; i < 256; i += 1) {
    if (hist[i] > clipThreshold) {
      excess += hist[i] - clipThreshold;
      hist[i] = clipThreshold;
    }
  }

  const increment = Math.floor(excess / 256);
  const remainder = excess % 256;
  if (increment > 0) {
    for (let i = 0; i < 256; i += 1) {
      hist[i] += increment;
    }
  }
  for (let i = 0; i < remainder; i += 1) {
    hist[i] += 1;
  }

  const cdf = new Uint32Array(256);
  let cumulative = 0;
  for (let i = 0; i < 256; i += 1) {
    cumulative += hist[i];
    cdf[i] = cumulative;
  }

  let cdfMin = 0;
  for (let i = 0; i < 256; i += 1) {
    if (cdf[i] > 0) {
      cdfMin = cdf[i];
      break;
    }
  }

  const denom = Math.max(1, tileArea - cdfMin);
  const lut = new Uint8Array(256);
  for (let i = 0; i < 256; i += 1) {
    lut[i] = clamp(Math.round(((cdf[i] - cdfMin) / denom) * 255));
  }
  return lut;
};

const applyClahe = (imgData, clipLimit = 2.0, tileGrid = 8) => {
  const w = imgData.width;
  const h = imgData.height;
  const src = imgData.data;

  const tilesX = Math.max(2, Math.min(16, Math.round(tileGrid)));
  const tilesY = tilesX;
  const tileW = Math.ceil(w / tilesX);
  const tileH = Math.ceil(h / tilesY);

  const luts = Array.from({ length: tilesY }, () => Array.from({ length: tilesX }, () => null));

  for (let ty = 0; ty < tilesY; ty += 1) {
    const y0 = ty * tileH;
    const y1 = Math.min(h, (ty + 1) * tileH);

    for (let tx = 0; tx < tilesX; tx += 1) {
      const x0 = tx * tileW;
      const x1 = Math.min(w, (tx + 1) * tileW);
      const tilePixels = new Uint8Array((y1 - y0) * (x1 - x0));
      let index = 0;

      for (let y = y0; y < y1; y += 1) {
        for (let x = x0; x < x1; x += 1) {
          const i = (y * w + x) * 4;
          tilePixels[index] = Math.round((src[i] + src[i + 1] + src[i + 2]) / 3);
          index += 1;
        }
      }

      luts[ty][tx] = buildClaheLut(tilePixels, clipLimit);
    }
  }

  const out = new Uint8ClampedArray(src.length);

  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const srcIdx = (y * w + x) * 4;
      const gray = Math.round((src[srcIdx] + src[srcIdx + 1] + src[srcIdx + 2]) / 3);

      const txFloat = (x + 0.5) / tileW - 0.5;
      const tyFloat = (y + 0.5) / tileH - 0.5;

      let xL = Math.floor(txFloat);
      let yT = Math.floor(tyFloat);
      let wx = txFloat - xL;
      let wy = tyFloat - yT;

      if (xL < 0) {
        xL = 0;
        wx = 0;
      }
      if (yT < 0) {
        yT = 0;
        wy = 0;
      }
      if (xL >= tilesX - 1) {
        xL = tilesX - 1;
        wx = 0;
      }
      if (yT >= tilesY - 1) {
        yT = tilesY - 1;
        wy = 0;
      }

      const xR = Math.min(tilesX - 1, xL + 1);
      const yB = Math.min(tilesY - 1, yT + 1);

      const lutTL = luts[yT][xL];
      const lutTR = luts[yT][xR];
      const lutBL = luts[yB][xL];
      const lutBR = luts[yB][xR];

      const top = lutTL[gray] * (1 - wx) + lutTR[gray] * wx;
      const bottom = lutBL[gray] * (1 - wx) + lutBR[gray] * wx;
      const mapped = clamp(Math.round(top * (1 - wy) + bottom * wy));

      out[srcIdx] = mapped;
      out[srcIdx + 1] = mapped;
      out[srcIdx + 2] = mapped;
      out[srcIdx + 3] = src[srcIdx + 3];
    }
  }

  return new ImageData(out, w, h);
};

const erodeBinary = (imgData, kernelSize = 3) => {
  const w = imgData.width;
  const h = imgData.height;
  const src = imgData.data;
  const out = new Uint8ClampedArray(src.length);
  const radius = Math.max(1, Math.floor(kernelSize / 2));

  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let minGray = 255;
      for (let ky = -radius; ky <= radius; ky += 1) {
        for (let kx = -radius; kx <= radius; kx += 1) {
          const px = x + kx;
          const py = y + ky;
          if (px < 0 || py < 0 || px >= w || py >= h) {
            continue;
          }
          const idx = (py * w + px) * 4;
          const gray = Math.round((src[idx] + src[idx + 1] + src[idx + 2]) / 3);
          minGray = Math.min(minGray, gray);
        }
      }

      const idx = (y * w + x) * 4;
      out[idx] = minGray;
      out[idx + 1] = minGray;
      out[idx + 2] = minGray;
      out[idx + 3] = 255;
    }
  }

  return new ImageData(out, w, h);
};

const dilateBinary = (imgData, kernelSize = 3) => {
  const w = imgData.width;
  const h = imgData.height;
  const src = imgData.data;
  const out = new Uint8ClampedArray(src.length);
  const radius = Math.max(1, Math.floor(kernelSize / 2));

  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let maxGray = 0;
      for (let ky = -radius; ky <= radius; ky += 1) {
        for (let kx = -radius; kx <= radius; kx += 1) {
          const px = x + kx;
          const py = y + ky;
          if (px < 0 || py < 0 || px >= w || py >= h) {
            continue;
          }
          const idx = (py * w + px) * 4;
          const gray = Math.round((src[idx] + src[idx + 1] + src[idx + 2]) / 3);
          maxGray = Math.max(maxGray, gray);
        }
      }

      const idx = (y * w + x) * 4;
      out[idx] = maxGray;
      out[idx + 1] = maxGray;
      out[idx + 2] = maxGray;
      out[idx + 3] = 255;
    }
  }

  return new ImageData(out, w, h);
};

const runMorphology = (imgData, type, kernelSize = 3) => {
  if (type === "opening") {
    return dilateBinary(erodeBinary(imgData, kernelSize), kernelSize);
  }
  return erodeBinary(dilateBinary(imgData, kernelSize), kernelSize);
};

const chooseMaxScore = (scores) => {
  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  return {
    label: entries[0][0],
    bestScore: entries[0][1],
    secondScore: entries[1][1]
  };
};

const extractClassificationFeatures = (imgData) => {
  const data = imgData.data;
  const w = imgData.width;
  const h = imgData.height;
  const pixelCount = w * h;

  const grayValues = new Float32Array(pixelCount);
  let totalGray = 0;
  let totalSqGray = 0;
  let breastPixels = 0;
  let breastGraySum = 0;
  let breastGraySqSum = 0;

  for (let i = 0; i < pixelCount; i += 1) {
    const idx = i * 4;
    const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
    grayValues[i] = gray;
    totalGray += gray;
    totalSqGray += gray * gray;
    if (gray > 14) {
      breastPixels += 1;
      breastGraySum += gray;
      breastGraySqSum += gray * gray;
    }
  }

  const imageMean = totalGray / pixelCount;
  const imageStdDev = Math.sqrt(Math.max(0, totalSqGray / pixelCount - imageMean * imageMean));
  const breastRatio = breastPixels / pixelCount;

  if (breastPixels < pixelCount * 0.06) {
    return {
      breastRatio,
      breastMean: 0,
      breastStdDev: 0,
      lesionRatio: 0,
      lesionMean: 0,
      lesionContrast: 0,
      lesionCircularity: 0,
      lesionRoughness: 0,
      lesionSpiculation: 0,
      lesionCompactness: 0,
      lesionEdgeSharpness: 0,
      imageMean,
      imageStdDev,
      lesionThreshold: 0,
      normalScore: 1,
      benignScore: 0,
      malignantScore: 0,
      lesionMinX: 0,
      lesionMinY: 0,
      lesionMaxX: w,
      lesionMaxY: h
    };
  }

  const breastMean = breastGraySum / breastPixels;
  const breastStdDev = Math.sqrt(Math.max(0, breastGraySqSum / breastPixels - breastMean * breastMean));
  const lesionThreshold = breastMean + Math.max(10, breastStdDev * 0.95);

  let lesionPixels = 0;
  let lesionSum = 0;
  let lesionMinX = w;
  let lesionMinY = h;
  let lesionMaxX = 0;
  let lesionMaxY = 0;

  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const gray = grayValues[y * w + x];
      if (gray < 14 || gray < lesionThreshold) {
        continue;
      }

      lesionPixels += 1;
      lesionSum += gray;
      if (x < lesionMinX) lesionMinX = x;
      if (y < lesionMinY) lesionMinY = y;
      if (x > lesionMaxX) lesionMaxX = x;
      if (y > lesionMaxY) lesionMaxY = y;
    }
  }

  if (lesionPixels < Math.max(25, breastPixels * 0.01)) {
    return {
      breastRatio,
      breastMean,
      breastStdDev,
      lesionRatio: lesionPixels / breastPixels,
      lesionMean: 0,
      lesionContrast: 0,
      lesionCircularity: 0,
      lesionRoughness: 0,
      lesionSpiculation: 0,
      lesionCompactness: 0,
      lesionEdgeSharpness: 0,
      imageMean,
      imageStdDev,
      lesionThreshold,
      normalScore: 1,
      benignScore: 0,
      malignantScore: 0,
      lesionMinX: 0,
      lesionMinY: 0,
      lesionMaxX: w,
      lesionMaxY: h
    };
  }

  const lesionMean = lesionSum / lesionPixels;
  const lesionContrast = clamp01((lesionMean - breastMean) / (breastStdDev * 2.2 + 1));
  const lesionRatio = lesionPixels / breastPixels;

  let perimeter = 0;
  let edgeGradientSum = 0;
  let edgeSampleCount = 0;
  let sumX = 0;
  let sumY = 0;

  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const gray = grayValues[y * w + x];
      if (gray < lesionThreshold) {
        continue;
      }

      sumX += x;
      sumY += y;

      let boundary = false;
      const neighbors = [
        [x - 1, y],
        [x + 1, y],
        [x, y - 1],
        [x, y + 1]
      ];

      for (const [nx, ny] of neighbors) {
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
          boundary = true;
          continue;
        }
        const nGray = grayValues[ny * w + nx];
        if (nGray < lesionThreshold) {
          boundary = true;
          edgeGradientSum += Math.abs(gray - nGray);
          edgeSampleCount += 1;
        }
      }

      if (boundary) {
        perimeter += 1;
      }
    }
  }

  const centroidX = sumX / lesionPixels;
  const centroidY = sumY / lesionPixels;
  const bboxWidth = Math.max(1, lesionMaxX - lesionMinX + 1);
  const bboxHeight = Math.max(1, lesionMaxY - lesionMinY + 1);
  const bboxArea = bboxWidth * bboxHeight;
  const compactness = lesionPixels / bboxArea;
  const circularity = perimeter > 0 ? (4 * Math.PI * lesionPixels) / (perimeter * perimeter) : 0;
  const roughness = perimeter / Math.sqrt(lesionPixels);

  let radialSum = 0;
  let radialSqSum = 0;
  let radialCount = 0;

  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const gray = grayValues[y * w + x];
      if (gray < lesionThreshold) {
        continue;
      }

      const dx = x - centroidX;
      const dy = y - centroidY;
      const radius = Math.sqrt(dx * dx + dy * dy);
      radialSum += radius;
      radialSqSum += radius * radius;
      radialCount += 1;
    }
  }

  const radialMean = radialSum / Math.max(1, radialCount);
  const radialVariance = Math.max(0, radialSqSum / Math.max(1, radialCount) - radialMean * radialMean);
  const radialStdDev = Math.sqrt(radialVariance);
  const spiculation = radialMean > 0 ? clamp01(radialStdDev / radialMean) : 0;
  const edgeSharpness = edgeSampleCount > 0 ? edgeGradientSum / edgeSampleCount : 0;

  const normalScore =
    0.34 * clamp01(1 - lesionRatio / 0.08) +
    0.22 * clamp01(circularity / 1.1) +
    0.16 * clamp01(1 - spiculation) +
    0.14 * clamp01(1 - roughness / 8) +
    0.14 * clamp01(1 - lesionContrast);

  const benignScore =
    0.22 * clamp01(1 - Math.abs(lesionRatio - 0.05) / 0.05) +
    0.25 * clamp01(circularity / 1.1) +
    0.18 * clamp01(1 - spiculation) +
    0.17 * clamp01(compactness / 0.65) +
    0.10 * lesionContrast +
    0.08 * clamp01(1 - roughness / 8);

  const malignantScore =
    0.24 * clamp01((lesionRatio - 0.03) / 0.14) +
    0.24 * spiculation +
    0.16 * clamp01(roughness / 8) +
    0.12 * clamp01(1 - circularity / 1.1) +
    0.08 * clamp01(edgeSharpness / 40) +
    0.06 * clamp01((breastStdDev - 20) / 16) +
    0.10 * clamp01(lesionContrast);

  return {
    breastRatio,
    breastMean,
    breastStdDev,
    lesionRatio,
    lesionMean,
    lesionContrast,
    lesionCircularity: clamp01(circularity / 1.1),
    lesionRoughness: clamp01(roughness / 8),
    lesionSpiculation: clamp01(spiculation),
    lesionCompactness: clamp01(compactness / 0.65),
    lesionEdgeSharpness: clamp01(edgeSharpness / 40),
    imageMean,
    imageStdDev,
    lesionThreshold,
    normalScore,
    benignScore,
    malignantScore,
    lesionMinX,
    lesionMinY,
    lesionMaxX,
    lesionMaxY
  };
};

const classifyImageDecisionTree = (imgData) => {
  const features = extractClassificationFeatures(imgData);

  if (features.breastRatio < 0.06) {
    return {
      label: "normal",
      confidence: 84,
      features
    };
  }

  const scores = {
    normal: features.normalScore,
    benign: features.benignScore,
    malignant: features.malignantScore
  };

  const choice = chooseMaxScore(scores);
  let label = choice.label;

  if (label === "malignant" && (
    features.lesionSpiculation < 0.34 ||
    features.lesionCircularity > 0.58 ||
    features.lesionRatio < 0.04 ||
    choice.bestScore - choice.secondScore < 0.08
  )) {
    label = "benign";
  }
  if (label === "benign" && features.lesionSpiculation > 0.52 && features.lesionRoughness > 0.46) {
    label = "malignant";
  }
  if (label !== "normal" && features.lesionRatio < 0.015) {
    label = "normal";
  }

  const scoreGap = Math.max(0, choice.bestScore - choice.secondScore);
  let confidence = Math.round(60 + scoreGap * 58 + Math.max(0, choice.bestScore - 0.45) * 16);

  if (label === "normal" && features.lesionCircularity > 0.5 && features.lesionSpiculation < 0.2) {
    confidence += 10;
  }
  if (label === "benign" && features.lesionCircularity > 0.5 && features.lesionSpiculation < 0.35) {
    confidence += 8;
  }
  if (label === "malignant" && features.lesionSpiculation > 0.38 && features.lesionRoughness > 0.28) {
    confidence += 8;
  }

  if (label === "malignant") {
    confidence -= 10;
  }
  if (label === "benign") {
    confidence += 4;
  }

  confidence = Math.max(52, Math.min(88, confidence));

  return {
    label,
    confidence,
    features
  };
};

const autoCropImage = (imgData, features, paddingFactor = 0.3) => {
  const w = imgData.width;
  const h = imgData.height;
  
  if (features.lesionMaxX <= features.lesionMinX || features.lesionMaxY <= features.lesionMinY) {
    return imgData; // No valid crop found
  }

  const boxW = features.lesionMaxX - features.lesionMinX;
  const boxH = features.lesionMaxY - features.lesionMinY;
  
  const padX = boxW * paddingFactor;
  const padY = boxH * paddingFactor;

  const minX = Math.max(0, Math.floor(features.lesionMinX - padX));
  const minY = Math.max(0, Math.floor(features.lesionMinY - padY));
  const maxX = Math.min(w, Math.ceil(features.lesionMaxX + padX));
  const maxY = Math.min(h, Math.ceil(features.lesionMaxY + padY));

  const cropW = maxX - minX;
  const cropH = maxY - minY;

  const cropped = new ImageData(cropW, cropH);
  for (let y = 0; y < cropH; y++) {
    for (let x = 0; x < cropW; x++) {
      const srcIdx = ((minY + y) * w + (minX + x)) * 4;
      const dstIdx = (y * cropW + x) * 4;
      cropped.data[dstIdx] = imgData.data[srcIdx];
      cropped.data[dstIdx + 1] = imgData.data[srcIdx + 1];
      cropped.data[dstIdx + 2] = imgData.data[srcIdx + 2];
      cropped.data[dstIdx + 3] = imgData.data[srcIdx + 3];
    }
  }
  return { croppedData: cropped, minX, minY, cropW, cropH };
};

const saveCanvas = (canvas, fileName = "image.png") => {
  const link = document.createElement("a");
  link.href = canvas.toDataURL("image/png");
  link.download = fileName;
  link.click();
};

const showPage = (id) => {
  ["page1", "page2", "page3", "page4"].forEach((pageId) => {
    document.getElementById(pageId).classList.add("hidden");
  });
  document.getElementById(id).classList.remove("hidden");
};

window.onload = () => {
  // Start loading ONNX model immediately in background
  loadOnnxModel().then((ok) => {
    if (ok) {
      console.log('[App] ONNX model loaded and ready');
    } else {
      console.warn('[App] ONNX model not available, will use rule-based fallback');
    }
  });

  const originalCanvas = document.getElementById("original");
  const editedCanvas = document.getElementById("edited");
  const page2InputCanvas = document.getElementById("page2Input");
  const page2OutputCanvas = document.getElementById("page2Output");
  const morphOriginalCanvas = document.getElementById("morphOriginal");
  const morphEditedCanvas = document.getElementById("morphEdited");
  const classifyCanvas = document.getElementById("classifyCanvas");

  const octx = originalCanvas.getContext("2d");
  const ectx = editedCanvas.getContext("2d");
  const p2iCtx = page2InputCanvas.getContext("2d");
  const p2oCtx = page2OutputCanvas.getContext("2d");
  const moCtx = morphOriginalCanvas.getContext("2d");
  const meCtx = morphEditedCanvas.getContext("2d");
  const cctx = classifyCanvas.getContext("2d");

  const uploadMain = document.getElementById("uploadMain");
  const uploadMorph = document.getElementById("uploadMorph");
  const uploadClassify = document.getElementById("uploadClassify");
  const noise = document.getElementById("noise");
  const noiseValue = document.getElementById("noiseValue");
  const noiseFilterSelect = document.getElementById("noiseFilterSelect");
  const noiseInfo = document.getElementById("noiseInfo");
  const brightness = document.getElementById("brightness");
  const brightnessValue = document.getElementById("brightnessValue");
  const contrast = document.getElementById("contrast");
  const contrastValue = document.getElementById("contrastValue");
  const claheClip = document.getElementById("claheClip");
  const claheClipValue = document.getElementById("claheClipValue");
  const claheTiles = document.getElementById("claheTiles");
  const claheInfo = document.getElementById("claheInfo");
  const page2ArrowMenu = document.getElementById("page2ArrowMenu");
  const btnArrowSave = document.getElementById("btnArrowSave");
  const btnArrowBack = document.getElementById("btnArrowBack");
  const btnArrowCancel = document.getElementById("btnArrowCancel");
  const toPage1Arrow = document.getElementById("toPage1");
  const btnUndoPage2 = document.getElementById("btnUndoPage2");
  const page2HistoryList = document.getElementById("page2HistoryList");
  const classifyResultBox = document.getElementById("classifyResult");
  let lastClassificationFeatures = null;

  const renderPage1 = () => {
    if (!state.originalImageData) {
      return;
    }

    const denoised = applyNoiseFilter(state.originalImageData, Number(noise.value), state.noiseFilter);
    const result = applyBrightness(denoised, Number(brightness.value));
    state.page1Result = result;
    drawImageData(originalCanvas, octx, state.originalImageData);
    drawImageData(editedCanvas, ectx, result);
  };

  const renderPage2FromBase = () => {
    if (!state.page2Base || !state.page2Working) {
      return;
    }

    const result = applyContrast(state.page2Working, Number(contrast.value));
    state.page2Result = result;
    drawImageData(page2InputCanvas, p2iCtx, state.page2Base);
    drawImageData(page2OutputCanvas, p2oCtx, result);
  };

  const updatePage2UndoButton = () => {
    btnUndoPage2.disabled = state.page2HistoryCursor <= 0;
  };

  const renderPage2History = () => {
    if (!page2HistoryList) {
      return;
    }

    page2HistoryList.innerHTML = "";
    if (state.page2History.length === 0) {
      const empty = document.createElement("span");
      empty.className = "history-empty";
      empty.textContent = "No history yet";
      page2HistoryList.appendChild(empty);
      return;
    }

    state.page2History.forEach((entry, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "history-step";
      if (index === state.page2HistoryCursor) {
        button.classList.add("active");
      }
      button.textContent = `${index + 1}. ${entry.label}`;
      button.addEventListener("click", () => {
        state.page2HistoryCursor = index;
        state.page2Working = copyImageData(entry.image);
        contrast.value = "0";
        contrastValue.textContent = "0";
        claheInfo.textContent = `History restored | Step ${index + 1}: ${entry.label}`;
        updatePage2UndoButton();
        renderPage2FromBase();
        renderPage2History();
      });
      page2HistoryList.appendChild(button);
    });
  };

  const commitPage2Operation = (label, imageData) => {
    if (!imageData) {
      return;
    }

    if (state.page2HistoryCursor < state.page2History.length - 1) {
      state.page2History = state.page2History.slice(0, state.page2HistoryCursor + 1);
    }

    state.page2Working = copyImageData(imageData);
    state.page2History.push({
      id: state.page2HistoryNextId,
      label,
      image: copyImageData(imageData)
    });
    state.page2HistoryNextId += 1;

    if (state.page2History.length > 20) {
      state.page2History.shift();
      state.page2HistoryCursor = Math.max(0, state.page2HistoryCursor - 1);
    }

    state.page2HistoryCursor = state.page2History.length - 1;
    updatePage2UndoButton();
    renderPage2History();
  };

  const syncPage2FromPage1 = () => {
    if (!state.page1Result) {
      return;
    }

    state.page2Base = copyImageData(state.page1Result);
    state.page2Working = copyImageData(state.page1Result);
    state.page2History = [{ id: 0, label: "Input", image: copyImageData(state.page1Result) }];
    state.page2HistoryCursor = 0;
    state.page2HistoryNextId = 1;
    contrast.value = "0";
    contrastValue.textContent = "0";
    claheInfo.textContent = "CLAHE ready | Clip Limit: 2.0 | Tile Grid: 8x8";
    updatePage2UndoButton();
    renderPage2FromBase();
    renderPage2History();
  };

  const resetPage1Workflow = () => {
    state.originalImageData = null;
    state.page1Result = null;
    state.page2Base = null;
    state.page2Working = null;
    state.page2Result = null;
    state.page2History = [];
    state.page2HistoryCursor = -1;
    state.page2HistoryNextId = 1;

    uploadMain.value = "";
    noise.value = "0";
    brightness.value = "0";
    contrast.value = "0";
    noiseValue.textContent = "0";
    brightnessValue.textContent = "0";
    contrastValue.textContent = "0";
    claheClip.value = "2.0";
    claheClipValue.textContent = "2.0";
    claheTiles.value = "8";
    claheInfo.textContent = "CLAHE ready | Clip Limit: 2.0 | Tile Grid: 8x8";

    state.noiseMode = "manual";
    state.noiseFilter = "mean";
    noiseFilterSelect.value = "mean";
    noiseInfo.textContent = "Workflow reset. Upload a new image for page 1.";

    clearCanvas(originalCanvas, octx);
    clearCanvas(editedCanvas, ectx);
    clearCanvas(page2InputCanvas, p2iCtx);
    clearCanvas(page2OutputCanvas, p2oCtx);
    updatePage2UndoButton();
    renderPage2History();
  };

  uploadMain.addEventListener("change", async (e) => {
    try {
      const imgData = await loadImageFile(e.target.files[0]);
      state.originalImageData = imgData;
      brightness.value = "0";
      noise.value = "0";
      contrast.value = "0";
      brightnessValue.textContent = "0";
      noiseValue.textContent = "0";
      contrastValue.textContent = "0";
      state.noiseMode = "manual";
      state.noiseFilter = "mean";
      noiseFilterSelect.value = "mean";
      noiseInfo.textContent = "Noise analyzer ready. Filters available: Mean, Gaussian, Median. Current: Mean Filter.";
      claheInfo.textContent = "CLAHE ready | Clip Limit: 2.0 | Tile Grid: 8x8";
      renderPage1();
      syncPage2FromPage1();
    } catch (err) {
      alert("ไม่สามารถโหลดรูปได้");
    }
  });

  noise.addEventListener("input", () => {
    state.noiseMode = "manual";
    state.noiseFilter = noiseFilterSelect.value;
    noiseValue.textContent = noise.value;
    noiseInfo.textContent = `Manual noise mode | Level: ${noise.value}/20 | Using: ${filterLabelMap[state.noiseFilter]} | Filters available: Mean, Gaussian, Median`;
    renderPage1();
    syncPage2FromPage1();
  });

  brightness.addEventListener("input", () => {
    brightnessValue.textContent = brightness.value;
    renderPage1();
    syncPage2FromPage1();
  });

  document.getElementById("btnResetPage1").addEventListener("click", () => {
    if (!state.originalImageData) {
      return;
    }
    resetPage1Workflow();
  });

  document.getElementById("btnAutoNoise").addEventListener("click", () => {
    if (!state.originalImageData) {
      alert("กรุณาอัปโหลดรูปก่อน");
      return;
    }

    const variance = estimateNoiseVariance(state.originalImageData);
    const suggestion = suggestNoiseFilter(variance);

    state.noiseMode = "auto";
    state.noiseFilter = suggestion.filter;
    noiseFilterSelect.value = suggestion.filter;
    noise.value = String(suggestion.level);
    noiseValue.textContent = noise.value;

    noiseInfo.textContent = `Auto noise ON | Variance: ${variance.toFixed(2)} | Applied: ${suggestion.label} | Level: ${suggestion.level}/20 | Filters available: Mean, Gaussian, Median`;

    renderPage1();
    syncPage2FromPage1();
  });

  noiseFilterSelect.addEventListener("change", () => {
    state.noiseMode = "manual";
    state.noiseFilter = noiseFilterSelect.value;
    noiseInfo.textContent = `Manual noise mode | Level: ${noise.value}/20 | Using: ${filterLabelMap[state.noiseFilter]} | Filters available: Mean, Gaussian, Median`;
    renderPage1();
    syncPage2FromPage1();
  });

  document.getElementById("toPage2").addEventListener("click", () => {
    if (!state.page1Result) {
      alert("กรุณาอัปโหลดรูปก่อน");
      return;
    }
    page2ArrowMenu.classList.add("hidden");
    page2ArrowMenu.setAttribute("aria-hidden", "true");
    syncPage2FromPage1();
    showPage("page2");
  });

  const closePage2ArrowMenu = () => {
    page2ArrowMenu.classList.add("hidden");
    page2ArrowMenu.setAttribute("aria-hidden", "true");
  };

  toPage1Arrow.addEventListener("click", (event) => {
    event.stopPropagation();
    const willOpen = page2ArrowMenu.classList.contains("hidden");
    if (willOpen) {
      page2ArrowMenu.classList.remove("hidden");
      page2ArrowMenu.setAttribute("aria-hidden", "false");
      return;
    }
    closePage2ArrowMenu();
  });

  page2ArrowMenu.addEventListener("click", (event) => {
    event.stopPropagation();
  });

  btnArrowSave.addEventListener("click", () => {
    if (!state.page2Result) {
      alert("ยังไม่มีภาพสำหรับบันทึก");
      return;
    }
    saveCanvas(page2OutputCanvas, "contrast-result.png");
    closePage2ArrowMenu();
  });

  btnArrowBack.addEventListener("click", () => {
    closePage2ArrowMenu();
    resetPage1Workflow();
    showPage("page1");
  });

  btnArrowCancel.addEventListener("click", () => {
    closePage2ArrowMenu();
  });

  document.addEventListener("click", () => {
    closePage2ArrowMenu();
  });

  document.getElementById("btnAutoContrast").addEventListener("click", () => {
    if (!state.page2Working) {
      alert("ยังไม่มีภาพสำหรับปรับ contrast");
      return;
    }

    const result = autoContrast(state.page2Working);
    commitPage2Operation("Auto Contrast", result);
    contrast.value = "0";
    contrastValue.textContent = "0";
    claheInfo.textContent = "Auto Contrast applied | Left image remains original input";
    renderPage2FromBase();
  });

  contrast.addEventListener("input", () => {
    contrastValue.textContent = contrast.value;
    renderPage2FromBase();
  });

  claheClip.addEventListener("input", () => {
    claheClipValue.textContent = Number(claheClip.value).toFixed(1);
  });

  document.getElementById("btnApplyClahe").addEventListener("click", () => {
    if (!state.page2Working) {
      alert("ยังไม่มีภาพสำหรับ CLAHE");
      return;
    }

    const clip = Number(claheClip.value);
    const tiles = 8;

    const result = applyClahe(state.page2Working, clip, tiles);
    commitPage2Operation(`CLAHE ${clip.toFixed(1)} / ${tiles}x${tiles}`, result);
    contrast.value = "0";
    contrastValue.textContent = "0";
    claheInfo.textContent = `CLAHE applied | Clip Limit: ${clip.toFixed(1)} | Tile Grid: ${tiles}x${tiles}`;
    renderPage2FromBase();
  });

  btnUndoPage2.addEventListener("click", () => {
    if (state.page2HistoryCursor <= 0) {
      return;
    }

    state.page2HistoryCursor -= 1;
    state.page2Working = copyImageData(state.page2History[state.page2HistoryCursor].image);
    contrast.value = "0";
    contrastValue.textContent = "0";
    claheInfo.textContent = "Undo applied | Right image reverted to previous step";
    updatePage2UndoButton();
    renderPage2FromBase();
    renderPage2History();
  });

  updatePage2UndoButton();
  renderPage2History();

  uploadMorph.addEventListener("change", async (e) => {
    try {
      const imgData = await loadImageFile(e.target.files[0]);
      state.morphInputImageData = imgData;
      state.morphResult = null;
      drawImageData(morphOriginalCanvas, moCtx, imgData);
      clearCanvas(morphEditedCanvas, meCtx);
    } catch (err) {
      alert("ไม่สามารถโหลดรูป morphology ได้");
    }
  });

  document.getElementById("btnMorph").addEventListener("click", () => {
    if (!state.morphInputImageData) {
      alert("ยังไม่มีภาพให้ทำ morphology");
      return;
    }

    const morphType = document.getElementById("morphType").value;
    const morphKernel = Number(document.getElementById("morphKernel").value || 3);
    const result = runMorphology(state.morphInputImageData, morphType, morphKernel);
    state.morphResult = result;
    drawImageData(morphOriginalCanvas, moCtx, state.morphInputImageData);
    drawImageData(morphEditedCanvas, meCtx, result);
  });

  document.getElementById("btnSaveMorph").addEventListener("click", () => {
    if (!state.morphResult) {
      alert("ยังไม่มีภาพ morphology สำหรับบันทึก");
      return;
    }
    saveCanvas(morphEditedCanvas, "morphology-result.png");
  });

  uploadClassify.addEventListener("change", async (e) => {
    try {
      const imgData = await loadImageFile(e.target.files[0]);
      state.classifyInputImageData = imgData;
      lastClassificationFeatures = null;
      classifyResultBox.textContent = "Result: ready to analyze.";
      drawImageData(classifyCanvas, cctx, imgData);
    } catch (err) {
      alert("ไม่สามารถโหลดรูป classification ได้");
    }
  });

  document.getElementById("btnClassify").addEventListener("click", () => {
    const classifyBtn = document.getElementById("btnClassify");

    if (!state.classifyInputImageData) {
      alert("ยังไม่มีภาพให้ classify");
      return;
    }

    drawImageData(classifyCanvas, cctx, state.classifyInputImageData);
    classifyBtn.disabled = true;
    classifyResultBox.textContent = "Result: processing...";
    
    // Apply CLAHE (Critical: model is trained on CLAHE images)
    let finalImgData = applyClahe(state.classifyInputImageData, 2.0, 8);
    
    // Use ONNX model (in-browser) with fallback to decision tree
    (async () => {
      try {
        const result = await classifyWithModel(finalImgData);
        lastClassificationFeatures = result.features || {};
        const probs = result.probabilities || {};
        const pNormal = Number(probs.normal || 0).toFixed(3);
        const pBenign = Number(probs.benign || 0).toFixed(3);
        const pMalignant = Number(probs.malignant || 0).toFixed(3);
        const decisionSource = result.decision_source || 'unknown';
        const timeInfo = result.inferenceTime ? ` | Time: ${result.inferenceTime}s` : '';
        classifyResultBox.textContent =
          `Result: ${result.label} | Confidence: ${result.confidence}% | Source: ${decisionSource}${timeInfo} | ` +
          `P(normal): ${pNormal}, P(benign): ${pBenign}, P(malignant): ${pMalignant}`;
      } catch (error) {
        console.error('Classify error:', error);
        classifyResultBox.textContent = `Result: error - ${error.message}`;
      } finally {
        classifyBtn.disabled = false;
      }
    })();
  });


  const drawer = document.getElementById("drawer");
  const drawerBackdrop = document.getElementById("drawerBackdrop");
  const menuBtn = document.getElementById("menuBtn");
  const closeDrawerBtn = document.getElementById("closeDrawer");

  const openDrawer = () => {
    drawer.classList.remove("hidden");
    drawerBackdrop.classList.remove("hidden");
    drawer.setAttribute("aria-hidden", "false");
    drawer.removeAttribute("inert");
    closeDrawerBtn.focus();
  };

  const closeDrawer = () => {
    // Move focus out of drawer before hiding it to avoid aria-hidden focus traps.
    if (drawer.contains(document.activeElement)) {
      menuBtn.focus();
    }
    drawer.classList.add("hidden");
    drawerBackdrop.classList.add("hidden");
    drawer.setAttribute("aria-hidden", "true");
    drawer.setAttribute("inert", "");
  };

  menuBtn.addEventListener("click", openDrawer);
  closeDrawerBtn.addEventListener("click", closeDrawer);
  drawerBackdrop.addEventListener("click", closeDrawer);

  document.querySelectorAll(".menu-link").forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.target;
      if (target === "page2" && !state.page1Result) {
        alert("เริ่มจากหน้า 1 ก่อน");
        return;
      }
      showPage(target);
      closeDrawer();
    });
  });
};
