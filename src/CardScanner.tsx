import { useCallback, useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import type { Worker } from "tesseract.js";

type Crop = { x: number; y: number; width: number; height: number };
type Point = { x: number; y: number };
type Candidate = { text: string; digits: string; confidence: number };
type AnchorCandidate = { points: Point[]; area: number; score: number; aligned: number };
type Cv = typeof import("@techstark/opencv-js");
type CvRuntime = Cv & { onRuntimeInitialized?: () => void };

declare global {
  interface Window {
    cv?: CvRuntime | Promise<CvRuntime>;
  }
}

const CODE_CROPS: Crop[] = [
  { x: 42, y: 77.7, width: 56.5, height: 10.4 },
  { x: 43, y: 82, width: 54.5, height: 9 },
];
const ACCESS_NUMBER_CROP: Crop = { x: 1, y: 0, width: 98, height: 74 };
const CAMERA_CROP_MARGIN = 0.04;
const CANONICAL_CARD_WIDTH = 1280;
const CANONICAL_CARD_HEIGHT = 808;

let openCvPromise: Promise<CvRuntime> | null = null;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function formatCode(value: string) {
  return value.match(/.{1,4}/g)?.join(" ") ?? value;
}

function distance(a: Point, b: Point) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function orderPoints(points: Point[]) {
  const bySum = [...points].sort((a, b) => a.x + a.y - (b.x + b.y));
  const byDiff = [...points].sort((a, b) => a.y - a.x - (b.y - b.x));
  return [bySum[0], byDiff[0], bySum[bySum.length - 1], byDiff[byDiff.length - 1]];
}

function rotateCanvas(source: HTMLCanvasElement, degrees: 90 | 180 | 270) {
  const swapsSides = degrees === 90 || degrees === 270;
  const canvas = document.createElement("canvas");
  canvas.width = swapsSides ? source.height : source.width;
  canvas.height = swapsSides ? source.width : source.height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("浏览器无法旋转图片");
  context.translate(canvas.width / 2, canvas.height / 2);
  context.rotate((degrees * Math.PI) / 180);
  context.drawImage(source, -source.width / 2, -source.height / 2);
  return canvas;
}

function cloneCanvas(source: HTMLCanvasElement) {
  const canvas = document.createElement("canvas");
  canvas.width = source.width;
  canvas.height = source.height;
  canvas.getContext("2d")?.drawImage(source, 0, 0);
  return canvas;
}

async function loadOpenCv() {
  if (openCvPromise) return openCvPromise;
  openCvPromise = new Promise<CvRuntime>((resolve, reject) => {
    const finish = async () => {
      try {
        const candidate = await Promise.resolve(window.cv);
        if (!candidate) throw new Error("OpenCV 没有完成初始化");
        if (candidate.Mat) {
          resolve(candidate);
          return;
        }
        candidate.onRuntimeInitialized = () => resolve(candidate);
      } catch (error) {
        reject(error);
      }
    };

    if (window.cv) {
      void finish();
      return;
    }
    const existing = document.querySelector<HTMLScriptElement>("script[data-opencv]");
    if (existing) {
      existing.addEventListener("load", () => void finish(), { once: true });
      existing.addEventListener("error", () => reject(new Error("OpenCV 加载失败")), {
        once: true,
      });
      return;
    }
    const script = document.createElement("script");
    script.src =
      "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@5.0.0-release.1/dist/opencv.js";
    script.async = true;
    script.dataset.opencv = "true";
    script.onload = () => void finish();
    script.onerror = () => reject(new Error("OpenCV 加载失败"));
    document.head.appendChild(script);
  });
  return openCvPromise;
}

function warpCard(sourceCanvas: HTMLCanvasElement, cv: CvRuntime, points: Point[]) {
  const ordered = orderPoints(points);
  const topWidth = distance(ordered[0], ordered[1]);
  const bottomWidth = distance(ordered[3], ordered[2]);
  const leftHeight = distance(ordered[0], ordered[3]);
  const rightHeight = distance(ordered[1], ordered[2]);
  const measuredWidth = Math.max(topWidth, bottomWidth);
  const measuredHeight = Math.max(leftHeight, rightHeight);
  const portrait = measuredHeight > measuredWidth;
  const outputWidth = portrait ? CANONICAL_CARD_HEIGHT : CANONICAL_CARD_WIDTH;
  const outputHeight = portrait ? CANONICAL_CARD_WIDTH : CANONICAL_CARD_HEIGHT;
  const source = cv.imread(sourceCanvas);
  const destination = new cv.Mat();
  const opaqueDestination = new cv.Mat();
  const sourcePoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
    ordered[0].x,
    ordered[0].y,
    ordered[1].x,
    ordered[1].y,
    ordered[2].x,
    ordered[2].y,
    ordered[3].x,
    ordered[3].y,
  ]);
  const destinationPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0,
    0,
    outputWidth - 1,
    0,
    outputWidth - 1,
    outputHeight - 1,
    0,
    outputHeight - 1,
  ]);
  const transform = cv.getPerspectiveTransform(sourcePoints, destinationPoints);
  cv.warpPerspective(
    source,
    destination,
    transform,
    new cv.Size(outputWidth, outputHeight),
    cv.INTER_LINEAR,
    cv.BORDER_REPLICATE,
  );
  cv.cvtColor(destination, opaqueDestination, cv.COLOR_RGBA2RGB);
  const canvas = document.createElement("canvas");
  cv.imshow(canvas, opaqueDestination);
  source.delete();
  destination.delete();
  opaqueDestination.delete();
  sourcePoints.delete();
  destinationPoints.delete();
  transform.delete();
  return portrait ? rotateCanvas(canvas, 90) : canvas;
}

function normalizeCardSize(source: HTMLCanvasElement) {
  const landscape = source.width >= source.height ? source : rotateCanvas(source, 90);
  const canvas = document.createElement("canvas");
  canvas.width = CANONICAL_CARD_WIDTH;
  canvas.height = CANONICAL_CARD_HEIGHT;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("浏览器无法归一化卡片尺寸");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(landscape, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function warpWideRegion(sourceCanvas: HTMLCanvasElement, cv: CvRuntime, points: Point[]) {
  let ordered = orderPoints(points);
  let measuredWidth = Math.max(
    distance(ordered[0], ordered[1]),
    distance(ordered[3], ordered[2]),
  );
  let measuredHeight = Math.max(
    distance(ordered[0], ordered[3]),
    distance(ordered[1], ordered[2]),
  );
  if (measuredHeight > measuredWidth) {
    ordered = [ordered[3], ordered[0], ordered[1], ordered[2]];
    [measuredWidth, measuredHeight] = [measuredHeight, measuredWidth];
  }
  const outputWidth = clamp(Math.round(measuredWidth), 1000, 1600);
  const outputHeight = Math.max(100, Math.round(outputWidth * measuredHeight / measuredWidth));
  const source = cv.imread(sourceCanvas);
  const destination = new cv.Mat();
  const opaqueDestination = new cv.Mat();
  const sourcePoints = cv.matFromArray(
    4,
    1,
    cv.CV_32FC2,
    ordered.flatMap((point) => [point.x, point.y]),
  );
  const destinationPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0,
    outputWidth - 1, 0,
    outputWidth - 1, outputHeight - 1,
    0, outputHeight - 1,
  ]);
  const transform = cv.getPerspectiveTransform(sourcePoints, destinationPoints);
  cv.warpPerspective(
    source,
    destination,
    transform,
    new cv.Size(outputWidth, outputHeight),
    cv.INTER_LINEAR,
    cv.BORDER_REPLICATE,
  );
  cv.cvtColor(destination, opaqueDestination, cv.COLOR_RGBA2RGB);
  const canvas = document.createElement("canvas");
  cv.imshow(canvas, opaqueDestination);
  source.delete();
  destination.delete();
  opaqueDestination.delete();
  sourcePoints.delete();
  destinationPoints.delete();
  transform.delete();
  return canvas;
}

function scoreAccessCodeBoxes(
  warped: InstanceType<CvRuntime["Mat"]>,
  cv: CvRuntime,
) {
  const grey = new cv.Mat();
  const binary = new cv.Mat();
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  const boxes: Array<{ x: number; y: number; width: number }> = [];
  try {
    cv.cvtColor(warped, grey, cv.COLOR_RGBA2GRAY);
    cv.adaptiveThreshold(
      grey,
      binary,
      255,
      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv.THRESH_BINARY,
      31,
      4,
    );
    cv.findContours(binary, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    for (let index = 0; index < contours.size(); index += 1) {
      const contour = contours.get(index);
      const rect = cv.boundingRect(contour);
      const area = Math.abs(cv.contourArea(contour));
      const widthRatio = rect.width / warped.cols;
      const heightRatio = rect.height / warped.rows;
      const aspect = rect.width / Math.max(1, rect.height);
      const fill = area / Math.max(1, rect.width * rect.height);
      if (
        widthRatio >= 0.018 && widthRatio <= 0.075 &&
        heightRatio >= 0.12 && heightRatio <= 0.48 &&
        aspect >= 0.42 && aspect <= 1.3 && fill >= 0.45
      ) {
        boxes.push({
          x: rect.x + rect.width / 2,
          y: rect.y + rect.height / 2,
          width: rect.width,
        });
      }
      contour.delete();
    }
    let bestAligned = 0;
    let bestScore = 0;
    for (const seed of boxes) {
      const row = boxes
        .filter((box) => Math.abs(box.y - seed.y) < warped.rows * 0.11)
        .sort((a, b) => a.x - b.x);
      const widths = row.map((box) => box.width).sort((a, b) => a - b);
      const medianWidth = widths[Math.floor(widths.length / 2)] ?? 1;
      const consistent = row.filter(
        (box) => box.width > medianWidth * 0.55 && box.width < medianWidth * 1.65,
      );

      // A printed letter box normally produces both an inner and an outer
      // contour. Count overlapping x positions once, otherwise the real ten
      // ACCESS CODE boxes are often miscounted as roughly twenty boxes.
      const unique: typeof consistent = [];
      for (const box of consistent) {
        const previous = unique[unique.length - 1];
        if (!previous || Math.abs(box.x - previous.x) > medianWidth * 0.65) {
          unique.push(box);
        }
      }
      const gaps = unique.slice(1).map((box, index) => box.x - unique[index].x);
      const sortedGaps = [...gaps].sort((a, b) => a - b);
      const medianGap = sortedGaps[Math.floor(sortedGaps.length / 2)] ?? 1;
      const regularGaps = gaps.filter(
        (gap) => gap > medianGap * 0.55 && gap < medianGap * 1.45,
      ).length;
      const aligned = unique.length;
      const countScore = Math.max(0, 10 - Math.abs(10 - aligned)) * 5;
      const spacingScore = gaps.length ? (regularGaps / gaps.length) * 20 : 0;
      const score = countScore + spacingScore;
      if (score > bestScore) {
        bestScore = score;
        bestAligned = aligned;
      }
    }
    return {
      aligned: bestAligned,
      score: bestScore,
    };
  } finally {
    grey.delete();
    binary.delete();
    contours.delete();
    hierarchy.delete();
  }
}

function detectAccessCodeRegion(sourceCanvas: HTMLCanvasElement, cv: CvRuntime) {
  const longestSide = Math.max(sourceCanvas.width, sourceCanvas.height);
  const detectionScale = Math.min(1, 1600 / longestSide);
  const detectionCanvas = document.createElement("canvas");
  detectionCanvas.width = Math.round(sourceCanvas.width * detectionScale);
  detectionCanvas.height = Math.round(sourceCanvas.height * detectionScale);
  const context = detectionCanvas.getContext("2d");
  if (!context) throw new Error("浏览器无法分析 ACCESS CODE 区域");
  context.drawImage(sourceCanvas, 0, 0, detectionCanvas.width, detectionCanvas.height);

  const source = cv.imread(detectionCanvas);
  const grey = new cv.Mat();
  const blurred = new cv.Mat();
  const edges = new cv.Mat();
  const closed = new cv.Mat();
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
  const candidates: AnchorCandidate[] = [];
  try {
    cv.cvtColor(source, grey, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(grey, blurred, new cv.Size(3, 3), 0);
    cv.Canny(blurred, edges, 30, 110);
    cv.morphologyEx(edges, closed, cv.MORPH_CLOSE, kernel);
    cv.findContours(closed, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    const imageArea = detectionCanvas.width * detectionCanvas.height;

    for (let index = 0; index < contours.size(); index += 1) {
      const contour = contours.get(index);
      const area = Math.abs(cv.contourArea(contour));
      const areaRatio = area / imageArea;
      if (areaRatio < 0.0015 || areaRatio > 0.12) {
        contour.delete();
        continue;
      }
      const perimeter = cv.arcLength(contour, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(contour, approx, perimeter * 0.02, true);
      if (approx.rows === 4 && cv.isContourConvex(approx)) {
        const points: Point[] = Array.from({ length: 4 }, (_, pointIndex) => ({
          x: approx.data32S[pointIndex * 2],
          y: approx.data32S[pointIndex * 2 + 1],
        }));
        const ordered = orderPoints(points);
        const width = Math.max(
          distance(ordered[0], ordered[1]),
          distance(ordered[3], ordered[2]),
        );
        const height = Math.max(
          distance(ordered[0], ordered[3]),
          distance(ordered[1], ordered[2]),
        );
        const ratio = Math.max(width, height) / Math.max(1, Math.min(width, height));
        if (ratio >= 3.4 && ratio <= 8) {
          const analysisPoints = points.map((point) => ({
            x: point.x,
            y: point.y,
          }));
          const warpedCanvas = warpWideRegion(detectionCanvas, cv, analysisPoints);
          const warped = cv.imread(warpedCanvas);
          const structure = scoreAccessCodeBoxes(warped, cv);
          warped.delete();
          if (
            structure.aligned >= 8 &&
            structure.aligned <= 12 &&
            structure.score >= 55
          ) {
            candidates.push({
              points: points.map((point) => ({
                x: point.x / detectionScale,
                y: point.y / detectionScale,
              })),
              area,
              ...structure,
            });
          }
        }
      }
      approx.delete();
      contour.delete();
    }
  } finally {
    source.delete();
    grey.delete();
    blurred.delete();
    edges.delete();
    closed.delete();
    contours.delete();
    hierarchy.delete();
    kernel.delete();
  }

  if (!candidates.length) return null;
  const highestScore = Math.max(...candidates.map((candidate) => candidate.score));
  const best = candidates
    .filter((candidate) => candidate.score >= highestScore - 8)
    .sort((a, b) => b.area - a.area)[0];
  return {
    canvas: warpWideRegion(sourceCanvas, cv, best.points),
    points: best.points,
    alignedBoxes: best.aligned,
  };
}

function detectCard(sourceCanvas: HTMLCanvasElement, cv: CvRuntime) {
  const longestSide = Math.max(sourceCanvas.width, sourceCanvas.height);
  const detectionScale = Math.min(1, 1200 / longestSide);
  const detectionCanvas = document.createElement("canvas");
  detectionCanvas.width = Math.round(sourceCanvas.width * detectionScale);
  detectionCanvas.height = Math.round(sourceCanvas.height * detectionScale);
  const detectionContext = detectionCanvas.getContext("2d");
  if (!detectionContext) throw new Error("浏览器无法分析图片");
  detectionContext.drawImage(sourceCanvas, 0, 0, detectionCanvas.width, detectionCanvas.height);

  const source = cv.imread(detectionCanvas);
  const grey = new cv.Mat();
  const blurred = new cv.Mat();
  const edges = new cv.Mat();
  const closed = new cv.Mat();
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(7, 7));
  let bestQuad: Point[] | null = null;
  let bestQuadArea = 0;
  let fallbackContour: InstanceType<CvRuntime["Mat"]> | null = null;
  let fallbackArea = 0;

  try {
    cv.cvtColor(source, grey, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(grey, blurred, new cv.Size(5, 5), 0);
    cv.Canny(blurred, edges, 45, 145);
    cv.morphologyEx(edges, closed, cv.MORPH_CLOSE, kernel);
    cv.findContours(closed, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    const imageArea = detectionCanvas.width * detectionCanvas.height;

    for (let index = 0; index < contours.size(); index += 1) {
      const contour = contours.get(index);
      const area = Math.abs(cv.contourArea(contour));
      if (area < imageArea * 0.1 || area > imageArea * 0.985) {
        contour.delete();
        continue;
      }
      if (area > fallbackArea) {
        fallbackContour?.delete();
        fallbackContour = contour.clone();
        fallbackArea = area;
      }
      const perimeter = cv.arcLength(contour, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(contour, approx, perimeter * 0.025, true);
      if (approx.rows === 4 && cv.isContourConvex(approx) && area > bestQuadArea) {
        const points: Point[] = [];
        for (let pointIndex = 0; pointIndex < 4; pointIndex += 1) {
          points.push({
            x: approx.data32S[pointIndex * 2] / detectionScale,
            y: approx.data32S[pointIndex * 2 + 1] / detectionScale,
          });
        }
        const ordered = orderPoints(points);
        const width = Math.max(distance(ordered[0], ordered[1]), distance(ordered[3], ordered[2]));
        const height = Math.max(distance(ordered[0], ordered[3]), distance(ordered[1], ordered[2]));
        const ratio = Math.max(width, height) / Math.max(1, Math.min(width, height));
        if (ratio >= 1.25 && ratio <= 1.9) {
          bestQuad = points;
          bestQuadArea = area;
        }
      }
      approx.delete();
      contour.delete();
    }

    if (!bestQuad && fallbackContour && fallbackArea > imageArea * 0.1) {
      const rect = cv.minAreaRect(fallbackContour);
      const rectPoints = cv.RotatedRect.points(rect);
      const ratio = Math.max(rect.size.width, rect.size.height) /
        Math.max(1, Math.min(rect.size.width, rect.size.height));
      if (ratio >= 1.2 && ratio <= 2) {
        bestQuad = rectPoints.map((point) => ({
          x: point.x / detectionScale,
          y: point.y / detectionScale,
        }));
      }
    }
  } finally {
    fallbackContour?.delete();
    source.delete();
    grey.delete();
    blurred.delete();
    edges.delete();
    closed.delete();
    contours.delete();
    hierarchy.delete();
    kernel.delete();
  }

  if (!bestQuad) return null;
  return { canvas: warpCard(sourceCanvas, cv, bestQuad), points: bestQuad };
}

function wholeImageFallback(source: HTMLCanvasElement) {
  const landscape = source.width >= source.height ? cloneCanvas(source) : rotateCanvas(source, 270);
  const ratio = landscape.width / landscape.height;
  if (ratio <= 2.05) return landscape;
  const canvas = document.createElement("canvas");
  const targetRatio = 1.6;
  canvas.height = landscape.height;
  canvas.width = Math.round(landscape.height * targetRatio);
  const context = canvas.getContext("2d");
  if (!context) return landscape;
  context.drawImage(
    landscape,
    Math.max(0, (landscape.width - canvas.width) / 2),
    0,
    Math.min(canvas.width, landscape.width),
    landscape.height,
    0,
    0,
    canvas.width,
    canvas.height,
  );
  return canvas;
}

function extractCrop(source: HTMLCanvasElement, crop: Crop) {
  const sx = Math.round((crop.x / 100) * source.width);
  const sy = Math.round((crop.y / 100) * source.height);
  const sw = Math.max(1, Math.min(source.width - sx, Math.round((crop.width / 100) * source.width)));
  const sh = Math.max(1, Math.min(source.height - sy, Math.round((crop.height / 100) * source.height)));
  const canvas = document.createElement("canvas");
  canvas.width = sw;
  canvas.height = sh;
  canvas.getContext("2d")?.drawImage(source, sx, sy, sw, sh, 0, 0, sw, sh);
  return canvas;
}

function otsuThreshold(histogram: Uint32Array, total: number) {
  let sum = 0;
  for (let i = 0; i < 256; i += 1) sum += i * histogram[i];
  let backgroundWeight = 0;
  let backgroundSum = 0;
  let bestVariance = 0;
  let threshold = 128;
  for (let i = 0; i < 256; i += 1) {
    backgroundWeight += histogram[i];
    if (!backgroundWeight) continue;
    const foregroundWeight = total - backgroundWeight;
    if (!foregroundWeight) break;
    backgroundSum += i * histogram[i];
    const backgroundMean = backgroundSum / backgroundWeight;
    const foregroundMean = (sum - backgroundSum) / foregroundWeight;
    const variance =
      backgroundWeight * foregroundWeight * (backgroundMean - foregroundMean) ** 2;
    if (variance > bestVariance) {
      bestVariance = variance;
      threshold = i;
    }
  }
  return threshold;
}

function prepareCrop(source: HTMLCanvasElement, crop: Crop, binary: boolean) {
  const sx = Math.round((crop.x / 100) * source.width);
  const sy = Math.round((crop.y / 100) * source.height);
  const sw = Math.max(1, Math.round((crop.width / 100) * source.width));
  const sh = Math.max(1, Math.round((crop.height / 100) * source.height));
  const scale = clamp(1500 / sw, 1.5, 4);
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(sw * scale);
  canvas.height = Math.round(sh * scale);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("浏览器无法创建图像处理画布");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(source, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
  const image = context.getImageData(0, 0, canvas.width, canvas.height);
  const histogram = new Uint32Array(256);
  const greys = new Uint8Array(canvas.width * canvas.height);
  let mean = 0;
  for (let pixel = 0, index = 0; pixel < image.data.length; pixel += 4, index += 1) {
    const grey = Math.round(
      image.data[pixel] * 0.299 +
        image.data[pixel + 1] * 0.587 +
        image.data[pixel + 2] * 0.114,
    );
    greys[index] = grey;
    histogram[grey] += 1;
    mean += grey;
  }
  const total = greys.length;
  const invert = mean / total < 128;
  let low = 0;
  let high = 255;
  let seen = 0;
  for (let i = 0; i < 256; i += 1) {
    seen += histogram[i];
    if (seen < total * 0.02) low = i;
    if (seen < total * 0.98) high = i;
  }
  const range = Math.max(24, high - low);
  const normalized = new Uint8Array(total);
  const normalizedHistogram = new Uint32Array(256);
  for (let index = 0; index < total; index += 1) {
    let value = clamp(Math.round(((greys[index] - low) / range) * 255), 0, 255);
    if (invert) value = 255 - value;
    normalized[index] = value;
    normalizedHistogram[value] += 1;
  }
  const threshold = otsuThreshold(normalizedHistogram, total);
  for (let pixel = 0, index = 0; pixel < image.data.length; pixel += 4, index += 1) {
    const value = binary ? (normalized[index] < threshold ? 0 : 255) : normalized[index];
    image.data[pixel] = value;
    image.data[pixel + 1] = value;
    image.data[pixel + 2] = value;
    image.data[pixel + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  const padded = document.createElement("canvas");
  padded.width = canvas.width + 100;
  padded.height = canvas.height + 48;
  const paddedContext = padded.getContext("2d");
  if (!paddedContext) return canvas;
  paddedContext.fillStyle = "#ffffff";
  paddedContext.fillRect(0, 0, padded.width, padded.height);
  paddedContext.drawImage(canvas, 50, 24);
  return padded;
}

function splitIntoCodeGroups(source: HTMLCanvasElement) {
  const outerPadding = 50;
  const contentWidth = source.width - outerPadding * 2;
  const groupWidth = contentWidth / 5;
  return Array.from({ length: 5 }, (_, index) => {
    const canvas = document.createElement("canvas");
    canvas.width = Math.ceil(groupWidth) + 24;
    canvas.height = source.height;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("浏览器无法拆分数字组");
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(
      source,
      outerPadding + index * groupWidth,
      0,
      groupWidth,
      source.height,
      12,
      0,
      groupWidth,
      source.height,
    );
    return canvas;
  });
}

function scoreCandidate(candidate: Candidate) {
  if (candidate.digits.length === 20) return -10_000 - candidate.confidence;
  return Math.abs(candidate.digits.length - 20) * 100 - candidate.confidence;
}

export default function CardScanner() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const cameraGuideRef = useRef<HTMLDivElement>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const [fileName, setFileName] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("请拍摄或选择一张包含完整卡片的照片");
  const [digits, setDigits] = useState("");
  const [rawText, setRawText] = useState("");
  const [confidence, setConfidence] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const [corrected, setCorrected] = useState(false);
  const [cardDetected, setCardDetected] = useState<boolean | null>(null);
  const [cameraAvailable, setCameraAvailable] = useState(false);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraStarting, setCameraStarting] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState("");
  const [cameraResolution, setCameraResolution] = useState("");

  const renderImage = useCallback(() => {
    const image = imageRef.current;
    const canvas = canvasRef.current;
    if (!image || !canvas) return;
    const scale = Math.min(1, 2000 / Math.max(image.naturalWidth, image.naturalHeight));
    canvas.width = Math.round(image.naturalWidth * scale);
    canvas.height = Math.round(image.naturalHeight * scale);
    canvas.getContext("2d")?.drawImage(image, 0, 0, canvas.width, canvas.height);
  }, []);

  const releaseCameraStream = useCallback(() => {
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    cameraStreamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const stopCamera = useCallback(() => {
    releaseCameraStream();
    setCameraOpen(false);
    setCameraStarting(false);
    setCameraError("");
    setCameras([]);
    setSelectedCameraId("");
    setCameraResolution("");
  }, [releaseCameraStream]);

  useEffect(() => {
    let cancelled = false;
    const detectCamera = async () => {
      if (!navigator.mediaDevices?.getUserMedia) return;
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameraKnown = devices.some((device) => device.kind === "videoinput");
        const cameraUnknown = devices.length === 0;
        if (!cancelled) setCameraAvailable(cameraKnown || cameraUnknown);
      } catch {
        if (!cancelled) setCameraAvailable(true);
      }
    };
    void detectCamera();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    const stream = cameraStreamRef.current;
    if (!cameraOpen || cameraStarting || !video || !stream) return;
    video.srcObject = stream;
    void video.play().catch(() => {
      setCameraError("摄像头画面无法播放，请检查浏览器权限或改用选择图片");
    });
  }, [cameraOpen, cameraStarting]);

  useEffect(() => {
    return () => {
      cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
      void workerRef.current?.terminate();
    };
  }, []);

  useEffect(() => {
    if (!fileName) return;
    const frame = window.requestAnimationFrame(renderImage);
    return () => window.cancelAnimationFrame(frame);
  }, [fileName, renderImage]);

  const loadImageFile = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      imageRef.current = image;
      setFileName(file.name);
      setDigits("");
      setRawText("");
      setConfidence(null);
      setCorrected(false);
      setCardDetected(null);
      setStatus("照片已就绪，将自动检测卡片、角度和 ACCESS CODE");
      URL.revokeObjectURL(url);
      requestAnimationFrame(renderImage);
    };
    image.onerror = () => {
      setStatus("无法读取这张图片，请换一张 JPEG、PNG 或 HEIC 照片");
      URL.revokeObjectURL(url);
    };
    image.src = url;
  }, [renderImage]);

  const selectFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    loadImageFile(file);
    event.target.value = "";
  };

  const startCamera = useCallback(async (deviceId?: string) => {
    setCameraStarting(true);
    setCameraError("");
    releaseCameraStream();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          ...(deviceId
            ? { deviceId: { exact: deviceId } }
            : { facingMode: { ideal: "environment" } }),
          width: { ideal: 4096 },
          height: { ideal: 3072 },
        },
        audio: false,
      });
      cameraStreamRef.current = stream;
      const track = stream.getVideoTracks()[0];
      const settings = track?.getSettings();
      let videoInputs: MediaDeviceInfo[] = [];
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        videoInputs = devices.filter((device) => device.kind === "videoinput");
      } catch {
        // Capturing can still work even when a browser withholds the device list.
      }
      setCameras(videoInputs);
      setSelectedCameraId(settings?.deviceId || deviceId || videoInputs[0]?.deviceId || "");
      setCameraResolution(
        settings?.width && settings.height ? `${settings.width} × ${settings.height}` : "",
      );
      setCameraStarting(false);
    } catch (error) {
      console.error(error);
      setCameraStarting(false);
      setCameraError("无法使用摄像头，请允许摄像头权限或改用选择图片");
    }
  }, [releaseCameraStream]);

  const openCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) return;
    setCameraOpen(true);
    await startCamera();
  };

  const changeCamera = async (event: ChangeEvent<HTMLSelectElement>) => {
    const deviceId = event.target.value;
    setSelectedCameraId(deviceId);
    await startCamera(deviceId);
  };

  const capturePhoto = () => {
    const video = videoRef.current;
    if (!video?.videoWidth || !video.videoHeight) {
      setCameraError("摄像头仍在准备中，请稍等后重试");
      return;
    }
    const guide = cameraGuideRef.current;
    const videoRect = video.getBoundingClientRect();
    const guideRect = guide?.getBoundingClientRect();
    const viewportWidth = videoRect.width;
    const viewportHeight = videoRect.height;
    if (!guideRect || !viewportWidth || !viewportHeight) {
      setCameraError("无法确定取景框位置，请旋转屏幕后重试");
      return;
    }

    // First reproduce the visible object-fit: cover preview using the whole
    // video frame. Cropping a canvas is more reliable on mobile Safari than
    // passing a source rectangle directly to drawImage(video, ...).
    const coverScale = Math.max(
      viewportWidth / video.videoWidth,
      viewportHeight / video.videoHeight,
    );
    const displayedWidth = video.videoWidth * coverScale;
    const displayedHeight = video.videoHeight * coverScale;
    const hiddenX = (displayedWidth - viewportWidth) / 2;
    const hiddenY = (displayedHeight - viewportHeight) / 2;
    const margin = viewportWidth * CAMERA_CROP_MARGIN;
    const visibleX = clamp(guideRect.left - videoRect.left - margin, 0, viewportWidth);
    const visibleY = clamp(guideRect.top - videoRect.top - margin, 0, viewportHeight);
    const visibleRight = clamp(
      guideRect.right - videoRect.left + margin,
      visibleX,
      viewportWidth,
    );
    const visibleBottom = clamp(
      guideRect.bottom - videoRect.top + margin,
      visibleY,
      viewportHeight,
    );
    const visibleWidth = visibleRight - visibleX;
    const visibleHeight = visibleBottom - visibleY;
    const renderScale = Math.min(4, Math.max(1, 1 / coverScale));

    const previewCanvas = document.createElement("canvas");
    previewCanvas.width = Math.round(viewportWidth * renderScale);
    previewCanvas.height = Math.round(viewportHeight * renderScale);
    previewCanvas.getContext("2d")?.drawImage(
      video,
      -hiddenX * renderScale,
      -hiddenY * renderScale,
      displayedWidth * renderScale,
      displayedHeight * renderScale,
    );

    const captureCanvas = document.createElement("canvas");
    captureCanvas.width = Math.round(visibleWidth * renderScale);
    captureCanvas.height = Math.round(visibleHeight * renderScale);
    captureCanvas.getContext("2d")?.drawImage(
      previewCanvas,
      visibleX * renderScale,
      visibleY * renderScale,
      visibleWidth * renderScale,
      visibleHeight * renderScale,
      0,
      0,
      captureCanvas.width,
      captureCanvas.height,
    );
    captureCanvas.toBlob(
      (blob) => {
        if (!blob) {
          setCameraError("拍摄失败，请重试或改用选择图片");
          return;
        }
        const photo = new File([blob], `camera-cropped-${Date.now()}.jpg`, {
          type: "image/jpeg",
        });
        stopCamera();
        loadImageFile(photo);
      },
      "image/jpeg",
      0.94,
    );
  };

  const recognize = async () => {
    const previewCanvas = canvasRef.current;
    const sourceImage = imageRef.current;
    if (!previewCanvas || !sourceImage || busy) return;

    // Always rebuild the recognition input from the original image. The preview
    // may show a corrected card after a successful run, but a second click must
    // still process exactly the same source pixels as the first click.
    const sourceCanvas = document.createElement("canvas");
    const sourceScale = Math.min(
      1,
      2000 / Math.max(sourceImage.naturalWidth, sourceImage.naturalHeight),
    );
    sourceCanvas.width = Math.round(sourceImage.naturalWidth * sourceScale);
    sourceCanvas.height = Math.round(sourceImage.naturalHeight * sourceScale);
    sourceCanvas.getContext("2d")?.drawImage(
      sourceImage,
      0,
      0,
      sourceCanvas.width,
      sourceCanvas.height,
    );
    setBusy(true);
    setDigits("");
    setRawText("");
    setConfidence(null);
    setProgress(0.03);
    setCorrected(false);
    setCardDetected(null);

    try {
      setStatus("正在加载视觉检测组件…");
      const [cv, tesseract] = await Promise.all([loadOpenCv(), import("tesseract.js")]);
      setProgress(0.12);
      setStatus("正在检测卡片边缘并建立固定坐标系…");
      const detection = detectCard(sourceCanvas, cv);
      const normalizedCard = normalizeCardSize(
        detection?.canvas ?? wholeImageFallback(sourceCanvas),
      );
      setCardDetected(Boolean(detection));

      if (!workerRef.current) {
        setStatus("首次使用：正在加载本地数字识别模型…");
        workerRef.current = await tesseract.createWorker(
          "eng",
          tesseract.OEM.LSTM_ONLY,
          {
            workerPath:
              "https://cdn.jsdelivr.net/npm/tesseract.js@7.0.0/dist/worker.min.js",
            corePath: "https://cdn.jsdelivr.net/npm/tesseract.js-core@7.0.0",
            langPath:
              "https://cdn.jsdelivr.net/npm/@tesseract.js-data/eng/4.0.0_best_int",
            gzip: true,
            logger(message) {
              if (message.status === "recognizing text") {
                setProgress(0.25 + message.progress * 0.7);
              }
            },
          },
        );
        await workerRef.current.setParameters({
          tessedit_pageseg_mode: tesseract.PSM.RAW_LINE,
          tessedit_char_whitelist: "0123456789 ",
          preserve_interword_spaces: "1",
          user_defined_dpi: "300",
        });
      }

      let best: Candidate | null = null;
      let attempt = 0;
      let usedAccessAnchor = false;
      let usedFixedLayout = false;
      let successfulPreview: HTMLCanvasElement | null = null;
      let groupReviewSource: HTMLCanvasElement | null = null;
      let groupReviewPreview: HTMLCanvasElement | null = null;
      let groupReviewScore = Number.POSITIVE_INFINITY;

      const recognizeLine = async (
        prepared: HTMLCanvasElement,
        preview: HTMLCanvasElement,
        label: string,
      ) => {
        attempt += 1;
        setStatus(`${label}（第 ${attempt} 次）…`);
        const result = await workerRef.current!.recognize(prepared);
        const candidate: Candidate = {
          text: result.data.text.trim(),
          digits: result.data.text.replace(/\D/g, ""),
          confidence: result.data.confidence,
        };
        if (!best || scoreCandidate(candidate) < scoreCandidate(best)) best = candidate;
        if (
          candidate.digits.length >= 15 &&
          candidate.digits.length < 20 &&
          scoreCandidate(candidate) < groupReviewScore
        ) {
          groupReviewSource = prepared;
          groupReviewPreview = preview;
          groupReviewScore = scoreCandidate(candidate);
        }
        if (candidate.digits.length !== 20) return false;
        successfulPreview = preview;
        return true;
      };

      if (detection) {
        const orientations = [normalizedCard, rotateCanvas(normalizedCard, 180)];
        fixedLayout: for (const binary of [false, true]) {
          for (const orientation of orientations) {
            for (const crop of CODE_CROPS) {
              const prepared = prepareCrop(orientation, crop, binary);
              if (
                await recognizeLine(
                  prepared,
                  orientation,
                  binary
                    ? "正在复核固定数字区域的二值化结果"
                    : "正在读取卡片底部固定数字区域",
                )
              ) {
                usedFixedLayout = true;
                break fixedLayout;
              }
            }
          }
        }
      }

      let accessRegions: NonNullable<ReturnType<typeof detectAccessCodeRegion>>[] = [];
      if (!successfulPreview) {
        setStatus("固定布局未得到 20 位结果，正在使用 ACCESS CODE 方格结构兜底定位…");
        accessRegions = [
          detection ? detectAccessCodeRegion(normalizedCard, cv) : null,
          detectAccessCodeRegion(sourceCanvas, cv),
        ]
          .filter((region): region is NonNullable<typeof region> => Boolean(region))
          .sort((a, b) => b.alignedBoxes - a.alignedBoxes)
          .slice(0, 2);
        setCardDetected(Boolean(detection || accessRegions.length));

        anchorSearch: for (const accessRegion of accessRegions) {
          for (const orientation of [
            accessRegion.canvas,
            rotateCanvas(accessRegion.canvas, 180),
          ]) {
            const prepared = prepareCrop(orientation, ACCESS_NUMBER_CROP, false);
            if (
              await recognizeLine(
                prepared,
                orientation,
                "已匹配 ACCESS CODE 方格结构，正在读取上方数字",
              )
            ) {
              usedAccessAnchor = true;
              break anchorSearch;
            }
          }
        }
      }

      if (!successfulPreview && groupReviewSource) {
        setStatus("整行结果接近 20 位，正在按五组四位结构进行一次最终复核…");
        await workerRef.current!.setParameters({
          tessedit_pageseg_mode: tesseract.PSM.SINGLE_WORD,
        });
        try {
          const groupResults: Candidate[] = [];
          for (const group of splitIntoCodeGroups(groupReviewSource)) {
            const result = await workerRef.current!.recognize(group);
            groupResults.push({
              text: result.data.text.trim(),
              digits: result.data.text.replace(/\D/g, ""),
              confidence: result.data.confidence,
            });
          }
          const grouped: Candidate = {
            text: groupResults.map((group) => group.text).join(" "),
            digits: groupResults.map((group) => group.digits).join(""),
            confidence:
              groupResults.reduce((sum, group) => sum + group.confidence, 0) /
              groupResults.length,
          };
          if (!best || scoreCandidate(grouped) < scoreCandidate(best)) best = grouped;
          if (grouped.digits.length === 20) {
            successfulPreview = groupReviewPreview ?? normalizedCard;
          }
        } finally {
          await workerRef.current!.setParameters({
            tessedit_pageseg_mode: tesseract.PSM.RAW_LINE,
          });
        }
      }

      // recognizeLine mutates this closure variable, which TypeScript's
      // control-flow analysis cannot infer across awaited nested calls.
      const finalCandidate = best as Candidate | null;
      const found = finalCandidate?.digits ?? "";
      const validFound = found.length === 20;
      setDigits(validFound ? found : "");
      setRawText(finalCandidate?.text ?? "");
      setConfidence(finalCandidate?.confidence ?? null);
      if (successfulPreview) {
        previewCanvas.width = successfulPreview.width;
        previewCanvas.height = successfulPreview.height;
        previewCanvas.getContext("2d")?.drawImage(successfulPreview, 0, 0);
        setCorrected(true);
      }
      setProgress(1);
      if (validFound) {
        setStatus(
          usedAccessAnchor
            ? "已通过 ACCESS CODE 字符框定位并识别出 20 位数字，请核对后使用"
            : usedFixedLayout
              ? "已在固定卡片坐标中识别出 20 位数字，请核对后使用"
              : "已按五组四位结构复核出 20 位数字，请核对后使用",
        );
      } else if (!accessRegions.length && !detection) {
        setStatus("未找到完整卡片边缘或 ACCESS CODE 方格结构；请确保卡片四角都在画面内");
      } else if (found.length) {
        setStatus(`已完成多阶段复核，但最佳候选只有 ${found.length} 位，未作为有效结果输出`);
      } else {
        setStatus("卡片已校正，但没有可靠识别到 ACCESS CODE；请换一张更清晰的照片");
      }
    } catch (error) {
      console.error(error);
      setStatus("自动识别没有完成，请刷新页面后重试");
    } finally {
      setBusy(false);
    }
  };

  const copyResult = async () => {
    if (!digits) return;
    await navigator.clipboard.writeText(digits);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  };

  const hasImage = Boolean(fileName);
  const validLength = digits.length === 20;

  return (
    <main className="shell">
      <header className="topbar">
        <div className="header-copy">
          <p className="eyebrow">ON-DEVICE CARD VISION</p>
          <h1>自动识别和提取 ACCESS CODE</h1>
        </div>
      </header>

      <section className="workspace">
        <div className="scanner-card">
          <div className="section-heading">
            <div><span>01</span><h2>拍摄完整卡片</h2></div>
            {fileName && <p title={fileName}>{fileName}</p>}
          </div>

          {!hasImage ? (
            <div className="dropzone">
              <div className="camera-icon" aria-hidden="true"><b /></div>
              <strong>拍照或选择卡片照片</strong>
              <span>方向和拍摄角度不限，但请让卡片四角都在画面内</span>
              <div className="source-actions">
                <label className="source-button source-button-primary">
                  <input type="file" accept="image/*" onChange={selectFile} />
                  选择图片
                </label>
                {cameraAvailable && (
                  <button type="button" className="source-button source-button-secondary" onClick={openCamera}>
                    打开相机
                  </button>
                )}
              </div>
            </div>
          ) : (
            <>
              <div className={`image-stage ${busy ? "is-scanning" : ""}`}>
                <canvas ref={canvasRef} aria-label="待识别的卡片照片" />
                {busy && <div className="scan-line" aria-hidden="true" />}
                {corrected && <span className="corrected-badge">✓ 已自动校正透视</span>}
              </div>
              <div className="image-actions">
                <label className="text-button">
                  <input type="file" accept="image/*" onChange={selectFile} />
                  更换图片
                </label>
                {cameraAvailable && (
                  <button type="button" className="text-button" onClick={openCamera}>重新拍照</button>
                )}
                <span className="auto-note">无需旋转或框选，系统会自动尝试</span>
              </div>
            </>
          )}
        </div>

        <aside className="result-card">
          <div className="section-heading">
            <div><span>02</span><h2>自动识别结果</h2></div>
          </div>

          <div className={`status-panel ${validLength ? "is-valid" : ""}`}>
            <span className="status-dot" />
            <p>{status}</p>
          </div>

          {busy && (
            <div className="progress" aria-label={`识别进度 ${Math.round(progress * 100)}%`}>
              <i style={{ width: `${Math.max(6, progress * 100)}%` }} />
            </div>
          )}

          {cardDetected !== null && (
            <div className="analysis-steps">
              <span className={cardDetected ? "done" : "warning"}>{cardDetected ? "✓ 卡片已检测" : "! 整图回退"}</span>
              <span className={corrected ? "done" : ""}>{corrected ? "✓ 透视已校正" : "— 未校正"}</span>
              <span className="done">✓ 已尝试多方向</span>
            </div>
          )}

          <div className={`code-output ${digits ? "has-value" : ""}`}>
            <label>ACCESS CODE</label>
            <div>{digits ? formatCode(digits) : "•••• •••• •••• •••• ••••"}</div>
            <footer>
              <span>{confidence === null ? "等待识别" : `OCR 置信度 ${Math.round(confidence)}%`}</span>
              <span className={validLength ? "length-ok" : ""}>{digits.length}/20 位</span>
            </footer>
          </div>

          {rawText && <details><summary>查看 OCR 原始结果</summary><pre>{rawText}</pre></details>}

          <button type="button" className="primary-button" disabled={!hasImage || busy} onClick={recognize}>
            {busy ? "正在自动识别…" : "自动检测并识别"}
          </button>
          <button type="button" className="secondary-button" disabled={!validLength} onClick={copyResult}>
            {copied ? "已复制" : "复制纯数字结果"}
          </button>

          <div className="tips">
            <strong>拍摄要求</strong>
            <p>卡片可以倾斜或旋转；请确保四个角没有出画，数字清晰，并尽量避免强烈反光。</p>
          </div>
        </aside>
      </section>

      {cameraOpen && (
        <div className="camera-overlay" role="dialog" aria-modal="true" aria-labelledby="camera-title">
          <div className="camera-dialog">
            <header>
              <div>
                <span>实时拍摄</span>
                <h2 id="camera-title">将完整卡片放入画面</h2>
              </div>
              <button type="button" className="camera-close" onClick={stopCamera} aria-label="关闭相机">×</button>
            </header>
            {(cameras.length > 1 || cameraResolution) && (
              <div className="camera-toolbar">
                {cameras.length > 1 && (
                  <label>
                    <span>摄像头</span>
                    <select value={selectedCameraId} disabled={cameraStarting} onChange={changeCamera}>
                      {cameras.map((camera, index) => (
                        <option key={camera.deviceId} value={camera.deviceId}>
                          {camera.label || `摄像头 ${index + 1}`}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                {cameraResolution && <small>实际画面 {cameraResolution}</small>}
              </div>
            )}
            <div className="camera-viewport">
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                onLoadedMetadata={(event) => {
                  const currentVideo = event.currentTarget;
                  setCameraResolution(`${currentVideo.videoWidth} × ${currentVideo.videoHeight}`);
                }}
              />
              <div ref={cameraGuideRef} className="camera-guide" aria-hidden="true" />
              {cameraStarting && <p>正在启动摄像头…</p>}
              {cameraError && <p className="camera-error">{cameraError}</p>}
            </div>
            <footer>
              <button type="button" className="camera-cancel" onClick={stopCamera}>取消</button>
              <button type="button" className="camera-capture" disabled={cameraStarting || Boolean(cameraError)} onClick={capturePhoto}>
                拍摄照片
              </button>
            </footer>
          </div>
        </div>
      )}
    </main>
  );
}
