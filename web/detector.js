// 女王蜂検出の前処理・後処理と「発見」の判定。
// ブラウザにも Node（テスト）にも依存しない純粋な処理だけを置く。

/** モデルの入力サイズ（学習・書き出しと同じ 640）。 */
export const INPUT_SIZE = 640;

/**
 * Ultralytics の letterbox と同じ配置（縦横比を保って縮小し、余白をグレー 114 で埋めて中央に置く）。
 * @param {number} sw 元画像（切り出し範囲）の幅
 * @param {number} sh 元画像（切り出し範囲）の高さ
 */
export function letterbox(sw, sh, size = INPUT_SIZE) {
  const r = Math.min(size / sw, size / sh);
  const nw = Math.round(sw * r);
  const nh = Math.round(sh * r);
  const dx = Math.round((size - nw) / 2 - 0.1);
  const dy = Math.round((size - nh) / 2 - 0.1);
  return { r, nw, nh, dx, dy, size };
}

/**
 * RGBA（canvas の getImageData）→ モデル入力（1×3×H×W、RGB、0〜1 の float32）。
 * @param {Uint8ClampedArray|Uint8Array} rgba
 * @param {Float32Array} [out] 使い回す出力バッファ
 */
export function rgbaToCHW(rgba, size = INPUT_SIZE, out = new Float32Array(3 * size * size)) {
  const n = size * size;
  const k = 1 / 255;
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    out[i] = rgba[j] * k;
    out[i + n] = rgba[j + 1] * k;
    out[i + 2 * n] = rgba[j + 2] * k;
  }
  return out;
}

/** 2 つの枠 [x1, y1, x2, y2] の IoU。 */
export function iou(a, b) {
  const w = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const h = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}

/** 重なった枠を 1 つにまとめる（確信度の高い順に残す）。 */
export function nms(boxes, iouThres = 0.5, maxDet = 100) {
  const sorted = [...boxes].sort((a, b) => b.conf - a.conf);
  const keep = [];
  for (const b of sorted) {
    if (keep.length >= maxDet) break;
    if (keep.every((k) => iou(k, b) <= iouThres)) keep.push(b);
  }
  return keep;
}

/**
 * モデルの生の出力（1×5×N: cx, cy, w, h, 確信度。クラスは女王蜂の 1 つ）を枠に変換する。
 * 座標は letterbox を戻して元画像（切り出し範囲）の画素で返す。
 * @param {Float32Array} out
 * @param {number} numAnchors N（640 入力なら 8400）
 * @param {{r:number, dx:number, dy:number}} lb letterbox() の戻り値
 * @param {number} sw 元画像の幅（枠を画像内に収める）
 * @param {number} sh 元画像の高さ
 */
export function decode(out, numAnchors, lb, sw, sh, { conf = 0.25, iouThres = 0.5 } = {}) {
  const n = numAnchors;
  const cand = [];
  for (let i = 0; i < n; i++) {
    const s = out[4 * n + i];
    if (s < conf) continue;
    const cx = out[i];
    const cy = out[n + i];
    const w = out[2 * n + i];
    const h = out[3 * n + i];
    const clampX = (v) => Math.min(Math.max(v, 0), sw);
    const clampY = (v) => Math.min(Math.max(v, 0), sh);
    cand.push({
      x1: clampX((cx - w / 2 - lb.dx) / lb.r),
      y1: clampY((cy - h / 2 - lb.dy) / lb.r),
      x2: clampX((cx + w / 2 - lb.dx) / lb.r),
      y2: clampY((cy + h / 2 - lb.dy) / lb.r),
      conf: s,
    });
  }
  return nms(cand, iouThres);
}

/**
 * 「女王蜂発見」の判定（スマホアプリの queen_judge.dart・PC 版 detect_stream.py と同じ考え方）。
 * 1 フレームの誤検出で騒がないよう、直近 window フレームのうち minHits フレーム以上で検出されたら「発見」。
 */
export class QueenJudge {
  constructor(window = 10, minHits = 4) {
    this.window = window;
    this.minHits = minHits;
    this.hits = [];
  }

  /** 1 フレーム分の結果（女王蜂が 1 匹以上検出されたか）を加え、発見状態を返す。 */
  add(hit) {
    this.hits.push(hit);
    while (this.hits.length > this.window) this.hits.shift();
    return this.found;
  }

  get hitCount() {
    return this.hits.filter(Boolean).length;
  }

  get found() {
    return this.hitCount >= this.minHits;
  }

  reset() {
    this.hits = [];
  }
}

/** 自動保存の間隔制御（「発見」状態が続いても cooldownMs ごとに 1 枚だけ保存する）。 */
export class SnapPolicy {
  constructor(cooldownMs = 10000) {
    this.cooldownMs = cooldownMs;
    this.last = null;
  }

  shouldSnap(now) {
    return this.last === null || now - this.last >= this.cooldownMs;
  }

  markSaved(now) {
    this.last = now;
  }
}
