// 女王蜂さがし（Web 版）。カメラ映像を、開いた端末の GPU（WebGPU）で推論する。
// 映像・画像はどこにも送らない（通信はページとモデルを最初に読み込むときだけ）。
import * as ort from './vendor/ort/ort.webgpu.min.mjs';
import { INPUT_SIZE, letterbox, rgbaToCHW, decode, QueenJudge, SnapPolicy } from './detector.js';
import * as store from './store.js';

const MODEL_URL = './models/queen.onnx';
const MAX_DIGITAL_ZOOM = 4;

// ---------------------------------------------------------------- 設定（端末に保存）
const DEFAULTS = {
  conf: 0.15, // 確信度のしきい値。低めにして見逃しを減らす（誤検出は「発見」の判定と目視で見分ける）
  window: 20, // 判定に使う直近フレーム数
  minHits: 1, // 「発見」とみなす検出フレーム数
  cooldownSec: 9, // 自動保存の間隔
  autoSave: false,
  sound: true,
  vibrate: false,
  maxFps: 0, // 推論の上限（0 = 上限なし）。発熱・電池を抑えたいときに下げる
  backend: 'auto', // auto / webgpu / wasm
};
// 端末には「初期値から変えた項目」だけを保存する（初期値を変えたとき、触っていない項目は新しい初期値になるように）
const SETTINGS_KEY = 'bqd-settings-v2';
// 以前（全項目を保存していた頃）の初期値。それと同じ値は「変えていない」とみなして捨てる
const OLD_DEFAULTS = { conf: 0.25, window: 10, minHits: 4, cooldownSec: 10, autoSave: true, sound: true, vibrate: true, maxFps: 0, backend: 'auto' };
function loadSettings() {
  const saved = localStorage.getItem(SETTINGS_KEY);
  if (saved) return JSON.parse(saved);
  const old = JSON.parse(localStorage.getItem('bqd-settings') || '{}');
  localStorage.removeItem('bqd-settings');
  return Object.fromEntries(Object.entries(old).filter(([k, v]) => k in DEFAULTS && v !== OLD_DEFAULTS[k]));
}
const settings = { ...DEFAULTS, ...loadSettings() };
if (settings.minHits > settings.window) settings.minHits = settings.window;
const saveSettings = () =>
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(Object.fromEntries(Object.entries(settings).filter(([k, v]) => v !== DEFAULTS[k]))));
saveSettings();

// ---------------------------------------------------------------- 画面の部品
const $ = (id) => document.getElementById(id);
let video = $('video'); // 最初の画面へ戻るたびに作り直す（iPhone で 2 回目の映像が小さく表示されるのを防ぐ）
const overlay = $('overlay');
const octx = overlay.getContext('2d');
const prep = document.createElement('canvas');
prep.width = prep.height = INPUT_SIZE;
const pctx = prep.getContext('2d', { willReadFrequently: true });
const inputBuf = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

const judge = new QueenJudge(settings.window, settings.minHits);
const snap = new SnapPolicy(settings.cooldownSec * 1000);

let session = null;
let backendName = '';
let modelPromise = null;
let stream = null;
let track = null;
let running = false;
let loopGen = 0; // 推論ループの世代（止めてすぐ再開したときに古いループを確実に終わらせる）
let hwZoom = null; // カメラ側のズーム範囲 {min, max}（使えなければ null → 映像を切り出して拡大）
let zoom = 1;
let digitalZoom = 1;
let latest = { boxes: [] };
let fps = 0;
let ms = 0;
let lastEnd = 0;
let saving = false;
let wakeLock = null;
let audioCtx = null;

// ---------------------------------------------------------------- Service Worker（オフライン対応）
async function registerSW() {
  if (!('serviceWorker' in navigator)) return;
  try {
    await navigator.serviceWorker.register('./sw.js');
    // 初回は、Service Worker が動き出してからモデルを読む（読んだモデルを端末に保存させるため）
    if (!navigator.serviceWorker.controller) {
      await Promise.race([
        new Promise((r) => navigator.serviceWorker.addEventListener('controllerchange', r, { once: true })),
        new Promise((r) => setTimeout(r, 3000)),
      ]);
    }
  } catch (e) {
    console.warn('Service Worker を登録できませんでした', e);
  }
}

// ---------------------------------------------------------------- モデル
async function fetchModel(onProgress) {
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`モデルを取得できません（${res.status}）`);
  // GitHub Pages は gzip で送るので、Content-Length は圧縮後の大きさ（届く量より小さい）。
  // 進み具合の目安にだけ使い、受け取ったものは貯めてから最後につなげる
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const chunks = [];
  const reader = res.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(Math.min(got / total, 0.99));
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  return buf;
}

async function hasWebGPU() {
  try {
    return !!(navigator.gpu && (await navigator.gpu.requestAdapter()));
  } catch {
    return false;
  }
}

async function createSession(model) {
  ort.env.wasm.wasmPaths = new URL('./vendor/ort/', import.meta.url).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  let order = settings.backend === 'auto' ? ['webgpu', 'wasm'] : [settings.backend];
  if (order.includes('webgpu') && !(await hasWebGPU())) {
    order = order.filter((b) => b !== 'webgpu');
    if (!order.length) order = ['wasm'];
  }
  let lastErr = null;
  for (const ep of order) {
    try {
      const s = await ort.InferenceSession.create(model, {
        executionProviders: [ep],
        graphOptimizationLevel: 'all',
      });
      // 1 回空で動かしておく（初回はシェーダーの準備などで遅いため）
      const t = new ort.Tensor('float32', new Float32Array(inputBuf.length), [1, 3, INPUT_SIZE, INPUT_SIZE]);
      const r = await s.run({ [s.inputNames[0]]: t });
      r[s.outputNames[0]].dispose?.();
      return { s, ep };
    } catch (e) {
      console.warn(`${ep} で動かせませんでした`, e);
      lastErr = e;
    }
  }
  throw lastErr;
}

function loadModel() {
  const bar = $('progress');
  const label = $('loadLabel');
  modelPromise = (async () => {
    label.textContent = 'モデルを読み込み中…';
    const model = await fetchModel((p) => {
      bar.value = p;
      label.textContent = `モデルを読み込み中… ${Math.round(p * 100)}%`;
    });
    bar.value = 1;
    label.textContent = '推論の準備中…';
    const { s, ep } = await createSession(model);
    session = s;
    backendName = ep === 'webgpu' ? 'GPU (WebGPU)' : `CPU (WASM${ort.env.wasm.numThreads > 1 ? ` ×${ort.env.wasm.numThreads}` : ''})`;
    label.textContent = '準備完了';
    $('backend').textContent = backendName;
    if (ep !== 'webgpu') $('gpuWarn').hidden = false;
  })();
  modelPromise.catch((e) => {
    label.textContent = `モデルを読み込めませんでした: ${e.message || e}`;
    label.classList.add('error');
  });
  return modelPromise;
}

// ---------------------------------------------------------------- 映像
async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
  });
  track = stream.getVideoTracks()[0];
  video.srcObject = stream;
  video.loop = false;
  await video.play();
  const caps = track.getCapabilities?.() || {};
  hwZoom = caps.zoom && caps.zoom.max > caps.zoom.min ? { min: caps.zoom.min, max: caps.zoom.max } : null;
  $('torchBtn').hidden = !caps.torch;
  torchOn = false; // 新しいカメラではライトは消えている
  $('torchBtn').classList.remove('on');
  track.addEventListener('ended', () => running && toast('カメラが止まりました。画面をタップして再開してください'));
}

async function startFile(file) {
  stopStream();
  video.srcObject = null;
  video.src = URL.createObjectURL(file);
  video.loop = true;
  await video.play();
  hwZoom = null;
  $('torchBtn').hidden = true;
}

function stopStream() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  track = null;
}

/** 画面に見えている範囲（映像の画素座標）。推論もこの範囲だけで行う（ズームすると小さな女王蜂も見やすくなる）。 */
function viewRect() {
  const W = overlay.clientWidth;
  const H = overlay.clientHeight;
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const s = Math.max(W / vw, H / vh) * digitalZoom;
  const sw = Math.min(vw, W / s);
  const sh = Math.min(vh, H / s);
  return { sx: (vw - sw) / 2, sy: (vh - sh) / 2, sw, sh, s, W, H };
}

// ---------------------------------------------------------------- 推論ループ
async function inferOnce() {
  const v = viewRect();
  const lb = letterbox(v.sw, v.sh);
  pctx.fillStyle = 'rgb(114,114,114)';
  pctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  pctx.drawImage(video, v.sx, v.sy, v.sw, v.sh, lb.dx, lb.dy, lb.nw, lb.nh);
  rgbaToCHW(pctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data, INPUT_SIZE, inputBuf);
  const t0 = performance.now();
  const input = new ort.Tensor('float32', inputBuf, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const res = await session.run({ [session.inputNames[0]]: input });
  const out = res[session.outputNames[0]];
  const data = out.data;
  ms = ms ? ms * 0.8 + (performance.now() - t0) * 0.2 : performance.now() - t0;
  const boxes = decode(data, out.dims[2], lb, v.sw, v.sh, { conf: settings.conf }).map((b) => ({
    ...b,
    x1: b.x1 + v.sx,
    y1: b.y1 + v.sy,
    x2: b.x2 + v.sx,
    y2: b.y2 + v.sy,
  }));
  out.dispose?.();
  return boxes;
}

async function loop(gen) {
  while (running && gen === loopGen) {
    if (document.hidden || video.readyState < 2 || !video.videoWidth) {
      await sleep(100);
      continue;
    }
    const start = performance.now();
    let boxes;
    try {
      boxes = await inferOnce();
    } catch (e) {
      console.error(e);
      toast(`推論に失敗しました: ${e.message || e}`);
      await sleep(1000);
      continue;
    }
    latest = { boxes };
    onResult(boxes);
    drawOverlay();
    const now = performance.now();
    if (lastEnd) fps = fps ? fps * 0.8 + (1000 / (now - lastEnd)) * 0.2 : 1000 / (now - lastEnd);
    lastEnd = now;
    if (settings.maxFps > 0) {
      const wait = 1000 / settings.maxFps - (performance.now() - start);
      if (wait > 0) await sleep(wait);
    } else {
      await new Promise((r) => requestAnimationFrame(r)); // 画面の更新に譲る
    }
  }
}

const sleep = (t) => new Promise((r) => setTimeout(r, t));

function onResult(boxes) {
  const wasFound = judge.found;
  const found = judge.add(boxes.length > 0);
  const maxConf = boxes.reduce((m, b) => Math.max(m, b.conf), 0);
  renderStatus(found, maxConf);
  if (found && !wasFound) alertFound();
  const now = Date.now();
  if (found && settings.autoSave && !saving && snap.shouldSnap(now)) {
    snap.markSaved(now);
    saveSnapshot(true);
  }
}

function renderStatus(found, maxConf) {
  const bar = $('status');
  bar.classList.toggle('found', found);
  $('statusText').textContent = found ? '女王蜂発見！' : '巣板を映してください';
  $('confText').textContent = maxConf > 0 ? `確信度 ${Math.round(maxConf * 100)}%` : '';
  $('fpsText').textContent = `${fps.toFixed(1)} fps・${Math.round(ms)} ms  ${judge.hitCount}/${judge.window}`;
}

// ---------------------------------------------------------------- 枠の描画
function resizeOverlay() {
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.round(overlay.clientWidth * dpr);
  overlay.height = Math.round(overlay.clientHeight * dpr);
  drawOverlay();
}

function drawBoxes(ctx, boxes, map, scale) {
  ctx.lineWidth = 3 * scale;
  ctx.font = `bold ${16 * scale}px system-ui, sans-serif`;
  ctx.textBaseline = 'bottom';
  for (const b of boxes) {
    const [x1, y1] = map(b.x1, b.y1);
    const [x2, y2] = map(b.x2, b.y2);
    ctx.strokeStyle = '#ffc107';
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `女王蜂 ${Math.round(b.conf * 100)}%`;
    const tw = ctx.measureText(label).width + 8 * scale;
    const th = 22 * scale;
    const ty = y1 - th < 0 ? y2 + th : y1;
    ctx.fillStyle = '#ffc107';
    ctx.fillRect(x1 - ctx.lineWidth / 2, ty - th, tw, th);
    ctx.fillStyle = '#000';
    ctx.fillText(label, x1 + 4 * scale - ctx.lineWidth / 2, ty - 3 * scale);
  }
}

function drawOverlay() {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!video.videoWidth) return;
  const v = viewRect();
  const dpr = overlay.width / (overlay.clientWidth || 1);
  drawBoxes(octx, latest.boxes, (x, y) => [(x - v.sx) * v.s * dpr, (y - v.sy) * v.s * dpr], dpr);
}

// ---------------------------------------------------------------- 発見の通知
function alertFound() {
  if (settings.vibrate && navigator.vibrate) navigator.vibrate([200, 100, 200]); // iPhone の Safari では使えない
  if (settings.sound && audioCtx) {
    const t = audioCtx.currentTime;
    for (const [i, f] of [880, 1175].entries()) {
      const o = audioCtx.createOscillator();
      const g = audioCtx.createGain();
      o.frequency.value = f;
      g.gain.setValueAtTime(0.0001, t + i * 0.18);
      g.gain.exponentialRampToValueAtTime(0.4, t + i * 0.18 + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + i * 0.18 + 0.16);
      o.connect(g).connect(audioCtx.destination);
      o.start(t + i * 0.18);
      o.stop(t + i * 0.18 + 0.17);
    }
  }
  const flash = $('flash');
  flash.classList.remove('on');
  void flash.offsetWidth;
  flash.classList.add('on');
}

// ---------------------------------------------------------------- 保存（記録）
async function makeSnapshot(boxes) {
  const v = viewRect();
  const k = Math.min(1, 1920 / Math.max(v.sw, v.sh));
  const c = document.createElement('canvas');
  c.width = Math.round(v.sw * k);
  c.height = Math.round(v.sh * k);
  const ctx = c.getContext('2d');
  ctx.drawImage(video, v.sx, v.sy, v.sw, v.sh, 0, 0, c.width, c.height);
  const scale = Math.max(1, c.width / 800);
  drawBoxes(ctx, boxes, (x, y) => [(x - v.sx) * k, (y - v.sy) * k], scale);
  ctx.font = `${14 * scale}px system-ui, sans-serif`;
  ctx.fillStyle = 'rgba(0,0,0,.5)';
  const stamp = new Date().toLocaleString('ja-JP');
  ctx.fillRect(0, c.height - 24 * scale, ctx.measureText(stamp).width + 12 * scale, 24 * scale);
  ctx.fillStyle = '#fff';
  ctx.textBaseline = 'bottom';
  ctx.fillText(stamp, 6 * scale, c.height - 5 * scale);
  return new Promise((r) => c.toBlob(r, 'image/jpeg', 0.9));
}

async function saveSnapshot(auto) {
  if (saving || !video.videoWidth) return;
  saving = true;
  try {
    const boxes = latest.boxes;
    const blob = await makeSnapshot(boxes);
    await store.add({ time: Date.now(), blob, auto, conf: boxes.reduce((m, b) => Math.max(m, b.conf), 0) });
    await updateRecordCount();
    toast(auto ? '女王蜂を見つけたので記録しました' : '記録しました（「記録」から写真に保存できます）');
  } catch (e) {
    toast(`保存に失敗しました: ${e.message || e}`);
  } finally {
    saving = false;
  }
}

async function updateRecordCount() {
  const n = await store.count();
  $('recordLabel').textContent = n ? `記録 ${n}` : '記録';
}

const fileName = (rec) => {
  const t = new Date(rec.time);
  const p = (v) => String(v).padStart(2, '0');
  return `queen_${t.getFullYear()}${p(t.getMonth() + 1)}${p(t.getDate())}_${p(t.getHours())}${p(t.getMinutes())}${p(t.getSeconds())}${rec.auto ? '_auto' : ''}.jpg`;
};

/** 共有シート（iPhone では「画像を保存」で写真へ）。使えなければダウンロード。 */
async function shareRecords(recs) {
  const files = recs.map((r) => new File([r.blob], fileName(r), { type: 'image/jpeg' }));
  if (navigator.canShare?.({ files })) {
    try {
      await navigator.share({ files });
    } catch (e) {
      if (e.name !== 'AbortError') toast(`共有できませんでした: ${e.message}`);
    }
    return;
  }
  for (const f of files) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(f);
    a.download = f.name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  }
}

let recordUrls = [];
async function openRecords() {
  const recs = (await store.all()).reverse();
  recordUrls.forEach((u) => URL.revokeObjectURL(u));
  recordUrls = [];
  const grid = $('recordGrid');
  grid.replaceChildren();
  $('recordEmpty').hidden = recs.length > 0;
  $('shareAll').disabled = $('clearAll').disabled = recs.length === 0;
  for (const rec of recs) {
    const url = URL.createObjectURL(rec.blob);
    recordUrls.push(url);
    const fig = document.createElement('button');
    fig.className = 'thumb';
    fig.innerHTML = `<img alt=""><span></span>`;
    fig.querySelector('img').src = url;
    fig.querySelector('span').textContent =
      `${new Date(rec.time).toLocaleString('ja-JP', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })}${rec.auto ? ' 自動' : ''}`;
    fig.onclick = () => openViewer(rec, url);
    grid.append(fig);
  }
  $('shareAll').onclick = () => shareRecords(recs);
  $('clearAll').onclick = async () => {
    if (!confirm(`記録 ${recs.length} 枚をすべて削除しますか？（写真に保存したものは消えません）`)) return;
    await store.clear();
    await updateRecordCount();
    openRecords();
  };
  $('records').showModal();
}

function openViewer(rec, url) {
  $('viewerImg').src = url;
  $('viewerShare').onclick = () => shareRecords([rec]);
  $('viewerDelete').onclick = async () => {
    await store.remove(rec.id);
    await updateRecordCount();
    $('viewer').close();
    openRecords();
  };
  $('viewer').showModal();
}

// ---------------------------------------------------------------- ズーム・ライト
let zoomBusy = false;
async function setZoom(z) {
  if (hwZoom) {
    zoom = Math.min(Math.max(z, hwZoom.min), hwZoom.max);
    digitalZoom = 1;
    if (!zoomBusy) {
      zoomBusy = true;
      try {
        await track.applyConstraints({ advanced: [{ zoom }] });
      } catch {
        hwZoom = null; // カメラ側のズームが効かなければ切り出しに切り替える
      } finally {
        zoomBusy = false;
      }
    }
  }
  if (!hwZoom) {
    zoom = Math.min(Math.max(z, 1), MAX_DIGITAL_ZOOM);
    digitalZoom = zoom;
  }
  video.style.transform = `scale(${digitalZoom})`;
  $('zoomBtn').textContent = `${zoom.toFixed(1)}x`;
  drawOverlay();
}

function setupPinch() {
  const stage = $('stage');
  let d0 = 0;
  let z0 = 1;
  const dist = (t) => Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
  stage.addEventListener('touchstart', (e) => {
    if (e.touches.length === 2) {
      d0 = dist(e.touches);
      z0 = zoom;
    }
  });
  stage.addEventListener(
    'touchmove',
    (e) => {
      if (e.touches.length === 2 && d0) {
        e.preventDefault();
        setZoom((z0 * dist(e.touches)) / d0);
      }
    },
    { passive: false },
  );
  stage.addEventListener('touchend', () => (d0 = 0));
}

let torchOn = false;
async function toggleTorch() {
  try {
    await track.applyConstraints({ advanced: [{ torch: !torchOn }] });
    torchOn = !torchOn;
    $('torchBtn').classList.toggle('on', torchOn);
  } catch (e) {
    toast('ライトを使えませんでした');
  }
}

// ---------------------------------------------------------------- 画面を消さない
async function keepAwake() {
  try {
    if ('wakeLock' in navigator && !document.hidden && running) wakeLock = await navigator.wakeLock.request('screen');
  } catch {
    /* 省電力モードなどでは使えない */
  }
}

// 画面から離れたらカメラと音を止める（Android ではバックグラウンドでもカメラが動き続け、電池と発熱の元になる）
let cameraPaused = false;
document.addEventListener('visibilitychange', async () => {
  if (!running) return;
  if (document.hidden) {
    audioCtx?.suspend?.();
    if (stream) {
      stopStream();
      cameraPaused = true;
    } else {
      video.pause(); // 動画ファイルの再生も止める
    }
    return;
  }
  // アプリから戻ったときに発見状態を引きずらない
  judge.reset();
  latest = { boxes: [] };
  drawOverlay();
  keepAwake();
  audioCtx?.resume?.();
  if (!cameraPaused && !stream && video.src) video.play().catch(() => {});
  else restartCameraIfEnded();
});

async function resumeCamera() {
  try {
    await startCamera();
    if (document.hidden || !running) {
      // 取り直している間にまた画面を離れた（または最初の画面に戻った）
      stopStream();
      cameraPaused = running;
      return;
    }
    await setZoom(zoom);
  } catch (e) {
    cameraPaused = true; // 画面をタップすればもう一度試す
    toast(`カメラを再開できませんでした（タップで再試行）: ${e.message}`);
  }
}

/** 自分で止めたカメラ、または iPhone がバックグラウンドで止めたカメラを取り直す。 */
async function restartCameraIfEnded() {
  if (cameraPaused) cameraPaused = false;
  else if (!stream || (track && track.readyState !== 'ended')) return;
  await resumeCamera();
}

// ---------------------------------------------------------------- 設定画面
const SETTING_FIELDS = [
  ['conf', 'number'],
  ['window', 'int'],
  ['minHits', 'int'],
  ['cooldownSec', 'int'],
  ['maxFps', 'int'],
  ['autoSave', 'bool'],
  ['sound', 'bool'],
  ['vibrate', 'bool'],
  ['backend', 'string'],
];

function fillSettingsForm() {
  for (const [k, t] of SETTING_FIELDS) {
    const el = $(`set-${k}`);
    if (t === 'bool') el.checked = settings[k];
    else el.value = settings[k];
  }
  $('set-minHits').max = settings.window;
  updateSettingLabels();
}

function updateSettingLabels() {
  $('lbl-conf').textContent = Number(settings.conf).toFixed(2);
  $('lbl-judge').textContent = `直近 ${settings.window} フレーム中 ${settings.minHits} フレーム以上で検出`;
  $('lbl-cooldown').textContent = `${settings.cooldownSec} 秒`;
  $('lbl-maxFps').textContent = settings.maxFps ? `${settings.maxFps} fps まで` : '上限なし';
}

function setupSettings() {
  const prevBackend = settings.backend;
  for (const [k, t] of SETTING_FIELDS) {
    $(`set-${k}`).addEventListener('input', (e) => {
      const el = e.target;
      settings[k] = t === 'bool' ? el.checked : t === 'int' ? parseInt(el.value, 10) : t === 'number' ? parseFloat(el.value) : el.value;
      if (k === 'window') {
        $('set-minHits').max = settings.window;
        if (settings.minHits > settings.window) settings.minHits = settings.window;
        $('set-minHits').value = settings.minHits;
      }
      if (k === 'autoSave') $('autoBtn').classList.toggle('on', settings.autoSave);
      updateSettingLabels();
      saveSettings();
      judge.window = settings.window;
      judge.minHits = settings.minHits;
      snap.cooldownMs = settings.cooldownSec * 1000;
      if (k === 'backend') $('backendNote').hidden = settings.backend === prevBackend;
    });
  }
  $('resetSettings').onclick = () => {
    Object.assign(settings, { ...DEFAULTS, backend: settings.backend });
    saveSettings();
    fillSettingsForm();
    $('autoBtn').classList.toggle('on', settings.autoSave);
  };
  $('settings').addEventListener('close', () => judge.reset());
}

// ---------------------------------------------------------------- その他
let toastTimer = 0;
function toast(msg) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 2500);
}

async function begin(useFile) {
  // 音はユーザー操作の中で準備しておく必要がある（iPhone）
  try {
    audioCtx ||= new (window.AudioContext || window.webkitAudioContext)();
    audioCtx.resume();
  } catch {
    /* 音なしで続ける */
  }
  const errEl = $('startError');
  errEl.hidden = true;
  try {
    if (useFile) await startFile(useFile);
    else await startCamera();
  } catch (e) {
    errEl.textContent =
      e.name === 'NotAllowedError'
        ? 'カメラの使用が許可されていません。ブラウザ（iPhone は「設定」→「Safari」→「カメラ」）で許可してください。'
        : `カメラを開けませんでした: ${e.message || e}`;
    errEl.hidden = false;
    return;
  }
  $('start').hidden = true;
  $('main').hidden = false;
  resizeOverlay();
  await setZoom(1);
  $('statusText').textContent = 'モデルを準備中…';
  try {
    await modelPromise;
  } catch {
    $('statusText').textContent = 'モデルを読み込めませんでした（再読み込みしてください）';
    return;
  }
  judge.reset();
  fps = ms = lastEnd = 0;
  running = true;
  keepAwake();
  loop(++loopGen);
}

function backToStart() {
  running = false;
  loopGen++;
  stopStream();
  video.pause();
  if (video.src.startsWith('blob:')) URL.revokeObjectURL(video.src);
  video.removeAttribute('src');
  video.srcObject = null;
  const fresh = video.cloneNode(false);
  fresh.style.transform = '';
  video.replaceWith(fresh);
  video = fresh;
  zoom = digitalZoom = 1;
  latest = { boxes: [] };
  $('settings').close();
  $('main').hidden = true;
  $('start').hidden = false;
  wakeLock?.release?.();
  cameraPaused = false;
  audioCtx?.suspend?.(); // 開始ボタンを押したときにまた resume する
}

async function init() {
  const standalone = matchMedia('(display-mode: standalone)').matches || navigator.standalone;
  const ios = /iPhone|iPad|iPod/.test(navigator.userAgent);
  $('installHint').hidden = !ios || standalone;
  if (!(await hasWebGPU())) $('gpuWarn').hidden = false;

  $('cameraBtn').onclick = () => begin(null);
  $('fileInput').onchange = (e) => {
    const f = e.target.files[0];
    e.target.value = ''; // 同じファイルをもう一度選んでも反応するように
    if (f) begin(f);
  };
  $('shutterBtn').onclick = () => saveSnapshot(false);
  $('zoomWrap').onclick = () => setZoom(zoom < 1.5 ? 2 : zoom < 2.5 ? 3 : 1);
  $('torchBtn').onclick = toggleTorch;
  $('autoBtn').classList.toggle('on', settings.autoSave);
  $('autoBtn').onclick = () => {
    settings.autoSave = !settings.autoSave;
    saveSettings();
    $('autoBtn').classList.toggle('on', settings.autoSave);
    toast(settings.autoSave ? '発見時に自動で記録します' : '自動記録をオフにしました');
  };
  $('recordBtn').onclick = openRecords;
  $('settingsBtn').onclick = () => {
    fillSettingsForm();
    $('backendNote').hidden = true;
    $('settings').showModal();
  };
  $('backToStart').onclick = backToStart;
  for (const d of document.querySelectorAll('dialog')) {
    d.addEventListener('click', (e) => e.target === d && d.close()); // 外側をタップで閉じる
  }
  setupSettings();
  setupPinch();
  $('stage').addEventListener('click', () => running && restartCameraIfEnded()); // 止まったカメラはタップで再開
  new ResizeObserver(resizeOverlay).observe(overlay);
  updateRecordCount();

  await registerSW();
  loadModel();
}

init();

// テスト・計測用（コンソールから参照）
window.bqd = { settings, get backend() { return backendName; }, get fps() { return fps; }, get ms() { return ms; }, get boxes() { return latest.boxes; } };
