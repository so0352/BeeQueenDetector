// オフライン対応（巣箱の前の圏外でも起動できるように、ページ・ONNX Runtime・モデルを端末に保存する）。
//
// モデルや ONNX Runtime を差し替えたときは VERSION を上げる（古い保存分が消え、読み込み直される）。
const VERSION = 'v1';
const CACHE = `bqd-${VERSION}`;

// 小さいファイルは最初に保存しておく。大きいもの（モデル・wasm）はページが読み込んだときに保存する
const SHELL = [
  './',
  './index.html',
  './style.css',
  './app.js',
  './detector.js',
  './store.js',
  './manifest.webmanifest',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './icons/apple-touch-icon.png',
  './vendor/ort/ort.webgpu.min.mjs',
];

// 大きく、VERSION を上げない限り変わらないもの → 端末の保存分を優先（通信しない）
const isHeavy = (url) => /\.(onnx|wasm)$/.test(url.pathname) || url.pathname.includes('/vendor/');

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('bqd-') && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

// WASM（CPU 推論）を複数スレッドで動かすには cross-origin isolation が要る。
// GitHub Pages はヘッダーを設定できないので、ここで付ける（WebGPU の速度には関係しない）
function isolate(res) {
  if (!res || res.status === 0 || res.type === 'opaque') return res;
  const h = new Headers(res.headers);
  h.set('Cross-Origin-Embedder-Policy', 'require-corp');
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  h.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
}

self.addEventListener('fetch', (e) => {
  const req = e.request;
  const url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== self.location.origin) return;

  e.respondWith(
    (async () => {
      const cache = await caches.open(CACHE);
      if (isHeavy(url)) {
        const hit = await cache.match(req);
        if (hit) return isolate(hit);
        const res = await fetch(req);
        if (res.ok) await cache.put(req, res.clone());
        return isolate(res);
      }
      // ページ・JS・CSS は新しいものを優先し、圏外なら保存分を使う
      try {
        const res = await fetch(req);
        if (res.ok) await cache.put(req, res.clone());
        return isolate(res);
      } catch (err) {
        const hit = await cache.match(req, { ignoreSearch: true });
        if (hit) return isolate(hit);
        throw err;
      }
    })(),
  );
});
