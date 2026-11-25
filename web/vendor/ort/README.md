# ONNX Runtime Web 1.31.1

https://github.com/microsoft/onnxruntime （MIT License, Copyright (c) Microsoft Corporation）

npm パッケージ `onnxruntime-web@1.31.1` の `dist/` から、WebGPU 版で使う 3 ファイルだけを置いています。

- `ort.webgpu.min.mjs`
- `ort-wasm-simd-threaded.asyncify.mjs`
- `ort-wasm-simd-threaded.asyncify.wasm`

更新するとき:

```bash
npm pack onnxruntime-web@<版> && tar xzf onnxruntime-web-<版>.tgz
cp package/dist/{ort.webgpu.min.mjs,ort-wasm-simd-threaded.asyncify.mjs,ort-wasm-simd-threaded.asyncify.wasm} web/vendor/ort/
```

更新したら `web/sw.js` の `VERSION` を上げてください（端末に保存された古いものが入れ替わります）。
