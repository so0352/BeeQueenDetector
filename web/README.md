# 女王蜂さがし（Web 版）

ブラウザで動く「女王蜂さがし」です。URL を開くだけで使え、iPhone でも App Store・Apple Developer Program なしで配れます。

- **推論は、ページを開いた端末の GPU で行います**（ONNX Runtime Web の WebGPU）。GPU が使えない端末では CPU（WASM）に自動で切り替わります。
- 映像・画像はどこにも送りません。通信はページとモデル（約 38 MB）を最初に読み込むときだけです。
- 一度開けば、圏外（巣箱の前）でも起動できます（Service Worker で端末に保存）。
- ビルド作業はありません（素の HTML / JS）。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `index.html` / `style.css` | 画面 |
| `help.html` | 使う人向けの使い方ページ（最初の画面と設定からリンク） |
| `app.js` | カメラ・推論ループ・枠の描画・通知・記録・設定 |
| `detector.js` | 前処理（letterbox）・後処理（NMS）・「発見」の判定。ブラウザに依存しない |
| `store.js` | 記録した画像の保存（端末内の IndexedDB） |
| `sw.js` | オフライン対応。CPU 推論を複数スレッドにするためのヘッダーも付ける |
| `manifest.webmanifest` / `icons/` | ホーム画面に追加したときの名前・アイコン |
| `models/queen.onnx` | 女王蜂モデル（`export_mobile.py --formats onnx` で書き出し） |
| `vendor/ort/` | ONNX Runtime Web 1.31.1（WebGPU 版）。更新方法は `vendor/ort/README.md` |

## 手元で動かす

```bash
cd web && python3 -m http.server 8765
```

パソコンのブラウザで `http://localhost:8765/` を開きます（カメラは `localhost` なら HTTP でも使えます）。
カメラがなければ「動画ファイルで試す」で `data/test_videos/*.webm` などを選んでください。

**スマホでは HTTPS が必要です。** `http://<Mac の IP>:8765/` ではカメラが使えないので、
スマホで試すときは GitHub Pages（下記）に公開してから開いてください。

## 公開（GitHub Pages）

`.github/workflows/pages.yml` が、`main` に `web/` の変更をプッシュすると `web/` を GitHub Pages に公開します。

1. GitHub のリポジトリを公開（Public）にする（無料プランの GitHub Pages は公開リポジトリのみ。AGPL-3.0 の条件もこれで満たせる）。
2. リポジトリの Settings → Pages → Source を **GitHub Actions** にする。
3. `main` にプッシュ（または Actions タブから「Deploy web to GitHub Pages」を手動実行）。
4. `https://<ユーザー名>.github.io/BeeQueenDetector/` で開ける。

### モデル・ONNX Runtime を差し替えたとき

`sw.js` の `VERSION` を上げてからプッシュしてください。上げないと、使う人の端末に保存された古いモデルが使われ続けます
（ページ・JS・CSS は通信できるときは自動で新しいものになります）。

```bash
.venv-onnx/bin/python export_mobile.py --formats onnx   # → web/models/queen.onnx
```

書き出し用の環境（本体の `.venv` とは分ける。`cv2.imshow` を壊さないため）:

```bash
uv venv .venv-onnx --python 3.12
uv pip install --python .venv-onnx/bin/python ultralytics onnx onnxslim onnxruntime pyyaml
```

## 使う人への案内（iPhone）

サイト内の「使い方」ページ（`help.html`）に、使う人向けの説明をまとめています。要点:

1. Safari で URL を開く → 「カメラで開始」→ カメラを許可。
2. 共有ボタン →「ホーム画面に追加」すると、アプリのように全画面で使える。
3. 女王蜂を見つけると、画面上部が赤くなり、画面の縁が点滅し、音が鳴る（マナーモードでは鳴らないことがある）。
4. シャッターまたは自動記録で、枠付きの画像が「記録」にたまる。写真に残すときは「記録」→「共有」→「画像を保存」。

## 確認済みのこと（2026-11-25、Mac の Chrome）

| 項目 | 結果 |
|---|---|
| ONNX の精度（test 880 枚、conf 0.25） | `.pt` と同じ（検出 91.7% / 誤検出 1.9%） |
| ブラウザの前処理・推論・NMS と Python（Ultralytics）の一致 | 30 枚中 30 枚で検出あり/なしが一致、先頭の枠の IoU 平均 0.993 |
| 速度（Apple Silicon の Mac、WebGPU） | 1 回約 20 ms、約 30 fps（画面の更新に合わせて頭打ち） |
| 速度（同、CPU / WASM 4 スレッド） | 1 回約 150 ms（1 スレッドだと約 400 ms。2 回目に開いたときから 4 スレッド） |
| オフライン | 通信を切って再読み込みしても起動・推論できる |

## 未確認（実機で確かめること）

- **iPhone の Safari での速度**（iOS 26 以降で WebGPU）。WEB_PLAN.md の基準では 8 fps 以上なら YOLO11s のままでよい。
  画面上部に fps と 1 回の推論時間（ms）が出るので、その数字を記録する。設定の「推論に使うもの」で GPU / CPU を比べられる。
- iPhone でのカメラのズーム・ライト（使えない場合、ズームは映像の切り出しに切り替わり、ライトのボタンは出ない）。
- ホーム画面から起動したときのカメラ・全画面表示、マナーモード時の音。
- 長く使ったときの発熱（熱くなるなら設定の「推論の回数」を 5 fps などに下げる）。

## Flutter 版との違い

| 機能 | Web 版 |
|---|---|
| 振動 | Android のみ（iPhone の Safari は非対応）。代わりに画面の点滅と音で知らせる |
| 写真への保存 | 自動では保存できない。いったん「記録」（端末内）にため、共有シートで写真に保存 |
| タップでピント | なし（ブラウザから指定できないため） |
| ズーム | カメラ側のズームが使えればそれを使い、使えなければ映像の中央を切り出して拡大（推論も見えている範囲だけで行う） |
