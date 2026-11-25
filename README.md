# BeeQueenDetector — 女王蜂リアルタイム検出

スマホのカメラ映像を PC（Mac の Apple Silicon、または WSL2 + NVIDIA GPU）で解析し、西洋ミツバチの巣板から**女王蜂を見つけて枠で示す**アプリです。

```
スマホ（カメラ配信アプリ） ──Wi-Fi(同じLAN) MJPEG/HTTP──▶ PC (Mac / WSL2 + GPU)
                                                     Ultralytics YOLO で推論
                                                     → 枠付き映像をウィンドウ or ブラウザに表示
```

## ファイル構成

| ファイル | 役割 |
|---|---|
| `check_env.py` | torch / CUDA・MPS / `cv2.imshow` の動作確認 |
| `datasets.txt` | 取得する Roboflow Universe データセットの一覧 |
| `download_datasets.py` | Roboflow REST API でデータ取得 → `data/raw/<name>/` |
| `merge_datasets.py` | 1 クラス `queen` に統合・重複除去 → `data/merged/` |
| `train.py` | YOLO の学習 → `runs/queen/<name>/weights/best.pt` |
| `start.sh` / `start_windows.bat` | スマホを自動で探して検出を起動（ワンコマンド / Windows でダブルクリック） |
| `detect_stream.py` | リアルタイム検出アプリ本体 |
| `tools/find_phone.py` | LAN 上のスマホ配信アプリ（8080 / 4747）を自動検索 |
| `collect_frames.py` | スマホ映像から追加注釈用のフレームを保存 |
| `eval_images.py` | しきい値ごとの「検出できた割合 / 誤検出の割合」（画像単位） |
| `eval_videos.py` | 複数モデルを動画で比較（検出率・誤検出の目安・確認用画像） |
| `stream_utils.py` | 映像入力・ブラウザ配信などの共通部品 |
| `tools/fake_camera.py` | 動画をスマホ配信アプリと同じ MJPEG で流す（スマホなしで動作確認） |
| `export_mobile.py` | スマホアプリ用にモデルを書き出す（Android: TFLite / iPhone: CoreML / Web 版: ONNX） |
| `app/` | スマホ単体アプリ（Flutter、Android / iPhone）。詳しくは `app/README.md` |
| `web/` | Web 版（ブラウザで動き、開いた端末の GPU で推論）。詳しくは `web/README.md` |
| `CREDITS.md` | 使用データの出典・ライセンス（自動生成） |
| `LICENSE` / `PRIVACY.md` | ライセンス（AGPL-3.0）・プライバシーポリシー |

## 1. 環境構築

```bash
cd ~/work/BeeQueenDetector
uv venv .venv --python 3.12          # または python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt   # または pip install -r requirements.txt
python check_env.py
```

- 使うデバイスは自動で選びます（NVIDIA GPU → `0`、Apple Silicon の Mac → `mps`、それ以外 → `cpu`）。
  `--device` で明示もできます。FP16 推論は CUDA のときだけ使います。
- `data/`・`runs/`・`mobile_models/`・事前学習済みの `yolo*.pt` は大きいので Git に入れていません
  （採用モデル `models/queen.pt` だけはリポジトリに含めています）。データは 2. の手順で作り直せます。

- 起動時に `QFontDatabase: Cannot find font directory ...` という警告が出ますが、無害なので無視して構いません。
- **`roboflow` パッケージは入れないでください。** `opencv-python-headless` を引き込み `cv2.imshow` が壊れます。
  入れてしまった場合: `pip uninstall -y opencv-python-headless && pip install --force-reinstall opencv-python`
- `cv2.imshow` のウィンドウが出ない場合は、検出時に `--web 8090` を付けてブラウザで
  `http://localhost:8090/` を開いてください。

## 2. データの取得と統合

1. [Roboflow](https://roboflow.com/) の無料アカウントを作り、Settings → API Keys の **Private API Key** を控える。
2. 取得と統合:

```bash
echo "ROBOFLOW_API_KEY=xxxxxxxx" > .env && chmod 600 .env   # または export ROBOFLOW_API_KEY=...
python download_datasets.py --list-only   # バージョン・ライセンス・クラスを確認
python download_datasets.py               # data/raw/ に YOLOv8 形式で展開
python merge_datasets.py                  # data/merged/ と data.yaml を作成、枚数を集計表示
```

- `datasets.txt` には Roboflow Universe の女王蜂データ 23 件（すべて CC BY 4.0）を登録済み。出典は `CREDITS.md`。
  追加したいときは Universe の URL の `<workspace>/<project>` を 1 行書き足して再実行します。
  **追加したら必ず画像を目で確認してください。** クラス名が `queen` でも、アリの女王やチェスの駒のデータが
  実際にありました（`datasets.txt` で除外済み）。
- 女王蜂のクラス（`queen` / `Bee_Queen` / `reina` など。`non-queen` や王台 `queen cell` は除外）を `queen` に統一し、
  他のクラス（働き蜂など）は捨てます。女王蜂のいない画像は背景（負例）として残します（train は正例数と同数まで）。
- 回転・反転に強い知覚ハッシュで重複画像を検出し、評価の水増しと train / val・test 間のリークを防ぎます（GPU で計算）。
- 評価（val/test）は「スマホで巣板を撮った実映像」に近いデータだけで行います。切り抜き・接写中心・固定カメラの
  データは学習専用（`merge_datasets.py` の `DEFAULT_TRAIN_ONLY`）。
- 自前で注釈したデータは `data/custom/<名前>/`（`train/images`, `train/labels`, `data.yaml` の YOLO 形式）に置けば一緒に統合されます。

## 3. 学習

```bash
# 現在の models/queen.pt を作ったときの設定（RTX 4070 SUPER で約 1.5 時間。Mac の mps ではかなり遅い）
python train.py --name v3_nc_y11s --model yolo11s.pt --epochs 50 --patience 15 --batch 16 --cache none --workers 10
cp runs/queen/v3_nc_y11s/weights/best.pt models/queen.pt

# 評価
python eval_images.py --models models/queen.pt                 # しきい値ごとの検出率・誤検出率
python eval_videos.py --models models/queen.pt --videos data/test_videos/*.webm
```

- 結果: `runs/queen/<name>/weights/best.pt`、指標は `summary.json`、検証画像への推論結果は `val_preds/`。
- **学習は 1 本ずつ**。2 本同時に回すと（RTX 4070 SUPER の場合）VRAM 12GB を超えて Windows の共有メモリにあふれ、極端に遅くなります。
- 公開データは自分の巣箱とは条件が違うため、mAP の数字よりも**実際の巣板映像での見逃し・誤検出**を重視してください。
- 採用したモデルは `models/queen.pt` に置くと `detect_stream.py` の既定になります。

### これまでの比較結果（2026-11-18）

評価データ（val 2293 枚・test 880 枚）はスマホで巣板を撮った画像に近いデータのみ。動画は Wikimedia Commons の 2 本。

| モデル | 学習データ | val mAP50 | test mAP50 | test 検出率 / 誤検出率 (conf 0.25) | 動画 検出率 (2本) |
|---|---|---|---|---|---|
| **YOLO11s（採用）** | 切り抜き除外 | 0.888 | 0.919 | **91.7% / 1.9%** | **100% / 79%** |
| YOLO26s | 切り抜き除外 | **0.896** | **0.928** | 90.0% / 1.7% | 99% / 70% |
| YOLO11s | 切り抜き込み | 0.890 | 0.891 | 89.2% / 1.3% | 100% / 75% |
| YOLO11s（初期・4 データのみ） | 509 枚 | 0.299 | — | — | 79% / 47% |

YOLO26s は mAP では僅差で上だが、画像単位の検出率と動画で YOLO11s が上回ったため YOLO11s を採用。
推論速度は 1 フレーム約 10〜20 ms（RTX 4070 SUPER、FP16）。

既知の弱点: 女王蜂が他の蜂に隠れている・頭を巣房に入れて産卵中だと見逃しやすい。学習データには
マーキング（背中の色付きの点）された女王が多いので、**マーキングなしの女王では精度が下がる可能性**があります。

## 4. リアルタイム検出

### スマホ側の準備（Android / iPhone どちらも可）

| スマホ | アプリ | コマンド |
|---|---|---|
| Android | **IP Webcam**（Google Play） | `--app ipwebcam --source <スマホのIP>` |
| Android / iPhone | **DroidCam**（Google Play / App Store） | `--app droidcam --source <スマホのIP>` |
| その他 | MJPEG/RTSP を配信できるアプリ | `--source <アプリに表示される URL>` |

- スマホと PC を**同じ Wi-Fi** につなぎ、アプリで配信を開始して画面に出る IP アドレスを使います。
- 解像度は 1280×720〜1920×1080 程度がおすすめ（女王蜂が小さく映るため、低すぎると見逃します）。

### かんたん起動（おすすめ）

スマホでアプリの配信を開始してから:

```bash
~/work/BeeQueenDetector/start.sh                 # スマホを LAN から自動で探して起動
~/work/BeeQueenDetector/start.sh 192.168.0.23    # IP を指定（アプリの種類は自動判定）
~/work/BeeQueenDetector/start.sh --tiles 2       # detect_stream.py のオプションもそのまま渡せる
```

- Windows からは `start_windows.bat` をダブルクリック（デスクトップにコピーして使えます。
  エクスプローラーで `\\wsl$\Ubuntu-24.04\home\<ユーザー名>\work\BeeQueenDetector` を開くと見つかります）。
- IP Webcam（8080）と DroidCam（4747）を自動で判別します。見つかるまで 5 秒ごとに探し続けます。
- 前回の IP を `.phone` に覚えるので、2 回目以降はすぐつながります。
- 検出画面のウィンドウに加え、ブラウザでも `http://localhost:8090/` で見られます（`WEB_PORT=0 ./start.sh` で無効）。
- 見つからないとき: スマホと PC が同じ Wi-Fi か、アプリが配信中か、DroidCam の PC アプリやブラウザで
  同時に開いていないか（DroidCam は 1 台しか接続できない）を確認してください。

### 実行（個別に指定する場合）

```bash
source .venv/bin/activate
python detect_stream.py --source hive.mp4                       # 動画ファイルで確認
python detect_stream.py --app ipwebcam --source 192.168.1.23    # Android（IP Webcam）
python detect_stream.py --app droidcam --source 192.168.1.23    # DroidCam
python detect_stream.py --app droidcam --source 192.168.1.23 --web 8090   # ブラウザ表示
```

- 直近 `--window` フレーム（既定 10）のうち `--min-hits` フレーム（既定 4）以上で検出されると「女王蜂発見！」を表示し、
  `snapshots/` に自動保存します（`--cooldown` 秒間隔）。
- キー: `s` 手動保存、`q` / `Esc` 終了。
- 既定のしきい値は `--conf 0.25`。誤検出が多いときは上げる（例 0.5）、見逃すときは下げる（例 0.15）。
- 巣板全体を遠目に映して女王蜂が小さいときは `--tiles 2`（画面を 2×2 の重なりタイルにも分けて推論。少し重くなる）。
- スマホなしで試すとき: `python tools/fake_camera.py data/test_videos/queen_bee.webm --port 8080` を別端末で動かし、
  `python detect_stream.py --source http://localhost:8080/video`。

### つながらないとき

- まず `curl -m 5 -s -o /dev/null -w '%{http_code}\n' http://<スマホIP>:8080/video` が `200` になるか確認。
- スマホと PC が同じ Wi-Fi（ゲスト用 Wi-Fi は端末間通信が禁止されていることが多い）か確認。
- Mac の場合、初回に「ローカルネットワークへのアクセス」の許可を求められたら許可する（システム設定 → プライバシーとセキュリティ → ローカルネットワーク）。

#### WSL2 の場合

- この PC の WSL2 は NAT モードです。WSL → LAN 上のスマホへの接続は通常そのまま可能です。
- まず WSL から `curl -m 5 -s -o /dev/null -w '%{http_code}\n' http://<スマホIP>:8080/video` が `200` になるか確認。
  ならない場合は、Windows 側のブラウザで同じ URL が見られるか → 見られるなら Windows ファイアウォール / VPN を確認。

## 5. 精度を上げる（自前データの追加）

公開データは少なく、自分の巣板・カメラ・照明とは条件が違います。**精度を上げる本命は自前データ**です。

```bash
# 2 秒おきに（ブレた画像・似た画像は除外して）保存。スペースキーで手動保存
python collect_frames.py --app ipwebcam --source 192.168.1.23 --interval 2
# 学習済みモデルが「迷っている」フレームだけを集める（注釈の効率が良い）
python collect_frames.py --source hive.mp4 --model models/queen.pt --uncertain
```

1. `data/collected/<日時>/` の画像を Roboflow（Web）や CVAT で注釈（クラス名 `queen`、女王蜂がいない画像もそのまま残す）。
2. YOLOv8 形式で書き出して `data/custom/<名前>/` に展開。
3. `python merge_datasets.py && python train.py` で再学習。

## 6. スマホ単体アプリ（`app/`）

PC なし・圏外でも、スマホの中だけで女王蜂を検出するアプリです（Flutter、Android / iPhone 共通のコード）。
ビルド・インストール・モデル更新の手順は **`app/README.md`** を参照してください。

```bash
.venv-export/bin/python export_mobile.py --formats tflite   # Android 用モデル（LiteRT w8a32, 約 10 MB）
.venv-coreml/bin/python export_mobile.py --formats coreml   # iPhone 用モデル（CoreML FP16, 約 18 MB）
cd app && flutter build apk --release --target-platform android-arm64
```

- 書き出し後の精度（test 880 枚、conf 0.25）: `.pt` 検出 91.7% / 誤検出 1.9% → `.tflite` 91.3% / 1.9%（ほぼ同じ）。
- iPhone 版のビルドには Mac（Xcode）が必要です。

## 7. Web 版（`web/`）

ブラウザで URL を開くだけで使える版です。**推論はページを開いた端末の GPU（WebGPU）で行い**、映像はどこにも送りません。
iPhone にも App Store なしで配れます（GitHub Pages で公開）。手順・確認結果は **`web/README.md`** を参照してください。

```bash
.venv-onnx/bin/python export_mobile.py --formats onnx   # Web 版用モデル（ONNX FP32, 約 38 MB）→ web/models/
cd web && python3 -m http.server 8765                   # http://localhost:8765/ で確認
```

- 書き出し後の精度（test 880 枚、conf 0.25）: `.onnx` 検出 91.7% / 誤検出 1.9%（`.pt` と同じ）。
- Apple Silicon の Mac の Chrome（WebGPU）で 1 回約 20 ms。iPhone での速度は実機で要確認。

## ライセンス

Copyright (c) 2026 so0352

このリポジトリ（コード・学習済みモデル）は [AGPL-3.0](LICENSE) で公開しています（Ultralytics YOLO / `ultralytics_yolo` が AGPL-3.0 のため）。

- 学習データ: Roboflow Universe の各データセット（CC BY 4.0）。出典は [CREDITS.md](CREDITS.md)。
- Web 版に同梱の ONNX Runtime Web: MIT License（[web/vendor/ort/LICENSE](web/vendor/ort/LICENSE)）。
