# 女王蜂さがし（スマホ単体アプリ）

スマホのカメラで巣板を映すと、**スマホの中だけで**女王蜂を検出して枠で示すアプリです。
PC もネットも不要なので、巣箱の前が圏外でも動きます。

- Android: LiteRT（TFLite）、`assets/models/queen.tflite`（GPU が使えれば GPU、だめなら CPU）
- iPhone: CoreML、`assets/models/queen.mlpackage.zip`（Neural Engine / GPU を自動で使用）
- モデルは PC 版と同じ `models/queen.pt`（YOLO11s, imgsz 640）を書き出したもの
- 推論・カメラ・枠の描画は Ultralytics 公式の Flutter プラグイン [`ultralytics_yolo`](https://pub.dev/packages/ultralytics_yolo)

## 画面と操作

| 操作 | 内容 |
|---|---|
| 上の帯 | 「巣板を映してください」→ 女王蜂が続けて検出されると赤い「女王蜂発見！」（振動） |
| ピンチ / ズームボタン | ズーム（ボタンは 1x → 2x → 3x） |
| 画面をタップ | そこにピントと明るさを合わせる |
| ライト | 巣箱の中が暗いときに |
| シャッター | 枠付きの画像を写真に保存（Android は「findBeeQueen」アルバム） |
| 自動保存 | 「発見」したら自動で保存（既定 10 秒に 1 枚まで） |
| 設定 | しきい値、「発見」の判定（直近 N フレーム中 K 回）、自動保存の間隔、振動 |

判定の考え方は PC 版 `detect_stream.py` と同じです（既定: しきい値 0.25、直近 10 フレーム中 4 フレーム以上）。

## Android: ビルドとインストール

開発ツールは `~/opt` に入っています（JDK 21 / Flutter 3.50 / Android SDK）。

```bash
source ~/opt/env.sh
cd ~/work/findBeeQueen/app
flutter build apk --release --target-platform android-arm64
# → build/app/outputs/flutter-apk/app-release.apk（約 38 MB）
```

スマホへの入れ方（どれか 1 つ）:

1. **Tailscale 経由でダウンロード**（スマホで Tailscale を ON にして）
   ```bash
   cd build/app/outputs/flutter-apk && python3 -m http.server 8000 --bind 100.69.53.14
   ```
   スマホのブラウザで `http://100.69.53.14:8000/app-release.apk` を開いてダウンロード → 開く
   （「提供元不明のアプリ」の許可を求められたら許可）。終わったら Ctrl+C で止める。
2. **USB / Google ドライブ**: Windows のエクスプローラーで
   `\\wsl$\Ubuntu-24.04\home\<ユーザー名>\work\findBeeQueen\app\build\app\outputs\flutter-apk\` から APK をスマホへコピー。
3. **ワイヤレスデバッグ（adb）**: スマホの「開発者向けオプション → ワイヤレス デバッグ → ペア設定コード」で
   ```bash
   adb pair <スマホのIP>:<ペア用ポート>   # 6 桁のコードを入力
   adb connect <スマホのIP>:<接続用ポート>
   flutter install --release          # または flutter run --release（ログが見える）
   ```

## iPhone: ビルドとインストール（Mac が必要）

iPhone 向けのビルドは Apple の決まりで macOS + Xcode でしかできません（MacBook Air を使います）。

1. Mac に Xcode（App Store）と Flutter を入れる（`brew install --cask flutter` など）。
2. このフォルダ（`app/`）を Mac にコピーする（`mobile_models/` は不要。モデルは `assets/models/` に入っています）。
3. iPhone を USB でつなぎ、iPhone の「設定 → プライバシーとセキュリティ → デベロッパモード」を ON。
4. ```bash
   cd app
   flutter pub get
   open ios/Runner.xcworkspace
   ```
   Xcode で Runner → Signing & Capabilities → Team に自分の Apple ID を設定し、
   Bundle Identifier を自分だけの名前（例 `com.<自分の名前>.findbeequeen`）に変更。
5. ```bash
   flutter run --release
   ```
   初回は iPhone の「設定 → 一般 → VPN とデバイス管理」で開発元を信頼する。

- 無料の Apple ID だと **7 日でアプリが起動しなくなる**ので、そのたびに 5. をやり直します
  （年 99 ドルの Apple Developer Program なら 1 年有効、TestFlight での配布も可能）。

## iPhone: 他の人に配る（TestFlight）

Apple Developer Program（年 99 ドル）への登録が必要です。受け取る人は App Store の「TestFlight」アプリを入れて、
招待リンクを開くだけでインストールできます（ケーブルや Mac は不要、最大 1 万人、1 ビルドは 90 日有効）。

1. **登録**: https://developer.apple.com/programs/ で Apple Developer Program に登録する（承認まで 1〜2 日）。
2. **Xcode の署名**: `open ios/Runner.xcworkspace` → Runner → Signing & Capabilities で
   Team を **有料の方のチーム** にする（Bundle ID `com.example.findbeequeen` のままで OK）。
3. **App Store Connect にアプリを作る**: https://appstoreconnect.apple.com → アプリ → ＋ → 新規 App
   - プラットフォーム iOS、名前「女王蜂さがし」（使われていたら「女王蜂さがし - ○○」など）、
     言語 日本語、バンドル ID `com.example.findbeequeen`、SKU は任意（例 `findbeequeen`）。
4. **ビルドしてアップロード**（Mac で）:
   ```bash
   cd app
   flutter build ipa --release
   # → build/ios/ipa/*.ipa
   ```
   出来た `.ipa` を Mac の「Transporter」アプリ（App Store で無料）にドラッグして「配信」。
   または `open build/ios/archive/Runner.xcarchive` → Xcode Organizer で Distribute App → App Store Connect。
   - 2 回目以降は `pubspec.yaml` の `version: 1.0.0+1` の `+` の後ろ（ビルド番号）を必ず増やす。
   - 暗号化の質問（輸出コンプライアンス）は `Info.plist` で「使用しない」と申告済みなので聞かれません。
5. **テスターを招待**（App Store Connect → TestFlight タブ。処理に 10〜30 分かかる）:
   - **外部テスト**（知り合いに配る場合はこちら）: 「外部テスト」のグループを作る →
     ビルドを追加 → テスト情報（説明、フィードバック用メール、プライバシーポリシー URL に
     リポジトリの `PRIVACY.md` の URL）を入力 → **ベータ版 App 審査**（初回だけ 1 日程度）→
     承認後「公開リンクを有効にする」で出る URL を LINE などで送る。
   - 内部テスト（App Store Connect に登録したメンバー最大 100 人、審査なし）は自分の確認用に。
6. 受け取る人: iPhone に「TestFlight」をインストール → 送られたリンクを開く → インストール。

ビルドの有効期限は 90 日なので、それまでに新しいビルドをアップロードしてください（使っている人には自動で届きます）。

## ライセンス

このアプリは [AGPL-3.0](../LICENSE) で公開しています（Ultralytics の `ultralytics_yolo` / YOLO11 が AGPL-3.0 のため）。
アプリを配布するときは、このリポジトリ（ソースコード）を公開しておく必要があります。
学習データのクレジットは `../CREDITS.md`（CC BY 4.0）を参照してください。プライバシーポリシーは [PRIVACY.md](../PRIVACY.md)。

## モデルを更新したとき

再学習して `models/queen.pt` を差し替えたら、書き出してからアプリをビルドし直します。

```bash
cd ~/work/findBeeQueen
.venv-export/bin/python export_mobile.py --formats tflite   # Android 用（LiteRT, w8a32）→ app/assets/models/
.venv-coreml/bin/python export_mobile.py --formats coreml   # iPhone 用（CoreML, FP16）→ app/assets/models/
.venv-export/bin/python eval_images.py --models models/queen.pt mobile_models/queen.tflite   # 書き出し後の精度確認
```

- 書き出し用の環境は本体の `.venv` と分けています（書き出しツールが `opencv-python-headless` を入れて
  `cv2.imshow` を壊すため）。CoreML は coremltools 9.0 が PyTorch 2.14 に未対応なので、PyTorch 2.7 の
  `.venv-coreml` を使います。
- 2026-11-18 時点の確認（test 880 枚、しきい値 0.25）: `.pt` 検出 91.7% / 誤検出 1.9%、
  `.tflite`（w8a32）検出 91.3% / 誤検出 1.9%。量子化による劣化はほぼなし。
  CoreML は PC（Linux）では推論できないため未確認です（FP16 なので劣化はほぼないはず）。

## 注意

- スマホの処理能力は PC より低いので、YOLO11s では 1 秒に数〜20 フレーム程度になる見込みです（機種による）。
  遅すぎる場合は `yolo11n` で学習し直すと速くなります（精度は少し下がる）。
- PC 版の `--tiles`（画面を分割して小さな女王蜂を探す）は未対応です。女王蜂が小さく映るときはズームを使ってください。
- 使用ライブラリ・データのライセンスは上の「ライセンス」を参照してください。
