"""学習済みモデルをスマホアプリ用に書き出す（Android: LiteRT/TFLite、iPhone: CoreML）。

本体の .venv ではなく、書き出し専用の環境で実行する
（書き出しツールが opencv-python-headless を入れて cv2.imshow を壊すため）。
  .venv-export : LiteRT 用（PyTorch 2.14）
  .venv-coreml : CoreML 用（coremltools 9.0 が PyTorch 2.14 に未対応のため PyTorch 2.7）

使い方:
  .venv-export/bin/python export_mobile.py --formats tflite   # Android 用 → mobile_models/ と app/assets/models/
  .venv-coreml/bin/python export_mobile.py --formats coreml   # iPhone 用
  .venv-export/bin/python export_mobile.py --formats tflite --tflite-quantize fp32 --app-assets ""   # 精度比較用

精度の確認（書き出したモデルを PC で評価。tflite は PC でも動く。CoreML は macOS のみ）:
  .venv-export/bin/python eval_images.py --models models/queen.pt mobile_models/queen.tflite
"""
import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def export_tflite(model_path, out_dir, imgsz, quantize):
    from ultralytics import YOLO
    kw = {"format": "litert", "imgsz": imgsz, "nms": None, "batch": 1}
    if quantize != "fp32":
        kw["quantize"] = quantize
    # nms=None: 生の出力を書き出し、NMS はアプリ側（プラグインの Android 実装）で行う（公式の書き出し設定と同じ）
    out = Path(YOLO(model_path).export(**kw))
    tfl = out if out.suffix == ".tflite" else sorted(out.rglob("*.tflite"))[0]
    dst = out_dir / "queen.tflite"
    shutil.copy2(tfl, dst)
    print(f"[書き出し] Android 用: {dst} ({dst.stat().st_size / 1e6:.1f} MB, {quantize})")
    return dst


def export_coreml(model_path, out_dir, imgsz, quantize):
    from ultralytics import YOLO
    kw = {"format": "coreml", "imgsz": imgsz, "nms": True, "batch": 1}
    if quantize:
        kw["quantize"] = quantize
    # YOLO11 は NMS をモデルに埋め込む（iOS の Vision がそのまま枠を返す。UltralyticsYOLO が対応する形式）
    out = Path(YOLO(model_path).export(**kw))
    pkg = out_dir / "queen.mlpackage"
    if pkg.exists():
        shutil.rmtree(pkg)
    shutil.copytree(out, pkg)
    # Flutter の assets には .mlpackage（フォルダ）を置けないので、フォルダごと zip にする（最上位が queen.mlpackage/）
    zpath = out_dir / "queen.mlpackage.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pkg.rglob("*")):
            z.write(f, f.relative_to(out_dir))
    print(f"[書き出し] iPhone 用: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    return zpath


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models/queen.pt"))
    ap.add_argument("--formats", nargs="+", default=["tflite", "coreml"], choices=["tflite", "coreml"])
    ap.add_argument("--imgsz", type=int, default=640, help="学習時と同じ 640")
    ap.add_argument("--tflite-quantize", default="w8a32", choices=["w8a32", "fp16", "fp32"],
                    help="w8a32 = 重みだけ int8（公式と同じ。GPU で動き、校正データ不要）")
    ap.add_argument("--coreml-quantize", type=int, default=16, choices=[0, 8, 16],
                    help="16 = FP16（精度を優先）。8 にすると半分のサイズ")
    ap.add_argument("--out", default=str(ROOT / "mobile_models"))
    ap.add_argument("--app-assets", default=str(ROOT / "app/assets/models"),
                    help="書き出したモデルのコピー先（空文字でコピーしない）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 書き出し時の一時ファイルが models/ に散らからないよう、作業用コピーから書き出す
    work = out_dir / "work"
    work.mkdir(exist_ok=True)
    src = work / "queen.pt"
    shutil.copy2(args.model, src)

    made = []
    if "tflite" in args.formats:
        made.append(export_tflite(src, out_dir, args.imgsz, args.tflite_quantize))
    if "coreml" in args.formats:
        made.append(export_coreml(src, out_dir, args.imgsz, args.coreml_quantize or None))

    if args.app_assets:
        dst = Path(args.app_assets)
        dst.mkdir(parents=True, exist_ok=True)
        for f in made:
            shutil.copy2(f, dst / f.name)
            print(f"[コピー] {dst / f.name}")


if __name__ == "__main__":
    main()
