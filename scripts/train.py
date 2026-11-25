"""女王蜂検出モデル（1クラス queen）を Ultralytics YOLO で学習する。

使い方:
  python scripts/train.py                                   # yolo11s.pt, imgsz=640
  python scripts/train.py --model yolo26s.pt --name y26s    # 別モデルで比較
  python scripts/train.py --epochs 3 --name smoke           # 動作確認用の短い学習

学習後:
  - runs/queen/<name>/weights/best.pt が最良モデル
  - runs/queen/<name>/val_preds/ に検証画像への推論結果（目視確認用）を保存
  - runs/queen/<name>/summary.json に指標を保存
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from stream_utils import default_device, use_half

ROOT = Path(__file__).resolve().parent.parent  # リポジトリのルート


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data/merged/data.yaml"))
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=640, help="公開データは 640x640 なので既定 640")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=float, default=-1, help="-1 で自動。0<x<1 で GPU メモリの割合")
    ap.add_argument("--device", default=None, help="0（CUDA）/ mps（Apple Silicon）/ cpu。省略時は自動")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default=None, help="runs/queen/<name>（省略時はモデル名_imgsz）")
    ap.add_argument("--degrees", type=float, default=10.0)
    ap.add_argument("--scale", type=float, default=0.5, help="拡大縮小の拡張の強さ（小さく映る女王蜂対策で大きめも試す）")
    ap.add_argument("--close-mosaic", type=int, default=10, help="最後の N エポックはモザイク拡張を止める")
    ap.add_argument("--cache", default="ram", help="ram / disk / none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-preview", type=int, default=40, help="推論結果を保存する検証画像の枚数")
    args = ap.parse_args()
    args.device = args.device or default_device()
    print(f"[デバイス] {args.device}")

    name = args.name or f"{Path(args.model).stem}_{args.imgsz}"
    batch = int(args.batch) if args.batch == -1 or args.batch >= 1 else args.batch
    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=batch,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "runs/queen"),
        name=name,
        exist_ok=True,
        seed=args.seed,
        cache=False if args.cache == "none" else args.cache,
        # 巣板の向きは任意なので上下・左右反転と小さな回転を入れる
        flipud=0.5,
        fliplr=0.5,
        degrees=args.degrees,
        scale=args.scale,
        close_mosaic=args.close_mosaic,
        plots=True,
    )

    run_dir = ROOT / "runs/queen" / name
    best = run_dir / "weights/best.pt"
    print(f"\n== 最良モデルで検証: {best}")
    m = YOLO(str(best))
    metrics = m.val(data=args.data, imgsz=args.imgsz, device=args.device, split="val",
                    project=str(run_dir), name="val_best", exist_ok=True, plots=True)
    summary = {
        "model": args.model, "imgsz": args.imgsz, "best": str(best),
        "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
    }
    if "test:" in Path(args.data).read_text():
        t = m.val(data=args.data, imgsz=args.imgsz, device=args.device, split="test",
                  project=str(run_dir), name="test_best", exist_ok=True)
        summary.update({"test_mAP50": float(t.box.map50), "test_mAP50-95": float(t.box.map),
                        "test_precision": float(t.box.mp), "test_recall": float(t.box.mr)})
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # 目視確認用: 正例を優先して検証画像に推論結果を描画
    import yaml
    d = yaml.safe_load(Path(args.data).read_text())
    val_imgs = sorted((Path(d["path"]) / d["val"]).glob("*"))
    lbl_dir = Path(d["path"]) / d["val"].replace("images", "labels")
    pos = [p for p in val_imgs if (lbl_dir / (p.stem + ".txt")).exists() and (lbl_dir / (p.stem + ".txt")).stat().st_size]
    pos_set = set(pos)
    neg = [p for p in val_imgs if p not in pos_set]
    import random
    rng = random.Random(0)  # データセット名順に偏らないよう無作為に選ぶ
    preview = rng.sample(pos, min(len(pos), args.n_preview * 3 // 4)) + rng.sample(neg, min(len(neg), args.n_preview // 4))
    if preview:
        m.predict([str(p) for p in preview], imgsz=args.imgsz, device=args.device, conf=0.25, quantize=16 if use_half(args.device) else None,
                  save=True, project=str(run_dir), name="val_preds", exist_ok=True, verbose=False)
        print(f"推論結果の画像: {run_dir / 'val_preds'}")


if __name__ == "__main__":
    main()
