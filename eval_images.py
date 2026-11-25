"""しきい値（--conf）を決めるための画像単位の評価。

「女王蜂がいる画像のうち検出できた割合」と「女王蜂がいない画像で誤検出した割合」を
しきい値ごとに表示する（女王蜂を見つけたいアプリでは mAP よりこちらが実用的な目安）。

使い方:
  python eval_images.py --models models/queen.pt
  python eval_images.py --models runs/queen/*/weights/best.pt --split val --conf 0.15 0.25 0.35 0.5
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/merged/data.yaml"))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--conf", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.5])
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO
    d = yaml.safe_load(Path(args.data).read_text())
    img_dir = Path(d["path"]) / d[args.split]
    imgs = sorted(str(p) for p in img_dir.iterdir())
    lbl = lambda p: Path(p.replace("/images/", "/labels/")).with_suffix(".txt")  # noqa: E731
    pos = [p for p in imgs if lbl(p).exists() and lbl(p).stat().st_size > 0]
    neg = [p for p in imgs if p not in set(pos)]
    print(f"{args.split}: 女王蜂あり {len(pos)} 枚 / なし {len(neg)} 枚")
    for mp in args.models:
        m = YOLO(mp)

        def max_conf(ps):
            out = []
            for i in range(0, len(ps), 32):
                for r in m.predict(ps[i:i + 32], imgsz=args.imgsz, conf=min(args.conf), quantize=16, verbose=False):
                    out.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
            return np.array(out)

        cp, cn = max_conf(pos), max_conf(neg)
        for t in args.conf:
            print(f"{mp}  conf>={t:.2f}: 女王蜂あり画像で検出 {np.mean(cp >= t):6.1%}"
                  f" / 女王蜂なし画像で誤検出 {np.mean(cn >= t):5.1%}")


if __name__ == "__main__":
    main()
