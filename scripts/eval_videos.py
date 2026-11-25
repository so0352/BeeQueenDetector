"""複数のモデルを動画で比較する（正解ラベルなしの簡易評価）。

女王蜂が映っている動画なら「検出ありフレーム率」が高いほど見逃しが少なく、
女王蜂が 1 匹なら「1 フレーム平均の個数」が 1 を大きく超えるほど誤検出が多い。
各動画から等間隔に 8 フレームを選び、検出結果を並べた画像も保存する（目視確認用）。

使い方:
  python scripts/eval_videos.py --models runs/queen/*/weights/best.pt --videos data/test_videos/*.webm
  python scripts/eval_videos.py --models models/queen.pt --videos my_hive.mp4 --imgsz 960 --tiles 2
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from detect_stream import draw, predict
from stream_utils import TextDrawer, default_device, use_half

ROOT = Path(__file__).resolve().parent.parent  # リポジトリのルート


def model_label(p):
    p = Path(p)
    return p.parent.parent.name if p.parent.name == "weights" else p.stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640])
    ap.add_argument("--tiles", type=int, nargs="+", default=[0])
    ap.add_argument("--conf", type=float, default=0.25, help="確信度のしきい値（テストで 0.25: 検出 92%%・誤検出 2%%/画像）")
    ap.add_argument("--stride", type=int, default=2, help="何フレームおきに評価するか")
    ap.add_argument("--device", default=None, help="0（CUDA）/ mps（Apple Silicon）/ cpu。省略時は自動")
    ap.add_argument("--out", default=str(ROOT / "runs/video_eval"))
    args = ap.parse_args()
    args.device = args.device or default_device()
    print(f"[デバイス] {args.device}")

    from ultralytics import YOLO
    drawer = TextDrawer()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mp in args.models:
        model = YOLO(mp)
        predict(model, np.zeros((720, 1280, 3), np.uint8), 640, args.conf, use_half(args.device), args.device)
        for imgsz in args.imgsz:
            for tiles in args.tiles:
                for vp in args.videos:
                    cap = cv2.VideoCapture(vp)
                    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    sheet_ids = set(np.linspace(0, n_total - 1, 10).astype(int)[1:-1].tolist())
                    sheet = []
                    n = det = nbox = 0
                    confs = []
                    t_inf = 0.0
                    fi = -1
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        fi += 1
                        want_sheet = fi in sheet_ids
                        if fi % args.stride and not want_sheet:
                            continue
                        t0 = time.time()
                        boxes = predict(model, frame, imgsz, args.conf, use_half(args.device), args.device, tiles)
                        t_inf += time.time() - t0
                        n += 1
                        det += bool(boxes)
                        nbox += len(boxes)
                        confs += [c for _, c in boxes]
                        if want_sheet:
                            vis = draw(frame.copy(), boxes, False, 0, drawer, f"#{fi}")
                            sheet.append((fi, cv2.resize(vis, (640, int(640 * vis.shape[0] / vis.shape[1])))))
                    cap.release()
                    tag = f"{model_label(mp)}_i{imgsz}_t{tiles}_{Path(vp).stem}"
                    if sheet:
                        tiles_img = [im for _, im in sheet] + [np.zeros_like(sheet[0][1])] * (8 - len(sheet))
                        grid = np.vstack([np.hstack(tiles_img[r * 4:(r + 1) * 4]) for r in range(2)])
                        cv2.imwrite(str(out / f"{tag}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    row = {"model": model_label(mp), "imgsz": imgsz, "tiles": tiles, "video": Path(vp).name,
                           "frames": n, "det_rate": det / max(n, 1), "boxes_per_frame": nbox / max(n, 1),
                           "median_conf": float(np.median(confs)) if confs else 0.0,
                           "ms_per_frame": 1000 * t_inf / max(n, 1)}
                    rows.append(row)
                    print(f"{row['model']:<22} i{imgsz:<5} t{tiles} {row['video']:<26} 検出率 {row['det_rate']:6.1%}"
                          f"  個/フレーム {row['boxes_per_frame']:5.2f}  conf中央値 {row['median_conf']:.2f}"
                          f"  {row['ms_per_frame']:5.1f} ms", flush=True)
    (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\n確認用画像: {out}/*.jpg")


if __name__ == "__main__":
    main()
