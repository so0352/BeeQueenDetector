"""女王蜂リアルタイム検出アプリ本体。

スマホのカメラ映像（MJPEG/HTTP）・PC のカメラ・動画ファイルを入力に、女王蜂を枠で示す。

使い方の例:
  python detect_stream.py --source sample.mp4                       # 動画ファイル
  python detect_stream.py --app ipwebcam --source 192.168.1.23      # Android「IP Webcam」
  python detect_stream.py --app droidcam --source 192.168.1.23      # DroidCam（Android / iPhone）
  python detect_stream.py --source http://192.168.1.23:8080/video   # URL を直接指定
  python detect_stream.py --source 0                                # PC のカメラ
  python detect_stream.py --source ... --web 8090                   # ブラウザ表示（http://localhost:8090/）

キー操作（ウィンドウ表示時）: s = 手動スナップショット、q / Esc = 終了
"""
import argparse
import collections
import time
from pathlib import Path

import cv2

from stream_utils import PRESETS, FrameSource, MJPEGServer, TextDrawer, default_device, resolve_source, use_half

ROOT = Path(__file__).resolve().parent


def default_model():
    cands = sorted((ROOT / "runs/queen").glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if (ROOT / "models/queen.pt").exists():
        return str(ROOT / "models/queen.pt")
    if cands:
        return str(cands[-1])
    raise SystemExit("モデルが見つかりません。--model で指定するか、train.py で学習してください。")


def tile_boxes(w, h, grid, overlap=0.25):
    """画像を grid×grid の重なりありタイルに分ける座標 [(x0, y0, x1, y1)]。"""
    tw, th = int(w / (grid - (grid - 1) * overlap)), int(h / (grid - (grid - 1) * overlap))
    xs = [round(i * (w - tw) / (grid - 1)) for i in range(grid)]
    ys = [round(i * (h - th) / (grid - 1)) for i in range(grid)]
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def predict(model, frame, imgsz, conf, half, device, tiles=0, iou=0.5):
    """[((x0, y0, x1, y1), conf), ...] を返す。tiles>=2 なら全体＋タイルをまとめて推論し NMS で統合する。"""
    q = 16 if half else None
    if tiles < 2:
        r = model.predict(frame, imgsz=imgsz, conf=conf, quantize=q, device=device, verbose=False)[0]
        return [(b.xyxy[0].tolist(), float(b.conf[0])) for b in r.boxes]
    import torch
    import torchvision
    h, w = frame.shape[:2]
    regions = [(0, 0, w, h)] + tile_boxes(w, h, tiles)
    crops = [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in regions]
    res = model.predict(crops, imgsz=imgsz, conf=conf, quantize=q, device=device, verbose=False)
    xyxy, scores = [], []
    for (x0, y0, x1, y1), r in zip(regions, res):
        if len(r.boxes):
            b = r.boxes.xyxy.float().cpu()
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            xyxy.append(b)
            scores.append(r.boxes.conf.float().cpu())
    if not xyxy:
        return []
    xyxy, scores = torch.cat(xyxy), torch.cat(scores)
    keep = torchvision.ops.nms(xyxy, scores, iou)
    return [(xyxy[k].tolist(), float(scores[k])) for k in keep]


def draw(frame, boxes, found, fps, drawer, info):
    for (x0, y0, x1, y1), conf in boxes:
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        t = max(2, frame.shape[1] // 400)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), t + 1)
        cv2.rectangle(frame, (x0 - t * 3, y0 - t * 3), (x1 + t * 3, y1 + t * 3), (0, 255, 255), t)
        label = f"queen {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t / 2, t)
        cv2.rectangle(frame, (x0, max(0, y0 - h - 10)), (x0 + w + 6, y0), (0, 0, 255), -1)
        cv2.putText(frame, label, (x0 + 3, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t / 2, (255, 255, 255), t,
                    cv2.LINE_AA)
    size = max(20, frame.shape[1] // 40)
    head = f"FPS {fps:4.1f}  {info}" if fps > 0 else info
    drawer.put(frame, head, head, (8, 8), size=size // 2 + 6, color=(255, 255, 255), bg=(40, 40, 40))
    if found:
        drawer.put(frame, "女王蜂発見！", "QUEEN FOUND!", (8, 8 + size + 8), size=size + 10,
                   color=(255, 255, 255), bg=(0, 0, 220))
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="カメラ番号 / URL / 動画ファイル（--app 指定時はスマホの IP）")
    ap.add_argument("--app", choices=sorted(PRESETS), help="スマホ配信アプリのプリセット")
    ap.add_argument("--model", default=None, help="省略時は models/queen.pt → runs/queen の最新 best.pt")
    ap.add_argument("--imgsz", type=int, default=640, help="推論解像度（学習時と同じ 640 が基本）")
    ap.add_argument("--conf", type=float, default=0.25, help="確信度のしきい値（テストで 0.25: 検出 92%%・誤検出 2%%/画像）")
    ap.add_argument("--device", default=None, help="0（CUDA）/ mps（Apple Silicon）/ cpu。省略時は自動")
    ap.add_argument("--tiles", type=int, default=0,
                    help="2 以上で画面を N×N の重なりタイルにも分けて推論（小さく映る女王蜂向け、重くなる）")
    ap.add_argument("--no-half", action="store_true", help="FP16 を使わない")
    ap.add_argument("--window", type=int, default=10, help="判定に使う直近フレーム数 N")
    ap.add_argument("--min-hits", type=int, default=4, help="N フレーム中 K フレーム以上検出で「発見」")
    ap.add_argument("--snap-dir", default=str(ROOT / "snapshots"))
    ap.add_argument("--cooldown", type=float, default=10.0, help="自動スナップショットの間隔（秒）")
    ap.add_argument("--no-autosnap", action="store_true")
    ap.add_argument("--web", type=int, default=0, metavar="PORT", help="ブラウザ表示用 MJPEG サーバのポート")
    ap.add_argument("--no-show", action="store_true", help="ウィンドウを出さない")
    ap.add_argument("--display-width", type=int, default=1280, help="表示の最大幅（推論解像度とは無関係）")
    ap.add_argument("--realtime", action="store_true", help="動画ファイルも実時間で再生（遅れたフレームは捨てる）")
    ap.add_argument("--save-video", default=None, help="注釈付き映像を mp4 で保存")
    ap.add_argument("--max-frames", type=int, default=0, help="このフレーム数で終了（テスト用）")
    args = ap.parse_args()
    args.device = args.device or default_device()
    print(f"[デバイス] {args.device}")

    from ultralytics import YOLO

    src = resolve_source(args.source, args.app)
    model_path = args.model or default_model()
    print(f"[モデル] {model_path}")
    model = YOLO(model_path)
    half = not args.no_half and use_half(args.device)

    import numpy as np
    predict(model, np.zeros((720, 1280, 3), np.uint8), args.imgsz, args.conf, half, args.device, args.tiles)  # ウォームアップ
    print(f"[入力] {src}")
    fs = FrameSource(src, realtime_file=args.realtime)
    drawer = TextDrawer()
    web = MJPEGServer(args.web) if args.web else None
    show = not args.no_show
    if show:
        cv2.namedWindow("BeeQueenDetector", cv2.WINDOW_NORMAL)
    snap_dir = Path(args.snap_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)
    writer = None

    hits = collections.deque(maxlen=args.window)
    t_last = time.time()
    fps = 0.0
    last_snap = 0.0
    fid = 0
    n = 0
    n_found = 0
    n_det = 0
    n_boxes = 0
    infer_ms = 0.0
    try:
        while True:
            fid, frame = fs.read(fid)
            if fid is None:
                print("[入力] 終了")
                break
            if frame is None:
                continue
            t0 = time.time()
            boxes = predict(model, frame, args.imgsz, args.conf, half, args.device, args.tiles)
            infer_ms = 0.9 * infer_ms + 0.1 * (time.time() - t0) * 1000 if n else (time.time() - t0) * 1000
            hits.append(1 if boxes else 0)
            n_det += bool(boxes)
            n_boxes += len(boxes)
            found = sum(hits) >= args.min_hits
            n_found += found

            now = time.time()
            dt = now - t_last
            t_last = now
            fps = 0.9 * fps + 0.1 * (1 / dt) if fps else 1 / max(dt, 1e-6)

            raw = frame.copy() if (found and not args.no_autosnap) else None
            vis = draw(frame, boxes, found, fps, drawer, f"推論 {infer_ms:.0f}ms" if drawer.japanese
                       else f"infer {infer_ms:.0f}ms")

            stamp = time.strftime("%Y%m%d_%H%M%S")
            if raw is not None and now - last_snap >= args.cooldown:
                cv2.imwrite(str(snap_dir / f"{stamp}_auto.jpg"), vis)
                cv2.imwrite(str(snap_dir / f"{stamp}_auto_raw.jpg"), raw)
                last_snap = now
                print(f"[発見] 女王蜂を検出 → スナップショット保存 {stamp}_auto.jpg（最大 conf "
                      f"{max(c for _, c in boxes) if boxes else 0:.2f}）")

            if args.save_video:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fs.fps_src if not fs.live else 15, (w, h))
                writer.write(vis)

            disp = vis
            if vis.shape[1] > args.display_width:
                s = args.display_width / vis.shape[1]
                disp = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if web:
                web.update(disp)
            if show:
                cv2.imshow("BeeQueenDetector", disp)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("s"):
                    cv2.imwrite(str(snap_dir / f"{stamp}_manual.jpg"), vis)
                    print(f"[保存] {snap_dir / f'{stamp}_manual.jpg'}")
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        fs.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        if web:
            web.close()
    n = max(n, 1)
    print(f"[統計] 処理 {n} フレーム / 検出あり {n_det} ({n_det / n:.1%}) / 「発見」状態 {n_found} ({n_found / n:.1%})"
          f" / 1フレーム平均 {n_boxes / n:.2f} 個 / 平均推論 {infer_ms:.1f} ms")


if __name__ == "__main__":
    main()
