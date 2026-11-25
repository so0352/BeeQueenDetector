"""スマホ映像などからフレームを保存する（追加注釈用）。

使い方:
  python scripts/collect_frames.py --app ipwebcam --source 192.168.1.23 --interval 2
  python scripts/collect_frames.py --source hive.mp4 --interval 1 --no-show
  # 学習済みモデルで「自信のないフレーム」を優先的に集める（注釈の効率化）
  python scripts/collect_frames.py --source hive.mp4 --model runs/queen/yolo11s_960/weights/best.pt --uncertain

キー操作: スペース / s = 今のフレームを保存、p = 自動保存の一時停止/再開、q / Esc = 終了
保存先: data/collected/<日時>/ （Roboflow / CVAT などで注釈して data/custom/ に YOLO 形式で置く）
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from stream_utils import PRESETS, FrameSource, TextDrawer, resolve_source

ROOT = Path(__file__).resolve().parent.parent  # リポジトリのルート


def sharpness(img):
    g = cv2.cvtColor(cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1]))), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source")
    ap.add_argument("--app", choices=sorted(PRESETS))
    ap.add_argument("--out", default=None, help="保存先（省略時は data/collected/<日時>）")
    ap.add_argument("--interval", type=float, default=2.0, help="自動保存の間隔（秒）。0 でキー操作のみ")
    ap.add_argument("--min-sharpness", type=float, default=60.0, help="これ未満のブレた画像は自動保存しない")
    ap.add_argument("--max-sim", type=float, default=0.97, help="前回保存と似すぎた画像は自動保存しない")
    ap.add_argument("--model", default=None, help="指定すると検出結果を表示し、--uncertain で選別に使う")
    ap.add_argument("--uncertain", action="store_true", help="conf が 0.1〜0.5 の検出があるフレームだけ自動保存")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--max-saves", type=int, default=0)
    args = ap.parse_args()

    src = resolve_source(args.source, args.app)
    out = Path(args.out) if args.out else ROOT / "data/collected" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    model = None
    if args.model:
        from ultralytics import YOLO
        model = YOLO(args.model)

    fs = FrameSource(src)
    drawer = TextDrawer()
    show = not args.no_show
    if show:
        cv2.namedWindow("collect_frames", cv2.WINDOW_NORMAL)
    last_t = 0.0
    last_thumb = None
    saved = 0
    paused = False
    fid = 0
    try:
        while True:
            fid, frame = fs.read(fid)
            if fid is None:
                break
            if frame is None:
                continue
            now = time.time() if fs.live else fid / fs.fps_src
            confs = []
            vis = frame.copy()
            if model is not None:
                r = model.predict(frame, imgsz=args.imgsz, conf=0.1, quantize=16, verbose=False)[0]
                for b in r.boxes:
                    x0, y0, x1, y1 = map(int, b.xyxy[0].tolist())
                    c = float(b.conf[0])
                    confs.append(c)
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255) if c >= 0.5 else (0, 200, 255), 2)

            want_auto = (args.interval > 0 and not paused and now - last_t >= args.interval)
            if want_auto and args.uncertain:
                want_auto = any(0.1 <= c < 0.5 for c in confs)
            reason = None
            if want_auto:
                thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32).ravel()
                sim = 0.0
                if last_thumb is not None:
                    a, b = thumb - thumb.mean(), last_thumb - last_thumb.mean()
                    sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
                if sharpness(frame) >= args.min_sharpness and sim < args.max_sim:
                    reason = "auto"
                    last_thumb = thumb

            key = -1
            if show:
                txt = f"保存 {saved} 枚 {'(一時停止)' if paused else ''}"
                drawer.put(vis, txt, f"saved {saved} {'(paused)' if paused else ''}", (8, 8), size=24,
                           bg=(40, 40, 40))
                disp = vis if vis.shape[1] <= 1280 else cv2.resize(vis, None, fx=1280 / vis.shape[1],
                                                                  fy=1280 / vis.shape[1])
                cv2.imshow("collect_frames", disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    paused = not paused
                if key in (ord(" "), ord("s")):
                    reason = "manual"

            if reason:
                name = f"{time.strftime('%Y%m%d_%H%M%S')}_{fid:06d}_{reason}.jpg"
                cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
                last_t = now
                print(f"[保存] {name}")
                if args.max_saves and saved >= args.max_saves:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        fs.release()
        if show:
            cv2.destroyAllWindows()
    print(f"保存 {saved} 枚 → {out}")


if __name__ == "__main__":
    main()
