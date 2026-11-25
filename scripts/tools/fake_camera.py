"""動画ファイルをスマホの配信アプリ（IP Webcam など）と同じ MJPEG/HTTP で流すテスト用サーバ。

スマホなしで detect_stream.py の URL 入力・再接続などを確認できる。

使い方:
  python scripts/tools/fake_camera.py data/test_videos/queen_bee.webm --port 8080
  python scripts/detect_stream.py --source http://localhost:8080/video
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stream_utils import MJPEGServer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--width", type=int, default=1280, help="配信する幅（スマホ配信の一般的な 720p 相当）")
    ap.add_argument("--loops", type=int, default=0, help="繰り返し回数（0 で無限）")
    args = ap.parse_args()

    srv = MJPEGServer(args.port, quality=85, path="/video")
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    loops = 0
    t0, n = time.time(), 0
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                loops += 1
                if args.loops and loops >= args.loops:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            if f.shape[1] > args.width:
                f = cv2.resize(f, (args.width, int(f.shape[0] * args.width / f.shape[1])), interpolation=cv2.INTER_AREA)
            srv.update(f)
            n += 1
            wait = t0 + n / fps - time.time()
            if wait > 0:
                time.sleep(wait)
    except KeyboardInterrupt:
        pass
    srv.close()


if __name__ == "__main__":
    main()
