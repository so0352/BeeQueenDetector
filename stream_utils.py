"""映像入力（カメラ番号 / URL / 動画ファイル）と表示（ウィンドウ / ブラウザ MJPEG）の共通部品。"""
import glob
import http.server
import os
import socketserver
import threading
import time


import cv2
import numpy as np

# スマホ配信アプリのプリセット（URL は各アプリの画面表示に従って --source で直接指定してもよい）
PRESETS = {
    "ipwebcam": "http://{ip}:8080/video",   # Android「IP Webcam」
    "droidcam": "http://{ip}:4747/video",   # DroidCam（Android / iPhone）
}


def default_device():
    """推論・学習に使うデバイス。NVIDIA GPU → "0"、Apple Silicon → "mps"、それ以外 → "cpu"。"""
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def use_half(device):
    """FP16 推論は CUDA のときだけ使う（MPS / CPU では使わない）。"""
    return str(device) not in ("cpu", "mps")


def resolve_source(source, preset=None):
    """--source と --app から OpenCV に渡す入力を決める。"""
    if preset:
        if not source:
            raise SystemExit("--app を使うときは --source にスマホの IP アドレスを指定してください")
        return PRESETS[preset].format(ip=source)
    if source is None:
        raise SystemExit("--source を指定してください（カメラ番号 / URL / 動画ファイル）")
    if str(source).isdigit():
        return int(source)
    return source


def is_live(src):
    return isinstance(src, int) or str(src).startswith(("http://", "https://", "rtsp://", "rtmp://", "udp://"))


class FrameSource:
    """ライブ映像は別スレッドで読み続け、最新の 1 枚だけを保持する（遅延を溜めない）。
    動画ファイルは全フレームを順に返す（--realtime でファイルも実時間で間引く）。"""

    def __init__(self, src, realtime_file=False, reconnect_sec=2.0):
        self.src = src
        self.live = is_live(src) or realtime_file
        self.reconnect_sec = reconnect_sec
        self.cap = self._open()
        self.fps_src = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame = None
        self.frame_id = 0
        self.ended = False
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        if self.live:
            self.th = threading.Thread(target=self._loop, daemon=True)
            self.th.start()

    def _open(self):
        cap = cv2.VideoCapture(self.src)
        if isinstance(self.src, int):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        file_mode = not is_live(self.src)
        t0 = time.time()
        n = 0
        while not self.ended:
            ok, f = self.cap.read()
            if not ok:
                if file_mode:
                    self.ended = True
                    break
                print(f"[入力] 映像が途切れました。{self.reconnect_sec}秒後に再接続します: {self.src}")
                self.cap.release()
                time.sleep(self.reconnect_sec)
                self.cap = self._open()
                continue
            with self.cond:
                self.frame = f
                self.frame_id += 1
                self.cond.notify_all()
            if file_mode:  # ファイルを実時間で再生
                n += 1
                wait = t0 + n / self.fps_src - time.time()
                if wait > 0:
                    time.sleep(wait)
        with self.cond:
            self.cond.notify_all()

    def read(self, last_id=0, timeout=5.0):
        """(frame_id, frame) を返す。終了時は (None, None)。"""
        if not self.live:
            ok, f = self.cap.read()
            if not ok:
                self.ended = True
                return None, None
            self.frame_id += 1
            return self.frame_id, f
        with self.cond:
            end = time.time() + timeout
            while self.frame_id == last_id and not self.ended:
                rem = end - time.time()
                if rem <= 0:
                    return last_id, None  # タイムアウト（再接続待ち）
                self.cond.wait(rem)
            if self.ended and self.frame_id == last_id:
                return None, None
            return self.frame_id, self.frame

    def release(self):
        self.ended = True
        self.cap.release()


class MJPEGServer:
    """注釈済みフレームをブラウザに MJPEG で配信する（cv2.imshow が使えない場合の代替）。
    WSL2 の場合、Windows のブラウザから http://localhost:<port>/ で見られる。"""

    def __init__(self, port=8090, quality=80, path="/stream"):
        self.quality = quality
        stream_path = path
        self.jpg = None
        self.cond = threading.Condition()
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/":
                    body = (b"<html><head><meta charset='utf-8'><title>findBeeQueen</title></head>"
                            b"<body style='margin:0;background:#111'>"
                            b"<img src='" + stream_path.encode() + b"' style='width:100%;height:100vh;object-fit:contain'></body></html>")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path.split("?")[0] != stream_path:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = None
                try:
                    while True:
                        with server.cond:
                            server.cond.wait_for(lambda: server.jpg is not last, timeout=5)
                            jpg = last = server.jpg
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        class TServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.httpd = TServer(("0.0.0.0", port), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        print(f"[配信] http://localhost:{port}/ （映像: http://localhost:{port}{stream_path}）")

    def update(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self.cond:
                self.jpg = buf.tobytes()
                self.cond.notify_all()

    def close(self):
        self.httpd.shutdown()


class TextDrawer:
    """日本語フォントがあれば PIL で日本語を描画、なければ英語で cv2.putText。"""

    FONT_GLOBS = [
        "/usr/share/fonts/**/NotoSansCJK*-Bold.ttc", "/usr/share/fonts/**/NotoSansCJK*.ttc",
        "/usr/share/fonts/**/NotoSansJP*.?tf", "/usr/share/fonts/**/ipag*.ttf", "/usr/share/fonts/**/IPA*.ttf",
        "/mnt/c/Windows/Fonts/meiryob.ttc", "/mnt/c/Windows/Fonts/meiryo.ttc", "/mnt/c/Windows/Fonts/YuGothB.ttc",
        "/mnt/c/Windows/Fonts/msgothic.ttc",
    ]

    def __init__(self):
        self.font_path = None
        for g in self.FONT_GLOBS:
            hits = sorted(glob.glob(g, recursive=True))
            if hits and os.path.exists(hits[0]):
                self.font_path = hits[0]
                break
        self._cache = {}

    @property
    def japanese(self):
        return self.font_path is not None

    def put(self, img, text_ja, text_en, org, size=28, color=(255, 255, 255), bg=(0, 0, 0)):
        """org は左上座標。color/bg は BGR。"""
        if self.font_path is None:
            scale = size / 30
            (w, h), base = cv2.getTextSize(text_en, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            x, y = org
            if bg is not None:
                cv2.rectangle(img, (x, y), (x + w + 10, y + h + base + 10), bg, -1)
            cv2.putText(img, text_en, (x + 5, y + h + 5), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
            return img
        from PIL import Image, ImageDraw, ImageFont
        font = self._cache.get(size)
        if font is None:
            font = self._cache[size] = ImageFont.truetype(self.font_path, size)
        x0, y0, x1, y1 = font.getbbox(text_ja)
        w, h = x1 - x0 + 12, y1 - y0 + 12
        x, y = org
        H, W = img.shape[:2]
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x or y2 <= y:
            return img
        roi = img[y:y2, x:x2]
        pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        if bg is not None:
            pil.paste(tuple(int(c) for c in bg[::-1]), (0, 0, pil.width, pil.height))
        ImageDraw.Draw(pil).text((6 - x0, 6 - y0), text_ja, font=font, fill=tuple(int(c) for c in color[::-1]))
        img[y:y2, x:x2] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        return img
