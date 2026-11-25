"""同じ LAN 上のスマホ配信アプリ（IP Webcam: 8080 / DroidCam: 4747）を自動で探す。

見つかったら「アプリ名 IP」を 1 行だけ標準出力に出す（start.sh が使う）。前回見つけた IP を
.phone に記録し、次回はまずそこを確認するので速い。

使い方:
  python tools/find_phone.py                 # 自動検索
  python tools/find_phone.py --ip 192.168.0.23   # この IP でどちらのアプリかだけ判定
"""
import argparse
import ipaddress
import re
import shutil
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".phone"
PORTS = {"ipwebcam": 8080, "droidcam": 4747}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def lan_networks():
    """スキャン対象の LAN（/24 まで）と自分の IP の一覧を返す。"""
    nets, own = [], set()
    # WSL2（NAT モード）では Linux 側から LAN が見えないので Windows の ipconfig から取る
    if shutil.which("ipconfig.exe"):
        try:
            out = subprocess.run(["ipconfig.exe"], capture_output=True, timeout=10).stdout.decode("cp932", "replace")
            for block in re.split(r"\r?\n(?=\S)", out):
                ip = re.search(r"IPv4[^:]*:\s*([\d.]+)", block)
                mask = re.search(r"(?:Subnet Mask|サブネット マスク)[^:]*:\s*([\d.]+)", block)
                gw = re.search(r"(?:Default Gateway|デフォルト ゲートウェイ)[^:]*:\s*([\d.]+)", block)
                if ip and mask and gw:  # ゲートウェイがある = 実際の LAN（WSL 用・Tailscale 等を除外）
                    own.add(ip.group(1))
                    nets.append(ipaddress.ip_interface(f"{ip.group(1)}/{mask.group(1)}"))
        except Exception as e:  # noqa: BLE001
            log(f"[検索] ipconfig.exe の読み取りに失敗: {e}")
    if not nets and sys.platform == "darwin":  # macOS: 既定経路のインターフェースの IP とネットマスク
        try:
            route = subprocess.run(["route", "-n", "get", "default"], capture_output=True, text=True).stdout
            dev = re.search(r"interface:\s*(\S+)", route)
            if dev:
                cfg = subprocess.run(["ifconfig", dev.group(1)], capture_output=True, text=True).stdout
                for m in re.finditer(r"inet ([\d.]+) netmask (0x[0-9a-f]+)", cfg):
                    mask = str(ipaddress.IPv4Address(int(m.group(2), 16)))
                    own.add(m.group(1))
                    nets.append(ipaddress.ip_interface(f"{m.group(1)}/{mask}"))
        except Exception as e:  # noqa: BLE001
            log(f"[検索] macOS のネットワーク情報の読み取りに失敗: {e}")
    if not nets and shutil.which("ip"):  # 通常の Linux / WSL ミラーモード
        out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True).stdout
        dev = re.search(r"default via \S+ dev (\S+)",
                        subprocess.run(["ip", "-4", "route"], capture_output=True, text=True).stdout)
        for m in re.finditer(r"\d+: (\S+)\s+inet ([\d./]+)", out):
            if dev and m.group(1) == dev.group(1):
                itf = ipaddress.ip_interface(m.group(2))
                own.add(str(itf.ip))
                nets.append(itf)
    result = []
    for itf in nets:
        net = itf.network if itf.network.prefixlen >= 24 else ipaddress.ip_interface(f"{itf.ip}/24").network
        if net not in result:
            result.append(net)
    return result, own


def port_open(ip, port, timeout):
    try:
        with socket.create_connection((str(ip), port), timeout=timeout):
            return True
    except OSError:
        return False


def is_video(ip, port, timeout=3.0):
    """/video が MJPEG を返すか確認（8080 は他の機器でもよく使われるため）。"""
    try:
        with socket.create_connection((str(ip), port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(f"GET /video HTTP/1.0\r\nHost: {ip}\r\n\r\n".encode())
            head = b""
            while b"\r\n\r\n" not in head and len(head) < 8192:
                chunk = s.recv(1024)
                if not chunk:
                    break
                head += chunk
    except OSError:
        return False
    h = head.decode("latin-1").lower()
    if "busy" in h:  # DroidCam は同時に 1 台しか接続できない
        log(f"[検索] {ip}:{port} は他の機器と接続中です（DroidCam の PC アプリやブラウザを閉じてください）")
    return h.startswith("http/") and " 200" in h.split("\r\n", 1)[0] and ("multipart" in h or "image/jpeg" in h)


def check_ip(ip, timeout=1.0):
    for app, port in PORTS.items():
        if port_open(ip, port, timeout) and is_video(ip, port):
            return app
    return None


def scan(nets, own, timeout=0.6):
    hosts = [(ip, port) for net in nets for ip in net.hosts() if str(ip) not in own for port in PORTS.values()]
    log(f"[検索] {', '.join(map(str, nets))} の {len(hosts) // len(PORTS)} 台を確認中…")
    with ThreadPoolExecutor(256) as ex:
        opened = [h for h, ok in zip(hosts, ex.map(lambda h: port_open(*h, timeout), hosts)) if ok]
    for ip, port in opened:
        if is_video(ip, port):
            return next(a for a, p in PORTS.items() if p == port), str(ip)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", help="IP を指定（アプリの種類だけ判定）")
    ap.add_argument("--no-cache", action="store_true", help="前回の IP を使わない")
    args = ap.parse_args()

    if args.ip:
        app = check_ip(args.ip, timeout=2.0)
        if not app:
            log(f"[検索] {args.ip} で配信アプリ（8080 / 4747 の /video）が見つかりません")
            return 1
        found = (app, args.ip)
    else:
        found = None
        if not args.no_cache and CACHE.exists():
            parts = CACHE.read_text().split()
            if len(parts) == 2:
                app = check_ip(parts[1])
                found = (app, parts[1]) if app else None
        if not found:
            nets, own = lan_networks()
            if not nets:
                log("[検索] LAN のアドレスが分かりませんでした")
                return 1
            found = scan(nets, own)
        if not found:
            return 1
    CACHE.write_text(f"{found[0]} {found[1]}\n")
    print(f"{found[0]} {found[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
