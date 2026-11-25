"""Roboflow Universe からデータセットを YOLOv8 形式で取得し data/raw/<name>/ に展開する。

roboflow パッケージは使わず（opencv-python-headless を引き込むため）、REST API を requests で直接叩く。

API（Roboflow Platform API, https://api.roboflow.com）:
  GET /{workspace}/{project}?api_key=...                    -> project 情報と versions 一覧
  GET /{workspace}/{project}/{version}/{format}?api_key=... -> 202: 書き出し生成中 {"ready": false, "progress": x}
                                                              200: {"export": {"link": "<zip URL>"}, ...}

使い方:
  export ROBOFLOW_API_KEY=xxxxx
  python download_datasets.py                 # datasets.txt の全件
  python download_datasets.py --list-only     # バージョン・ライセンス等の情報表示のみ
  python download_datasets.py nectar/bees-ytrmp-o8kmc@2
"""
import argparse
import io
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

API = "https://api.roboflow.com"
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"


def parse_spec(line):
    line = line.split("#", 1)[0].strip()
    if not line:
        return None
    version = None
    if "@" in line:
        line, version = line.split("@", 1)
        version = int(version)
    ws, proj = line.strip("/").split("/")
    return ws, proj, version


def get_json(url, params, retries=5):
    for i in range(retries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(5 * (i + 1))
            continue
        return r
    return r


def project_info(ws, proj, key):
    r = get_json(f"{API}/{ws}/{proj}", {"api_key": key})
    if r.status_code != 200:
        raise RuntimeError(f"{ws}/{proj}: プロジェクト情報の取得に失敗 HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def wait_export(ws, proj, version, fmt, key, timeout=900):
    url = f"{API}/{ws}/{proj}/{version}/{fmt}"
    t0 = time.time()
    while True:
        r = get_json(url, {"api_key": key, "nocache": "true"})
        if r.status_code == 200:
            data = r.json()
            if "export" in data and data["export"].get("link"):
                print()
                return data["export"]["link"]
            if data.get("ready") is False:
                pass
            else:
                raise RuntimeError(f"想定外のレスポンス: {json.dumps(data)[:300]}")
        elif r.status_code == 202:
            data = r.json()
        else:
            raise RuntimeError(f"書き出し失敗 HTTP {r.status_code}: {r.text[:300]}")
        prog = float(data.get("progress") or 0.0)
        print(f"\r  書き出し生成中... {prog * 100:5.1f}%", end="", flush=True)
        if time.time() - t0 > timeout:
            raise TimeoutError("書き出しの生成がタイムアウトしました")
        time.sleep(3)


def download_zip(link, dest: Path):
    r = requests.get(link, stream=True, timeout=300)
    r.raise_for_status()
    total = int(r.headers.get("content-length") or 0)
    buf = io.BytesIO()
    for chunk in r.iter_content(1 << 20):
        buf.write(chunk)
        if total:
            print(f"\r  ダウンロード {buf.tell() / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(buf) as z:
        z.extractall(dest)


def load_dotenv(path=ROOT / ".env"):
    """プロジェクト直下の .env（KEY=VALUE 形式）を環境変数に読み込む（既存の環境変数を優先）。"""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def write_credits():
    """取得済みデータセットの出典・ライセンス一覧 CREDITS.md を作る（CC BY は出典表示が必要）。"""
    rows = []
    for meta_path in sorted(RAW.glob("*/_source.json")):
        m = json.loads(meta_path.read_text())
        rows.append(f"| {m.get('name')} | [{m['workspace']}/{m['project']}]({m['url']}) | v{m['version']} | {m.get('license')} |")
    vids = ROOT / "data/test_videos/SOURCES.txt"
    text = ("# 使用データの出典\n\n学習に使った Roboflow Universe のデータセット（`download_datasets.py` が自動生成）。\n\n"
            "| 名前 | 出典 | バージョン | ライセンス |\n|---|---|---|---|\n" + "\n".join(rows) + "\n")
    if vids.exists():
        text += "\n## 動作確認用の動画（Wikimedia Commons）\n\n```\n" + vids.read_text() + "```\n"
    (ROOT / "CREDITS.md").write_text(text, encoding="utf-8")
    print(f"出典一覧: {ROOT / 'CREDITS.md'}")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="*", help="workspace/project[@version]（省略時は datasets.txt）")
    ap.add_argument("--list", default=str(ROOT / "datasets.txt"))
    ap.add_argument("--format", default="yolov8")
    ap.add_argument("--list-only", action="store_true", help="情報表示のみでダウンロードしない")
    ap.add_argument("--force", action="store_true", help="既に取得済みでも再取得")
    args = ap.parse_args()

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("環境変数 ROBOFLOW_API_KEY を設定してください（Roboflow の Settings → API Keys → Private API Key）")

    lines = args.specs or Path(args.list).read_text(encoding="utf-8").splitlines()
    specs = [s for s in map(parse_spec, lines) if s]
    summary = []
    for ws, proj, version in specs:
        name = f"{ws}__{proj}"
        print(f"=== {ws}/{proj}")
        try:
            info = project_info(ws, proj, key)
        except Exception as e:
            print(f"  エラー: {e}")
            summary.append((name, "ERROR", str(e)[:80]))
            continue
        p = info.get("project", {})
        versions = info.get("versions", [])
        vnums = sorted(int(str(v["id"]).rsplit("/", 1)[-1]) for v in versions)
        print(f"  名前: {p.get('name')}  画像数: {p.get('images')}  ライセンス: {p.get('license')}")
        print(f"  クラス: {p.get('classes')}")
        print(f"  バージョン: {vnums}")
        if not vnums:
            summary.append((name, "NO VERSION", ""))
            continue
        v = version or vnums[-1]
        vinfo = next((x for x in versions if str(x["id"]).endswith(f"/{v}")), {})
        print(f"  使用バージョン: {v}  (画像 {vinfo.get('images')} 枚, splits={vinfo.get('splits')})")
        if args.list_only:
            summary.append((name, f"v{v}", p.get("license")))
            continue

        dest = RAW / name
        meta_path = dest / "_source.json"
        if dest.exists() and meta_path.exists() and not args.force:
            old = json.loads(meta_path.read_text())
            if old.get("version") == v:
                print("  取得済み（--force で再取得）")
                summary.append((name, f"v{v}", "skip"))
                continue
        try:
            link = wait_export(ws, proj, v, args.format, key)
            download_zip(link, dest)
        except Exception as e:
            print(f"  エラー: {e}")
            summary.append((name, "ERROR", str(e)[:80]))
            continue
        meta = {
            "workspace": ws, "project": proj, "version": v, "format": args.format,
            "license": p.get("license"), "name": p.get("name"), "classes": p.get("classes"),
            "url": f"https://universe.roboflow.com/{ws}/{proj}/dataset/{v}",
            "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        n = sum(1 for _ in dest.rglob("*.jpg")) + sum(1 for _ in dest.rglob("*.png"))
        print(f"  展開完了: {dest}  (画像 {n} 枚)")
        summary.append((name, f"v{v}", p.get("license")))

    write_credits()
    print("\n=== まとめ")
    for row in summary:
        print("  " + " | ".join(str(x) for x in row))


if __name__ == "__main__":
    main()
