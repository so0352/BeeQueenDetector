"""複数の YOLO 形式データセットを 1 クラス（queen）に統合して data/merged/ を作る。

入力:
  data/raw/<name>/      download_datasets.py で取得したもの（Roboflow YOLOv8 形式）
  data/custom/<name>/   自前で注釈したデータ（同じ YOLO 形式: <split>/images, <split>/labels, data.yaml）

処理:
  - data.yaml の names のうち女王蜂のクラス（is_queen_class: 'queen'/'reina' などを含み、'non-queen'・王台などを除く）
    → クラス 0 'queen'。他は捨てる。
  - ポリゴン形式のラベル行は外接矩形に変換。
  - 画像名に <データセット名>__ のプレフィックスを付けて衝突回避。
  - 回転・反転に強い知覚ハッシュ(pHash × 8 通りの向き)で重複画像を検出（GPU があれば GPU で計算）。
      * val/test は似た場面を 1 枚にする（評価の水増し防止）。
      * val/test の画像に似た train 画像は捨てる（train と val/test のリーク防止）。
      * ほぼ同一の画像で「女王蜂あり」と「なし」が食い違う場合は「なし」側を捨てる（付け忘れ対策）。
      * train では同じ向きのほぼ同一画像のみ除く（回転コピーは学習用の拡張として残す）。
  - --train-only のデータセットは全分割を train に入れる（評価には使わない）。
  - 女王蜂のいない画像は背景（負例）として残す（train は --neg-ratio で上限）。

使い方:
  python merge_datasets.py
  python merge_datasets.py --exclude --out data/merged_all   # 除外なしで全データ
"""
import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}
# 既定で使わないデータセット。queenbee-detector（強調した切り抜き画像）は入れると test の mAP50 が
# 0.919 → 0.891、動画での検出率も下がった（2026-11-18 の比較、runs/queen/v3_all_y11s と v3_nc_y11s）。
DEFAULT_EXCLUDE = ["abullrahman-dawshi__queenbee-detector"]

# train にだけ使い、評価(val/test)には使わないデータセット。評価はスマホで巣板を撮った実映像に近いものに限る。
DEFAULT_TRAIN_ONLY = [
    "abullrahman-dawshi__queenbee-detector",          # 拡大・強調した切り抜き画像が中心
    "sbl2022__queen-local",                           # 固定カメラ・白黒の観察巣箱（ほぼ同じ場面）
    "roberto-saif-al-abarb-sciamanna__queen-worker",  # 以下、女王蜂が画面の半分以上を占める接写中心
    "fateme228498gmailcom__bee-x0xxu",
    "machine-learning-and-ai-umw0l__bee-detection-omvbs",
    "red-hawk-pcduc__bee-detector-f3ivv",
    "queen-bee-detection-by-arpan__queen-vw0wv",
    "rocket-ts144__queen-bee-c9kzm",
    "apicultura__reina",                              # 128px の低解像度
]


QUEEN_WORDS = ("queen", "reina", "königin", "koenigin", "matka", "regina", "reine")
NOT_QUEEN_WORDS = ("non", "not", "no_", "no-", "cell", "cup", "larva", "worker", "drone", "mite", "anne")


def is_queen_class(name: str) -> bool:
    """クラス名が女王蜂か。'non-queen'・'queen cell'（王台）・複合クラスなどは除く。"""
    n = name.lower()
    return any(w in n for w in QUEEN_WORDS) and not any(w in n for w in NOT_QUEEN_WORDS)


def read_names(ds_dir: Path):
    for cand in [ds_dir / "data.yaml", *ds_dir.glob("*.yaml")]:
        if cand.exists():
            y = yaml.safe_load(cand.read_text(encoding="utf-8"))
            names = y.get("names")
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            if isinstance(names, list):
                return {i: str(n) for i, n in enumerate(names)}
    return None


def convert_label_file(path: Path, queen_ids):
    """queen ボックスのリスト [(cx, cy, w, h)] を返す（正規化座標）。"""
    boxes = []
    if not path.exists():
        return boxes
    for parts in (ln.split() for ln in path.read_text().splitlines() if ln.strip()):
        if int(float(parts[0])) not in queen_ids:
            continue
        vals = list(map(float, parts[1:]))
        if len(vals) == 4:
            cx, cy, w, h = vals
        elif len(vals) >= 6 and len(vals) % 2 == 0:  # ポリゴン → 外接矩形
            xs, ys = vals[0::2], vals[1::2]
            cx, cy, w, h = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
        else:
            continue
        x0, y0 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x1, y1 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        if x1 - x0 > 1e-4 and y1 - y0 > 1e-4:
            boxes.append(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return boxes


# ---------------------------------------------------------------- 知覚ハッシュ（8 通りの向き）
_N = 32
_DCT = np.cos(np.pi * np.outer(np.arange(_N), 2 * np.arange(_N) + 1) / (2 * _N))  # DCT-II 行列


def _phash_bits(a):
    low = (_DCT @ a @ _DCT.T)[:8, :8]
    return (low > np.median(low)).ravel()


def dihedral_phash(path):
    """(8, 64) bool: 元画像・回転・反転の 8 通りの pHash（imagehash.phash と同じ定義）。"""
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((_N, _N), Image.LANCZOS), dtype=np.float64)
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out.append(_phash_bits(r))
        out.append(_phash_bits(r[:, ::-1]))
    return np.stack(out)


def near_pairs(H, thresh, same_orient_only=False, chunk=1024):
    """H: (N, 8, 64) bool。距離 <= thresh のペア (i, j, d), i < j を返す。
    距離は「i の 8 通りの向き」と「j の元の向き」のハミング距離の最小値。"""
    n = len(H)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dt = torch.float16 if dev in ("cuda", "mps") else torch.float32
        V = torch.from_numpy(H).to(dev).to(dt) * 2 - 1           # ±1
        B = V[:, 0, :].contiguous()                                 # (N, 64)
        pairs = []
        for s in range(0, n, chunk):
            q = V[s:s + chunk, :1 if same_orient_only else 8, :]
            c, k, _ = q.shape
            ham = (64 - q.reshape(c * k, 64) @ B.T) / 2               # (c*k, N)
            ham = ham.reshape(c, k, n).amin(dim=1)                    # (c, N)
            ii = torch.arange(s, s + c, device=dev)[:, None]
            jj = torch.arange(n, device=dev)[None, :]
            mask = (ham <= thresh) & (jj > ii)
            idx = mask.nonzero()
            d = ham[idx[:, 0], idx[:, 1]]
            for (a, b), dd in zip(idx.tolist(), d.tolist()):
                pairs.append((s + a, b, int(round(dd))))
        print(f"    距離計算: {dev}")
        return pairs
    except ImportError:
        pass
    V = H.astype(np.float32) * 2 - 1
    B = V[:, 0, :]
    pairs = []
    for s in range(0, n, chunk):
        q = V[s:s + chunk, :1 if same_orient_only else 8, :]
        c, k, _ = q.shape
        ham = ((64 - q.reshape(c * k, 64) @ B.T) / 2).reshape(c, k, n).min(axis=1)
        for a, b in zip(*np.nonzero(ham <= thresh)):
            if b > s + a:
                pairs.append((s + a, int(b), int(round(ham[a, b]))))
    return pairs


def collect(ds_dirs, train_only):
    items = []
    for ds in ds_dirs:
        names = read_names(ds)
        if names is None:
            print(f"  [skip] {ds}: data.yaml が見つかりません")
            continue
        queen_ids = {i for i, n in names.items() if is_queen_class(n)}
        short = ds.name
        flag = "（train のみ）" if short in train_only else ""
        print(f"  {short}{flag}: names={list(names.values())} → queen: {[names[i] for i in queen_ids]}")
        for img_dir in sorted(ds.rglob("images")):
            if not img_dir.is_dir():
                continue
            split = SPLIT_ALIASES.get(img_dir.parent.name.lower(), "train")
            if short in train_only:
                split = "train"
            lbl_dir = img_dir.parent / "labels"
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                items.append({
                    "ds": short, "split": split, "img": img,
                    "boxes": convert_label_file(lbl_dir / (img.stem + ".txt"), queen_ids),
                    "out_name": f"{short}__{img.stem}{img.suffix.lower()}",
                    "eval_ok": short not in train_only,
                })
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="*", default=["data/raw", "data/custom"],
                    help="データセットを含む親ディレクトリ（各サブディレクトリが 1 データセット）")
    ap.add_argument("--out", default="data/merged")
    ap.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE,
                    help="使わないデータセット名（ディレクトリ名）。既定は queenbee-detector。全部使うには --exclude のみ指定")
    ap.add_argument("--train-only", nargs="*", default=DEFAULT_TRAIN_ONLY,
                    help="train にのみ使うデータセット名（既定: queenbee-detector）。空にするには --train-only のみ指定")
    ap.add_argument("--hash-thresh", type=int, default=6, help="pHash のハミング距離がこれ以下なら重複とみなす")
    ap.add_argument("--neg-ratio", type=float, default=1.0,
                    help="train の負例（女王蜂なし画像）の上限 = 正例数 × この値（負の値で無制限）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    rng = random.Random(args.seed)

    ds_dirs = []
    for parent in args.inputs:
        p = ROOT / parent
        if p.exists():
            ds_dirs += sorted(d for d in p.iterdir() if d.is_dir() and d.name not in args.exclude)
    if not ds_dirs:
        raise SystemExit("データセットが見つかりません。先に download_datasets.py を実行してください。")

    print("== 収集")
    items = collect(ds_dirs, set(args.train_only))
    print(f"  画像 {len(items)} 枚")

    print("== 重複検出（回転・反転を考慮した pHash）")
    with ProcessPoolExecutor(args.workers) as ex:
        H = np.stack(list(ex.map(dihedral_phash, [it["img"] for it in items], chunksize=64)))
    pairs = near_pairs(H, args.hash_thresh)
    print(f"  近似重複ペア（距離 <= {args.hash_thresh}）: {len(pairs)}")
    # 推移的にまとめると似た場面の連鎖で巨大グループができるため、直接の近傍だけを見る（貪欲法）
    nbr = defaultdict(set)
    for i, j, d in pairs:
        nbr[i].add(j)
        nbr[j].add(i)
    drop = Counter()

    def mark(k, why):
        if k not in dropped:
            dropped.add(k)
            drop[why] += 1

    dropped = set()
    # 1) ほぼ同一画像（距離 <= 2）で女王蜂あり/なしが食い違う → なし側を捨てる（付け忘れ対策）
    for i, j, d in pairs:
        if d <= 2 and bool(items[i]["boxes"]) != bool(items[j]["boxes"]):
            mark(j if items[i]["boxes"] else i, "label_conflict")
    # 2) 評価用(val→test の順): 既に残した評価画像に近いものは捨てる（1 場面 1 枚）
    kept_eval = set()
    for split in ("val", "test"):
        cand = [k for k in range(len(items)) if items[k]["split"] == split and k not in dropped]
        cand.sort(key=lambda k: (-len(items[k]["boxes"]), k))
        for k in cand:
            if nbr[k] & kept_eval:
                mark(k, f"{split}_dup")
            else:
                kept_eval.add(k)
    # 3) train: 評価画像に近いものは捨てる（リーク防止）
    for k in range(len(items)):
        if items[k]["split"] == "train" and k not in dropped and nbr[k] & kept_eval:
            mark(k, "train_leak")
    # 4) train: 同じ向きでほぼ同一（距離 <= 2）の重複は 1 枚に
    exact = near_pairs(H, 2, same_orient_only=True)
    for i, j, _ in exact:
        if i not in dropped and j not in dropped and items[i]["split"] == items[j]["split"] == "train":
            mark(j if len(items[i]["boxes"]) >= len(items[j]["boxes"]) else i, "train_exact_dup")
    cross = sum(1 for i, j, _ in pairs if items[i]["ds"] != items[j]["ds"])
    print(f"  データセット間の近似重複ペア: {cross}")
    print(f"  除外の内訳: {dict(drop)}  計 {len(dropped)} 枚")
    drop = dropped

    # train の負例を間引く
    keep = [i for i in range(len(items)) if i not in drop]
    if args.neg_ratio >= 0:
        tr_pos = [i for i in keep if items[i]["split"] == "train" and items[i]["boxes"]]
        tr_neg = [i for i in keep if items[i]["split"] == "train" and not items[i]["boxes"]]
        limit = int(len(tr_pos) * args.neg_ratio)
        if len(tr_neg) > limit:
            rng.shuffle(tr_neg)
            cut = set(tr_neg[limit:])
            keep = [i for i in keep if i not in cut]
            print(f"  train の負例を {len(tr_neg)} → {limit} 枚に間引き（--neg-ratio {args.neg_ratio}）")

    print("== 書き出し")
    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val", "test"):
        (out / s / "images").mkdir(parents=True)
        (out / s / "labels").mkdir(parents=True)
    stats = defaultdict(Counter)
    for i in keep:
        it = items[i]
        s = it["split"]
        dst = out / s / "images" / it["out_name"]
        try:
            os.link(it["img"], dst)
        except OSError:
            shutil.copy2(it["img"], dst)
        (out / s / "labels" / (Path(it["out_name"]).stem + ".txt")).write_text(
            "".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in it["boxes"]))
        for key in (s, f"{s}/{it['ds']}"):
            st = stats[key]
            st["images"] += 1
            st["boxes"] += len(it["boxes"])
            st["positive" if it["boxes"] else "negative"] += 1

    data_yaml = {"path": str(out), "train": "train/images", "val": "val/images"}
    if stats["test"]["images"]:
        data_yaml["test"] = "test/images"
    data_yaml["names"] = {0: "queen"}
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False))

    sizes = [max(w, h) for i in keep for (_, _, w, h) in items[i]["boxes"]]
    summary = {k: dict(v) for k, v in sorted(stats.items())}
    if sizes:
        summary["box_rel_size"] = {"p10": float(np.percentile(sizes, 10)), "median": float(np.median(sizes)),
                                   "p90": float(np.percentile(sizes, 90))}
    summary["args"] = vars(args)
    (out / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n== 集計")
    print(f"{'分割':<6} {'画像':>7} {'正例':>7} {'負例':>7} {'queen box':>10}")
    for s in ("train", "val", "test"):
        st = stats[s]
        print(f"{s:<6} {st['images']:>7} {st['positive']:>7} {st['negative']:>7} {st['boxes']:>10}")
    print("\n内訳（分割/データセット: 画像, 正例, box）")
    for k in sorted(k for k in stats if "/" in k):
        print(f"  {k}: {stats[k]['images']}, {stats[k]['positive']}, {stats[k]['boxes']}")
    if sizes:
        b = summary["box_rel_size"]
        print(f"\n女王蜂ボックスの大きさ（画像辺に対する比） p10={b['p10']:.3f} 中央値={b['median']:.3f} p90={b['p90']:.3f}")
    print(f"\n出力: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
