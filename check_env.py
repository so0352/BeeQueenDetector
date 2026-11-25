"""環境確認: torch / CUDA・MPS / OpenCV GUI(cv2.imshow) / ultralytics の動作確認。

使い方:
    python check_env.py            # 全チェック（imshow はウィンドウを 2 秒表示）
    python check_env.py --no-gui   # GUI チェックを省略
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gui", action="store_true", help="cv2.imshow のチェックを省略")
    args = ap.parse_args()
    ok = True

    print(f"Python: {sys.version.split()[0]}")

    import torch
    print(f"torch: {torch.__version__} (CUDA build: {torch.version.cuda})")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        _ = (x @ x).sum().item()
        print(f"CUDA: OK  -> {name} ({mem:.1f} GB), FP16 行列積も成功")
    elif torch.backends.mps.is_available():
        x = torch.randn(2048, 2048, device="mps")
        _ = (x @ x).sum().item()
        print("MPS: OK  -> Apple Silicon の GPU が使えます（CUDA なし。学習は NVIDIA GPU より遅くなります）")
    else:
        print("GPU: NG  -> CUDA も MPS も使えません（CPU で動作）")
        ok = False

    import cv2
    print(f"OpenCV: {cv2.__version__}")
    build = cv2.getBuildInformation()
    gui_line = next((l.strip() for l in build.splitlines() if l.strip().startswith("GUI:")), "GUI: 不明")
    print(f"  {gui_line}")
    try:
        from importlib.metadata import distributions
        cv_pkgs = sorted({d.metadata["Name"] for d in distributions() if "opencv" in (d.metadata["Name"] or "").lower()})
        print(f"  入っている OpenCV パッケージ: {cv_pkgs}")
        if any("headless" in p for p in cv_pkgs):
            print("  警告: headless 版が入っています。imshow が使えない場合は\n"
                  "        pip uninstall -y opencv-python-headless && pip install --force-reinstall opencv-python")
    except Exception:
        pass

    if not args.no_gui:
        print(f"  DISPLAY={os.environ.get('DISPLAY')}  WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY')}")
        try:
            import numpy as np
            img = np.zeros((240, 480, 3), np.uint8)
            cv2.putText(img, "findBeeQueen: imshow OK", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("check_env", img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            print("cv2.imshow: OK（ウィンドウが表示されたか目視で確認してください）")
        except Exception as e:
            print(f"cv2.imshow: NG -> {e}\n  → detect_stream.py の --web オプション（ブラウザ表示）を使ってください")
            ok = False

    import ultralytics
    print(f"ultralytics: {ultralytics.__version__}")

    print("\n結果:", "すべて OK" if ok else "一部 NG（上記参照）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
