"""
PickUP フォルダの red_roi 画像に対して 4_mask_core_fit 手法で楕円フィッティングを行う。

手法 (save_masks.py / adaptdog.py の 4_mask_core_fit と同一):
  red_roi (グレースケール, RedEnhance済み)
    → DoG (σ_s=1.5, σ_l=minor_est*0.75)
    → Otsu → pick_central_blob (mask_core)
    → _fit_ellipse_on_mask(mask_core)  ← この楕円を採用

出力:
  各 PickUP/<subject>/<LEFT|RIGHT>/ellipse/  に overlay PNG
  各 PickUP/<subject>/<LEFT|RIGHT>/ellipse_results.csv  に maj/min/ratio/angle

Run:
  python experiments/pickup_mask_core_fit.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from src.ellipse.adaptdog import (
    stretch_to_255,
    _otsu_mask, _pick_central_blob,
    _fit_ellipse_on_mask, _estimate_minor,
)

PICKUP_DIR = Path("data/Repeatability/PickUP")


def fit_from_red_roi(gray: np.ndarray) -> dict | None:
    """red_roi グレースケール画像から mask_core に直接 fitEllipse する。"""
    minor_est = _estimate_minor(gray)
    sigma_l   = max(8.0, minor_est * 0.75)

    blur_s = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    blur_l = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma_l)
    dog    = stretch_to_255(np.clip(blur_s - blur_l, 0, None))

    mask_raw  = _otsu_mask(dog)
    mask_core = _pick_central_blob(mask_raw)

    return _fit_ellipse_on_mask(mask_core)


def draw_overlay(gray: np.ndarray, e: dict | None) -> np.ndarray:
    """グレースケール画像に楕円を描画して BGR で返す。"""
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if e is None:
        return out
    color = (0, 255, 80)
    cv2.ellipse(out, e["raw"], color, 2)
    cv2.drawMarker(out, (int(e["cx"]), int(e["cy"])), color,
                   cv2.MARKER_CROSS, 10, 1)
    label = (f"maj={e['major']:.1f} min={e['minor']:.1f} "
             f"r={e['minor']/e['major']:.3f}")
    cv2.putText(out, label, (4, out.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return out


def process_eye_dir(eye_dir: Path) -> None:
    pngs = sorted(eye_dir.glob("*.png"))
    if not pngs:
        return

    out_dir = eye_dir / "ellipse"
    out_dir.mkdir(exist_ok=True)

    rows = []
    n_ok, n_fail = 0, 0

    for png_path in pngs:
        gray = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        stem = png_path.stem

        e = fit_from_red_roi(gray)
        overlay = draw_overlay(gray, e)
        cv2.imwrite(str(out_dir / f"{stem}.png"), overlay)

        if e:
            rows.append({
                "stem":  stem,
                "major": f"{e['major']:.2f}",
                "minor": f"{e['minor']:.2f}",
                "ratio": f"{e['minor']/e['major']:.4f}",
                "angle": f"{e['angle']:.2f}",
            })
            n_ok += 1
        else:
            rows.append({"stem": stem, "major": "", "minor": "",
                         "ratio": "", "angle": ""})
            n_fail += 1

    csv_path = eye_dir / "ellipse_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "major", "minor", "ratio", "angle"])
        w.writeheader()
        w.writerows(rows)

    print(f"    {eye_dir.parent.name}/{eye_dir.name}: "
          f"{n_ok} ok, {n_fail} no-fit  → ellipse/ + ellipse_results.csv")


def main():
    subject_dirs = sorted([d for d in PICKUP_DIR.iterdir() if d.is_dir()])
    print(f"PickUP: {len(subject_dirs)} subject folders\n")

    for subj_dir in subject_dirs:
        print(f"  {subj_dir.name}")
        for eye in ("LEFT", "RIGHT"):
            eye_dir = subj_dir / eye
            if eye_dir.exists():
                process_eye_dir(eye_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
