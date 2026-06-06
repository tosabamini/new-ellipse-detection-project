"""
red_roi/ 画像に楕円オーバーレイを重ねて ellipse_red_roi/ に保存。

per_image.csv に cx/cy が保存されていないため、
roi/ 画像に AdaptDoG を再実行して楕円パラメータを取得し、
red_roi/ のグレースケール画像に描画する。

Run:
  python experiments/make_ellipse_red_roi.py
  python experiments/make_ellipse_red_roi.py --run_name repeatability_0603_sim_ratio
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src.ellipse.adaptdog import red_channel, stretch_to_255, run_adaptive_dog

RUN_DIR = Path("data/processed/pipeline_runs")


def draw_ellipse_on_gray(gray: np.ndarray, e: dict,
                         color=(0, 255, 80), thickness=2) -> np.ndarray:
    """グレースケール画像をBGRに変換して楕円を描画"""
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if e is None:
        return out
    cx, cy   = int(e["cx"]), int(e["cy"])
    axes     = (max(1, int(e["major"] / 2)), max(1, int(e["minor"] / 2)))
    angle    = e["angle"]
    cv2.ellipse(out, (cx, cy), axes, angle, 0, 360, color, thickness)
    cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 10, 1)
    # major / minor / ratio テキスト
    label = f"maj={e['major']:.1f} min={e['minor']:.1f} r={e['minor']/e['major']:.3f}"
    cv2.putText(out, label, (4, out.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return out


def process_patient(patient_dir: Path) -> None:
    roi_dir     = patient_dir / "roi"
    red_roi_dir = patient_dir / "red_roi"
    out_dir     = patient_dir / "ellipse_red_roi"

    if not roi_dir.exists() or not red_roi_dir.exists():
        print(f"  [SKIP] {patient_dir.name}: roi/ or red_roi/ missing")
        return

    out_dir.mkdir(exist_ok=True)
    n_ok, n_fail = 0, 0

    for roi_path in sorted(roi_dir.glob("*.png")):
        stem = roi_path.stem.replace("_roi", "")

        # red_roi 画像を読み込む
        rr_path = red_roi_dir / f"{stem}_red_roi.png"
        if not rr_path.exists():
            continue
        gray = cv2.imread(str(rr_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        # roi から AdaptDoG で楕円を取得
        roi_bgr = cv2.imread(str(roi_path))
        if roi_bgr is None:
            continue
        red_roi_str = stretch_to_255(red_channel(roi_bgr))
        e = run_adaptive_dog(red_roi_str)

        overlay = draw_ellipse_on_gray(gray, e,
                                       color=(0, 255, 80) if e else (0, 80, 255))
        out_path = out_dir / f"{stem}_ellipse_red_roi.png"
        cv2.imwrite(str(out_path), overlay)

        if e:
            n_ok += 1
        else:
            n_fail += 1

    print(f"  [{patient_dir.name}] {n_ok} ok, {n_fail} no-fit  -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="repeatability_0603_sim_ratio")
    args = parser.parse_args()

    run_dir = RUN_DIR / args.run_name
    patient_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir()])

    print(f"Run: {args.run_name}  ({len(patient_dirs)} patients)")
    for pd in patient_dirs:
        process_patient(pd)
    print("Done.")


if __name__ == "__main__":
    main()
