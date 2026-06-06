"""
Simulation roi/ 画像に対して mask_core → fitEllipse を実行し、
既知の D 値・フィット角度・ratio を重ねて保存。

Run:
  python experiments/sim_mask_core_fit.py
  python experiments/sim_mask_core_fit.py --groups p20 p30 p40 --run_name sim_run01
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from src.ellipse.adaptdog import (
    red_channel, stretch_to_255,
    _otsu_mask, _pick_central_blob, _fit_ellipse_on_mask, _estimate_minor,
)

SIM_RUN_DIR = Path("data/processed/simulation_runs")


def parse_D(stem: str) -> float | None:
    m = re.search(r"_D(m?p?)(\d+)", stem)
    if not m:
        return None
    s, v = m.group(1), int(m.group(2)) / 100.0
    if s == "m": return -v
    if s == "p": return  v
    return 0.0


def mask_core_fit(red_str: np.ndarray):
    """dilation なし: mask_core → fitEllipse"""
    minor_est = _estimate_minor(red_str)
    sigma_l   = max(8.0, minor_est * 0.75)
    blur_s = cv2.GaussianBlur(red_str.astype(np.float32), (0, 0), 1.5)
    blur_l = cv2.GaussianBlur(red_str.astype(np.float32), (0, 0), sigma_l)
    dog    = stretch_to_255(np.clip(blur_s - blur_l, 0, None))
    mask_raw  = _otsu_mask(dog)
    mask_core = _pick_central_blob(mask_raw)
    e = _fit_ellipse_on_mask(mask_core)
    return mask_core, e


def process_group(group: str, run_name: str):
    roi_dir = SIM_RUN_DIR / run_name / group / "roi"
    out_dir = SIM_RUN_DIR / run_name / group / "mask_core_fit"
    out_dir.mkdir(exist_ok=True)

    files = sorted(roi_dir.glob("*.png"))
    print(f"  [{group}] {len(files)} images")

    for roi_path in files:
        stem = roi_path.stem  # e.g. camera_p30_Dm300_roi
        d_val = parse_D(stem)
        if d_val is None:
            continue

        roi_bgr = cv2.imread(str(roi_path))
        if roi_bgr is None:
            continue

        red_str  = stretch_to_255(red_channel(roi_bgr))
        mask_core, e = mask_core_fit(red_str)

        # mask_core をBGRに変換してオーバーレイ
        overlay = cv2.cvtColor(mask_core, cv2.COLOR_GRAY2BGR)

        if e:
            cv2.ellipse(overlay, e['raw'], (0, 255, 80), 2)
            cv2.drawMarker(overlay,
                           (int(e['cx']), int(e['cy'])),
                           (0, 255, 80), cv2.MARKER_CROSS, 10, 1)
            line1 = f"D={d_val:+.2f}D  angle={e['angle']:.1f}deg"
            line2 = f"maj={e['major']:.1f}  min={e['minor']:.1f}  r={e['minor']/e['major']:.3f}"
        else:
            line1 = f"D={d_val:+.2f}D  [no fit]"
            line2 = ""

        h = overlay.shape[0]
        cv2.putText(overlay, line1, (4, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, line2, (4, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 80),  1, cv2.LINE_AA)

        out_name = stem.replace("_roi", "") + "_core_fit.png"
        cv2.imwrite(str(out_dir / out_name), overlay)

    print(f"    -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="sim_run01")
    parser.add_argument("--groups",   nargs="+", default=["p20", "p30", "p40"])
    args = parser.parse_args()

    print(f"Run: {args.run_name}  groups: {args.groups}")
    for g in args.groups:
        process_group(g, args.run_name)
    print("Done.")


if __name__ == "__main__":
    main()
