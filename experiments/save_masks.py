"""
AdaptDoG の中間マスクを保存する。

保存内容 (mask_debug/ フォルダ):
  *_dog.png        DoG フィルタ後
  *_mask_raw.png   Otsu 二値化直後
  *_mask_core.png  pick_central_blob 後
  *_mask_final.png dilation/close 後（fitEllipse の入力）

Run:
  python experiments/save_masks.py --patient aiswarya_RIGHT
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.ellipse.adaptdog import (
    red_channel, stretch_to_255,
    _otsu_mask, _pick_central_blob, _fit_ellipse_on_mask, _estimate_minor,
)

RUN_DIR = Path("data/processed/pipeline_runs/repeatability_0603_sim_ratio")


def process(patient: str):
    roi_dir   = RUN_DIR / patient / "roi"
    out_dir   = RUN_DIR / patient / "mask_debug"
    out_dir.mkdir(exist_ok=True)

    for roi_path in sorted(roi_dir.glob("*.png")):
        stem = roi_path.stem.replace("_roi", "")
        roi_bgr = cv2.imread(str(roi_path))
        if roi_bgr is None:
            continue

        red_str = stretch_to_255(red_channel(roi_bgr))

        # ── 中間マスク再現 ─────────────────────────────────────────────
        minor_est = _estimate_minor(red_str)
        sigma_l   = max(8.0, minor_est * 0.75)

        blur_s = cv2.GaussianBlur(red_str.astype(np.float32), (0, 0), 1.5)
        blur_l = cv2.GaussianBlur(red_str.astype(np.float32), (0, 0), sigma_l)
        dog    = stretch_to_255(np.clip(blur_s - blur_l, 0, None))

        mask_raw  = _otsu_mask(dog)
        mask_core = _pick_central_blob(mask_raw)

        e_core     = _fit_ellipse_on_mask(mask_core)
        core_ratio = e_core['ratio'] if e_core else 0.0
        core_angle = e_core['angle'] if e_core else 90.0

        if core_ratio < 0.20:
            dil_w = max(3,  int(minor_est * 0.33))
            dil_h = max(15, int(minor_est * 1.20))
            if core_angle < 45 or core_angle > 135:
                dil_w, dil_h = dil_h, dil_w
            mask_final = cv2.dilate(
                mask_core,
                cv2.getStructuringElement(cv2.MORPH_RECT, (dil_w, dil_h)))
        else:
            close_k = max(5, int(minor_est * 0.20)) | 1
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
            mask_final = cv2.morphologyEx(mask_core, cv2.MORPH_CLOSE, k)

        # ── mask_core に fitEllipse を直接重ねる ──────────────────────
        e_core_fit = _fit_ellipse_on_mask(mask_core)
        core_overlay = cv2.cvtColor(mask_core, cv2.COLOR_GRAY2BGR)
        if e_core_fit:
            cv2.ellipse(core_overlay, e_core_fit['raw'], (0, 255, 80), 2)
            cv2.drawMarker(core_overlay,
                           (int(e_core_fit['cx']), int(e_core_fit['cy'])),
                           (0, 255, 80), cv2.MARKER_CROSS, 10, 1)
            label = (f"maj={e_core_fit['major']:.1f} "
                     f"min={e_core_fit['minor']:.1f} "
                     f"r={e_core_fit['minor']/e_core_fit['major']:.3f}")
            cv2.putText(core_overlay, label, (4, core_overlay.shape[0] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 80), 1, cv2.LINE_AA)

        # ── 保存 ──────────────────────────────────────────────────────
        for subdir, img in [
            ("1_dog",              dog),
            ("2_mask_raw",         mask_raw),
            ("3_mask_core",        mask_core),
            ("4_mask_core_fit",    core_overlay),
            ("5_mask_final",       mask_final),
        ]:
            (out_dir / subdir).mkdir(exist_ok=True)
            cv2.imwrite(str(out_dir / subdir / f"{stem}.png"), img)

    print(f"Done: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="aiswarya_RIGHT")
    args = parser.parse_args()
    process(args.patient)


if __name__ == "__main__":
    main()
