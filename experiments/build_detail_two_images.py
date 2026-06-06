"""
Detail figure for two specific 101_RIGHT images.

For each image, shows:
  • Top:    5 stage thumbnails (Raw → Red → DoG → Mask → Ellipse)
  • Bottom: full numerical chain
            (major, minor, angle) → ratio
                                  → area_scaled = major·minor·SF²
                                  → p_est        (quadratic in p)
                                  → D1, D2       (quadratic in D)
                                  → adopted_D    (= D2)

One PNG per image, saved under experiments/explainer_v150526_101right/details/.

Run:
  python -m experiments.build_detail_two_images
"""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.ellipse.adaptdog import (
    red_channel, stretch_to_255,
    _otsu_mask, _pick_central_blob, _estimate_minor,
    _fit_ellipse_on_mask, draw_ellipse_overlay,
)
from src.preprocessing.preprocess_utils import center_crop
from src.analysis.pupil_estimator import (
    estimate_pupil, SCALE_FACTOR, S0, S1, S2, I0, I1, I2, P_MIN, P_MAX,
)
from src.analysis.build_patient_model import estimate_D_from_ratio_and_p, _abc


PROJECT_ROOT_MAIN = Path(r"C:\Users\issas\Desktop\new_ellipse_detection_project")

# (patient_label, raw_dir, output_dir, stems)
JOBS = [
    (
        "101_RIGHT",
        PROJECT_ROOT_MAIN / "data" / "raw" / "patient_data" / "101" / "RIGHT",
        PROJECT_ROOT_MAIN / "experiments" / "explainer_v150526_101right" / "details",
        [
            "abhilekh_3D_IMG_20260513_142018_436",
            "IMG_20260513_142017_635",
        ],
    ),
    (
        "103_LEFT",
        PROJECT_ROOT_MAIN / "data" / "raw" / "patient_data" / "103" / "LEFT",
        PROJECT_ROOT_MAIN / "experiments" / "explainer_v150526_103_LEFT" / "details",
        [
            "IMG_20260513_123924_083",
            "Rohith_3D_IMG_20260513_123924_932",
        ],
    ),
]


def adaptdog_with_intermediates(red_str):
    minor_est = _estimate_minor(red_str)
    sigma_l   = max(8.0, minor_est * 0.75)
    blur_s = cv2.GaussianBlur(red_str.astype(np.float32), (0, 0), 1.5)
    blur_l = cv2.GaussianBlur(red_str.astype(np.float32), (0, 0), sigma_l)
    dog    = stretch_to_255(np.clip(blur_s - blur_l, 0, None))
    mask_core = _pick_central_blob(_otsu_mask(dog))
    e_core    = _fit_ellipse_on_mask(mask_core)
    core_ratio = e_core["ratio"] if e_core else 0.0
    core_angle = e_core["angle"] if e_core else 90.0
    if core_ratio < 0.20:
        dil_w = max(3,  int(minor_est * 0.33))
        dil_h = max(15, int(minor_est * 1.20))
        if core_angle < 45 or core_angle > 135:
            dil_w, dil_h = dil_h, dil_w
        final_mask = cv2.dilate(
            mask_core, cv2.getStructuringElement(cv2.MORPH_RECT, (dil_w, dil_h)))
        morph_branch = "dilate-major"
    else:
        close_k = max(5, int(minor_est * 0.20)) | 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        final_mask = cv2.morphologyEx(mask_core, cv2.MORPH_CLOSE, k)
        morph_branch = "ellipse-close"
    return {
        "minor_est": minor_est, "sigma_l": sigma_l,
        "dog": dog, "final_mask": final_mask,
        "ellipse": _fit_ellipse_on_mask(final_mask),
        "morph_branch": morph_branch,
    }


def make_detail_figure(stem: str, raw_dir: Path, patient_label: str,
                       out_path: Path) -> None:
    raw_path = raw_dir / f"{stem}.jpg"
    img_bgr = cv2.imread(str(raw_path))
    if img_bgr is None:
        raise FileNotFoundError(raw_path)

    roi_bgr     = center_crop(img_bgr)
    red_roi_str = stretch_to_255(red_channel(roi_bgr))
    inter   = adaptdog_with_intermediates(red_roi_str)
    e       = inter["ellipse"]
    overlay = draw_ellipse_overlay(roi_bgr, e, color=(0, 255, 120), thickness=2)

    # ── numerical chain (up to ratio) ────────────────────────────────────
    major, minor, angle = e["major"], e["minor"], e["angle"]
    ratio = minor / major
    area_scaled = major * minor * SCALE_FACTOR ** 2

    # ── figure layout: 2 rows (images, text) ─────────────────────────────
    fig = plt.figure(figsize=(18, 6.2))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 0.55],
                          hspace=0.30, wspace=0.12,
                          left=0.04, right=0.98, top=0.90, bottom=0.05)

    # Stage thumbnails
    def show(ax, img, title, cmap=None):
        if cmap is None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    h, w = img_bgr.shape[:2]
    s = 700 / max(h, w)
    raw_thumb = cv2.resize(img_bgr, (int(w * s), int(h * s)))

    show(fig.add_subplot(gs[0, 0]), raw_thumb, "1. Raw")
    show(fig.add_subplot(gs[0, 1]), red_roi_str,
         "2. RedEnhance on ROI", cmap="gray")
    show(fig.add_subplot(gs[0, 2]), inter["dog"],
         f"3. DoG  σ_L={inter['sigma_l']:.1f}\n(from minor_est={inter['minor_est']:.1f})",
         cmap="gray")
    show(fig.add_subplot(gs[0, 3]), inter["final_mask"],
         f"4. Mask\n({inter['morph_branch']})", cmap="gray")
    show(fig.add_subplot(gs[0, 4]), overlay,
         f"5. Ellipse\nmajor={major:.1f} minor={minor:.1f} ang={angle:.1f}°")

    # ── text panel spanning all 5 columns ────────────────────────────────
    ax_t = fig.add_subplot(gs[1, :])
    ax_t.axis("off")

    angle_bin = (
        "90deg" if 70 <= angle % 180 < 110 else
        "45deg" if 30 <= angle % 180 < 60  else
        "0deg"  if (angle % 180 < 20 or angle % 180 >= 160) else
        "other"
    )

    text = (
        f"ELLIPSE PARAMETERS  (output of Stage 5  cv2.fitEllipse):\n"
        f"   major = {major:.4f} px\n"
        f"   minor = {minor:.4f} px\n"
        f"   angle = {angle:.4f}°    →    angle_bin = {angle_bin}\n"
        f"\n"
        f"DERIVED:\n"
        f"   ratio        = minor / major\n"
        f"                = {minor:.4f} / {major:.4f}\n"
        f"                = {ratio:.6f}\n"
        f"\n"
        f"   area_scaled  = major · minor · SCALE_FACTOR²        (SCALE_FACTOR = {SCALE_FACTOR})\n"
        f"                = {major:.4f} · {minor:.4f} · {SCALE_FACTOR}²\n"
        f"                = {area_scaled:.4f}"
    )

    ax_t.text(0.0, 1.0, text, ha="left", va="top",
              family="DejaVu Sans Mono", fontsize=10.5,
              transform=ax_t.transAxes,
              bbox=dict(facecolor="#f7f7f7", edgecolor="#bbb",
                        boxstyle="round,pad=0.6"))

    fig.suptitle(f"{patient_label}  /  {stem}", fontsize=13, y=0.985)
    fig.savefig(out_path, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")

    # Print numerical summary to stdout for verification
    print(f"    major={major:.4f}  minor={minor:.4f}  angle={angle:.4f}")
    print(f"    ratio={ratio:.6f}  area_scaled={area_scaled:.4f}")


def main():
    for patient_label, raw_dir, out_dir, stems in JOBS:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {patient_label} ===")
        for stem in stems:
            print(f"processing {stem} ...")
            make_detail_figure(stem, raw_dir, patient_label,
                               out_dir / f"detail_{stem}.png")
        print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
