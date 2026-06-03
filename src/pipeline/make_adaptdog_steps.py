"""
Generate Step 4.1 ~ 4.4 intermediate images for AdaptDoG pipeline explanation.
Target: Patient 102 LEFT, IMG_20260513_143643_210

Output: data/processed/pipeline_runs/pipeline_v150526_101_106/102_LEFT/pipeline_steps/
  step4_1_dog.png
  step4_2_otsu.png
  step4_3_blob.png
  step4_4_ellipse.png
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.paths import PATIENT_DATA_DIR, PROCESSED_DIR
from src.preprocessing.preprocess_utils import center_crop
from src.ellipse.adaptdog import (
    red_channel, stretch_to_255,
    _estimate_minor, _otsu_mask, _pick_central_blob, _fit_ellipse_on_mask,
    draw_ellipse_overlay,
)

# ── config ─────────────────────────────────────────────────────────────────────
TARGET_STEM = "IMG_20260513_143643_210"
PATIENT_DIR = PATIENT_DATA_DIR / "102" / "LEFT"
OUT = (PROCESSED_DIR / "pipeline_runs" / "pipeline_v150526_101_106"
       / "102_LEFT" / "pipeline_steps")
OUT.mkdir(parents=True, exist_ok=True)
DPI = 130

# ── load and preprocess ────────────────────────────────────────────────────────
target_path = PATIENT_DIR / f"{TARGET_STEM}.jpg"
img_bgr  = cv2.imread(str(target_path))
roi_bgr  = center_crop(img_bgr)
red_roi  = red_channel(roi_bgr)
red_str  = stretch_to_255(red_roi)   # input to AdaptDoG

# ── reproduce AdaptDoG internals step by step ──────────────────────────────────
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
        cv2.getStructuringElement(cv2.MORPH_RECT, (dil_w, dil_h))
    )
    morph_label = f"Directional dilation  ({dil_w}×{dil_h} px kernel)\nalong major axis to recover dim tips"
else:
    close_k = max(5, int(minor_est * 0.20)) | 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask_final = cv2.morphologyEx(mask_core, cv2.MORPH_CLOSE, k)
    morph_label = f"Morphological close  (kernel={close_k}px)\nto fill small gaps in the blob"

e_final = _fit_ellipse_on_mask(mask_final)


# ── Step 4.1 — Difference of Gaussians ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

axes[0].imshow(red_str, cmap="gray")
axes[0].set_title(f"Input: RedEnhance ROI\n(grayscale)", fontsize=11)
axes[0].axis("off")

axes[1].imshow(blur_s, cmap="gray")
axes[1].set_title(f"GaussianBlur  σ = 1.5\n(small scale — preserves edges)", fontsize=11)
axes[1].axis("off")

axes[2].imshow(dog, cmap="inferno")
axes[2].set_title(
    f"DoG = Blur(σ=1.5) − Blur(σ={sigma_l:.1f})\n"
    f"σ_large = max(8.0,  minor_est × 0.75) = {sigma_l:.1f}",
    fontsize=11
)
axes[2].axis("off")

plt.suptitle(
    "Step 4.1 — Difference of Gaussians (DoG)\n"
    "Suppresses low-frequency background halo; retains sharp reflex core",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUT / "step4_1_dog.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Step 4.1 saved.")


# ── Step 4.2 — Otsu Thresholding ──────────────────────────────────────────────
thresh_val, otsu_vis = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].imshow(dog, cmap="inferno")
axes[0].set_title("DoG output (input to thresholding)", fontsize=11)
axes[0].axis("off")

axes[1].imshow(otsu_vis, cmap="gray")
axes[1].set_title(f"Otsu binary mask\n(auto threshold = {int(thresh_val)} / 255)", fontsize=11)
axes[1].axis("off")

plt.suptitle(
    "Step 4.2 — Otsu Thresholding\n"
    "Automatically separates bright reflex pixels from background",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUT / "step4_2_otsu.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Step 4.2 saved.")


# ── Step 4.3 — Blob Selection + Morphological Processing ─────────────────────
# overlay the selected blob contour on the ROI for visualisation
blob_vis = roi_bgr.copy()
cnts, _ = cv2.findContours(mask_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(blob_vis, cnts, -1, (0, 200, 255), 2)

final_vis = roi_bgr.copy()
cnts_f, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(final_vis, cnts_f, -1, (0, 255, 80), 2)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].imshow(mask_core, cmap="gray")
axes[0].set_title(
    "Central blob after\nOpen(5px) + Close(9px) + centroid selection",
    fontsize=11
)
axes[0].axis("off")

axes[1].imshow(mask_final, cmap="gray")
axes[1].set_title(f"After morphological post-processing\n{morph_label}", fontsize=11)
axes[1].axis("off")

axes[2].imshow(cv2.cvtColor(final_vis, cv2.COLOR_BGR2RGB))
axes[2].set_title("Post-processed mask\noverlaid on ROI", fontsize=11)
axes[2].axis("off")

plt.suptitle(
    "Step 4.3 — Central Blob Selection & Morphological Processing\n"
    "Noise removal, gap filling, and tip recovery for elongated reflexes",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUT / "step4_3_blob.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Step 4.3 saved.")


# ── Step 4.4 — Ellipse Fitting ────────────────────────────────────────────────
overlay_final = draw_ellipse_overlay(roi_bgr, e_final, color=(0, 255, 80), thickness=2)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].imshow(mask_final, cmap="gray")
axes[0].set_title("Final binary mask\n(input to cv2.fitEllipse)", fontsize=11)
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(overlay_final, cv2.COLOR_BGR2RGB))
if e_final:
    axes[1].set_title(
        f"Fitted ellipse\n"
        f"major = {e_final['major']:.1f} px   "
        f"minor = {e_final['minor']:.1f} px   "
        f"angle = {e_final['angle']:.1f}°",
        fontsize=11
    )
axes[1].axis("off")

plt.suptitle(
    "Step 4.4 — Ellipse Fitting  (cv2.fitEllipse)\n"
    "Least-squares ellipse fit to the final mask contour",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUT / "step4_4_ellipse.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Step 4.4 saved.")

print(f"\nAll saved to: {OUT}")
print(f"minor_est = {minor_est:.1f} px,  sigma_large = {sigma_l:.1f}")
print(f"core_ratio = {core_ratio:.3f}  ->  {'dilation' if core_ratio < 0.20 else 'close'}")
if e_final:
    print(f"Final ellipse: major={e_final['major']:.1f}  minor={e_final['minor']:.1f}  angle={e_final['angle']:.1f}")
