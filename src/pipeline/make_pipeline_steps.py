"""
One-off script: generate step-by-step pipeline visualisation images for a
single image from 102_LEFT (IMG_20260513_143643_210).

Output: data/processed/pipeline_runs/pipeline_v150526_101_106/102_LEFT/pipeline_steps/
  step1_raw.png
  step2_redenhance.png
  step3_centercrop.png
  step4_adaptdog.png
  step5_iqr.png
  step6_pupil.png
  step7_d_estimation.png
"""

import glob as _glob
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.paths import PATIENT_DATA_DIR, PROCESSED_DIR
from src.preprocessing.preprocess_utils import center_crop
from src.ellipse.adaptdog import (
    red_channel, stretch_to_255, run_adaptive_dog,
    draw_ellipse_overlay, iqr_filter,
)
from src.analysis.pupil_estimator import (
    estimate_pupil, SCALE_FACTOR, P_MIN, P_MAX,
    S2, S1, S0, I2, I1, I0,
)
from src.analysis.build_patient_model import estimate_D_from_ratio_and_p

# ── config ────────────────────────────────────────────────────────────────────
TARGET_STEM = "IMG_20260513_143643_210"
PATIENT_DIR = PATIENT_DATA_DIR / "102" / "LEFT"
OUT = (PROCESSED_DIR / "pipeline_runs" / "pipeline_v150526_101_106"
       / "102_LEFT" / "pipeline_steps")
OUT.mkdir(parents=True, exist_ok=True)
EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

DPI = 130

# ── load all images (dedup) ───────────────────────────────────────────────────
seen: set[str] = set()
raw_paths: list[Path] = []
for ext in EXTENSIONS:
    for p in _glob.glob(str(PATIENT_DIR / ext)):
        key = Path(p).name.lower()
        if key not in seen:
            seen.add(key)
            raw_paths.append(Path(p))
raw_paths.sort()

target_path = next(p for p in raw_paths if p.stem == TARGET_STEM)
img_bgr = cv2.imread(str(target_path))

# ── STEP 1: Raw image ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
ax.set_title("Step 1: Raw input image", fontsize=13, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "step1_raw.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Step 1 saved.")

# ── STEP 2: RedEnhance ────────────────────────────────────────────────────────
red_raw = red_channel(img_bgr)
red_str = stretch_to_255(red_raw)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original (RGB)", fontsize=12)
axes[0].axis("off")
axes[1].imshow(red_str, cmap="inferno")
axes[1].set_title("RedEnhance output\nR - 0.5G - 0.5B  (stretched to 0-255)", fontsize=12)
axes[1].axis("off")
plt.suptitle("Step 2: Red-channel Enhancement", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "step2_redenhance.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Step 2 saved.")

# ── STEP 3: Center crop ───────────────────────────────────────────────────────
roi_bgr = center_crop(img_bgr)
h_full, w_full = img_bgr.shape[:2]
h_roi,  w_roi  = roi_bgr.shape[:2]
x0 = (w_full - w_roi) // 2
y0 = (h_full - h_roi) // 2
vis = img_bgr.copy()
cv2.rectangle(vis, (x0, y0), (x0 + w_roi, y0 + h_roi), (0, 255, 0), 4)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
axes[0].set_title(f"Full image with ROI box  ({w_full}x{h_full} px)", fontsize=11)
axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
axes[1].set_title(f"Center-cropped ROI  ({w_roi}x{h_roi} px)", fontsize=11)
axes[1].axis("off")
plt.suptitle("Step 3: Center Crop (central 20% of image)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "step3_centercrop.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Step 3 saved.")

# ── STEP 4: AdaptDoG ellipse fit ──────────────────────────────────────────────
red_roi     = red_channel(roi_bgr)
red_roi_str = stretch_to_255(red_roi)
e = run_adaptive_dog(red_roi_str)
overlay = draw_ellipse_overlay(roi_bgr, e, color=(0, 255, 80), thickness=2)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].imshow(red_roi_str, cmap="gray")
axes[0].set_title("RedEnhance ROI (grayscale input to AdaptDoG)", fontsize=11)
axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
if e:
    axes[1].set_title(
        f"Fitted ellipse overlay\n"
        f"major={e['major']:.1f} px   minor={e['minor']:.1f} px   angle={e['angle']:.1f} deg",
        fontsize=11,
    )
else:
    axes[1].set_title("AdaptDoG: no ellipse found", fontsize=11)
axes[1].axis("off")
plt.suptitle("Step 4: Adaptive DoG + Ellipse Fitting (AdaptDoG)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "step4_adaptdog.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Step 4 saved.")

# ── STEP 5: IQR filter ────────────────────────────────────────────────────────
all_stems, all_ellipses = [], []
for raw_path in raw_paths:
    img_b = cv2.imread(str(raw_path))
    if img_b is None:
        continue
    roi_b = center_crop(img_b)
    ee = run_adaptive_dog(stretch_to_255(red_channel(roi_b)))
    all_stems.append(raw_path.stem)
    all_ellipses.append(ee)

keep_mask = iqr_filter(all_ellipses, k=0.5)
all_majors = np.array([ee["major"] if ee else 0.0 for ee in all_ellipses])

kept_majors = all_majors[np.array(keep_mask, dtype=bool)]
q1, q3  = np.percentile(kept_majors, 25), np.percentile(kept_majors, 75)
iqr_val = q3 - q1
lo, hi  = q1 - 0.5 * iqr_val, q3 + 0.5 * iqr_val
target_idx = next(i for i, s in enumerate(all_stems) if s == TARGET_STEM)

fig, ax = plt.subplots(figsize=(11, 4))
bar_colors = ["#2ecc71" if k else "#e74c3c" for k in keep_mask]
ax.bar(range(len(all_majors)), all_majors, color=bar_colors, edgecolor="none", width=0.8)
ax.bar([target_idx], [all_majors[target_idx]],
       color="#f39c12", edgecolor="black", linewidth=1.5, width=0.8,
       label="Selected image")
ax.axhline(lo, color="#c0392b", linewidth=1.8, linestyle="--",
           label=f"Lower fence  Q1 - 0.5*IQR = {lo:.1f} px")
ax.axhline(hi, color="#2980b9", linewidth=1.8, linestyle="--",
           label=f"Upper fence  Q3 + 0.5*IQR = {hi:.1f} px")
from matplotlib.patches import Patch
legend_handles = [
    Patch(color="#2ecc71", label=f"Kept ({sum(keep_mask)})"),
    Patch(color="#e74c3c", label=f"Excluded ({sum(1 for k in keep_mask if not k)})"),
    Patch(color="#f39c12", label="Selected image"),
    plt.Line2D([0], [0], color="#c0392b", linestyle="--", label=f"Lower fence = {lo:.1f} px"),
    plt.Line2D([0], [0], color="#2980b9", linestyle="--", label=f"Upper fence = {hi:.1f} px"),
]
ax.legend(handles=legend_handles, fontsize=9, loc="upper right")
ax.set_xlabel("Image index", fontsize=11)
ax.set_ylabel("Major axis length (px)", fontsize=11)
ax.set_title(
    f"Step 5: IQR Filter on major axis (k=0.5)\n"
    f"Kept {sum(keep_mask)}/{len(all_majors)} images", fontsize=12, fontweight="bold"
)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "step5_iqr.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Step 5 saved.")

# ── STEP 6: Pupil estimation ──────────────────────────────────────────────────
ratio_t       = e["minor"] / e["major"]
area_scaled_t = e["major"] * e["minor"] * SCALE_FACTOR ** 2
p_est_t       = estimate_pupil(ratio_t, area_scaled_t)

ratio_range = np.linspace(0.08, 0.55, 400)
p_curve = []
for r in ratio_range:
    a_c = S2 * r + I2
    b_c = S1 * r + I1
    c_c = S0 * r + I0 - area_scaled_t
    disc = b_c ** 2 - 4 * a_c * c_c
    if disc < 0:
        p_curve.append(np.nan)
        continue
    roots = [(-b_c + np.sqrt(disc)) / (2 * a_c),
             (-b_c - np.sqrt(disc)) / (2 * a_c)]
    valid = [x for x in roots if P_MIN <= x <= P_MAX]
    p_curve.append(valid[0] if valid else np.nan)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# left: calibration curve ratio -> p
axes[0].plot(ratio_range, p_curve, color="#3498db", linewidth=2.5,
             label="p solution curve")
axes[0].axvline(ratio_t, color="#e74c3c", linewidth=1.8, linestyle="--",
                label=f"ratio = {ratio_t:.3f}")
if p_est_t:
    axes[0].axhline(p_est_t, color="#2ecc71", linewidth=1.8, linestyle="--",
                    label=f"p = {p_est_t:.2f} mm")
    axes[0].scatter([ratio_t], [p_est_t], color="#f39c12", s=150, zorder=6,
                    edgecolors="black", linewidths=1.5, label="Solution point")
axes[0].set_xlabel("ratio (minor / major)", fontsize=11)
axes[0].set_ylabel("Pupil diameter p (mm)", fontsize=11)
axes[0].set_title("Calibration curve: ratio → p\n(area_scaled fixed at target value)", fontsize=11)
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)
axes[0].set_ylim(P_MIN - 0.3, P_MAX + 0.3)

# right: text summary
ax2 = axes[1]
ax2.axis("off")
summary = (
    "Quadratic formula:\n\n"
    "  [S₂·r + I₂]·p² + [S₁·r + I₁]·p\n"
    "  + [S₀·r + I₀ − area_scaled] = 0\n\n"
    f"  ratio        = {ratio_t:.4f}\n"
    f"  area_scaled  = {area_scaled_t:.1f} px²\n\n"
    f"  → p = {p_est_t:.2f} mm\n"
    f"     (valid range: {P_MIN}–{P_MAX} mm)"
)
ax2.text(0.05, 0.5, summary, transform=ax2.transAxes,
         fontsize=11, va="center", family="monospace",
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#ecf0f1", alpha=0.9))
ax2.set_title("Numerical result", fontsize=11)
plt.suptitle("Step 6: Pupil Diameter Estimation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "step6_pupil.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Step 6 saved.")

# ── STEP 7: D estimation ──────────────────────────────────────────────────────
d1_t, d2_t = estimate_D_from_ratio_and_p(ratio_t, p_est_t)

ratio_range2 = np.linspace(0.05, 0.60, 400)
D1_curve, D2_curve = [], []
for r in ratio_range2:
    dd1, dd2 = estimate_D_from_ratio_and_p(r, p_est_t)
    D1_curve.append(dd1 if dd1 is not None else np.nan)
    D2_curve.append(dd2 if dd2 is not None else np.nan)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ratio_range2, D1_curve, color="#3498db", linewidth=2.2,
        label="D₁ (hyperopic solution)")
ax.plot(ratio_range2, D2_curve, color="#e67e22", linewidth=2.2,
        label="D₂ (myopic solution — adopted)")
ax.axvline(ratio_t, color="#e74c3c", linewidth=1.8, linestyle="--",
           label=f"ratio = {ratio_t:.3f}")
ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
if d2_t is not None:
    ax.scatter([ratio_t], [d2_t], color="#e67e22", s=180, zorder=6,
               edgecolors="black", linewidths=1.5)
    ax.annotate(f"D₂ = {d2_t:.2f} D  (adopted)",
                xy=(ratio_t, d2_t),
                xytext=(ratio_t + 0.05, d2_t + 0.6),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
if d1_t is not None:
    ax.scatter([ratio_t], [d1_t], color="#3498db", s=140, zorder=6,
               edgecolors="black", linewidths=1.5)
    ax.annotate(f"D₁ = {d1_t:.2f} D",
                xy=(ratio_t, d1_t),
                xytext=(ratio_t + 0.05, d1_t - 0.8),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
ax.set_xlabel("ratio (minor / major)", fontsize=12)
ax.set_ylabel("Refractive power D (diopters)", fontsize=12)
ax.set_title(
    f"Step 7: D Estimation from Calibration Model\n"
    f"p = {p_est_t:.2f} mm  →  D₂ (adopted) = {d2_t:.2f} D",
    fontsize=12, fontweight="bold",
)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "step7_d_estimation.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Step 7 saved.")

print(f"\nAll images saved to:\n  {OUT}")
print(f"\nTarget image parameters:")
print(f"  major       = {e['major']:.1f} px")
print(f"  minor       = {e['minor']:.1f} px")
print(f"  ratio       = {ratio_t:.4f}")
print(f"  angle       = {e['angle']:.1f} deg")
print(f"  area_scaled = {area_scaled_t:.1f} px^2")
print(f"  p_est       = {p_est_t:.2f} mm")
print(f"  D1          = {d1_t:.3f} D")
print(f"  D2 (adopted)= {d2_t:.3f} D")
