"""
Pipeline v150526 — Raw image → S / C / A
Geometry-based only, no ML.

Steps:
  1. RedEnhance  : R - 0.5G - 0.5B
  2. Center crop : ROI
  3. AdaptDoG    : ellipse fitting
  4. IQR filter  : exclude images with very small red reflex (k=0.5)
  5. Pupil est.  : (ratio, area_scaled) → p via quadratic formula
  6. D est.      : (ratio, p) → D from model-eye calibration
  7. D-IQR filter: per angle-bin outlier removal (k=1.5)
  8. SCA fit     : D(α) = P0 + P1·cos(2α) + P2·sin(2α) → S, C, A

Outputs per patient  (<run_dir>/<patient_id>/):
  red/              RedEnhance images (grayscale PNG)
  roi/              ROI images (colour PNG)
  ellipse/          Ellipse overlay images
  per_image.csv     Per-image results
  sca.csv           S, C, A, SE, R², n
  cos_curve.png     Cosine fit plot
  angle_dist.png    Angle distribution histogram
  ellipse_grid.png  Grid view of ellipse overlays

Run:
  python -m src.pipeline.pipeline_v150526 \\
      --patient_ids 101_LEFT 101_RIGHT \\
      --run_name pipeline_v150526_run01
"""

import argparse
import csv
import glob
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.paths import PATIENT_DATA_DIR, PROCESSED_DIR
from src.preprocessing.preprocess_utils import center_crop, process_red_by_mode
from src.ellipse.adaptdog import (
    red_channel, stretch_to_255,
    run_adaptive_dog, draw_ellipse_overlay,
    iqr_filter, d_iqr_filter,
)
from src.analysis.pupil_estimator import estimate_pupil, SCALE_FACTOR
from src.analysis.build_patient_model import estimate_D_from_ratio_and_p
from src.analysis.refraction_estimator import fit_sca

# ── settings ──────────────────────────────────────────────────────────────────

EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

IQR_K   = 0.5    # major-axis IQR fence multiplier
D_IQR_K = 1.5    # D-value IQR fence multiplier (per angle bin)

ANGLE_BINS = {
    "90deg": {"range": (70, 110),  "color": "#2980b9", "label": "90deg"},
    "45deg": {"range": (30,  60),  "color": "#f39c12", "label": "45deg"},
    "0deg":  {"range": None,       "color": "#e74c3c", "label": "0/180deg"},
    "other": {"range": None,       "color": "#95a5a6", "label": "other"},
}

GRID_COLS   = 6    # columns in ellipse_grid.png
GRID_CELL   = 140  # px per cell


# ── helpers ───────────────────────────────────────────────────────────────────

def classify_angle(deg: float) -> str:
    a = float(deg) % 180
    if 70 <= a < 110: return "90deg"
    if 30 <= a < 60:  return "45deg"
    if a < 20 or a >= 160: return "0deg"
    return "other"


def find_raw_images(patient_id: str) -> list[Path]:
    parts = patient_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("LEFT", "RIGHT"):
        patient_dir = PATIENT_DATA_DIR / parts[0] / parts[1]
    else:
        patient_dir = PATIENT_DATA_DIR / patient_id
    if not patient_dir.exists():
        raise FileNotFoundError(f"patient folder not found: {patient_dir}")
    paths = []
    for ext in EXTENSIONS:
        paths.extend(glob.glob(str(patient_dir / ext)))
    return sorted(Path(p) for p in paths)


def _resize_cell(img_bgr: np.ndarray, cell: int = GRID_CELL) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = min(cell / h, cell / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_bgr, (nw, nh))
    canvas = np.zeros((cell, cell, 3), dtype=np.uint8)
    oh = (cell - nh) // 2
    ow = (cell - nw) // 2
    canvas[oh:oh + nh, ow:ow + nw] = resized
    return canvas


# ── per-patient processing ────────────────────────────────────────────────────

def process_patient(patient_id: str, run_dir: Path,
                    exclude_prefixes: tuple = ()) -> dict | None:
    """
    Full pipeline for one patient.
    Returns SCA result dict, or None on failure.
    """
    out = run_dir / patient_id
    for sub in ("red", "roi", "ellipse"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ── Step 1 & 2: RedEnhance + ROI crop ────────────────────────────────────
    raw_paths = find_raw_images(patient_id)
    raw_paths = [p for p in raw_paths
                 if not (exclude_prefixes and p.name.startswith(exclude_prefixes))]
    if not raw_paths:
        print(f"  [{patient_id}] no raw images found")
        return None

    print(f"  [{patient_id}] {len(raw_paths)} raw images")

    stems, ellipses, roi_bgrs = [], [], []
    for raw_path in raw_paths:
        img_bgr = cv2.imread(str(raw_path))
        if img_bgr is None:
            continue
        stem = raw_path.stem

        # RedEnhance
        red_raw = red_channel(img_bgr)
        red_str = stretch_to_255(red_raw)
        cv2.imwrite(str(out / "red" / f"{stem}_red.png"), red_str)

        # ROI crop (centre 20% of original)
        roi_bgr = center_crop(img_bgr)
        cv2.imwrite(str(out / "roi" / f"{stem}_roi.png"), roi_bgr)

        # AdaptDoG on red-enhanced ROI
        red_roi     = red_channel(roi_bgr)
        red_roi_str = stretch_to_255(red_roi)
        e = run_adaptive_dog(red_roi_str)

        stems.append(stem)
        ellipses.append(e)
        roi_bgrs.append(roi_bgr)

    if not stems:
        print(f"  [{patient_id}] no images loaded")
        return None

    # ── Step 3: save ellipse overlays ─────────────────────────────────────────
    for stem, e, roi_bgr in zip(stems, ellipses, roi_bgrs):
        overlay = draw_ellipse_overlay(roi_bgr, e)
        cv2.imwrite(str(out / "ellipse" / f"{stem}_ellipse.png"), overlay)

    # ── Step 4: IQR filter ────────────────────────────────────────────────────
    keep_mask = iqr_filter(ellipses, k=IQR_K)
    n_kept    = sum(keep_mask)
    n_total   = len(stems)
    print(f"  [{patient_id}] IQR filter: kept {n_kept}/{n_total}")

    # ── Step 5 & 6: pupil + D estimation ─────────────────────────────────────
    per_image = []
    n_no_p    = 0
    for stem, e, keep in zip(stems, ellipses, keep_mask):
        if not keep or not e:
            continue
        ratio       = e["minor"] / e["major"]
        area_scaled = e["major"] * e["minor"] * SCALE_FACTOR ** 2
        p_est       = estimate_pupil(ratio, area_scaled)
        if p_est is None:
            n_no_p += 1
            continue
        d1, d2 = estimate_D_from_ratio_and_p(ratio, p_est)
        if d2 is None:
            continue
        per_image.append({
            "stem":      stem,
            "major":     e["major"],
            "minor":     e["minor"],
            "ratio":     ratio,
            "angle":     e["angle"],
            "angle_bin": classify_angle(e["angle"]),
            "area_scaled": area_scaled,
            "p_est":     p_est,
            "d1":        float(d1) if d1 is not None else None,
            "d2":        float(d2),
            "adopted_D": float(d2),
        })

    if n_no_p:
        print(f"  [{patient_id}] {n_no_p} images: no valid pupil estimate")

    # ── Step 7: D-IQR filter per angle bin ───────────────────────────────────
    d_mask   = d_iqr_filter(per_image, k=D_IQR_K)
    valid    = [x for x, k in zip(per_image, d_mask) if k]
    n_d_excl = sum(1 for k in d_mask if not k)
    if n_d_excl:
        print(f"  [{patient_id}] D-IQR: removed {n_d_excl} outlier(s)")

    if len(valid) < 3:
        print(f"  [{patient_id}] ERROR: fewer than 3 valid images for SCA fit")
        return None

    # ── Step 8: SCA fit ───────────────────────────────────────────────────────
    alpha_arr = np.array([x["angle"]     for x in valid])
    D_arr     = np.array([x["adopted_D"] for x in valid])
    sca       = fit_sca(alpha_arr, D_arr)
    p_med     = float(np.median([x["p_est"] for x in valid]))

    print(f"  [{patient_id}] S={sca['S']:+.2f}  C={sca['C']:+.2f}  "
          f"A={sca['A']:.1f}deg  R2={sca['R2']:.3f}  p_med={p_med:.1f}mm  n={sca['n']}")

    # ── outputs ───────────────────────────────────────────────────────────────
    _save_per_image_csv(per_image, out / "per_image.csv")
    _save_sca_csv(patient_id, sca, p_med, n_total, n_kept, n_no_p, n_d_excl, out / "sca.csv")
    _plot_cos_curve(patient_id, valid, alpha_arr, D_arr, sca, out / "cos_curve.png")
    _plot_angle_dist(patient_id, per_image, out / "angle_dist.png")
    _plot_ellipse_grid(patient_id, stems, ellipses, roi_bgrs, keep_mask, out / "ellipse_grid.png")

    return {
        "patient_id": patient_id,
        **{k: sca[k] for k in ("S", "C", "A", "SE", "R2", "n")},
        "p_med": p_med,
        "n_total": n_total, "n_kept": n_kept,
        "n_no_p": n_no_p, "n_d_excl": n_d_excl,
    }


# ── output helpers ────────────────────────────────────────────────────────────

def _save_per_image_csv(per_image: list, path: Path) -> None:
    fields = ["stem", "major", "minor", "ratio", "angle", "angle_bin",
              "area_scaled", "p_est", "d1", "d2", "adopted_D"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in per_image:
            w.writerow({k: (f"{row[k]:.4f}" if isinstance(row[k], float) else row[k])
                        for k in fields})


def _save_sca_csv(patient_id, sca, p_med, n_total, n_kept, n_no_p, n_d_excl, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "S_D", "C_D", "A_deg", "SE_D", "R2", "n",
                    "p_med_mm", "n_total", "n_kept", "n_no_p", "n_d_excl", "scale_factor"])
        w.writerow([patient_id,
                    f"{sca['S']:.3f}", f"{sca['C']:.3f}", f"{sca['A']:.1f}",
                    f"{sca['SE']:.3f}", f"{sca['R2']:.4f}", sca["n"],
                    f"{p_med:.2f}", n_total, n_kept, n_no_p, n_d_excl, SCALE_FACTOR])


def _plot_cos_curve(patient_id, valid, alpha_arr, D_arr, sca, path: Path) -> None:
    a_fine = np.linspace(0, 180, 360)
    a_rad  = np.deg2rad(alpha_arr)
    X      = np.column_stack([np.ones(len(a_rad)), np.cos(2 * a_rad), np.sin(2 * a_rad)])
    P, *_  = np.linalg.lstsq(X, D_arr, rcond=None)
    D_fit  = P[0] + P[1] * np.cos(2 * np.deg2rad(a_fine)) + P[2] * np.sin(2 * np.deg2rad(a_fine))

    fig, ax = plt.subplots(figsize=(9, 5))
    for bin_key, binfo in ANGLE_BINS.items():
        pts = [(x["angle"], x["adopted_D"]) for x in valid if x["angle_bin"] == bin_key]
        if pts:
            xa, ya = zip(*pts)
            ax.scatter(xa, ya, color=binfo["color"],
                       label=f"{binfo['label']} (n={len(pts)})",
                       s=55, zorder=3, edgecolors="white", linewidths=0.4)
    ax.plot(a_fine, D_fit, color="black", linewidth=2, label="cos fit", zorder=2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Major axis angle (deg)")
    ax.set_ylabel("D (diopters)")
    ax.set_title(f"{patient_id}\n"
                 f"S={sca['S']:+.2f}D  C={sca['C']:+.2f}D  "
                 f"A={sca['A']:.1f}deg  R²={sca['R2']:.3f}  n={sca['n']}")
    ax.set_xlim(0, 180)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_angle_dist(patient_id, per_image, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    angles = [x["angle"] for x in per_image]
    ax.hist(angles, bins=18, range=(0, 180), color="#3498db",
            edgecolor="white", linewidth=0.5)
    ax.axvspan(70, 110, alpha=0.15, color="#2980b9", label="90deg")
    ax.axvspan(30,  60, alpha=0.15, color="#f39c12", label="45deg")
    ax.set_xlabel("Major axis angle (deg)")
    ax.set_ylabel("Count")
    ax.set_title(f"{patient_id} — angle distribution (after IQR filter)")
    ax.set_xlim(0, 180)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_ellipse_grid(patient_id, stems, ellipses, roi_bgrs,
                       keep_mask, path: Path) -> None:
    n      = len(stems)
    n_cols = GRID_COLS
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.8, n_rows * 1.8),
                             facecolor="#111")
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx in range(n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        ax.set_facecolor("#111")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

        if idx < n:
            e      = ellipses[idx]
            kept   = keep_mask[idx]
            roi    = roi_bgrs[idx]
            overlay = draw_ellipse_overlay(roi, e,
                                           color=(0, 255, 120) if kept else (0, 80, 255))
            cell    = _resize_cell(overlay, GRID_CELL)
            ax.imshow(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
            border_col = "#2ecc71" if kept else "#e74c3c"
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(border_col); sp.set_linewidth(1.5)
            lbl = f"a={e['angle']:.0f}" if e else "no fit"
            ax.set_title(lbl, fontsize=5.5,
                         color="#2ecc71" if kept else "#e74c3c", pad=1)

    plt.suptitle(f"{patient_id} — ellipse grid (green=kept, red=excluded)",
                 color="white", fontsize=9)
    plt.tight_layout(pad=0.2)
    plt.savefig(path, dpi=110, facecolor="#111", bbox_inches="tight")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Pipeline v150526: raw → SCA")
    p.add_argument("--patient_ids", nargs="+", required=True,
                   help="e.g. 101_LEFT 101_RIGHT 104_LEFT")
    p.add_argument("--run_name",    required=True,
                   help="e.g. pipeline_v150526_run01")
    p.add_argument("--exclude_prefixes", nargs="*", default=[],
                   help="filename prefixes to exclude (e.g. r_3D_ samarth_3D_)")
    return p.parse_args()


def main():
    args     = parse_args()
    run_dir  = PROCESSED_DIR / "pipeline_runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    excl     = tuple(args.exclude_prefixes)

    print(f"Run: {args.run_name}")
    print(f"Output: {run_dir}\n")

    summary = []
    for patient_id in args.patient_ids:
        result = process_patient(patient_id, run_dir, excl)
        if result:
            summary.append(result)

    # summary CSV
    if summary:
        fields = ["patient_id", "S", "C", "A", "SE", "R2", "n",
                  "p_med", "n_total", "n_kept", "n_no_p", "n_d_excl"]
        with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in summary:
                w.writerow({k: (f"{r[k]:.3f}" if isinstance(r[k], float) else r[k])
                            for k in fields if k in r})

        print(f"\n{'='*55}")
        print(f"{'Patient':<14}  {'S':>6}  {'C':>6}  {'A':>6}  {'R2':>6}")
        print("-" * 45)
        for r in summary:
            print(f"{r['patient_id']:<14}  {r['S']:>+6.2f}  {r['C']:>+6.2f}  "
                  f"{r['A']:>6.1f}  {r['R2']:>6.3f}")
        print(f"\nSaved: {run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
