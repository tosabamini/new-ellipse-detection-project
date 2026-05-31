"""
Simulation pipeline — Raw PNG → ellipse fitting.
Geometry-based only, no ML.

Input layout:
  data/Simulation/p<N>/camera_p<N>_<D>.png
  (p<N> = pupil radius in units; D = refraction label)

Steps:
  1. RedEnhance  : R - 0.5G - 0.5B
  2. Center crop : 60% keep, shifted 10% left
  3. AdaptDoG    : ellipse fitting
  4. IQR filter  : exclude outliers by major axis (k=0.5)

Outputs per pupil group  (<run_dir>/<pupil_group>/):
  red/              RedEnhance images (grayscale PNG)
  roi/              Cropped colour images (PNG)
  ellipse/          Ellipse overlay images
  per_image.csv     Per-image: stem, major, minor, ratio, angle
  ellipse_grid.png  Grid view of ellipse overlays

Run:
  python -m src.pipeline.pipeline_simulation --run_name sim_run01
  python -m src.pipeline.pipeline_simulation --run_name sim_run01 --pupil_groups p10 p20 p30
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.paths import SIMULATION_DIR, SIMULATION_RUNS_DIR
from src.preprocessing.preprocess_utils import center_crop, process_red_by_mode, get_mean_brightness, classify_brightness
from src.ellipse.adaptdog import (
    red_channel, stretch_to_255,
    run_adaptive_dog, draw_ellipse_overlay,
    iqr_filter,
)

# ── crop settings for Simulation data ────────────────────────────────────────

CROP_RATIO = 0.60
LEFT_SHIFT = 0.10

# ── display settings ──────────────────────────────────────────────────────────

GRID_COLS  = 6
GRID_CELL  = 140
IQR_K      = 0.5


# ── helpers ───────────────────────────────────────────────────────────────────

def find_png_files(pupil_dir: Path) -> list[Path]:
    return sorted(p for p in pupil_dir.iterdir()
                  if p.suffix.lower() == ".png")


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


# ── per-group processing ──────────────────────────────────────────────────────

def process_group(pupil_group: str, run_dir: Path) -> dict | None:
    pupil_dir = SIMULATION_DIR / pupil_group
    if not pupil_dir.exists():
        print(f"  [{pupil_group}] directory not found: {pupil_dir}")
        return None

    out = run_dir / pupil_group
    for sub in ("red", "roi", "ellipse"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    png_paths = find_png_files(pupil_dir)
    if not png_paths:
        print(f"  [{pupil_group}] no PNG files found")
        return None

    print(f"  [{pupil_group}] {len(png_paths)} images")

    stems, ellipses, roi_bgrs = [], [], []

    for png_path in png_paths:
        img_bgr = cv2.imread(str(png_path))
        if img_bgr is None:
            print(f"    WARNING: could not load {png_path.name}")
            continue
        stem = png_path.stem

        # RedEnhance (full image)
        red_raw = red_channel(img_bgr)
        red_str = stretch_to_255(red_raw)
        cv2.imwrite(str(out / "red" / f"{stem}_red.png"), red_str)

        # Crop: 60% keep, 10% left shift
        roi_bgr = center_crop(img_bgr, crop_ratio=CROP_RATIO, left_shift=LEFT_SHIFT)
        cv2.imwrite(str(out / "roi" / f"{stem}_roi.png"), roi_bgr)

        # AdaptDoG on red-enhanced ROI
        red_roi     = red_channel(roi_bgr)
        red_roi_str = stretch_to_255(red_roi)
        e = run_adaptive_dog(red_roi_str)

        stems.append(stem)
        ellipses.append(e)
        roi_bgrs.append(roi_bgr)

    if not stems:
        print(f"  [{pupil_group}] no images loaded")
        return None

    # Save ellipse overlays
    for stem, e, roi_bgr in zip(stems, ellipses, roi_bgrs):
        overlay = draw_ellipse_overlay(roi_bgr, e)
        cv2.imwrite(str(out / "ellipse" / f"{stem}_ellipse.png"), overlay)

    # IQR filter
    keep_mask = iqr_filter(ellipses, k=IQR_K)
    n_kept  = sum(keep_mask)
    n_total = len(stems)
    print(f"  [{pupil_group}] IQR filter: kept {n_kept}/{n_total}")

    # per_image.csv
    _save_per_image_csv(stems, ellipses, keep_mask, out / "per_image.csv")

    # ellipse grid
    _plot_ellipse_grid(pupil_group, stems, ellipses, roi_bgrs, keep_mask,
                       out / "ellipse_grid.png")

    n_no_fit = sum(1 for e in ellipses if e is None)
    return {
        "pupil_group": pupil_group,
        "n_total": n_total,
        "n_kept": n_kept,
        "n_no_fit": n_no_fit,
    }


# ── output helpers ────────────────────────────────────────────────────────────

def _save_per_image_csv(stems, ellipses, keep_mask, path: Path) -> None:
    fields = ["stem", "kept", "major", "minor", "ratio", "angle"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for stem, e, kept in zip(stems, ellipses, keep_mask):
            if e:
                w.writerow({
                    "stem":  stem,
                    "kept":  int(kept),
                    "major": f"{e['major']:.2f}",
                    "minor": f"{e['minor']:.2f}",
                    "ratio": f"{e['ratio']:.4f}",
                    "angle": f"{e['angle']:.2f}",
                })
            else:
                w.writerow({"stem": stem, "kept": 0,
                            "major": "", "minor": "", "ratio": "", "angle": ""})


def _plot_ellipse_grid(pupil_group, stems, ellipses, roi_bgrs,
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
        for sp in ax.spines.values():
            sp.set_visible(False)

        if idx < n:
            e       = ellipses[idx]
            kept    = keep_mask[idx]
            roi     = roi_bgrs[idx]
            color   = (0, 255, 120) if kept else (0, 80, 255)
            overlay = draw_ellipse_overlay(roi, e, color=color)
            cell    = _resize_cell(overlay, GRID_CELL)
            ax.imshow(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
            border_col = "#2ecc71" if kept else "#e74c3c"
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(border_col)
                sp.set_linewidth(1.5)
            if e:
                lbl = f"r={e['ratio']:.2f} a={e['angle']:.0f}"
            else:
                lbl = "no fit"
            ax.set_title(lbl, fontsize=5.0,
                         color="#2ecc71" if kept else "#e74c3c", pad=1)

    plt.suptitle(f"{pupil_group} — ellipse grid (green=kept, red=excluded)",
                 color="white", fontsize=9)
    plt.tight_layout(pad=0.2)
    plt.savefig(path, dpi=110, facecolor="#111", bbox_inches="tight")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Simulation pipeline: PNG → ellipse fitting")
    p.add_argument("--run_name", required=True,
                   help="e.g. sim_run01")
    p.add_argument("--pupil_groups", nargs="*", default=None,
                   help="pupil group folders to process (default: all p* dirs)")
    return p.parse_args()


def main():
    args    = parse_args()
    run_dir = SIMULATION_RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.pupil_groups:
        groups = args.pupil_groups
    else:
        groups = sorted(d.name for d in SIMULATION_DIR.iterdir()
                        if d.is_dir() and d.name.startswith("p"))

    print(f"Run: {args.run_name}")
    print(f"Output: {run_dir}")
    print(f"Groups: {groups}\n")

    summary = []
    for group in groups:
        result = process_group(group, run_dir)
        if result:
            summary.append(result)

    if summary:
        print(f"\n{'='*45}")
        print(f"{'Group':<10}  {'total':>6}  {'kept':>6}  {'no_fit':>6}")
        print("-" * 35)
        for r in summary:
            print(f"{r['pupil_group']:<10}  {r['n_total']:>6}  "
                  f"{r['n_kept']:>6}  {r['n_no_fit']:>6}")
        print(f"\nSaved: {run_dir}")


if __name__ == "__main__":
    main()
