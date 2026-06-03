"""
run_myovshyper.py — Ellipse analysis for data/myovshyper/{group}/ folders.

Outputs per group:
  red/           RedEnhance images
  roi/           Centre-crop ROI images
  ellipse/       Ellipse overlay images
  per_image.csv  stem, major, minor, ratio, angle, iqr_kept

Run:
  python -m src.pipeline.run_myovshyper --run_name myovshyper_run01
  python -m src.pipeline.run_myovshyper --run_name myovshyper_run01 --groups m2 h4
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.paths import DATA_DIR, PROCESSED_DIR
from src.preprocessing.preprocess_utils import center_crop
from src.ellipse.adaptdog import (
    red_channel, stretch_to_255,
    run_adaptive_dog, draw_ellipse_overlay,
    iqr_filter,
)

MYOVSHYPER_DIR = DATA_DIR / "myovshyper"
EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

GRID_COLS = 4
GRID_CELL = 200


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


def process_group(group_id: str, run_dir: Path) -> list[dict]:
    src_dir = MYOVSHYPER_DIR / group_id
    if not src_dir.exists():
        raise FileNotFoundError(f"group folder not found: {src_dir}")

    out = run_dir / group_id
    for sub in ("red", "roi", "ellipse"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    import glob as _glob
    seen: set[str] = set()
    raw_paths = []
    for ext in EXTENSIONS:
        for p_str in _glob.glob(str(src_dir / ext)):
            key = Path(p_str).name.lower()
            if key not in seen:
                seen.add(key)
                raw_paths.append(Path(p_str))
    raw_paths.sort()
    if not raw_paths:
        print(f"  [{group_id}] no images found")
        return []

    print(f"  [{group_id}] {len(raw_paths)} images")

    stems, ellipses, roi_bgrs = [], [], []
    for raw_path in raw_paths:
        img_bgr = cv2.imread(str(raw_path))
        if img_bgr is None:
            continue
        stem = raw_path.stem

        red_raw = red_channel(img_bgr)
        red_str = stretch_to_255(red_raw)
        cv2.imwrite(str(out / "red" / f"{stem}_red.png"), red_str)

        roi_bgr = center_crop(img_bgr)
        cv2.imwrite(str(out / "roi" / f"{stem}_roi.png"), roi_bgr)

        red_roi = red_channel(roi_bgr)
        red_roi_str = stretch_to_255(red_roi)
        e = run_adaptive_dog(red_roi_str)

        stems.append(stem)
        ellipses.append(e)
        roi_bgrs.append(roi_bgr)

    keep_mask = iqr_filter(ellipses, k=0.5)
    n_kept = sum(keep_mask)
    print(f"  [{group_id}] IQR filter: kept {n_kept}/{len(stems)}")

    per_image = []
    for stem, e, kept, roi_bgr in zip(stems, ellipses, keep_mask, roi_bgrs):
        overlay = draw_ellipse_overlay(roi_bgr, e, color=(0, 255, 120) if kept else (0, 80, 255))
        cv2.imwrite(str(out / "ellipse" / f"{stem}_ellipse.png"), overlay)

        row: dict = {"stem": stem, "iqr_kept": int(kept)}
        if e:
            row.update({
                "major": round(e["major"], 2),
                "minor": round(e["minor"], 2),
                "ratio": round(e["minor"] / e["major"], 4),
                "angle": round(e["angle"], 2),
            })
        else:
            row.update({"major": None, "minor": None, "ratio": None, "angle": None})
        per_image.append(row)

    fields = ["stem", "major", "minor", "ratio", "angle", "iqr_kept"]
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_image)

    _plot_ellipse_grid(group_id, stems, ellipses, roi_bgrs, keep_mask,
                       out / "ellipse_grid.png")

    return per_image


def _plot_ellipse_grid(group_id: str, stems, ellipses, roi_bgrs,
                       keep_mask, path: Path) -> None:
    n = len(stems)
    n_cols = min(GRID_COLS, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.5, n_rows * 2.5),
                             facecolor="#111",
                             squeeze=False)

    for idx in range(n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        ax.set_facecolor("#111")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

        if idx < n:
            e = ellipses[idx]
            kept = keep_mask[idx]
            overlay = draw_ellipse_overlay(roi_bgrs[idx], e,
                                           color=(0, 255, 120) if kept else (0, 80, 255))
            cell = _resize_cell(overlay, GRID_CELL)
            ax.imshow(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
            border_col = "#2ecc71" if kept else "#e74c3c"
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(border_col); sp.set_linewidth(2)
            if e:
                lbl = f"maj={e['major']:.1f}  min={e['minor']:.1f}\nratio={e['minor']/e['major']:.3f}  a={e['angle']:.0f}°"
            else:
                lbl = "no fit"
            ax.set_title(lbl, fontsize=7,
                         color="#2ecc71" if kept else "#e74c3c", pad=2)
            ax.set_xlabel(stems[idx], fontsize=6, color="#aaa")

    plt.suptitle(f"{group_id} — ellipse grid (green=IQR kept, red=excluded)",
                 color="white", fontsize=10)
    plt.tight_layout(pad=0.4)
    plt.savefig(path, dpi=120, facecolor="#111", bbox_inches="tight")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Ellipse analysis for myovshyper data")
    p.add_argument("--run_name", required=True)
    p.add_argument("--groups", nargs="*", default=None,
                   help="subfolders to process (default: all in data/myovshyper/)")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = PROCESSED_DIR / "myovshyper_runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {args.run_name}")
    print(f"Output: {run_dir}\n")

    if args.groups:
        groups = args.groups
    else:
        groups = sorted(p.name for p in MYOVSHYPER_DIR.iterdir() if p.is_dir())

    all_rows = []
    for group_id in groups:
        rows = process_group(group_id, run_dir)
        for r in rows:
            all_rows.append({"group": group_id, **r})

    # combined summary CSV
    fields = ["group", "stem", "major", "minor", "ratio", "angle", "iqr_kept"]
    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n{'='*62}")
    print(f"{'group':<8}  {'stem':<40}  {'major':>6}  {'minor':>6}  {'ratio':>6}  {'angle':>6}")
    print("-" * 62)
    for r in all_rows:
        major = f"{r['major']:.1f}" if r["major"] is not None else "  ---"
        minor = f"{r['minor']:.1f}" if r["minor"] is not None else "  ---"
        ratio = f"{r['ratio']:.3f}" if r["ratio"] is not None else " ---"
        angle = f"{r['angle']:.1f}" if r["angle"] is not None else "  ---"
        print(f"{r['group']:<8}  {r['stem']:<40}  {major:>6}  {minor:>6}  {ratio:>6}  {angle:>6}")

    print(f"\nSaved: {run_dir / 'summary.csv'}")
    print(f"Images: {run_dir}/<group>/ellipse/")


if __name__ == "__main__":
    main()
