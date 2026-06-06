"""
Pipeline sim_ratio — Raw image → S / C / A
pipeline_v150526 の Step 5+6 を Simulation ベースの ratio→D 式に差し替えたバージョン。

差分 (v150526 との比較):
  旧 Step 5+6: (ratio, area_scaled) → pupil_est → estimate_D_from_ratio_and_p (モデルアイ校正)
  新 Step 5+6: ratio → estimate_D_from_ratio_sim (Simulation C⁰ Logistic, 暫定)

【注意】 D推定モデルが発展途上 (src/analysis/sim_ratio_model.py 参照)。
  - p20/p30/p40 の 3 グループ平均のみ
  - 瞳孔径依存性を無視
  - 遠視 D > +2.0 D は範囲外

その他のステップ (RedEnhance / AdaptDoG / IQR / D-IQR / SCA fit) は v150526 と同一。

Run:
  python -m src.pipeline.pipeline_sim_ratio \\
      --patient_ids 101_LEFT 101_RIGHT \\
      --run_name sim_ratio_run01
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
from src.analysis.sim_ratio_model import estimate_D_from_ratio_sim   # ← 差し替え箇所
from src.analysis.refraction_estimator import fit_sca

# ── settings ──────────────────────────────────────────────────────────────────

EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

IQR_K   = 0.5
D_IQR_K = 1.5

ANGLE_BINS = {
    "90deg": {"range": (70, 110),  "color": "#2980b9", "label": "90deg"},
    "45deg": {"range": (30,  60),  "color": "#f39c12", "label": "45deg"},
    "0deg":  {"range": None,       "color": "#e74c3c", "label": "0/180deg"},
    "other": {"range": None,       "color": "#95a5a6", "label": "other"},
}

GRID_COLS = 6
GRID_CELL = 140


# ── helpers (v150526 と同一) ──────────────────────────────────────────────────

def classify_angle(deg: float) -> str:
    a = float(deg) % 180
    if 70 <= a < 110: return "90deg"
    if 30 <= a < 60:  return "45deg"
    if a < 20 or a >= 160: return "0deg"
    return "other"


def find_raw_images(patient_id: str, data_dir: Path | None = None) -> list[Path]:
    base = data_dir if data_dir else PATIENT_DATA_DIR
    parts = patient_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("LEFT", "RIGHT"):
        patient_dir = base / parts[0] / parts[1]
    else:
        patient_dir = base / patient_id
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
                    exclude_prefixes: tuple = (),
                    data_dir: Path | None = None) -> dict | None:
    out = run_dir / patient_id
    for sub in ("red", "roi", "red_roi", "ellipse"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ── Step 1 & 2: RedEnhance + ROI crop ─────────────────────────────────────
    raw_paths = find_raw_images(patient_id, data_dir=data_dir)
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

        red_raw = red_channel(img_bgr)
        red_str = stretch_to_255(red_raw)
        cv2.imwrite(str(out / "red" / f"{stem}_red.png"), red_str)

        roi_bgr = center_crop(img_bgr)
        cv2.imwrite(str(out / "roi" / f"{stem}_roi.png"), roi_bgr)

        red_roi     = red_channel(roi_bgr)
        red_roi_str = stretch_to_255(red_roi)
        cv2.imwrite(str(out / "red_roi" / f"{stem}_red_roi.png"), red_roi_str)
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

    # ── Step 5 & 6: Simulation ratio→D (瞳孔径推定なし) ──────────────────────
    per_image  = []
    n_no_d     = 0
    for stem, e, keep in zip(stems, ellipses, keep_mask):
        if not keep or not e:
            continue
        ratio = e["minor"] / e["major"]

        d_myo, d_hyp = estimate_D_from_ratio_sim(ratio)
        if d_myo is None:
            n_no_d += 1
            continue

        per_image.append({
            "stem":      stem,
            "major":     e["major"],
            "minor":     e["minor"],
            "ratio":     ratio,
            "angle":     e["angle"],
            "angle_bin": classify_angle(e["angle"]),
            "d_myo":     float(d_myo),
            "d_hyp":     float(d_hyp) if d_hyp is not None else None,
            "adopted_D": float(d_myo),   # 近視側を採用 (v150526 の d2 に相当)
        })

    if n_no_d:
        print(f"  [{patient_id}] {n_no_d} images: no valid D estimate from sim model")

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

    print(f"  [{patient_id}] S={sca['S']:+.2f}  C={sca['C']:+.2f}  "
          f"A={sca['A']:.1f}deg  R2={sca['R2']:.3f}  n={sca['n']}")

    # ── outputs ───────────────────────────────────────────────────────────────
    _save_per_image_csv(per_image, out / "per_image.csv")
    _save_sca_csv(patient_id, sca, n_total, n_kept, n_no_d, n_d_excl, out / "sca.csv")
    _plot_cos_curve(patient_id, valid, alpha_arr, D_arr, sca, out / "cos_curve.png")
    _plot_angle_dist(patient_id, per_image, out / "angle_dist.png")
    _plot_ellipse_grid(patient_id, stems, ellipses, roi_bgrs, keep_mask, out / "ellipse_grid.png")

    return {
        "patient_id": patient_id,
        **{k: sca[k] for k in ("S", "C", "A", "SE", "R2", "n")},
        "n_total": n_total, "n_kept": n_kept,
        "n_no_d": n_no_d, "n_d_excl": n_d_excl,
    }


# ── output helpers ────────────────────────────────────────────────────────────

def _save_per_image_csv(per_image: list, path: Path) -> None:
    fields = ["stem", "major", "minor", "ratio", "angle", "angle_bin",
              "d_myo", "d_hyp", "adopted_D"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in per_image:
            w.writerow({k: (f"{row[k]:.4f}" if isinstance(row[k], float) else row[k])
                        for k in fields})


def _save_sca_csv(patient_id, sca, n_total, n_kept, n_no_d, n_d_excl, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "S_D", "C_D", "A_deg", "SE_D", "R2", "n",
                    "n_total", "n_kept", "n_no_d", "n_d_excl",
                    "D_model", "note"])
        w.writerow([patient_id,
                    f"{sca['S']:.3f}", f"{sca['C']:.3f}", f"{sca['A']:.1f}",
                    f"{sca['SE']:.3f}", f"{sca['R2']:.4f}", sca["n"],
                    n_total, n_kept, n_no_d, n_d_excl,
                    "sim_ratio_unified",
                    "PRELIMINARY: pupil-size dependence ignored"])


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
    ax.set_title(f"{patient_id}  [sim_ratio model — PRELIMINARY]\n"
                 f"S={sca['S']:+.2f}D  C={sca['C']:+.2f}D  "
                 f"A={sca['A']:.1f}deg  R2={sca['R2']:.3f}  n={sca['n']}")
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
            e       = ellipses[idx]
            kept    = keep_mask[idx]
            roi     = roi_bgrs[idx]
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
    p = argparse.ArgumentParser(
        description="Pipeline sim_ratio: raw → SCA using simulation-based ratio–D model")
    p.add_argument("--patient_ids", nargs="+", required=True)
    p.add_argument("--run_name",    required=True)
    p.add_argument("--exclude_prefixes", nargs="*", default=[])
    p.add_argument("--data_dir", default=None,
                   help="Custom patient data root (default: data/raw/patient_data)")
    return p.parse_args()


def main():
    args     = parse_args()
    run_dir  = PROCESSED_DIR / "pipeline_runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    excl     = tuple(args.exclude_prefixes)
    data_dir = Path(args.data_dir) if args.data_dir else None

    print(f"Run: {args.run_name}  [D model: sim_ratio - PRELIMINARY]")
    print(f"Output: {run_dir}\n")

    summary = []
    for pid in args.patient_ids:
        result = process_patient(pid, run_dir, exclude_prefixes=excl, data_dir=data_dir)
        if result:
            summary.append(result)

    if summary:
        summary_path = run_dir / "summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            fields = ["patient_id", "S", "C", "A", "SE", "R2", "n",
                      "n_total", "n_kept", "n_no_d", "n_d_excl"]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in summary:
                w.writerow({k: (f"{row[k]:.3f}" if isinstance(row[k], float) else row[k])
                            for k in fields})
        print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
