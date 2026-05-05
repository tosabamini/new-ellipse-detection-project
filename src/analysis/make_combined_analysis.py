"""
3瞳孔径 × 10屈折力の統合解析

1. 楕円オーバーレイ画像グリッド（3行×10列）
   行: 3.0mm / 5.0mm / 7.0mm
   列: +4D ... -5D（遠視→近視）

2. 3瞳孔径の Ratio vs 屈折力フィット曲線を1グラフに重ねて表示
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_EYE_RUNS_DIR = PROJECT_ROOT / "data/processed/model_eye_runs"
OUT_DIR = MODEL_EYE_RUNS_DIR / "combined_analysis"
OUT_DIR.mkdir(exist_ok=True)

IMG_H, IMG_W = 600, 800

PUPILS = [
    {"label": "3.0 mm", "run": "model_eye_3mm_v001", "color": "#1f77b4"},
    {"label": "5.0 mm", "run": "model_eye_5mm_v001", "color": "#ff7f0e"},
    {"label": "7.0 mm", "run": "model_eye_v001",     "color": "#2ca02c"},
]


# ── ユーティリティ ────────────────────────────────────────

def folder_to_diopter(name: str) -> float:
    parts = name.split("_")
    sign  = parts[1]
    major = int(parts[2])
    minor = int(parts[3][:-1])
    val   = major + minor / 100.0
    return -val if sign == "M" else (0.0 if sign == "Z" else val)


def load_ellipse_data(run_name: str) -> list[dict]:
    """各フォルダの ellipse_results.csv を読んで集計"""
    run_root = MODEL_EYE_RUNS_DIR / run_name
    records = []
    for folder_dir in sorted(run_root.iterdir()):
        csv_path = folder_dir / "ellipse_results.csv"
        if not csv_path.exists():
            continue
        ratios, majors, minors = [], [], []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["status"] != "ok":
                    continue
                major = float(row["major_axis"])
                minor = float(row["minor_axis"])
                majors.append(major)
                minors.append(minor)
                ratios.append(minor / major)
        if not ratios:
            continue
        records.append({
            "folder":     folder_dir.name,
            "diopter":    folder_to_diopter(folder_dir.name),
            "ratio_mean": float(np.mean(ratios)),
            "ratio_std":  float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0,
            "ratios":     ratios,
            "folder_dir": folder_dir,
        })
    records.sort(key=lambda x: x["diopter"])
    return records


def pick_representative(folder_dir: Path, mean_ratio: float) -> dict | None:
    """平均 Ratio に最も近い画像のパス群を返す"""
    csv_path = folder_dir / "ellipse_results.csv"
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok":
                continue
            major = float(row["major_axis"])
            minor = float(row["minor_axis"])
            rows.append({"filename": row["filename"], "ratio": minor / major})
    if not rows:
        return None
    best = min(rows, key=lambda r: abs(r["ratio"] - mean_ratio))
    json_stem  = Path(best["filename"]).stem          # IMG_..._red
    base_stem  = json_stem[:-4] if json_stem.endswith("_red") else json_stem
    return {
        "ell_path": folder_dir / "ellipse_overlay" / f"{json_stem}_ellipse.png",
        "roi_path": folder_dir / "roi"             / f"{base_stem}_roi.png",
    }


def load_bgr2rgb(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── 1. 楕円オーバーレイ グリッド ────────────────────────

def make_ellipse_grid():
    # 列順（遠視→近視）は最初のpupilのデータから決める
    ref_records = load_ellipse_data(PUPILS[0]["run"])
    diopters_sorted = sorted([r["diopter"] for r in ref_records], reverse=True)  # +4 ... -5
    n_cols = len(diopters_sorted)
    n_rows = len(PUPILS)

    cell_w = 2.0
    cell_h = cell_w * (IMG_H / IMG_W)
    label_w = 0.9
    col_label_h = 0.35

    fig_w = label_w + cell_w * n_cols
    fig_h = col_label_h + cell_h * n_rows

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    gs = gridspec.GridSpec(
        n_rows + 1, n_cols + 1,
        figure=fig,
        left=label_w / fig_w,
        right=1.0,
        top=1.0 - col_label_h / fig_h,
        bottom=0.0,
        wspace=0.02,
        hspace=0.02,
        height_ratios=[col_label_h / cell_h] + [1.0] * n_rows,
        width_ratios=[0.0] + [1.0] * n_cols,
    )

    # 列ラベル（屈折力）
    for col_i, d in enumerate(diopters_sorted):
        ax = fig.add_subplot(gs[0, col_i + 1])
        ax.axis("off")
        ax.text(0.5, 0.5, f"{d:+.2f}D", ha="center", va="center",
                fontsize=9, fontweight="bold", transform=ax.transAxes)

    # 各行（瞳孔径）
    for row_i, pupil in enumerate(PUPILS):
        records = load_ellipse_data(pupil["run"])
        rec_by_d = {r["diopter"]: r for r in records}

        # 行ラベル
        ax_rl = fig.add_subplot(gs[row_i + 1, 0])
        ax_rl.axis("off")
        ax_rl.text(1.0, 0.5, pupil["label"], ha="right", va="center",
                   fontsize=9, fontweight="bold", transform=ax_rl.transAxes)

        for col_i, d in enumerate(diopters_sorted):
            ax = fig.add_subplot(gs[row_i + 1, col_i + 1])
            ax.axis("off")
            rec = rec_by_d.get(d)
            img = None
            if rec:
                paths = pick_representative(rec["folder_dir"], rec["ratio_mean"])
                if paths:
                    img = load_bgr2rgb(paths["ell_path"])
            if img is not None:
                ax.imshow(img, interpolation="bilinear")
            else:
                ax.set_facecolor("#222")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        color="white", transform=ax.transAxes)

    out_path = OUT_DIR / "ellipse_grid_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Ellipse grid: {out_path}")


# ── 2. 3本の Ratio フィット曲線を重ねてプロット ─────────

def make_combined_fit_plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    D_fine = np.linspace(-5.5, 4.5, 400)

    for pupil in PUPILS:
        records = load_ellipse_data(pupil["run"])
        D_vals  = np.array([r["diopter"]    for r in records])
        R_means = np.array([r["ratio_mean"] for r in records])
        R_stds  = np.array([r["ratio_std"]  for r in records])

        # フィット
        coeffs = np.polyfit(D_vals, R_means, deg=2)
        a, b, c = coeffs
        poly = np.poly1d(coeffs)
        R_pred = poly(D_vals)
        ss_res = np.sum((R_means - R_pred) ** 2)
        ss_tot = np.sum((R_means - np.mean(R_means)) ** 2)
        r2 = 1 - ss_res / ss_tot

        col = pupil["color"]
        label_fit = (f"{pupil['label']}:  "
                     f"{a:.4f}D² + ({b:.4f})D + {c:.4f}   R²={r2:.4f}")

        ax.errorbar(D_vals, R_means, yerr=R_stds,
                    fmt="o", color=col, capsize=4, markersize=5, zorder=5)
        ax.plot(D_fine, poly(D_fine), "-", color=col, linewidth=2, label=label_fit)

    ax.set_xlabel("Refraction power (D)", fontsize=12)
    ax.set_ylabel("Minor / Major axis ratio", fontsize=12)
    ax.set_title("Ellipse ratio vs Refraction power — all pupil diameters", fontsize=13)
    ax.set_xticks(np.arange(-5, 5))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()

    out_path = OUT_DIR / "fit_curves_combined.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Combined fit plot: {out_path}")


# ── メイン ────────────────────────────────────────────────

def main():
    make_ellipse_grid()
    make_combined_fit_plot()
    print("Done.")


if __name__ == "__main__":
    main()
