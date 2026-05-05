"""
屈折力別サマリグリッド画像の生成

- 列：遠視（左）→ 近視（右）  +4D … -5D
- 行1：ROI、行2：RED（RedEnhance）、行3：RED + 楕円
- 各列：平均Ratioに最も近い画像を採用
- 縦横比は維持（800×600, 4:3）
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="例: model_eye_3mm_v001")
    return parser.parse_args()

IMG_H, IMG_W = 600, 800  # 実測サイズ


def folder_to_diopter(name: str) -> float:
    parts = name.split("_")
    sign  = parts[1]
    major = int(parts[2])
    minor = int(parts[3][:-1])
    val   = major + minor / 100.0
    return -val if sign == "M" else (0.0 if sign == "Z" else val)


def pick_representative(folder_dir: Path) -> dict | None:
    """Ratioが平均値に最も近い画像のステム情報を返す"""
    csv_path = folder_dir / "ellipse_results.csv"
    if not csv_path.exists():
        return None

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

    mean_ratio = np.mean([r["ratio"] for r in rows])
    best = min(rows, key=lambda r: abs(r["ratio"] - mean_ratio))

    # filename例: IMG_20260505_150751_777_red.json
    json_stem = Path(best["filename"]).stem          # IMG_..._red
    base_stem  = json_stem[:-4] if json_stem.endswith("_red") else json_stem  # IMG_...

    return {
        "diopter":   folder_to_diopter(folder_dir.name),
        "folder":    folder_dir.name,
        "ratio":     best["ratio"],
        "mean_ratio": mean_ratio,
        "roi_path":  folder_dir / "roi"             / f"{base_stem}_roi.png",
        "red_path":  folder_dir / "red"             / f"{json_stem}.png",
        "ell_path":  folder_dir / "ellipse_overlay" / f"{json_stem}_ellipse.png",
    }


def load_bgr2rgb(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    args = parse_args()
    run_root = MODEL_EYE_RUNS_DIR / args.run_name
    out_dir = run_root / "analysis"
    out_dir.mkdir(exist_ok=True)

    folders = sorted(
        d for d in run_root.iterdir()
        if d.is_dir() and (d / "ellipse_results.csv").exists()
    )
    recs = [r for d in folders if (r := pick_representative(d)) is not None]
    # 左が遠視(+), 右が近視(-): 降順ソート
    recs.sort(key=lambda x: x["diopter"], reverse=True)

    n_cols = len(recs)   # 10
    n_rows = 3

    # ── 図のサイズ：セルを IMG_W × IMG_H の比率に合わせる ──
    cell_w = 2.0          # inch/列
    cell_h = cell_w * (IMG_H / IMG_W)   # = 1.5 inch/行
    label_h = 0.35        # 列ラベル用の余白（inch）
    row_label_w = 0.55    # 行ラベル用の余白（inch）

    fig_w = row_label_w + cell_w * n_cols
    fig_h = label_h + cell_h * n_rows

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)

    # GridSpec: 行ラベル列 + 画像列×10、上部ラベル行 + 画像行×3
    gs = gridspec.GridSpec(
        n_rows + 1, n_cols + 1,
        figure=fig,
        left=row_label_w / fig_w,
        right=1.0,
        top=1.0 - label_h / fig_h,
        bottom=0.0,
        wspace=0.02,
        hspace=0.02,
        height_ratios=[label_h / cell_h] + [1.0] * n_rows,
        width_ratios=[0.0] + [1.0] * n_cols,
    )

    row_labels = ["ROI", "RED\n(RedEnhance)", "RED\n+ Ellipse"]

    for col_i, rec in enumerate(recs):
        # ── 列ラベル（屈折力） ──────────────────────────
        ax_label = fig.add_subplot(gs[0, col_i + 1])
        ax_label.axis("off")
        d = rec["diopter"]
        label_str = f"{d:+.2f}D"
        ax_label.text(0.5, 0.5, label_str,
                      ha="center", va="center",
                      fontsize=9, fontweight="bold",
                      transform=ax_label.transAxes)

        imgs = [
            load_bgr2rgb(rec["roi_path"]),
            load_bgr2rgb(rec["red_path"]),
            load_bgr2rgb(rec["ell_path"]),
        ]

        for row_i, img in enumerate(imgs):
            ax = fig.add_subplot(gs[row_i + 1, col_i + 1])
            if img is not None:
                ax.imshow(img, interpolation="bilinear")
            else:
                ax.set_facecolor("gray")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="white")
            ax.axis("off")

    # ── 行ラベル ─────────────────────────────────────────
    for row_i, label in enumerate(row_labels):
        ax_rl = fig.add_subplot(gs[row_i + 1, 0])
        ax_rl.axis("off")
        ax_rl.text(1.0, 0.5, label,
                   ha="right", va="center",
                   fontsize=8, fontweight="bold",
                   transform=ax_rl.transAxes)

    out_path = out_dir / "summary_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # 採用画像のログ
    print(f"\n{'Diopter':>10}  {'Folder':>20}  {'mean_ratio':>10}  {'used_ratio':>10}")
    for r in recs:
        print(f"{r['diopter']:>+10.2f}  {r['folder']:>20}  {r['mean_ratio']:>10.4f}  {r['ratio']:>10.4f}")


if __name__ == "__main__":
    main()
