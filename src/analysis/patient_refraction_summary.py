"""
各患者の楕円フィッティング結果と屈折力推定をまとめた画像を生成

フィルタ: status==ok, p_est < 10mm (ノイズ除去), D2 が実数解あり
採用解: D2 (近視側)

出力: data/processed/pipeline_runs/<run_name>/refraction_summary/
      patient_<id>_summary.png  (患者ごと)
"""

import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.build_patient_model import estimate_D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_RUNS_DIR = PROJECT_ROOT / "data/processed/pipeline_runs"

PATIENTS = ["52", "63", "66", "67"]
RUN_NAME = "pipeline_run_v001"

N_COLS = 4          # 1行あたりの画像数
P_EST_MAX = 10.0    # ノイズ除去: p_est がこれ以上は除外
SCALE_FACTOR = 1.3  # 患者画像のpxスケール補正（model eye比）


def load_valid_records(run_root: Path, patient_id: str) -> list[dict]:
    csv_path = run_root / patient_id / "results.csv"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))

    # 重複除去
    seen = set()
    unique = []
    for r in rows:
        if r["filename"] not in seen:
            seen.add(r["filename"])
            unique.append(r)

    valid = []
    for r in unique:
        if r["status"] != "ok":
            continue
        mj = float(r["pred_major_axis_px"])
        mn = float(r["pred_minor_axis_px"])
        ratio = mn / mj
        mj_scaled = mj * SCALE_FACTOR
        p_est, d1, d2 = estimate_D(mj_scaled, ratio)

        if p_est >= P_EST_MAX:      # ノイズ除去
            continue
        if d2 is None:              # 実数解なし
            continue

        valid.append({
            "filename":  r["filename"],
            "stem":      Path(r["filename"]).stem,
            "major":     mj,
            "major_scaled": mj * SCALE_FACTOR,
            "minor":     mn,
            "ratio":     ratio,
            "p_est":     p_est,
            "d1":        d1,
            "d2":        d2,
            "adopted_D": d2,
        })

    valid.sort(key=lambda x: x["filename"])
    return valid


def load_image_rgb(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_patient_summary(run_root: Path, patient_id: str, out_dir: Path):
    records = load_valid_records(run_root, patient_id)
    if not records:
        print(f"  [patient {patient_id}] 有効データなし")
        return

    ell_dir = run_root / patient_id / "ellipse_overlay"
    n = len(records)
    n_rows = (n + N_COLS - 1) // N_COLS

    cell_w = 3.0        # inch/列
    cell_h = cell_w * (360 / 800) + 0.9   # 画像比率 + テキスト余白
    header_h = 0.4

    fig_w = cell_w * N_COLS
    fig_h = header_h + cell_h * n_rows

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)

    fig.suptitle(
        f"Patient {patient_id}  —  refraction estimate (adopted: D2, myopic side)  "
        f"[n={n}]",
        fontsize=11, fontweight="bold", y=1.0 - header_h / (2 * fig_h)
    )

    for idx, rec in enumerate(records):
        row_i = idx // N_COLS
        col_i = idx  % N_COLS

        ax = fig.add_axes([
            col_i / N_COLS,
            1.0 - header_h / fig_h - (row_i + 1) * cell_h / fig_h,
            1.0 / N_COLS,
            cell_h / fig_h,
        ])

        # 楕円オーバーレイ画像
        img_path = ell_dir / f"{rec['stem']}_ellipse.png"
        img = load_image_rgb(img_path)

        img_h_ratio = (cell_h - 0.9) / cell_h
        ax_img = ax
        ax_img.set_position([
            col_i / N_COLS,
            1.0 - header_h / fig_h - (row_i + 1) * cell_h / fig_h + 0.9 / fig_h,
            1.0 / N_COLS,
            (cell_h - 0.9) / fig_h,
        ])

        if img is not None:
            ax_img.imshow(img)
        else:
            ax_img.set_facecolor("#444")
            ax_img.text(0.5, 0.5, "image\nnot found",
                        ha="center", va="center", color="white",
                        transform=ax_img.transAxes, fontsize=7)
        ax_img.axis("off")

        # テキストブロック
        ax_txt = fig.add_axes([
            col_i / N_COLS,
            1.0 - header_h / fig_h - (row_i + 1) * cell_h / fig_h,
            1.0 / N_COLS,
            0.9 / fig_h,
        ])
        ax_txt.axis("off")

        txt = (
            f"{rec['stem']}\n"
            f"major={rec['major']:.1f}px (x{SCALE_FACTOR}={rec['major_scaled']:.1f})  ratio={rec['ratio']:.4f}\n"
            f"p_est={rec['p_est']:.2f}mm\n"
            f"D1={rec['d1']:+.2f}D   D2={rec['d2']:+.2f}D\n"
            f"Adopted: {rec['adopted_D']:+.2f}D"
        )
        ax_txt.text(0.5, 1.0, txt,
                    ha="center", va="top",
                    fontsize=5.5,
                    transform=ax_txt.transAxes,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f4ff",
                              edgecolor="#aaa", linewidth=0.5))

        # 採用D の色分け枠 (近視 = 赤, 遠視 = 青, 0付近 = 緑)
        D_adopted = rec["adopted_D"]
        if D_adopted < -0.5:
            frame_col = "#cc2222"
        elif D_adopted > 0.5:
            frame_col = "#2255cc"
        else:
            frame_col = "#22aa44"

        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(frame_col)
            spine.set_linewidth(2.5)

    out_path = out_dir / f"patient_{patient_id}_summary.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path}")
    return records


def main():
    run_root = PIPELINE_RUNS_DIR / RUN_NAME
    out_dir  = run_root / "refraction_summary"
    out_dir.mkdir(exist_ok=True)

    all_records = {}
    for pid in PATIENTS:
        print(f"[patient {pid}]")
        recs = make_patient_summary(run_root, pid, out_dir)
        if recs:
            all_records[pid] = recs

    # テキストサマリ
    print("\n===== 屈折力推定サマリ =====")
    for pid, recs in all_records.items():
        adopted = [r["adopted_D"] for r in recs]
        print(f"patient {pid}: n={len(recs)}  "
              f"mean={np.mean(adopted):+.2f}D  "
              f"std={np.std(adopted):.2f}D  "
              f"range=[{np.min(adopted):+.2f}, {np.max(adopted):+.2f}]D")

    print(f"\n出力: {out_dir}")


if __name__ == "__main__":
    main()
