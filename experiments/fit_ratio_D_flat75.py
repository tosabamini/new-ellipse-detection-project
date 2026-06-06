"""
binary_mask_flat75 の楕円フィット結果から ratio-D 関係式を各瞳孔径ごとに導出。

出力:
  data/simu_masked/ellipse_flat75/fitting/
    individual/<group>_ratio_D.png   各グループ個別グラフ
    all_groups_ratio_D.png           全グループ重ねグラフ
    fit_summary.csv                  フィットパラメータ一覧

Run:
  python experiments/fit_ratio_D_flat75.py
"""

import csv
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ELLIPSE_DIR = Path("data/simu_masked/ellipse_flat75")
OUT_DIR     = ELLIPSE_DIR / "fitting"
GROUPS      = ["p10", "p15", "p20", "p25", "p30", "p35", "p40", "p45"]

GROUP_COLORS = {
    "p10": "#e74c3c", "p15": "#e67e22", "p20": "#f1c40f", "p25": "#2ecc71",
    "p30": "#3498db", "p35": "#9b59b6", "p40": "#1abc9c", "p45": "#e91e63",
}


# ── モデル ────────────────────────────────────────────────────────────────────

def logistic_c0(abs_d, a, k, x0, ratio_0):
    offset = ratio_0 - a / (1.0 + np.exp(k * x0))
    return a / (1.0 + np.exp(-k * (abs_d - x0))) + offset


def fit_side(abs_d_arr, ratio_arr, ratio_0, label=""):
    def model(x, a, k, x0):
        return logistic_c0(x, a, k, x0, ratio_0)
    p0     = [0.9, 0.6, 3.5]
    bounds = ([0.01, 0.01, 0.5], [2.0, 5.0, 10.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(model, abs_d_arr, ratio_arr,
                            p0=p0, bounds=bounds, maxfev=10000)
    a, k, x0 = popt
    pred  = model(abs_d_arr, *popt)
    ss_res = np.sum((ratio_arr - pred) ** 2)
    ss_tot = np.sum((ratio_arr - np.mean(ratio_arr)) ** 2)
    r2    = 1.0 - ss_res / ss_tot
    return a, k, x0, r2


# ── データ読み込み ────────────────────────────────────────────────────────────

def load_group(group: str):
    csv_path = ELLIPSE_DIR / group / "per_image_ellipse.csv"
    d_ratio  = {}
    for row in csv.DictReader(open(csv_path, encoding="utf-8")):
        if row["status"] != "ok" or not row["D"] or not row["ratio"]:
            continue
        d_ratio[float(row["D"])] = float(row["ratio"])
    return d_ratio


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    (OUT_DIR / "individual").mkdir(parents=True, exist_ok=True)

    summary_rows = []
    fit_results  = {}  # group -> (ratio_0, a_m, k_m, x0_m, r2_m, a_h, k_h, x0_h, r2_h)

    for group in GROUPS:
        d_ratio = load_group(group)
        ds      = np.array(sorted(d_ratio.keys()))
        ratios  = np.array([d_ratio[d] for d in ds])
        ratio_0 = d_ratio.get(0.0, float(np.mean([v for d, v in d_ratio.items() if abs(d) < 0.1])))

        # 近視側
        myo_mask = ds <= 0.0
        a_m, k_m, x0_m, r2_m = fit_side(np.abs(ds[myo_mask]), ratios[myo_mask], ratio_0)
        # 遠視側
        hyp_mask = ds >= 0.0
        a_h, k_h, x0_h, r2_h = fit_side(ds[hyp_mask], ratios[hyp_mask], ratio_0)

        fit_results[group] = (ratio_0, a_m, k_m, x0_m, r2_m, a_h, k_h, x0_h, r2_h)

        print(f"[{group}] ratio_0={ratio_0:.4f}")
        print(f"  myo:  a={a_m:.4f} k={k_m:.4f} x0={x0_m:.4f} R2={r2_m:.4f}")
        print(f"  hyp:  a={a_h:.4f} k={k_h:.4f} x0={x0_h:.4f} R2={r2_h:.4f}")

        summary_rows.append({"group": group, "side": "myopia",
                             "ratio_0": ratio_0, "a": a_m, "k": k_m, "x0": x0_m, "R2": r2_m})
        summary_rows.append({"group": group, "side": "hyperopia",
                             "ratio_0": ratio_0, "a": a_h, "k": k_h, "x0": x0_h, "R2": r2_h})

        # ── 個別グラフ ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        col = GROUP_COLORS[group]
        ax.scatter(ds, ratios, color=col, s=40, zorder=3, label="data")

        d_fine_m = np.linspace(0, 8, 300)
        d_fine_h = np.linspace(0, 8, 300)
        ax.plot(-d_fine_m,
                logistic_c0(d_fine_m, a_m, k_m, x0_m, ratio_0),
                color=col, linewidth=2,
                label=f"myo: a={a_m:.3f} k={k_m:.3f} x0={x0_m:.3f} R2={r2_m:.3f}")
        ax.plot(d_fine_h,
                logistic_c0(d_fine_h, a_h, k_h, x0_h, ratio_0),
                color=col, linewidth=2, linestyle="--",
                label=f"hyp: a={a_h:.3f} k={k_h:.3f} x0={x0_h:.3f} R2={r2_h:.3f}")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("D (Diopter)")
        ax.set_ylabel("ratio (minor/major)")
        ax.set_title(f"{group}  ratio-D fit  (binary_mask_flat75)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "individual" / f"{group}_ratio_D.png", dpi=150)
        plt.close(fig)

    # ── 全グループ重ねグラフ ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ratio-D fit per pupil group  (binary_mask_flat75)", fontsize=11)

    for group in GROUPS:
        col = GROUP_COLORS[group]
        d_ratio = load_group(group)
        ds      = np.array(sorted(d_ratio.keys()))
        ratios  = np.array([d_ratio[d] for d in ds])
        ratio_0, a_m, k_m, x0_m, r2_m, a_h, k_h, x0_h, r2_h = fit_results[group]

        d_fine = np.linspace(0, 8, 300)

        # 左: 生データ点
        axes[0].scatter(ds, ratios, color=col, s=20, alpha=0.7, label=group)

        # 右: フィット曲線のみ
        axes[1].plot(-d_fine,
                     logistic_c0(d_fine, a_m, k_m, x0_m, ratio_0),
                     color=col, linewidth=2, label=f"{group} myo")
        axes[1].plot(d_fine,
                     logistic_c0(d_fine, a_h, k_h, x0_h, ratio_0),
                     color=col, linewidth=2, linestyle="--")

    for ax in axes:
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("D (Diopter)")
        ax.set_ylabel("ratio (minor/major)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_title("Raw data points")
    axes[1].set_title("Fit curves  (solid=myo, dashed=hyp)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "all_groups_ratio_D.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR / 'all_groups_ratio_D.png'}")

    # ── CSV ───────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "fit_summary.csv"
    fields   = ["group", "side", "ratio_0", "a", "k", "x0", "R2"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: (f"{row[k]:.6f}" if isinstance(row[k], float) else row[k])
                        for k in fields})
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
