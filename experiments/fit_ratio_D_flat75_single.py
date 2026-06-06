"""
binary_mask_flat75 の ratio-D データを -8D〜+8D 全点一括で1本のロジスティックにフィット。
近視・遠視を分けず |D| を変数として使用。

出力:
  data/simu_masked/ellipse_flat75/fitting_single/
    individual/<group>_ratio_D_single.png
    all_groups_ratio_D_single.png
    fit_summary_single.csv

Run:
  python experiments/fit_ratio_D_flat75_single.py
"""

import csv
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ELLIPSE_DIR = Path("data/simu_masked/ellipse_flat75")
OUT_DIR     = ELLIPSE_DIR / "fitting_single"
GROUPS      = ["p10", "p15", "p20", "p25", "p30", "p35", "p40", "p45"]

GROUP_COLORS = {
    "p10": "#e74c3c", "p15": "#e67e22", "p20": "#f1c40f", "p25": "#2ecc71",
    "p30": "#3498db", "p35": "#9b59b6", "p40": "#1abc9c", "p45": "#e91e63",
}


def logistic_c0(abs_d, a, k, x0, ratio_0):
    offset = ratio_0 - a / (1.0 + np.exp(k * x0))
    return a / (1.0 + np.exp(-k * (abs_d - x0))) + offset


def fit_single(abs_d_arr, ratio_arr, ratio_0):
    def model(x, a, k, x0):
        return logistic_c0(x, a, k, x0, ratio_0)
    p0     = [0.9, 0.6, 3.5]
    bounds = ([0.01, 0.01, 0.5], [2.0, 5.0, 10.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(model, abs_d_arr, ratio_arr,
                            p0=p0, bounds=bounds, maxfev=10000)
    a, k, x0 = popt
    pred   = model(abs_d_arr, *popt)
    ss_res = np.sum((ratio_arr - pred) ** 2)
    ss_tot = np.sum((ratio_arr - np.mean(ratio_arr)) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    return a, k, x0, r2


def load_group(group):
    csv_path = ELLIPSE_DIR / group / "per_image_ellipse.csv"
    d_ratio  = {}
    for row in csv.DictReader(open(csv_path, encoding="utf-8")):
        if row["status"] != "ok" or not row["D"] or not row["ratio"]:
            continue
        d_ratio[float(row["D"])] = float(row["ratio"])
    return d_ratio


def main():
    (OUT_DIR / "individual").mkdir(parents=True, exist_ok=True)

    summary_rows = []
    fit_results  = {}

    for group in GROUPS:
        d_ratio = load_group(group)
        ds      = np.array(sorted(d_ratio.keys()))
        ratios  = np.array([d_ratio[d] for d in ds])
        ratio_0 = d_ratio.get(0.0, ratios[np.argmin(np.abs(ds))].item())

        # 全点を |D| に変換して一括フィット
        abs_ds  = np.abs(ds)
        a, k, x0, r2 = fit_single(abs_ds, ratios, ratio_0)

        fit_results[group] = (ratio_0, a, k, x0, r2)
        print(f"[{group}] ratio_0={ratio_0:.4f}  a={a:.4f} k={k:.4f} x0={x0:.4f}  R2={r2:.4f}")

        summary_rows.append({
            "group": group, "ratio_0": ratio_0,
            "a": a, "k": k, "x0": x0, "R2": r2,
            "formula": f"ratio = {a:.4f}/(1+exp(-{k:.4f}*(|D|-{x0:.4f}))) + offset",
        })

        # ── 個別グラフ ────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        col = GROUP_COLORS[group]

        ax.scatter(ds, ratios, color=col, s=40, zorder=3, label="data")

        d_fine = np.linspace(0, 8, 300)
        curve  = logistic_c0(d_fine, a, k, x0, ratio_0)
        ax.plot(-d_fine, curve, color=col, linewidth=2.5)
        ax.plot( d_fine, curve, color=col, linewidth=2.5,
                 label=f"a={a:.3f} k={k:.3f} x0={x0:.3f}\nR2={r2:.4f}")

        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("D (Diopter)")
        ax.set_ylabel("ratio (minor/major)")
        ax.set_title(f"{group}  single ratio-D fit  (|D|, all points)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "individual" / f"{group}_ratio_D_single.png", dpi=150)
        plt.close(fig)

    # ── 全グループ重ねグラフ ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Single ratio-D fit per pupil group  (binary_mask_flat75, all points)", fontsize=11)

    d_fine = np.linspace(0, 8, 300)

    for group in GROUPS:
        col = GROUP_COLORS[group]
        d_ratio = load_group(group)
        ds      = np.array(sorted(d_ratio.keys()))
        ratios  = np.array([d_ratio[d] for d in ds])
        ratio_0, a, k, x0, r2 = fit_results[group]
        curve = logistic_c0(d_fine, a, k, x0, ratio_0)

        axes[0].scatter(ds, ratios, color=col, s=15, alpha=0.7, label=group)
        axes[1].plot(-d_fine, curve, color=col, linewidth=2, label=f"{group} (R2={r2:.3f})")
        axes[1].plot( d_fine, curve, color=col, linewidth=2, linestyle="--")

    for ax in axes:
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("D (Diopter)")
        ax.set_ylabel("ratio (minor/major)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_title("Raw data points")
    axes[1].set_title("Fit curves  (solid=myo side, dashed=hyp side)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "all_groups_ratio_D_single.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR / 'all_groups_ratio_D_single.png'}")

    # ── CSV ───────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "fit_summary_single.csv"
    fields   = ["group", "ratio_0", "a", "k", "x0", "R2", "formula"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: (f"{row[k]:.6f}" if isinstance(row[k], float) else row[k])
                        for k in fields})
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
