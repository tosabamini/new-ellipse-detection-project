"""
3種類の関数形で ratio-D を -8D〜+8D 全点一括フィット比較。

Model A: 2ロジスティック和  f(D) = L_myo(|D|) * (D<=0) + L_hyp(D) * (D>=0)  ← C0連続
Model B: べき乗            f(D) = a * |D|^n + ratio_0
Model C: 反転ガウシアン    f(D) = ratio_0 + a * (1 - exp(-b * D^2))

出力:
  data/simu_masked/ellipse_flat75/fitting_compare/
    individual/<group>_compare.png
    all_groups_<model>.png  (A/B/C 各1枚)
    fit_summary_compare.csv

Run:
  python experiments/fit_ratio_D_flat75_compare.py
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
OUT_DIR     = ELLIPSE_DIR / "fitting_compare"
GROUPS      = ["p10", "p15", "p20", "p25", "p30", "p35", "p40", "p45"]
GROUP_COLORS = {
    "p10": "#e74c3c", "p15": "#e67e22", "p20": "#f1c40f", "p25": "#2ecc71",
    "p30": "#3498db", "p35": "#9b59b6", "p40": "#1abc9c", "p45": "#e91e63",
}


# ── Model A: 2ロジスティック和 (C0連続) ──────────────────────────────────────
def model_A(D, a_m, k_m, x0_m, a_h, k_h, x0_h, ratio_0):
    """近視側ロジスティック + 遠視側ロジスティック、D=0でC0連続"""
    off_m = ratio_0 - a_m / (1 + np.exp(k_m * x0_m))   # D=0: exp(k_m*x0_m)
    off_h = ratio_0 - a_h / (1 + np.exp(k_h * x0_h))   # D=0: exp(+k_h*x0_h) ← 符号修正
    myo = a_m / (1 + np.exp( k_m * (D + x0_m))) + off_m
    hyp = a_h / (1 + np.exp(-k_h * (D - x0_h))) + off_h
    return np.where(D <= 0, myo, hyp)

def fit_A(ds, ratios, ratio_0):
    def f(D, a_m, k_m, x0_m, a_h, k_h, x0_h):
        return model_A(D, a_m, k_m, x0_m, a_h, k_h, x0_h, ratio_0)
    p0     = [0.9, 0.7, 3.0, 0.7, 0.4, 3.5]
    bounds = ([0.01]*3 + [0.01]*3, [2.0, 5.0, 10.0]*2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(f, ds, ratios, p0=p0, bounds=bounds, maxfev=20000)
    pred = f(ds, *popt)
    r2   = 1 - np.sum((ratios - pred)**2) / np.sum((ratios - ratios.mean())**2)
    return popt, r2

def curve_A(D_fine, popt, ratio_0):
    return model_A(D_fine, *popt, ratio_0)


# ── Model B: べき乗 ──────────────────────────────────────────────────────────
def model_B(D, a, n, ratio_0):
    return a * np.abs(D)**n + ratio_0

def fit_B(ds, ratios, ratio_0):
    def f(D, a, n):
        return model_B(D, a, n, ratio_0)
    p0     = [0.05, 1.2]
    bounds = ([0.001, 0.1], [1.0, 5.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(f, ds, ratios, p0=p0, bounds=bounds, maxfev=10000)
    pred = f(ds, *popt)
    r2   = 1 - np.sum((ratios - pred)**2) / np.sum((ratios - ratios.mean())**2)
    return popt, r2

def curve_B(D_fine, popt, ratio_0):
    return model_B(D_fine, popt[0], popt[1], ratio_0)


# ── Model C: 反転ガウシアン ───────────────────────────────────────────────────
def model_C(D, a, b, ratio_0):
    return ratio_0 + a * (1 - np.exp(-b * D**2))

def fit_C(ds, ratios, ratio_0):
    def f(D, a, b):
        return model_C(D, a, b, ratio_0)
    p0     = [0.8, 0.05]
    bounds = ([0.01, 0.001], [2.0, 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(f, ds, ratios, p0=p0, bounds=bounds, maxfev=10000)
    pred = f(ds, *popt)
    r2   = 1 - np.sum((ratios - pred)**2) / np.sum((ratios - ratios.mean())**2)
    return popt, r2

def curve_C(D_fine, popt, ratio_0):
    return model_C(D_fine, popt[0], popt[1], ratio_0)


# ── データ読み込み ────────────────────────────────────────────────────────────
def load_group(group):
    d_ratio = {}
    for row in csv.DictReader(open(ELLIPSE_DIR / group / "per_image_ellipse.csv", encoding="utf-8")):
        if row["status"] != "ok" or not row["D"] or not row["ratio"]:
            continue
        d_ratio[float(row["D"])] = float(row["ratio"])
    return d_ratio


# ── メイン ────────────────────────────────────────────────────────────────────
def main():
    (OUT_DIR / "individual").mkdir(parents=True, exist_ok=True)

    all_results = {}  # group -> {A, B, C: (popt, r2, ratio_0)}
    summary_rows = []

    D_fine = np.linspace(-8, 8, 400)

    for group in GROUPS:
        d_ratio = load_group(group)
        ds      = np.array(sorted(d_ratio.keys()))
        ratios  = np.array([d_ratio[d] for d in ds])
        ratio_0 = d_ratio.get(0.0, ratios[np.argmin(np.abs(ds))].item())

        popt_A, r2_A = fit_A(ds, ratios, ratio_0)
        popt_B, r2_B = fit_B(ds, ratios, ratio_0)
        popt_C, r2_C = fit_C(ds, ratios, ratio_0)

        all_results[group] = {
            "A": (popt_A, r2_A, ratio_0),
            "B": (popt_B, r2_B, ratio_0),
            "C": (popt_C, r2_C, ratio_0),
        }

        print(f"[{group}]  A: R2={r2_A:.4f}  B: R2={r2_B:.4f}  C: R2={r2_C:.4f}")

        summary_rows.append({
            "group": group,
            "model": "A_dual_logistic",
            "R2": r2_A,
            "params": str(dict(zip(["a_m","k_m","x0_m","a_h","k_h","x0_h"], popt_A.round(4)))),
            "ratio_0": ratio_0,
        })
        summary_rows.append({
            "group": group, "model": "B_power",
            "R2": r2_B,
            "params": f"a={popt_B[0]:.4f} n={popt_B[1]:.4f}",
            "ratio_0": ratio_0,
        })
        summary_rows.append({
            "group": group, "model": "C_inv_gaussian",
            "R2": r2_C,
            "params": f"a={popt_C[0]:.4f} b={popt_C[1]:.4f}",
            "ratio_0": ratio_0,
        })

        # ── 個別グラフ（3モデル重ね）──────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        col = GROUP_COLORS[group]
        ax.scatter(ds, ratios, color="black", s=25, zorder=4, label="data", alpha=0.7)
        ax.plot(D_fine, curve_A(D_fine, popt_A, ratio_0),
                color="#e74c3c", linewidth=2,
                label=f"A dual-logistic  R2={r2_A:.4f}")
        ax.plot(D_fine, curve_B(D_fine, popt_B, ratio_0),
                color="#3498db", linewidth=2, linestyle="--",
                label=f"B power |D|^n     R2={r2_B:.4f}")
        ax.plot(D_fine, curve_C(D_fine, popt_C, ratio_0),
                color="#2ecc71", linewidth=2, linestyle=":",
                label=f"C inv-Gaussian    R2={r2_C:.4f}")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("D (Diopter)")
        ax.set_ylabel("ratio (minor/major)")
        ax.set_title(f"{group}  model comparison (ratio_0={ratio_0:.4f})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "individual" / f"{group}_compare.png", dpi=150)
        plt.close(fig)

    # ── 全グループ重ねグラフ × 3モデル ───────────────────────────────
    for model_key, curve_fn, title, ls in [
        ("A", curve_A, "Model A: Dual Logistic",   "-"),
        ("B", curve_B, "Model B: Power |D|^n",     "--"),
        ("C", curve_C, "Model C: Inv-Gaussian",    ":"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"{title}  —  all pupil groups", fontsize=11)
        for group in GROUPS:
            col = GROUP_COLORS[group]
            d_ratio = load_group(group)
            ds      = np.array(sorted(d_ratio.keys()))
            ratios  = np.array([d_ratio[d] for d in ds])
            popt, r2, ratio_0 = all_results[group][model_key]
            axes[0].scatter(ds, ratios, color=col, s=15, alpha=0.7, label=group)
            axes[1].plot(D_fine, curve_fn(D_fine, popt, ratio_0),
                         color=col, linewidth=2, linestyle=ls,
                         label=f"{group} R2={r2:.3f}")
        for ax in axes:
            ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_xlabel("D (Diopter)")
            ax.set_ylabel("ratio (minor/major)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        axes[0].set_title("Raw data")
        axes[1].set_title("Fit curves")
        plt.tight_layout()
        fig.savefig(OUT_DIR / f"all_groups_model{model_key}.png", dpi=150)
        plt.close(fig)
        print(f"Saved: all_groups_model{model_key}.png")

    # ── CSV ───────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "fit_summary_compare.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group","model","R2","params","ratio_0"])
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: (f"{row[k]:.6f}" if isinstance(row[k], float) else row[k])
                        for k in ["group","model","R2","params","ratio_0"]})
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
