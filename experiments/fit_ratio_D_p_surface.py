"""
p20〜p45 の ratio-D データを統合し、ratio = f(D, p) の2変数関数を構築。

手法:
  1. 各グループで Model A (Dual Logistic) をフィット → パラメータ取得
  2. 各パラメータを p の関数として多項式フィット
  3. ratio = f(D, p) の統合サーフェスモデルを構築
  4. グラフ: 3Dサーフェス、等高線、D別スライス、p別スライス

Run:
  python experiments/fit_ratio_D_p_surface.py
"""

import csv
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

ELLIPSE_DIR = Path("data/simu_masked/ellipse_flat75")
OUT_DIR     = ELLIPSE_DIR / "fitting_surface"
GROUPS      = ["p20", "p25", "p30", "p35", "p40", "p45"]
P_VALUES    = np.array([20, 25, 30, 35, 40, 45], dtype=float)

GROUP_COLORS = {
    "p20": "#f1c40f", "p25": "#2ecc71",
    "p30": "#3498db", "p35": "#9b59b6",
    "p40": "#1abc9c", "p45": "#e91e63",
}


# ── Model A: Dual Logistic ────────────────────────────────────────────────────

def logistic_myo(D, a_m, k_m, x0_m, ratio_0):
    off = ratio_0 - a_m / (1 + np.exp(k_m * x0_m))
    return a_m / (1 + np.exp(k_m * (D + x0_m))) + off  # D<0側

def logistic_hyp(D, a_h, k_h, x0_h, ratio_0):
    off = ratio_0 - a_h / (1 + np.exp(k_h * x0_h))   # D=0でratio_0になるよう符号修正
    return a_h / (1 + np.exp(-k_h * (D - x0_h))) + off  # D>0側

def model_A(D, a_m, k_m, x0_m, a_h, k_h, x0_h, ratio_0):
    return np.where(D <= 0,
                    logistic_myo(D, a_m, k_m, x0_m, ratio_0),
                    logistic_hyp(D, a_h, k_h, x0_h, ratio_0))

def fit_A(ds, ratios, ratio_0):
    def f(D, a_m, k_m, x0_m, a_h, k_h, x0_h):
        return model_A(D, a_m, k_m, x0_m, a_h, k_h, x0_h, ratio_0)
    p0     = [0.9, 0.7, 3.0, 0.7, 0.4, 3.5]
    bounds = ([0.01, 0.01, 0.5, 0.01, 0.01, 0.5],
              [2.0,  5.0, 10.0, 2.0,  5.0, 10.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(f, ds, ratios, p0=p0, bounds=bounds, maxfev=20000)
    pred = f(ds, *popt)
    r2   = 1 - np.sum((ratios-pred)**2) / np.sum((ratios-ratios.mean())**2)
    return popt, r2


def load_group(group):
    d_ratio = {}
    for row in csv.DictReader(open(ELLIPSE_DIR / group / "per_image_ellipse.csv", encoding="utf-8")):
        if row["status"] != "ok" or not row["D"] or not row["ratio"]:
            continue
        d_ratio[float(row["D"])] = float(row["ratio"])
    return d_ratio


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 各グループで Model A フィット ──────────────────────────
    param_names = ["a_m", "k_m", "x0_m", "a_h", "k_h", "x0_h"]
    params_per_group = {}   # group -> popt (6 params)
    ratio0_per_group = {}

    print("=== Step 1: Model A fit per group ===")
    for group, p_val in zip(GROUPS, P_VALUES):
        d_ratio = load_group(group)
        ds      = np.array(sorted(d_ratio.keys()))
        ratios  = np.array([d_ratio[d] for d in ds])
        ratio_0 = d_ratio.get(0.0, ratios[np.argmin(np.abs(ds))].item())
        popt, r2 = fit_A(ds, ratios, ratio_0)
        params_per_group[group] = popt
        ratio0_per_group[group] = ratio_0
        named = dict(zip(param_names, popt.round(4)))
        print(f"  {group}: R2={r2:.4f}  ratio_0={ratio_0:.4f}  {named}")

    # ── Step 2: 各パラメータを p の関数として多項式フィット ─────────────
    print("\n=== Step 2: Parameter vs p polynomial fit ===")
    p_arr   = P_VALUES
    poly_coeffs = {}   # param_name -> coeffs (quadratic)

    fig_params, axes_p = plt.subplots(2, 4, figsize=(16, 7))
    axes_p = axes_p.flatten()

    all_param_vals = {name: np.array([params_per_group[g][i] for g in GROUPS])
                      for i, name in enumerate(param_names)}
    ratio0_arr = np.array([ratio0_per_group[g] for g in GROUPS])
    all_param_vals["ratio_0"] = ratio0_arr

    for ax_i, pname in enumerate(param_names + ["ratio_0"]):
        vals   = all_param_vals[pname]
        coeffs = np.polyfit(p_arr, vals, deg=2)
        poly_coeffs[pname] = coeffs
        pred   = np.polyval(coeffs, p_arr)
        r2_p   = 1 - np.sum((vals-pred)**2) / np.sum((vals-vals.mean())**2)
        print(f"  {pname}: poly2 R2={r2_p:.4f}  coeffs={coeffs.round(6)}")

        ax = axes_p[ax_i]
        ax.scatter(p_arr, vals, color="#3498db", zorder=3)
        p_fine = np.linspace(18, 47, 100)
        ax.plot(p_fine, np.polyval(coeffs, p_fine), "r-", linewidth=1.5)
        ax.set_title(f"{pname}  R2={r2_p:.3f}", fontsize=9)
        ax.set_xlabel("p")
        ax.grid(alpha=0.3)

    axes_p[-1].set_visible(False)
    plt.suptitle("Parameter vs pupil size p (quadratic fit)", fontsize=11)
    plt.tight_layout()
    fig_params.savefig(OUT_DIR / "params_vs_p.png", dpi=150)
    plt.close(fig_params)

    # ── Step 3: 統合モデル f(D, p) ────────────────────────────────────
    def unified_model(D, p):
        a_m  = np.polyval(poly_coeffs["a_m"],  p)
        k_m  = np.polyval(poly_coeffs["k_m"],  p)
        x0_m = np.polyval(poly_coeffs["x0_m"], p)
        a_h  = np.polyval(poly_coeffs["a_h"],  p)
        k_h  = np.polyval(poly_coeffs["k_h"],  p)
        x0_h = np.polyval(poly_coeffs["x0_h"], p)
        r0   = np.polyval(poly_coeffs["ratio_0"], p)
        return model_A(D, a_m, k_m, x0_m, a_h, k_h, x0_h, r0)

    # ── 残差確認 ──────────────────────────────────────────────────────
    print("\n=== Step 3: Unified model residuals ===")
    for group, p_val in zip(GROUPS, P_VALUES):
        d_ratio = load_group(group)
        ds      = np.array(sorted(d_ratio.keys()))
        ratios  = np.array([d_ratio[d] for d in ds])
        pred    = np.array([unified_model(d, p_val) for d in ds])
        r2      = 1 - np.sum((ratios-pred)**2) / np.sum((ratios-ratios.mean())**2)
        print(f"  {group}: R2={r2:.4f}")

    # ── グラフ1: 3Dサーフェス ─────────────────────────────────────────
    D_grid = np.linspace(-8, 8, 80)
    P_grid = np.linspace(20, 45, 60)
    DD, PP = np.meshgrid(D_grid, P_grid)
    RR     = unified_model(DD, PP)

    fig3d = plt.figure(figsize=(10, 7))
    ax3d  = fig3d.add_subplot(111, projection="3d")
    ax3d.plot_surface(DD, PP, RR, cmap="viridis", alpha=0.8)

    for group, p_val in zip(GROUPS, P_VALUES):
        d_ratio = load_group(group)
        ds  = np.array(sorted(d_ratio.keys()))
        rs  = np.array([d_ratio[d] for d in ds])
        col = GROUP_COLORS[group]
        ax3d.scatter(ds, np.full_like(ds, p_val), rs,
                     color=col, s=15, zorder=5, label=group)

    ax3d.set_xlabel("D (Diopter)")
    ax3d.set_ylabel("p (pupil size)")
    ax3d.set_zlabel("ratio")
    ax3d.set_title("Unified surface: ratio = f(D, p)")
    ax3d.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    fig3d.savefig(OUT_DIR / "surface_3d.png", dpi=150)
    plt.close(fig3d)

    # ── グラフ2: 等高線 ───────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    cp = ax2.contourf(DD, PP, RR, levels=20, cmap="viridis")
    plt.colorbar(cp, ax=ax2, label="ratio")
    ax2.contour(DD, PP, RR, levels=20, colors="white", linewidths=0.4, alpha=0.4)
    for group, p_val in zip(GROUPS, P_VALUES):
        d_ratio = load_group(group)
        ds = np.array(sorted(d_ratio.keys()))
        rs = np.array([d_ratio[d] for d in ds])
        ax2.scatter(ds, np.full_like(ds, p_val),
                    color=GROUP_COLORS[group], s=12, zorder=5, label=group)
    ax2.set_xlabel("D (Diopter)")
    ax2.set_ylabel("p (pupil size)")
    ax2.set_title("ratio = f(D, p)  contour")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.2)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / "surface_contour.png", dpi=150)
    plt.close(fig2)

    # ── グラフ3: p別スライス (D軸) ───────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    D_fine = np.linspace(-8, 8, 300)
    for group, p_val in zip(GROUPS, P_VALUES):
        col = GROUP_COLORS[group]
        d_ratio = load_group(group)
        ds  = np.array(sorted(d_ratio.keys()))
        rs  = np.array([d_ratio[d] for d in ds])
        ax3.scatter(ds, rs, color=col, s=20, alpha=0.6)
        ax3.plot(D_fine, unified_model(D_fine, p_val),
                 color=col, linewidth=2, label=f"{group} (p={int(p_val)})")
    ax3.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax3.set_xlabel("D (Diopter)")
    ax3.set_ylabel("ratio")
    ax3.set_title("Unified model: ratio vs D  per pupil size")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    fig3.savefig(OUT_DIR / "slice_by_p.png", dpi=150)
    plt.close(fig3)

    # ── グラフ4: D別スライス (p軸) ───────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(9, 6))
    P_fine   = np.linspace(20, 45, 200)
    D_slices = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
    cmap4    = plt.cm.coolwarm
    for d_s in D_slices:
        col = cmap4((d_s + 8) / 16)
        ax4.plot(P_fine, unified_model(d_s, P_fine),
                 color=col, linewidth=2, label=f"D={d_s:+.0f}")
    ax4.set_xlabel("p (pupil size)")
    ax4.set_ylabel("ratio")
    ax4.set_title("Unified model: ratio vs p  per D value")
    ax4.legend(fontsize=8, ncol=3)
    ax4.grid(alpha=0.3)
    plt.tight_layout()
    fig4.savefig(OUT_DIR / "slice_by_D.png", dpi=150)
    plt.close(fig4)

    print(f"\nAll saved to: {OUT_DIR}")

    # ── CSV: poly_coeffs ─────────────────────────────────────────────
    csv_path = OUT_DIR / "poly_coeffs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["param", "c2", "c1", "c0"])
        for pname in param_names + ["ratio_0"]:
            c = poly_coeffs[pname]
            w.writerow([pname, f"{c[0]:.8f}", f"{c[1]:.8f}", f"{c[2]:.8f}"])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
