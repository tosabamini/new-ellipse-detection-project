"""
p20〜p45 の area(=major×minor)-D データを統合し、area = g(D, p) の2変数関数を構築。

手法は fit_ratio_D_p_surface.py と同一:
  1. 各グループで Dual Logistic をフィット (area_0 固定)
  2. 各パラメータを p の関数として多項式フィット
  3. area = g(D, p) の統合サーフェスモデルを構築
  4. グラフ: 3Dサーフェス、等高線、D別スライス、p別スライス

出力先: data/simu_masked/ellipse_flat75/fitting_surface_area/

Run:
  python experiments/fit_area_D_p_surface.py
"""

import csv
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ELLIPSE_DIR = Path("data/simu_masked/ellipse_flat75")
OUT_DIR     = ELLIPSE_DIR / "fitting_surface_area"
GROUPS      = ["p20", "p25", "p30", "p35", "p40", "p45"]
P_VALUES    = np.array([20, 25, 30, 35, 40, 45], dtype=float)

GROUP_COLORS = {
    "p20": "#f1c40f", "p25": "#2ecc71",
    "p30": "#3498db", "p35": "#9b59b6",
    "p40": "#1abc9c", "p45": "#e91e63",
}


# ── Dual Logistic (ratio モデルと同形、変数が area になるだけ) ───────────────

def logistic_myo(D, a_m, k_m, x0_m, area_0):
    off = area_0 - a_m / (1 + np.exp(k_m * x0_m))
    return a_m / (1 + np.exp(k_m * (D + x0_m))) + off  # D<=0 側

def logistic_hyp(D, a_h, k_h, x0_h, area_0):
    off = area_0 - a_h / (1 + np.exp(k_h * x0_h))
    return a_h / (1 + np.exp(-k_h * (D - x0_h))) + off  # D>=0 側

def model_dual(D, a_m, k_m, x0_m, a_h, k_h, x0_h, area_0):
    return np.where(D <= 0,
                    logistic_myo(D, a_m, k_m, x0_m, area_0),
                    logistic_hyp(D, a_h, k_h, x0_h, area_0))

def fit_dual(ds, areas, area_0):
    """area_0 を固定してフィット (C⁰ 連続)"""
    def f(D, a_m, k_m, x0_m, a_h, k_h, x0_h):
        return model_dual(D, a_m, k_m, x0_m, a_h, k_h, x0_h, area_0)

    # 初期値: area は ratio の ~100倍スケールなので a の初期値を大きめに
    a_scale = (areas.max() - area_0) * 0.9
    p0     = [a_scale, 0.5, 3.0, a_scale * 0.8, 0.4, 3.5]
    bounds = ([area_0 * 0.01, 0.01, 0.5,  area_0 * 0.01, 0.01, 0.5],
              [area_0 * 50,   5.0, 12.0,  area_0 * 50,   5.0,  12.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(f, ds, areas, p0=p0, bounds=bounds, maxfev=40000)
    pred = f(ds, *popt)
    ss_res = np.sum((areas - pred) ** 2)
    ss_tot = np.sum((areas - areas.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return popt, r2


def load_group(group):
    """D -> area (=major*minor) の辞書を返す"""
    d_area = {}
    for row in csv.DictReader(open(ELLIPSE_DIR / group / "per_image_ellipse.csv", encoding="utf-8")):
        if row["status"] != "ok" or not row["D"] or not row["major"] or not row["minor"]:
            continue
        d_area[float(row["D"])] = float(row["major"]) * float(row["minor"])
    return d_area


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 各グループで Dual Logistic フィット ───────────────────────
    param_names = ["a_m", "k_m", "x0_m", "a_h", "k_h", "x0_h"]
    params_per_group = {}
    area0_per_group  = {}

    print("=== Step 1: Dual Logistic fit per group (area) ===")
    for group, p_val in zip(GROUPS, P_VALUES):
        d_area  = load_group(group)
        ds      = np.array(sorted(d_area.keys()))
        areas   = np.array([d_area[d] for d in ds])
        area_0  = d_area.get(0.0, areas[np.argmin(np.abs(ds))].item())
        popt, r2 = fit_dual(ds, areas, area_0)
        params_per_group[group] = popt
        area0_per_group[group]  = area_0
        named = dict(zip(param_names, popt.round(2)))
        print(f"  {group}: R2={r2:.4f}  area_0={area_0:.1f}  {named}")

    # ── Step 2: 各パラメータを p の関数として多項式フィット ──────────────
    print("\n=== Step 2: Parameter vs p polynomial fit ===")
    p_arr = P_VALUES

    fig_params, axes_p = plt.subplots(2, 4, figsize=(16, 7))
    axes_p = axes_p.flatten()

    all_param_vals = {name: np.array([params_per_group[g][i] for g in GROUPS])
                      for i, name in enumerate(param_names)}
    area0_arr = np.array([area0_per_group[g] for g in GROUPS])
    all_param_vals["area_0"] = area0_arr

    poly_coeffs = {}
    for ax_i, pname in enumerate(param_names + ["area_0"]):
        vals   = all_param_vals[pname]
        coeffs = np.polyfit(p_arr, vals, deg=2)
        poly_coeffs[pname] = coeffs
        pred   = np.polyval(coeffs, p_arr)
        ss_res = np.sum((vals - pred) ** 2)
        ss_tot = np.sum((vals - vals.mean()) ** 2)
        r2_p = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  {pname}: poly2 R2={r2_p:.4f}  coeffs={coeffs.round(4)}")

        ax = axes_p[ax_i]
        ax.scatter(p_arr, vals, color="#3498db", zorder=3)
        p_fine = np.linspace(18, 47, 100)
        ax.plot(p_fine, np.polyval(coeffs, p_fine), "r-", linewidth=1.5)
        ax.set_title(f"{pname}  R2={r2_p:.3f}", fontsize=9)
        ax.set_xlabel("p")
        ax.grid(alpha=0.3)

    axes_p[-1].set_visible(False)
    plt.suptitle("Area model: Parameter vs pupil size p (quadratic fit)", fontsize=11)
    plt.tight_layout()
    fig_params.savefig(OUT_DIR / "params_vs_p.png", dpi=150)
    plt.close(fig_params)

    # ── Step 3: 統合モデル g(D, p) ───────────────────────────────────────
    def unified_model(D, p):
        a_m   = np.polyval(poly_coeffs["a_m"],   p)
        k_m   = np.polyval(poly_coeffs["k_m"],   p)
        x0_m  = np.polyval(poly_coeffs["x0_m"],  p)
        a_h   = np.polyval(poly_coeffs["a_h"],   p)
        k_h   = np.polyval(poly_coeffs["k_h"],   p)
        x0_h  = np.polyval(poly_coeffs["x0_h"],  p)
        a0    = np.polyval(poly_coeffs["area_0"], p)
        return model_dual(D, a_m, k_m, x0_m, a_h, k_h, x0_h, a0)

    print("\n=== Step 3: Unified model residuals ===")
    for group, p_val in zip(GROUPS, P_VALUES):
        d_area = load_group(group)
        ds     = np.array(sorted(d_area.keys()))
        areas  = np.array([d_area[d] for d in ds])
        pred   = np.array([unified_model(d, p_val) for d in ds])
        ss_res = np.sum((areas - pred) ** 2)
        ss_tot = np.sum((areas - areas.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  {group}: R2={r2:.4f}")

    # ── グラフ1: 3Dサーフェス ─────────────────────────────────────────────
    D_grid = np.linspace(-8, 8, 80)
    P_grid = np.linspace(20, 45, 60)
    DD, PP = np.meshgrid(D_grid, P_grid)
    AA     = unified_model(DD, PP)

    fig3d = plt.figure(figsize=(10, 7))
    ax3d  = fig3d.add_subplot(111, projection="3d")
    ax3d.plot_surface(DD, PP, AA, cmap="plasma", alpha=0.8)
    for group, p_val in zip(GROUPS, P_VALUES):
        d_area = load_group(group)
        ds  = np.array(sorted(d_area.keys()))
        ars = np.array([d_area[d] for d in ds])
        ax3d.scatter(ds, np.full_like(ds, p_val), ars,
                     color=GROUP_COLORS[group], s=15, zorder=5, label=group)
    ax3d.set_xlabel("D (Diopter)")
    ax3d.set_ylabel("p (pupil size)")
    ax3d.set_zlabel("area (px²)")
    ax3d.set_title("Unified surface: area = g(D, p)")
    ax3d.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    fig3d.savefig(OUT_DIR / "surface_3d.png", dpi=150)
    plt.close(fig3d)

    # ── グラフ2: 等高線 ───────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    cp = ax2.contourf(DD, PP, AA, levels=20, cmap="plasma")
    plt.colorbar(cp, ax=ax2, label="area (px²)")
    ax2.contour(DD, PP, AA, levels=20, colors="white", linewidths=0.4, alpha=0.4)
    for group, p_val in zip(GROUPS, P_VALUES):
        d_area = load_group(group)
        ds  = np.array(sorted(d_area.keys()))
        ax2.scatter(ds, np.full_like(ds, p_val),
                    color=GROUP_COLORS[group], s=12, zorder=5, label=group)
    ax2.set_xlabel("D (Diopter)")
    ax2.set_ylabel("p (pupil size)")
    ax2.set_title("area = g(D, p)  contour")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.2)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / "surface_contour.png", dpi=150)
    plt.close(fig2)

    # ── グラフ3: p別スライス (D軸) ───────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    D_fine = np.linspace(-8, 8, 300)
    for group, p_val in zip(GROUPS, P_VALUES):
        col    = GROUP_COLORS[group]
        d_area = load_group(group)
        ds  = np.array(sorted(d_area.keys()))
        ars = np.array([d_area[d] for d in ds])
        ax3.scatter(ds, ars, color=col, s=20, alpha=0.6)
        ax3.plot(D_fine, unified_model(D_fine, p_val),
                 color=col, linewidth=2, label=f"{group} (p={int(p_val)})")
    ax3.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax3.set_xlabel("D (Diopter)")
    ax3.set_ylabel("area = major × minor  (px²)")
    ax3.set_title("Unified model: area vs D  per pupil size")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    fig3.savefig(OUT_DIR / "slice_by_p.png", dpi=150)
    plt.close(fig3)

    # ── グラフ4: D別スライス (p軸) ───────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(9, 6))
    P_fine   = np.linspace(20, 45, 200)
    D_slices = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
    cmap4    = plt.cm.coolwarm
    for d_s in D_slices:
        col = cmap4((d_s + 8) / 16)
        ax4.plot(P_fine, unified_model(d_s, P_fine),
                 color=col, linewidth=2, label=f"D={d_s:+.0f}")
    ax4.set_xlabel("p (pupil size)")
    ax4.set_ylabel("area (px²)")
    ax4.set_title("Unified model: area vs p  per D value")
    ax4.legend(fontsize=8, ncol=3)
    ax4.grid(alpha=0.3)
    plt.tight_layout()
    fig4.savefig(OUT_DIR / "slice_by_D.png", dpi=150)
    plt.close(fig4)

    # ── CSV保存 ───────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "poly_coeffs_area.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["param", "c2", "c1", "c0"])
        for pname in param_names + ["area_0"]:
            c = poly_coeffs[pname]
            w.writerow([pname, f"{c[0]:.8f}", f"{c[1]:.8f}", f"{c[2]:.8f}"])
    print(f"\nAll saved to: {OUT_DIR}")
    print(f"Coefficients: {csv_path}")


if __name__ == "__main__":
    main()
