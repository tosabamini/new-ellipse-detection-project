"""
Simulation ratio / area を 0D で分割しない「1本の連続式」 f(D, p) で近似する。

方式 (D=0 での近視/遠視分割は廃止):
  1. 各瞳孔径グループで D の単一多項式 (deg=DEGD) をフィット
       y = sum_i c_i(p) * Dn^i     (Dn = D/8 に正規化)
  2. 各係数 c_i を p の多項式 (deg=DEGP) で補間
       c_i(p) = sum_j b_ij * pn^j   (pn = (p-27.5)/17.5 に正規化)
  => f(D,p) = sum_i [ sum_j b_ij pn^j ] Dn^i   ← 単一の閉形式 2 変数式

DEGD = 8 と 10 の両方を出力し、グラフで比較できるようにする。

出力: data/simu_masked/ellipse_flat75/fitting_unified/
  slice_deg8_ratio.png   slice_deg8_area.png
  slice_deg10_ratio.png  slice_deg10_area.png
  surface3d_deg{8,10}_{ratio,area}.png
  coeffs_deg{8,10}_{ratio,area}.csv

Run:
  python experiments/fit_unified_single_surface.py
"""

import csv
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ELLIPSE_DIR = Path("data/simu_masked/ellipse_flat75")
OUT_DIR     = ELLIPSE_DIR / "fitting_unified"

# 正規化定数 (数値安定化のため)
D_SCALE = 8.0
P_CEN, P_SCALE = 27.5, 17.5   # p10..p45 -> ~[-1, 1]

DEGP = 4   # p 方向の係数補間次数 (8 グループに対し 5 パラメータ)

GROUP_COLORS = {
    10: "#e74c3c", 15: "#e67e22", 20: "#f1c40f", 25: "#2ecc71",
    30: "#1abc9c", 35: "#3498db", 40: "#9b59b6", 45: "#e91e63",
}


def normD(D): return np.asarray(D) / D_SCALE
def normP(p): return (np.asarray(p) - P_CEN) / P_SCALE
def r2(y, yhat):
    ss = np.sum((y - yhat) ** 2); tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss / tot if tot > 0 else float("nan")


def load_groups():
    groups = {}
    for f in sorted(glob.glob(str(ELLIPSE_DIR / "p*" / "per_image_ellipse.csv"))):
        pg = int(re.search(r"p(\d+)", f).group(1))
        rows = [r for r in csv.DictReader(open(f, encoding="utf-8")) if r["status"] == "ok"]
        D = np.array([float(r["D"]) for r in rows])
        major = np.array([float(r["major"]) for r in rows])
        minor = np.array([float(r["minor"]) for r in rows])
        idx = np.argsort(D)
        groups[pg] = dict(D=D[idx], ratio=(minor / major)[idx], area=(major * minor)[idx])
    return groups


def build_surface(groups, var, DEGD):
    """f(D,p) を構築し、(評価関数, 係数行列 b[i,j], 各グループR2) を返す"""
    P_LIST = np.array(sorted(groups))
    # Step1: グループごとに deg-DEGD in normD
    coeffs_per_group = np.array([
        np.polyfit(normD(groups[pg]["D"]), groups[pg][var], DEGD)  # 高次→低次, 長さ DEGD+1
        for pg in P_LIST
    ])
    # Step2: 各係数を p の多項式で補間  -> b[i] = polyfit(pn, coeff_i)
    pn = normP(P_LIST)
    coeff_polys = [np.polyfit(pn, coeffs_per_group[:, k], DEGP) for k in range(DEGD + 1)]

    def f(D, p):
        cvec = np.array([np.polyval(coeff_polys[k], normP(p)) for k in range(DEGD + 1)])
        return np.polyval(cvec, normD(D))

    per_r2 = {pg: r2(groups[pg][var], f(groups[pg]["D"], pg)) for pg in P_LIST}
    return f, coeff_polys, per_r2


def save_coeffs(coeff_polys, DEGD, var, tag):
    """b[i,j]: D^i の係数を与える pn 多項式の係数 (高次→低次)"""
    path = OUT_DIR / f"coeffs_{tag}_{var}.csv"
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["# y = sum_i ci(pn) * Dn^i ; Dn=D/8 ; pn=(p-27.5)/17.5"])
        w.writerow(["D_power_i"] + [f"pn^{DEGP-j}" for j in range(DEGP + 1)])
        for i in range(DEGD + 1):
            d_power = DEGD - i   # polyval は高次→低次なので index 0 が D^DEGD
            w.writerow([d_power] + [f"{v:.8e}" for v in coeff_polys[i]])
    return path


def plot_slices(groups, f, per_r2, var, tag, ylabel):
    P_LIST = sorted(groups)
    D_fine = np.linspace(-8, 8, 400)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for pg in P_LIST:
        col = GROUP_COLORS[pg]
        g = groups[pg]
        ax.scatter(g["D"], g[var], color=col, s=16, alpha=0.7, zorder=3)
        ax.plot(D_fine, f(D_fine, pg), color=col, lw=1.8,
                label=f"p{pg}  R2={per_r2[pg]:.4f}")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("D (Diopter)")
    ax.set_ylabel(ylabel)
    worst = min(per_r2, key=per_r2.get)
    ax.set_title(f"Unified single equation f(D,p)  [{tag}]  {var}\n"
                 f"no D=0 split | worst = p{worst} (R2={per_r2[worst]:.4f})")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = OUT_DIR / f"slice_{tag}_{var}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_surface3d(groups, f, var, tag, zlabel):
    D_grid = np.linspace(-8, 8, 80)
    P_grid = np.linspace(10, 45, 70)
    DD, PP = np.meshgrid(D_grid, P_grid)
    ZZ = np.array([[f(d, p) for d in D_grid] for p in P_grid])
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(DD, PP, ZZ, cmap="viridis", alpha=0.8)
    for pg in sorted(groups):
        g = groups[pg]
        ax.scatter(g["D"], np.full_like(g["D"], pg), g[var],
                   color=GROUP_COLORS[pg], s=12, zorder=5, label=f"p{pg}")
    ax.set_xlabel("D"); ax.set_ylabel("p (sim units)"); ax.set_zlabel(zlabel)
    ax.set_title(f"Unified surface f(D,p)  [{tag}]  {var}")
    ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    path = OUT_DIR / f"surface3d_{tag}_{var}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    groups = load_groups()

    specs = [("ratio", "ratio = minor/major"), ("area", "area = major*minor (px^2)")]

    for DEGD in [8, 10]:
        tag = f"deg{DEGD}"
        print(f"\n========== DEGD={DEGD}, DEGP={DEGP} ==========")
        for var, ylabel in specs:
            f, coeff_polys, per_r2 = build_surface(groups, var, DEGD)
            worst = min(per_r2, key=per_r2.get)
            print(f"  {var:>5}: worst=p{worst} R2={per_r2[worst]:.4f} | " +
                  " ".join(f"p{pg}:{per_r2[pg]:.3f}" for pg in sorted(groups)))
            plot_slices(groups, f, per_r2, var, tag, ylabel)
            plot_surface3d(groups, f, var, tag, ylabel)
            save_coeffs(coeff_polys, DEGD, var, tag)

    print(f"\nAll saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
