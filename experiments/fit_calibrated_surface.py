"""
Simulation モデルを実測 pixel 空間にキャリブレーションした
Ratio・Area の統合式を構築する。

キャリブレーション根拠:
  - 近視安定域 (D=-5, -4, -3) での Real/Sim area 比 k を算出
  - k は D によらず p_mm だけの関数であることを確認済み
  - p15=3mm: k=0.4480,  p25=5mm: k=0.5023,  p35=7mm: k=0.6824

変換式:
  ratio_real(D, p_mm) = ratio_sim(D, p_mm * 5)          [スケール不変]
  area_real(D, p_mm)  = k(p_mm) * area_sim(D, p_mm * 5) [実 px² に変換]

  k(p_mm) は 3 点を通る 2 次多項式で補間。

出力:
  data/simu_masked/ellipse_flat75/fitting_calibrated/
    slice_by_pmm_ratio.png
    slice_by_pmm_area.png
    calibration_k.csv

Run:
  python experiments/fit_calibrated_surface.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── 依存モジュール読み込み ────────────────────────────────────────────────
# ratio surface model (poly_coeffs.csv)
RATIO_COEFFS_PATH = Path("data/simu_masked/ellipse_flat75/fitting_surface/poly_coeffs.csv")
# area surface model (poly_coeffs_area.csv)
AREA_COEFFS_PATH  = Path("data/simu_masked/ellipse_flat75/fitting_surface_area/poly_coeffs_area.csv")
OUT_DIR           = Path("data/simu_masked/ellipse_flat75/fitting_calibrated")
ELLIPSE_DIR       = Path("data/simu_masked/ellipse_flat75")

# ── キャリブレーション定数 ────────────────────────────────────────────────
# 近視安定域 (D=-5,-4,-3) の平均 k = area_real / area_sim
CALIB_P_MM  = np.array([3.0, 5.0, 7.0])
CALIB_K     = np.array([0.4480, 0.5023, 0.6824])
K_POLY      = np.polyfit(CALIB_P_MM, CALIB_K, deg=2)  # 3点通過の2次多項式

# 模型眼の実データ (ellipse_summary.csv から)
REAL_DATA = {
    3.0: {-5.0: 9097.3, -4.0: 7927.9, -3.0: 6460.7, -2.0: 4131.2,
          -1.0: 3146.1,  0.0: 2388.5,  1.0: 2822.9,  2.0: 3796.8,
           3.0: 5010.9,  4.0: 6089.3},
    5.0: {-5.0: 24756.3, -4.0: 21569.4, -3.0: 16513.8, -2.0: 11950.2,
          -1.0: 7338.6,   0.0: 3631.8,   1.0: 3285.2,   2.0: 5260.9,
           3.0: 8396.2,   4.0: 10688.4},
    7.0: {-5.0: 45225.0, -4.0: 39835.4, -3.0: 28162.8, -2.0: 17344.5,
          -1.0: 7320.5,   0.0: 4613.0,   1.0: 5084.1,   2.0: 8046.9,
           3.0: 11641.8,  4.0: 14702.4},
}

# 模型眼の実 ratio データ (per_image_ellipse から直接計算)
def load_real_ratio():
    """模型眼 ellipse_summary から ratio を取得"""
    import pandas as pd, os

    def folder_to_D(folder):
        parts = folder.split('_')
        sign = parts[1]
        d1 = int(parts[2]); d2 = int(parts[3].replace('D',''))
        val = d1 + d2/100
        return -val if sign == 'M' else val

    result = {}
    for mm, pg in [(3.0, 'p15'), (5.0, 'p25'), (7.0, 'p35')]:
        key = 'model_v001' if mm == 7.0 else f'model_{int(mm)}mm_v001'
        path = f'data/processed/model_eye_runs/model_eye_{key}/ellipse_summary.csv'
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['D'] = df['folder'].apply(folder_to_D)
        df['ratio'] = df['mean_minor_axis'] / df['mean_major_axis']
        result[mm] = df.set_index('D')['ratio'].to_dict()
    return result


# ── Ratio surface model ───────────────────────────────────────────────────

def load_ratio_poly():
    coeffs = {}
    for row in csv.DictReader(open(RATIO_COEFFS_PATH, encoding="utf-8")):
        coeffs[row["param"]] = [float(row["c2"]), float(row["c1"]), float(row["c0"])]
    return coeffs

def logistic_myo(D, a_m, k_m, x0_m, r0):
    off = r0 - a_m / (1 + np.exp(k_m * x0_m))
    return a_m / (1 + np.exp(k_m * (D + x0_m))) + off

def logistic_hyp(D, a_h, k_h, x0_h, r0):
    off = r0 - a_h / (1 + np.exp(k_h * x0_h))
    return a_h / (1 + np.exp(-k_h * (D - x0_h))) + off

def ratio_sim(D, p_sim, rc):
    a_m  = np.polyval(rc["a_m"],    p_sim)
    k_m  = np.polyval(rc["k_m"],    p_sim)
    x0_m = np.polyval(rc["x0_m"],   p_sim)
    a_h  = np.polyval(rc["a_h"],    p_sim)
    k_h  = np.polyval(rc["k_h"],    p_sim)
    x0_h = np.polyval(rc["x0_h"],   p_sim)
    r0   = np.polyval(rc["ratio_0"], p_sim)
    return np.where(D <= 0,
                    logistic_myo(D, a_m, k_m, x0_m, r0),
                    logistic_hyp(D, a_h, k_h, x0_h, r0))

def ratio_real(D, p_mm, rc):
    """スケール不変 → p_sim = p_mm * 5 に変換するだけ"""
    return ratio_sim(D, p_mm * 5.0, rc)


# ── Area surface model ────────────────────────────────────────────────────

def load_area_poly():
    coeffs = {}
    for row in csv.DictReader(open(AREA_COEFFS_PATH, encoding="utf-8")):
        coeffs[row["param"]] = [float(row["c2"]), float(row["c1"]), float(row["c0"])]
    return coeffs

def logistic_myo_a(D, a_m, k_m, x0_m, a0):
    off = a0 - a_m / (1 + np.exp(k_m * x0_m))
    return a_m / (1 + np.exp(k_m * (D + x0_m))) + off

def logistic_hyp_a(D, a_h, k_h, x0_h, a0):
    off = a0 - a_h / (1 + np.exp(k_h * x0_h))
    return a_h / (1 + np.exp(-k_h * (D - x0_h))) + off

def area_sim(D, p_sim, ac):
    a_m  = np.polyval(ac["a_m"],   p_sim)
    k_m  = np.polyval(ac["k_m"],   p_sim)
    x0_m = np.polyval(ac["x0_m"],  p_sim)
    a_h  = np.polyval(ac["a_h"],   p_sim)
    k_h  = np.polyval(ac["k_h"],   p_sim)
    x0_h = np.polyval(ac["x0_h"],  p_sim)
    a0   = np.polyval(ac["area_0"], p_sim)
    return np.where(D <= 0,
                    logistic_myo_a(D, a_m, k_m, x0_m, a0),
                    logistic_hyp_a(D, a_h, k_h, x0_h, a0))

def k_of_p(p_mm):
    """k(p_mm) = キャリブレーション係数 (近視安定域から導出)"""
    return np.polyval(K_POLY, p_mm)

def area_real(D, p_mm, ac):
    """area_real = k(p_mm) * area_sim(D, p_mm*5)"""
    return k_of_p(p_mm) * area_sim(D, p_mm * 5.0, ac)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rc = load_ratio_poly()
    ac = load_area_poly()
    real_ratio = load_real_ratio()

    D_fine = np.linspace(-8, 8, 300)
    p_mm_list  = [3.0, 5.0, 7.0]
    colors     = {3.0: "#3498db", 5.0: "#2ecc71", 7.0: "#e74c3c"}

    # ── グラフ1: Ratio (calibrated) vs D ────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for pm in p_mm_list:
        col  = colors[pm]
        pred = ratio_real(D_fine, pm, rc)
        ax.plot(D_fine, pred, color=col, linewidth=2,
                label=f"model p={pm}mm  [ratio_real(D, {pm}mm)]")
        # 実データ点
        if pm in real_ratio:
            ds  = np.array(sorted(real_ratio[pm].keys()))
            rs  = np.array([real_ratio[pm][d] for d in ds])
            ax.scatter(ds, rs, color=col, s=40, zorder=5, marker="o", edgecolors="k", linewidths=0.5)

    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("D (Diopter)")
    ax.set_ylabel("ratio = minor / major")
    ax.set_title("Calibrated Ratio model  ratio_real(D, p_mm)\n"
                 "= ratio_sim(D, p_mm×5)   [dots = model eye real data]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "slice_by_pmm_ratio.png", dpi=150)
    plt.close(fig)
    print("Saved: slice_by_pmm_ratio.png")

    # ── グラフ2: Area (calibrated) vs D ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for pm in p_mm_list:
        col  = colors[pm]
        pred = area_real(D_fine, pm, ac)
        ax.plot(D_fine, pred, color=col, linewidth=2,
                label=f"model p={pm}mm  k={k_of_p(pm):.4f}")
        # 実データ点
        if pm in REAL_DATA:
            ds  = np.array(sorted(REAL_DATA[pm].keys()))
            ars = np.array([REAL_DATA[pm][d] for d in ds])
            ax.scatter(ds, ars, color=col, s=40, zorder=5, marker="o", edgecolors="k", linewidths=0.5)

    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("D (Diopter)")
    ax.set_ylabel("area = major × minor  (px²)")
    ax.set_title("Calibrated Area model  area_real(D, p_mm)\n"
                 "= k(p_mm) × area_sim(D, p_mm×5)   [dots = model eye real data]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "slice_by_pmm_area.png", dpi=150)
    plt.close(fig)
    print("Saved: slice_by_pmm_area.png")

    # ── CSV: k calibration ───────────────────────────────────────────────
    csv_path = OUT_DIR / "calibration_k.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["p_mm", "k_mean", "k_poly2_c2", "k_poly2_c1", "k_poly2_c0"])
        for pm, km in zip(CALIB_P_MM, CALIB_K):
            w.writerow([pm, f"{km:.6f}", "", "", ""])
        w.writerow(["poly_coeffs", "", f"{K_POLY[0]:.8f}", f"{K_POLY[1]:.8f}", f"{K_POLY[2]:.8f}"])
    print(f"Saved: {csv_path}")

    # ── グラフ3: 3D サーフェス (area_real) ──────────────────────────────
    D_grid   = np.linspace(-8, 8, 80)
    Pmm_grid = np.linspace(2.0, 8.0, 60)
    DD, PP   = np.meshgrid(D_grid, Pmm_grid)
    AA       = area_real(DD, PP, ac)

    fig3d = plt.figure(figsize=(10, 7))
    ax3d  = fig3d.add_subplot(111, projection="3d")
    ax3d.plot_surface(DD, PP, AA, cmap="plasma", alpha=0.8)
    for pm in p_mm_list:
        col = colors[pm]
        ds  = np.array(sorted(REAL_DATA[pm].keys()))
        ars = np.array([REAL_DATA[pm][d] for d in ds])
        ax3d.scatter(ds, np.full_like(ds, pm), ars,
                     color=col, s=20, zorder=5, label=f"{pm}mm")
    ax3d.set_xlabel("D (Diopter)")
    ax3d.set_ylabel("p (mm)")
    ax3d.set_zlabel("area_real (px^2)")
    ax3d.set_title("Calibrated surface: area_real(D, p_mm)\n= k(p_mm) x area_sim(D, p_mm x 5)")
    ax3d.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    fig3d.savefig(OUT_DIR / "surface_3d_area.png", dpi=150)
    plt.close(fig3d)
    print("Saved: surface_3d_area.png")

    # ── グラフ4: 等高線 (area_real) ──────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    cp = ax2.contourf(DD, PP, AA, levels=20, cmap="plasma")
    plt.colorbar(cp, ax=ax2, label="area_real (px^2)")
    ax2.contour(DD, PP, AA, levels=20, colors="white", linewidths=0.4, alpha=0.4)
    for pm in p_mm_list:
        col = colors[pm]
        ds  = np.array(sorted(REAL_DATA[pm].keys()))
        ax2.scatter(ds, np.full_like(ds, pm), color=col, s=20, zorder=5, label=f"{pm}mm")
    ax2.set_xlabel("D (Diopter)")
    ax2.set_ylabel("p (mm)")
    ax2.set_title("Calibrated area_real(D, p_mm)  contour")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / "surface_contour_area.png", dpi=150)
    plt.close(fig2)
    print("Saved: surface_contour_area.png")

    print(f"\nk(p_mm) = {K_POLY[0]:.6f}*p^2 + {K_POLY[1]:.6f}*p + {K_POLY[2]:.6f}")
    print(f"  k(3mm)={k_of_p(3.0):.4f}  k(5mm)={k_of_p(5.0):.4f}  k(7mm)={k_of_p(7.0):.4f}")
    print(f"\nAll saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
