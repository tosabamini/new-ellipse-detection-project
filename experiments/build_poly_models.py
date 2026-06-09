"""
10次 2D 多項式モデルをシミュレーショングリッドからフィットして NPZ に保存。

  ratio_poly(D, p_mm): ratio = minor/major
  area_poly(D, p_mm):  area_real = alpha(p) * k(p) * area_sim   [px^2]

alpha(p) = 0.2742 + 0.0419*p  (患者補正係数)
k(p)     = 0.015732*p^2 - 0.098734*p + 0.602635  (模型眼校正係数)

Run:
  python experiments/build_poly_models.py
"""

import sys, io
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NPZ_RATIO = Path("data/simu_masked/ellipse_flat75/fitting_calibrated_spline/ratio_model.npz")
NPZ_AREA  = Path("data/simu_masked/ellipse_flat75/fitting_calibrated_spline/area_model.npz")
OUT_NPZ   = Path("data/simu_masked/ellipse_flat75/fitting_calibrated_spline/poly_model.npz")

DEG = 10
A_COEF, B_COEF = 0.2742, 0.0419   # alpha(p) = A_COEF + B_COEF*p


def alpha_of_p(p): return A_COEF + B_COEF * p


def poly2d_matrix(D_arr, P_arr, deg=10):
    D = np.asarray(D_arr, dtype=float).ravel()
    P = np.asarray(P_arr, dtype=float).ravel()
    cols = [D**i * P**j for i in range(deg + 1) for j in range(deg + 1 - i)]
    return np.column_stack(cols)


def main():
    # ── ratio モデル ──────────────────────────────────────────────────────
    d_ratio = np.load(NPZ_RATIO)
    p_arr   = d_ratio["p_arr"]   # p_sim 単位 (10,15,...,45)
    D_arr   = d_ratio["D_arr"]   # 65点
    Z_ratio = d_ratio["Z"]       # (8,65)

    p_mm_arr = p_arr / 5.0       # mm 換算 (2,3,...,9)

    # グリッド全点を展開
    DD, PP = np.meshgrid(D_arr, p_mm_arr)   # shape (8,65)
    D_flat = DD.ravel()
    P_flat = PP.ravel()
    R_flat = Z_ratio.ravel()

    M = poly2d_matrix(D_flat, P_flat, DEG)
    coef_ratio, _, _, _ = np.linalg.lstsq(M, R_flat, rcond=None)

    R_pred = M @ coef_ratio
    rmse_r = float(np.sqrt(np.mean((R_pred - R_flat)**2)))
    maxe_r = float(np.max(np.abs(R_pred - R_flat)))
    r2_r   = float(1 - np.var(R_pred - R_flat) / np.var(R_flat))
    print(f"ratio  poly deg={DEG}: RMSE={rmse_r:.5f}  MaxE={maxe_r:.5f}  R2={r2_r:.6f}")

    # ── area モデル (alpha*k込み) ─────────────────────────────────────────
    d_area  = np.load(NPZ_AREA)
    k_poly  = d_area["k_poly"]   # (3,) 2次多項式係数
    Z_sim   = d_area["Z_sim"]    # (8,65) sim area

    # area_real = alpha(p_mm) * k(p_mm) * area_sim
    alpha_vec = np.array([alpha_of_p(p) for p in p_mm_arr])   # (8,)
    k_vec     = np.array([np.polyval(k_poly, p) for p in p_mm_arr])   # (8,)
    scale_2d  = (alpha_vec * k_vec)[:, None]   # (8,1)
    Z_area    = scale_2d * Z_sim               # (8,65)

    A_flat = Z_area.ravel()

    # 同じ D_flat, P_flat を再利用
    coef_area, _, _, _ = np.linalg.lstsq(M, A_flat, rcond=None)

    A_pred = M @ coef_area
    rmse_a = float(np.sqrt(np.mean((A_pred - A_flat)**2)))
    maxe_a = float(np.max(np.abs(A_pred - A_flat)))
    r2_a   = float(1 - np.var(A_pred - A_flat) / np.var(A_flat))
    print(f"area   poly deg={DEG}: RMSE={rmse_a:.1f} px2  MaxE={maxe_a:.1f} px2  R2={r2_a:.6f}")

    # ── 保存 ──────────────────────────────────────────────────────────────
    np.savez(OUT_NPZ,
             coef_ratio=coef_ratio,
             coef_area=coef_area,
             deg=np.int32(DEG),
             D_min=np.float64(D_arr.min()),
             D_max=np.float64(D_arr.max()),
             p_mm_min=np.float64(p_mm_arr.min()),
             p_mm_max=np.float64(p_mm_arr.max()),
             A_COEF=np.float64(A_COEF),
             B_COEF=np.float64(B_COEF),
             k_poly=k_poly)
    print(f"Saved: {OUT_NPZ}")
    print(f"  coef_ratio: {len(coef_ratio)} terms")
    print(f"  coef_area:  {len(coef_area)} terms")


if __name__ == "__main__":
    main()
