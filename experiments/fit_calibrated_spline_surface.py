"""
Simulation area を RegularGridInterpolator (linear) で保存し、
Model Eye 実測との比較で変換係数 k(p_mm) を求めて実ピクセル版 area モデルを構築。

キャリブレーション根拠 (近視安定域 D=-5,-4,-3):
    k(3mm)=area_real/area_sim  →  3点を 2 次多項式で補間

Run:
  python experiments/fit_calibrated_spline_surface.py
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
from scipy.interpolate import RegularGridInterpolator

ELLIPSE_DIR = Path("data/simu_masked/ellipse_flat75")
OUT_DIR     = ELLIPSE_DIR / "fitting_calibrated_spline"
P_LIST      = [10, 15, 20, 25, 30, 35, 40, 45]
STABLE_D    = [-5.0, -4.0, -3.0]
CALIB       = [(3.0, "p15"), (5.0, "p25"), (7.0, "p35")]

GROUP_COLORS = {10:"#e74c3c",15:"#e67e22",20:"#f1c40f",25:"#2ecc71",
                30:"#1abc9c",35:"#3498db",40:"#9b59b6",45:"#e91e63"}


def folder_to_D(folder):
    parts = folder.split("_")
    sign = parts[1]; d1 = int(parts[2]); d2 = int(parts[3].replace("D",""))
    val = d1 + d2 / 100
    return -val if sign == "M" else val


def load_sim_groups():
    groups = {}
    for f in sorted(glob.glob(str(ELLIPSE_DIR / "p*" / "per_image_ellipse.csv"))):
        pg = int(re.search(r"p(\d+)", f).group(1))
        rows = [r for r in csv.DictReader(open(f, encoding="utf-8")) if r["status"] == "ok"]
        D   = np.array([float(r["D"])     for r in rows])
        maj = np.array([float(r["major"]) for r in rows])
        mino= np.array([float(r["minor"]) for r in rows])
        idx = np.argsort(D)
        groups[pg] = dict(D=D[idx], area=(maj*mino)[idx])
    return groups


def load_model_eye_area():
    import os
    out = {}
    src = {3.0: "model_eye_3mm_v001", 5.0: "model_eye_5mm_v001", 7.0: "model_eye_v001"}
    for p_mm, run in src.items():
        path = f"data/processed/model_eye_runs/{run}/ellipse_summary.csv"
        if not os.path.exists(path):
            continue
        d = {}
        for r in csv.DictReader(open(path, encoding="utf-8-sig")):
            d[folder_to_D(r["folder"])] = float(r["mean_major_axis"]) * float(r["mean_minor_axis"])
        out[p_mm] = d
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    groups = load_sim_groups()
    real   = load_model_eye_area()

    D_arr = groups[10]["D"]                                     # 共通 65点
    p_arr = np.array(P_LIST, dtype=float)
    Z_sim = np.array([groups[pg]["area"] for pg in P_LIST])    # (8, 65)

    # RegularGridInterpolator: 入力 (p_sim, D)
    interp_sim = RegularGridInterpolator(
        (p_arr, D_arr), Z_sim, method="linear", bounds_error=False, fill_value=None)

    def area_sim_val(D, p_sim):
        return float(interp_sim([[p_sim, D]]).ravel()[0])

    # ── キャリブレーション係数 k(p_mm) ──────────────────────────────────
    print("=== 変換係数 k = area_real / area_sim (近視安定域) ===")
    k_per_pmm = {}
    for p_mm, pg_name in CALIB:
        p_sim = p_mm * 5.0
        ks = [real[p_mm][D] / area_sim_val(D, p_sim) for D in STABLE_D]
        k_mean = float(np.mean(ks))
        k_per_pmm[p_mm] = k_mean
        detail = "  ".join(f"D={d:+.0f}:{kk:.4f}" for d, kk in zip(STABLE_D, ks))
        print(f"  {p_mm}mm ({pg_name}): {detail}  -> k_mean={k_mean:.4f}")

    pmm_arr = np.array([c[0] for c in CALIB])
    k_arr   = np.array([k_per_pmm[c[0]] for c in CALIB])
    K_POLY  = np.polyfit(pmm_arr, k_arr, 2)
    print(f"\nk(p_mm) = {K_POLY[0]:.6f}*p^2 + {K_POLY[1]:.6f}*p + {K_POLY[2]:.6f}")

    # ── モデル保存 ────────────────────────────────────────────────────────
    model_path = OUT_DIR / "area_model.npz"
    np.savez(model_path,
             p_arr=p_arr, D_arr=D_arr, Z_sim=Z_sim,
             k_poly=K_POLY,
             p_sim_to_mm=5.0,
             p_sim_range=np.array([10.0, 45.0]),
             d_range=np.array([-8.0, 8.0]),
             calib_p_mm=pmm_arr, calib_k=k_arr)
    print(f"Saved model: {model_path}")

    # ── 検証スライス ─────────────────────────────────────────────────────
    def k_of_pmm(p_mm): return np.polyval(K_POLY, p_mm)
    def area_real_val(D, p_mm):
        return k_of_pmm(p_mm) * area_sim_val(D, p_mm * 5.0)

    Dfine = np.linspace(-8, 8, 300)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (p_mm, pg_name) in zip(axes, CALIB):
        model = [area_real_val(d, p_mm) for d in Dfine]
        ax.plot(Dfine, model, "r-", lw=2, label="real-px model")
        ds  = np.array(sorted(real[p_mm].keys()))
        ars = np.array([real[p_mm][d] for d in ds])
        ax.scatter(ds, ars, c="k", s=35, zorder=5, label="model eye real")
        for d in STABLE_D:
            ax.axvline(d, color="green", lw=0.5, ls=":", alpha=0.5)
        ax.axvline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(f"{p_mm}mm ({pg_name})  k={k_of_pmm(p_mm):.4f}")
        ax.set_xlabel("D"); ax.set_ylabel("area (px^2)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.suptitle("Verify: real-pixel area model (linear interp) vs Model Eye", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "verify_slices_area_real.png", dpi=130)
    plt.close(fig)
    print("Saved: verify_slices_area_real.png")

    # ── CSV ───────────────────────────────────────────────────────────────
    with open(OUT_DIR / "k_calibration.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["p_mm", "p_sim", "k_mean"])
        for p_mm, pg in CALIB:
            w.writerow([p_mm, int(p_mm*5), f"{k_per_pmm[p_mm]:.6f}"])
        w.writerow(["poly2_c2", "poly2_c1", "poly2_c0"])
        w.writerow([f"{K_POLY[0]:.8f}", f"{K_POLY[1]:.8f}", f"{K_POLY[2]:.8f}"])
    print(f"All saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
