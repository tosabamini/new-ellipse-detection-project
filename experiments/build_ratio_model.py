"""
Simulation ratio を RegularGridInterpolator (linear) で保存。
振動なし・連続・予測可能。

Run:
  python experiments/build_ratio_model.py
"""

import csv
import glob
import re
from pathlib import Path

import numpy as np

ELLIPSE_DIR = Path("data/simu_masked/ellipse_flat75")
OUT_PATH    = ELLIPSE_DIR / "fitting_calibrated_spline" / "ratio_model.npz"
P_LIST      = [10, 15, 20, 25, 30, 35, 40, 45]


def load_sim_groups():
    groups = {}
    for f in sorted(glob.glob(str(ELLIPSE_DIR / "p*" / "per_image_ellipse.csv"))):
        pg = int(re.search(r"p(\d+)", f).group(1))
        rows = [r for r in csv.DictReader(open(f, encoding="utf-8")) if r["status"] == "ok"]
        D   = np.array([float(r["D"])     for r in rows])
        maj = np.array([float(r["major"]) for r in rows])
        mino= np.array([float(r["minor"]) for r in rows])
        idx = np.argsort(D)
        groups[pg] = dict(D=D[idx], ratio=(mino/maj)[idx])
    return groups


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    groups = load_sim_groups()
    D_arr = groups[10]["D"]                                      # 共通 65点
    p_arr = np.array(P_LIST, dtype=float)
    Z     = np.array([groups[pg]["ratio"] for pg in P_LIST])    # (8, 65)

    np.savez(OUT_PATH,
             p_arr=p_arr, D_arr=D_arr, Z=Z,
             p_sim_to_mm=5.0,
             p_sim_range=np.array([10.0, 45.0]),
             d_range=np.array([-8.0, 8.0]))
    print(f"Saved: {OUT_PATH}  shape={Z.shape}")


if __name__ == "__main__":
    main()
