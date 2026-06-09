"""
Calibrated REAL-pixel area model  —  10th-degree 2D polynomial.

    area_real(D, p_mm) = poly10(D, p_mm)

where the polynomial was fit to:
    alpha(p) * k(p) * area_sim(D, p_sim)
    alpha(p) = 0.2742 + 0.0419*p   (patient correction)
    k(p)     = quadratic from model-eye calibration

Polynomial fit on simulation grid (RMSE=311 px^2, R2=0.9993).
"""

from functools import lru_cache

import numpy as np

from src.common.paths import POLY_MODEL_NPZ


@lru_cache(maxsize=1)
def _load():
    d = np.load(POLY_MODEL_NPZ)
    return d["coef_area"], int(d["deg"])


def area_real(D, p_mm):
    """実ピクセル換算 area (alpha*k 込み)。スカラ入力でスカラを返す。"""
    coef, deg = _load()
    D = float(D); P = float(p_mm)
    vals = [D**i * P**j for i in range(deg + 1) for j in range(deg + 1 - i)]
    return float(np.dot(vals, coef))


if __name__ == "__main__":
    print("area_model.py self-test (poly deg=10)")
    for d, p in [(-4.0, 5.0), (0.0, 3.0), (-5.0, 7.0)]:
        print(f"  area_real(D={d:+.1f}, {p}mm) = {area_real(d, p):.1f}")
