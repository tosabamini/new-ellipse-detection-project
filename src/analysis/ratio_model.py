"""
Ratio reference model  —  10th-degree 2D polynomial.

    ratio_real(D, p_mm) = poly10(D, p_mm)

Polynomial fit on simulation grid (RMSE=0.01251, R2=0.9975).
"""

from functools import lru_cache

import numpy as np

from src.common.paths import POLY_MODEL_NPZ


@lru_cache(maxsize=1)
def _load():
    d = np.load(POLY_MODEL_NPZ)
    return d["coef_ratio"], int(d["deg"])


def ratio_real(D, p_mm):
    """ratio = minor/major の参照値。スカラ入力でスカラを返す。"""
    coef, deg = _load()
    D = float(D); P = float(p_mm)
    vals = [D**i * P**j for i in range(deg + 1) for j in range(deg + 1 - i)]
    return float(np.dot(vals, coef))


if __name__ == "__main__":
    print("ratio_model.py self-test (poly deg=10)")
    for d, p in [(0.0, 3.0), (-4.0, 5.0), (-8.0, 7.0)]:
        print(f"  ratio_real(D={d:+.1f}, {p}mm) = {ratio_real(d, p):.4f}")
