"""
Pupil diameter estimation from (ratio, area_scaled).

Model:
  area = slope(p) * ratio + intercept(p)
  slope(p)     = S2*p^2 + S1*p + S0
  intercept(p) = I2*p^2 + I1*p + I0

  Derived from hand-labeled model eye calibration at p = 3, 5, 7 mm.
  area = major * minor * SCALE_FACTOR^2  (model-eye scale)

Inversion:
  [S2*ratio + I2]*p^2 + [S1*ratio + I1]*p + [S0*ratio + I0 - area] = 0
  → solve quadratic in p, keep root in [P_MIN, P_MAX].
"""

import numpy as np

SCALE_FACTOR = 1.3   # 暫定: 患者画像 / モデル眼画像のpxスケール比

P_MIN = 2.0          # 有効な瞳孔径下限 (mm)
P_MAX = 9.0          # 有効な瞳孔径上限 (mm)

# slope(p) = S2*p^2 + S1*p + S0
S2, S1, S0 =   928.28, 1780.95,  -872.10
# intercept(p) = I2*p^2 + I1*p + I0
I2, I1, I0 =  -462.23, 3344.24, -4477.24


def estimate_pupil(ratio: float, area_scaled: float) -> float | None:
    """
    (ratio, area_scaled) から瞳孔径 p (mm) を推定する。

    area_scaled = major_px * minor_px * SCALE_FACTOR^2

    Returns p in [P_MIN, P_MAX], or None if no valid real root.
    When two valid roots exist, the larger is returned.
    """
    a_c = S2 * ratio + I2
    b_c = S1 * ratio + I1
    c_c = S0 * ratio + I0 - area_scaled

    disc = b_c ** 2 - 4 * a_c * c_c
    if disc < 0:
        return None

    sq    = np.sqrt(disc)
    roots = [(-b_c + sq) / (2 * a_c), (-b_c - sq) / (2 * a_c)]
    valid = [r for r in roots if P_MIN <= r <= P_MAX]
    return float(max(valid)) if valid else None
