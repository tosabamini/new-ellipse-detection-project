import csv
import numpy as np
from scipy.optimize import brentq
from pathlib import Path

# ── poly_coeffs 読み込み ──────────────────────────────────────────────────────
coeffs = {}
for row in csv.DictReader(open(
        "data/simu_masked/ellipse_flat75/fitting_surface/poly_coeffs.csv",
        encoding="utf-8")):
    coeffs[row["param"]] = [float(row["c2"]), float(row["c1"]), float(row["c0"])]

def get_param(name, p):
    return np.polyval(coeffs[name], p)

# ── モデル ────────────────────────────────────────────────────────────────────
def model_myo(D, p):
    a_m  = get_param("a_m",  p)
    k_m  = get_param("k_m",  p)
    x0_m = get_param("x0_m", p)
    r0   = get_param("ratio_0", p)
    off  = r0 - a_m / (1 + np.exp(k_m * x0_m))
    return a_m / (1 + np.exp(k_m * (D + x0_m))) + off

def model_hyp(D, p):
    a_h  = get_param("a_h",  p)
    k_h  = get_param("k_h",  p)
    x0_h = get_param("x0_h", p)
    r0   = get_param("ratio_0", p)
    off  = r0 - a_h / (1 + np.exp(k_h * x0_h))
    return a_h / (1 + np.exp(-k_h * (D - x0_h))) + off

# ── 逆算 ──────────────────────────────────────────────────────────────────────
ratio_obs = 0.5236

for p in [15.0, 20.0]:
    print(f"========== p={p} {'(外挿・モデル範囲外)' if p < 20 else ''} ==========")
    r0 = get_param("ratio_0", p)
    print(f"ratio_0 (D=0): {r0:.4f}")

    try:
        D_myo = brentq(lambda D: model_myo(D, p) - ratio_obs, -8.0, 0.0)
        print(f"近視側解: D = {D_myo:+.3f} D")
    except ValueError:
        print(f"近視側解: 範囲外 (D=0: {model_myo(0,p):.4f}, D=-8: {model_myo(-8,p):.4f})")

    try:
        D_hyp = brentq(lambda D: model_hyp(D, p) - ratio_obs, 0.0, 8.0)
        print(f"遠視側解: D = {D_hyp:+.3f} D")
    except ValueError:
        print(f"遠視側解: 範囲外 (D=0: {model_hyp(0,p):.4f}, D=+8: {model_hyp(8,p):.4f})")
    print()

if False:  # dummy to avoid syntax error
    p = 20.0

print(f"観測 ratio: {ratio_obs}")
print(f"--- 参考: 旧 sim_ratio モデル (p無視) ---")
print(f"  d_myo = -3.590 D  /  d_hyp = +5.610 D")
