"""
Simulation-based ratio → D estimator — 0607 暫定補正版

【変更点】
  RATIO_0 を 0.0318 → 0.0869 に変更。
  根拠: data/Repeatability/PickUP/11_Aslaha/LEFT の
        IMG_20260603_145405_263_red_roi.png (ratio=0.0869) が
        正視被験者の最小 ratio 画像として観測されたため、
        これを D=0 アンカーとして再設定。

  実機（患者・スマホカメラ）では光学収差等により、
  同じ屈折力でも Simulation より楕円が太くなる（ratio 大）と推測される。
  この補正はその系統誤差を暫定的に吸収する。

【注意】
  - あくまで 2026-06-07 時点の暫定キャリブレーション
  - 遠視側の精度は D > +2.0 D で保証なし（同上）
  - 瞳孔径依存性は未考慮（同上）
  - 正式なモデル更新時には sim_ratio_model.py 本体を改訂すること

フィット式 (C⁰ Logistic, D=0 アンカー): sim_ratio_model.py と同一
  Myopia  (D ≤ 0): a=1.0164, k=0.7606, x0=3.2696
  Hyperopia(D ≥ 0): a=1.9279, k=0.3092, x0=7.7889
"""

import numpy as np

# ── フィットパラメータ ────────────────────────────────────────────────────────
RATIO_0 = 0.0869   # 0607 補正: 実機観測正視画像の ratio (旧値: 0.0318)

_MYO_A  = 1.0164
_MYO_K  = 0.7606
_MYO_X0 = 3.2696

_HYP_A  = 1.9279
_HYP_K  = 0.3092
_HYP_X0 = 7.7889

D_MAX_HYPEROPIA = 2.0


def _logistic(abs_d: float, a: float, k: float, x0: float) -> float:
    offset = RATIO_0 - a / (1.0 + np.exp(k * x0))
    return a / (1.0 + np.exp(-k * (abs_d - x0))) + offset


def _invert_logistic(ratio: float, a: float, k: float, x0: float) -> float | None:
    offset = RATIO_0 - a / (1.0 + np.exp(k * x0))
    y = ratio - offset
    if y <= 0.0 or y >= a:
        return None
    return x0 - np.log(a / y - 1.0) / k


def estimate_D_from_ratio_sim(ratio: float) -> tuple[float | None, float | None]:
    """
    ratio から近視側・遠視側それぞれの D を逆算する (0607 補正版)。

    Returns
    -------
    (D_myopia, D_hyperopia)
    """
    abs_d_myo = _invert_logistic(ratio, _MYO_A, _MYO_K, _MYO_X0)
    abs_d_hyp = _invert_logistic(ratio, _HYP_A, _HYP_K, _HYP_X0)

    d_myo = -abs_d_myo if abs_d_myo is not None else None
    d_hyp =  abs_d_hyp if abs_d_hyp is not None else None

    return d_myo, d_hyp
