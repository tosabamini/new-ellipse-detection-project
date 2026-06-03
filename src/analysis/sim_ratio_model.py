"""
Simulation-based ratio → D estimator.

【暫定モデル — 使用注意】
  - データソース: p20/p30/p40 の 3 グループ平均のみ (p10/p15/p25/p35/p45 未反映)
  - 瞳孔径依存性を意図的に無視した近似式
  - 遠視側 D > +2.0 D は範囲外のため精度保証なし
  - 今後グループ追加・再フィッティングが必要な発展途上モデル

フィット式 (C⁰ Logistic, D=0 アンカー):
  f(|D|) = a / (1 + exp(−k·(|D| − x0))) + offset
  offset = RATIO_0 − a / (1 + exp(k·x0))

Myopia  (D ≤ 0): a=1.0164, k=0.7606, x0=3.2696  R²=0.9975
Hyperopia(D ≥ 0): a=1.9279, k=0.3092, x0=7.7889  R²=0.9984  (valid ≤ +2.0 D)
"""

import numpy as np

# ── フィットパラメータ (experiments/simulation_unified_fit.py で導出) ──────────
RATIO_0 = 0.0318   # D=0 アンカー (p20/p30/p40 平均)

_MYO_A  = 1.0164
_MYO_K  = 0.7606
_MYO_X0 = 3.2696

_HYP_A  = 1.9279
_HYP_K  = 0.3092
_HYP_X0 = 7.7889

D_MAX_HYPEROPIA = 2.0   # 遠視側の信頼できる上限

# ── 内部関数 ──────────────────────────────────────────────────────────────────

def _logistic(abs_d: float, a: float, k: float, x0: float) -> float:
    offset = RATIO_0 - a / (1.0 + np.exp(k * x0))
    return a / (1.0 + np.exp(-k * (abs_d - x0))) + offset


def _forward_myo(abs_d: float) -> float:
    return _logistic(abs_d, _MYO_A, _MYO_K, _MYO_X0)


def _forward_hyp(abs_d: float) -> float:
    return _logistic(abs_d, _HYP_A, _HYP_K, _HYP_X0)


def _invert_logistic(ratio: float, a: float, k: float, x0: float) -> float | None:
    """ratio → |D| の逆算。解なし/範囲外なら None"""
    offset = RATIO_0 - a / (1.0 + np.exp(k * x0))
    y = ratio - offset
    if y <= 0.0 or y >= a:
        return None
    return x0 - np.log(a / y - 1.0) / k


# ── 公開 API ──────────────────────────────────────────────────────────────────

def estimate_D_from_ratio_sim(ratio: float) -> tuple[float | None, float | None]:
    """
    ratio から近視側・遠視側それぞれの D を逆算する。

    Returns
    -------
    (D_myopia, D_hyperopia) : どちらも解なしの場合は None

    Notes
    -----
    - D_myopia  ≤ 0 D
    - D_hyperopia ≥ 0 D (D_MAX_HYPEROPIA = +2.0 D までのみ信頼性あり)
    - 現パイプラインでは D_myopia を採用 (D2 に相当)
    """
    abs_d_myo = _invert_logistic(ratio, _MYO_A, _MYO_K, _MYO_X0)
    abs_d_hyp = _invert_logistic(ratio, _HYP_A, _HYP_K, _HYP_X0)

    d_myo = -abs_d_myo if abs_d_myo is not None else None
    d_hyp =  abs_d_hyp if abs_d_hyp is not None else None

    return d_myo, d_hyp
