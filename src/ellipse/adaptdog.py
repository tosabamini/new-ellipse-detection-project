"""
AdaptDoG ellipse fitting and quality filters.
Geometry-based, no ML.
"""

import cv2
import numpy as np
from collections import defaultdict


# ── image helpers ─────────────────────────────────────────────────────────────

def stretch_to_255(src: np.ndarray) -> np.ndarray:
    a = src.astype(np.float32)
    lo, hi = a.min(), a.max()
    if hi == lo:
        return np.zeros_like(a, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255).astype(np.uint8)


def red_channel(bgr: np.ndarray) -> np.ndarray:
    """R - 0.5G - 0.5B, clipped to [0, 255]."""
    f = bgr.astype(np.float32)
    return np.clip(f[:, :, 2] - 0.5 * f[:, :, 1] - 0.5 * f[:, :, 0], 0, 255).astype(np.uint8)


# ── core DoG helpers ──────────────────────────────────────────────────────────

def _otsu_mask(gray: np.ndarray) -> np.ndarray:
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m


def _pick_central_blob(binary: np.ndarray, min_area: int = 30,
                       open_k: int = 5, close_k: int = 9) -> np.ndarray:
    """Morph clean-up + keep the component closest to image centre (area-weighted)."""
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k,  open_k))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    m = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_open)
    m = cv2.morphologyEx(m,      cv2.MORPH_CLOSE, k_close)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return m
    h, w = m.shape
    cx0, cy0 = w / 2, h / 2
    best, bs = None, None
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        score = a - 0.5 * ((cx - cx0) ** 2 + (cy - cy0) ** 2)
        if bs is None or score > bs:
            best, bs = c, score
    if best is None:
        return m
    out = np.zeros_like(m)
    cv2.drawContours(out, [best], -1, 255, cv2.FILLED)
    return out


def _fit_ellipse_on_mask(mask: np.ndarray) -> dict | None:
    """Fit ellipse to largest contour. Returns dict or None."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    e = cv2.fitEllipse(cnt)
    (cx, cy), (a1, a2), ang = e
    major = max(a1, a2)
    minor = min(a1, a2)
    if a2 > a1:
        ang += 90
    return dict(cx=cx, cy=cy, major=major, minor=minor,
                angle=ang % 180, ratio=minor / major if major > 0 else 0, raw=e)


def _estimate_minor(red_str: np.ndarray, top_pct: float = 0.005) -> float:
    """Coarse estimate of minor axis from the very brightest pixels."""
    thresh = float(np.percentile(red_str, 100 * (1 - top_pct)))
    _, coarse = cv2.threshold(red_str, thresh, 255, cv2.THRESH_BINARY)
    blob = _pick_central_blob(coarse, min_area=10, open_k=3, close_k=3)
    e = _fit_ellipse_on_mask(blob)
    return max(6.0, e['minor']) if e else 12.0


# ── AdaptDoG main ─────────────────────────────────────────────────────────────

def run_adaptive_dog(red_str: np.ndarray) -> dict | None:
    """
    Adaptive Difference-of-Gaussians ellipse fitting.

    Steps:
      1. Estimate minor axis from brightest pixels (sigma scale reference).
      2. DoG = GaussianBlur(σ=1.5) - GaussianBlur(σ=minor*0.75).
      3. Otsu threshold → central blob.
      4. If core is elongated (ratio < 0.20): dilate along major axis to recover dim tips.
         If core is round/oval: morphological close to fill gaps.
      5. Fit ellipse on final mask.

    Returns ellipse dict (cx, cy, major, minor, angle, ratio, raw) or None.
    """
    minor_est = _estimate_minor(red_str)
    sigma_l   = max(8.0, minor_est * 0.75)

    blur_s = cv2.GaussianBlur(red_str.astype(np.float32), (0, 0), 1.5)
    blur_l = cv2.GaussianBlur(red_str.astype(np.float32), (0, 0), sigma_l)
    dog    = stretch_to_255(np.clip(blur_s - blur_l, 0, None))

    mask_raw  = _otsu_mask(dog)
    mask_core = _pick_central_blob(mask_raw)

    e_core     = _fit_ellipse_on_mask(mask_core)
    core_ratio = e_core['ratio'] if e_core else 0.0
    core_angle = e_core['angle'] if e_core else 90.0

    if core_ratio < 0.20:
        dil_w = max(3,  int(minor_est * 0.33))
        dil_h = max(15, int(minor_est * 1.20))
        if core_angle < 45 or core_angle > 135:
            dil_w, dil_h = dil_h, dil_w
        mask = cv2.dilate(mask_core,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (dil_w, dil_h)))
    else:
        close_k = max(5, int(minor_est * 0.20)) | 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask_core, cv2.MORPH_CLOSE, k)

    # TODO: dilation/close ステップは新パイプラインでは廃止予定。
    # 副作用（斜め方向への誤膨張・短軸の過大算出）が目的（tips補完）を上回るため。
    # 新パイプラインでは mask_core を直接 fitEllipse に渡す。
    return _fit_ellipse_on_mask(mask)


def draw_ellipse_overlay(bgr: np.ndarray, e: dict | None,
                         color: tuple = (0, 255, 120), thickness: int = 2) -> np.ndarray:
    """Draw fitted ellipse on a BGR image copy."""
    out = bgr.copy()
    if e:
        cv2.ellipse(out, e['raw'], color, thickness)
        cv2.circle(out, (int(e['cx']), int(e['cy'])), 4, color, -1)
    return out


# ── quality filters ───────────────────────────────────────────────────────────

def iqr_filter(ellipses: list, k: float = 0.5) -> list[bool]:
    """
    Keep images whose major axis >= Q1 - k*IQR.
    k=0.5 (aggressive): excludes images with very small red reflex.
    ratio is NOT filtered — near-circular is valid near emmetropia.
    """
    majors = np.array([e['major'] for e in ellipses if e and e.get('major')])
    if len(majors) < 4:
        return [True] * len(ellipses)
    q1    = float(np.percentile(majors, 25))
    iqr   = float(np.percentile(majors, 75) - q1)
    fence = q1 - k * iqr
    return [(e is not None and e.get('major', 0) >= fence) for e in ellipses]


def d_iqr_filter(records: list, k: float = 1.5) -> list[bool]:
    """
    IQR filter on adopted_D within each angle_bin separately.
    Astigmatism makes D vary sinusoidally with angle — global filtering
    would remove legitimate extremes of the cosine curve.
    Bins with < 4 images are kept as-is.
    """
    bins = defaultdict(list)
    for i, r in enumerate(records):
        bins[r['angle_bin']].append(i)

    keep = [True] * len(records)
    for bin_name, idxs in bins.items():
        if len(idxs) < 4:
            continue
        vals = np.array([records[i]['adopted_D'] for i in idxs], dtype=float)
        q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        iqr_v  = q3 - q1
        lo, hi = q1 - k * iqr_v, q3 + k * iqr_v
        for i, v in zip(idxs, vals):
            if not (lo <= v <= hi):
                keep[i] = False
    return keep
