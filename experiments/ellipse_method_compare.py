"""
3-way ellipse extraction comparison: Frangi vs Sato vs Adaptive-DoG.
All methods are standalone (no src.* imports).

Edit INPUT_IMAGE, then:
    python experiments/ellipse_method_compare.py
"""

import cv2
import numpy as np
from pathlib import Path

from skimage.filters import frangi, sato

# ── paths ──────────────────────────────────────────────────────────────────────
INPUT_IMAGE = Path(r"C:\Users\issas\Desktop\new_ellipse_detection_project\data\processed\pipeline_runs\pipeline_run_101_106_v001\102_LEFT\roi\IMG_20260513_143554_905_roi.png")
OUTPUT_DIR  = Path(r"C:\Users\issas\Desktop\new_ellipse_detection_project\experiments\method_compare_output")

# ── shared helpers ─────────────────────────────────────────────────────────────

def red_enhance(bgr: np.ndarray) -> np.ndarray:
    f = bgr.astype(np.float32)
    return np.clip(f[:,:,2] - 0.5*f[:,:,1] - 0.5*f[:,:,0], 0, 255).astype(np.uint8)

def stretch_to_255(src) -> np.ndarray:
    a = src.astype(np.float32)
    lo, hi = a.min(), a.max()
    if hi == lo: return a.astype(np.uint8)
    return ((a - lo) / (hi - lo) * 255).astype(np.uint8)

def otsu_mask(gray: np.ndarray) -> np.ndarray:
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m

def pick_central_blob(binary: np.ndarray, min_area: int = 30,
                      open_k: int = 5, close_k: int = 9) -> np.ndarray:
    """Morph clean-up + keep the component closest to image centre (area-weighted)."""
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k,  open_k))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    m = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_open)
    m = cv2.morphologyEx(m,      cv2.MORPH_CLOSE, k_close)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return m
    h, w = m.shape; cx0, cy0 = w/2, h/2
    best, bs = None, None
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: continue
        M = cv2.moments(c)
        if M['m00'] == 0: continue
        cx = M['m10']/M['m00']; cy = M['m01']/M['m00']
        s = a - 0.5*((cx-cx0)**2 + (cy-cy0)**2)
        if bs is None or s > bs: best, bs = c, s
    if best is None: return m
    out = np.zeros_like(m)
    cv2.drawContours(out, [best], -1, 255, cv2.FILLED)
    return out

def fit_ellipse(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5: return None
    e = cv2.fitEllipse(cnt)
    (cx, cy), (a1, a2), ang = e
    major = max(a1, a2); minor = min(a1, a2)
    if a2 > a1: ang += 90
    return dict(cx=cx, cy=cy, major=major, minor=minor,
                angle=ang%180, ratio=minor/major if major>0 else 0, raw=e)

def to_bgr(g: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if g.ndim == 2 else g

def draw_overlay(img: np.ndarray, e, label: str = "") -> np.ndarray:
    c = to_bgr(img).copy()
    if e:
        cv2.ellipse(c, e['raw'], (255, 0, 255), 2)
        cv2.circle(c, (int(e['cx']), int(e['cy'])), 4, (255, 0, 255), -1)
    if label:
        cv2.putText(c, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2, cv2.LINE_AA)
    return c

# ── Method A: Frangi + adaptive dilation ──────────────────────────────────────

def run_frangi(red_str: np.ndarray):
    """
    Frangi detects the very thin bright ridge accurately.
    Problem: the binary mask after Otsu is just a few px wide → standard closing can't fill it.
    Fix: after picking the central blob, apply the same adaptive dilation as AdaptDoG
    (scaled by the minor estimate from the coarse pre-pass).
    """
    img_f = red_str.astype(np.float64) / 255.0
    vessel = frangi(img_f, sigmas=range(2, 22, 2), black_ridges=False)
    feature = stretch_to_255(vessel)

    mask_raw  = otsu_mask(feature)
    mask_core = pick_central_blob(mask_raw, close_k=13)

    # Frangi core is very thin → use coarse minor estimate to size the dilation
    minor_est = _estimate_minor(red_str)
    e_core = fit_ellipse(mask_core)
    core_angle = e_core['angle'] if e_core else 90.0
    dil_w = max(5,  int(minor_est * 0.33))
    dil_h = max(15, int(minor_est * 1.20))
    if core_angle < 45 or core_angle > 135:
        dil_w, dil_h = dil_h, dil_w
    mask = cv2.dilate(mask_core, cv2.getStructuringElement(cv2.MORPH_RECT, (dil_w, dil_h)))

    e = fit_ellipse(mask)
    return feature, mask, e, minor_est

# ── Method B: Sato + adaptive dilation ───────────────────────────────────────

def run_sato(red_str: np.ndarray):
    """
    Sato tends to capture too much halo → add the same adaptive DoG pre-filter
    to suppress diffuse background before computing Sato on the cleaned image.
    """
    minor_est = _estimate_minor(red_str)
    sigma_l   = max(8.0, minor_est * 0.75)

    # Pre-suppress halo with a soft DoG, then run Sato on the cleaned image
    blur_s = cv2.GaussianBlur(red_str.astype(np.float32), (0,0), 1.5)
    blur_l = cv2.GaussianBlur(red_str.astype(np.float32), (0,0), sigma_l)
    cleaned = stretch_to_255(np.clip(blur_s - blur_l, 0, None))

    img_f = cleaned.astype(np.float64) / 255.0
    ridge = sato(img_f, sigmas=range(2, 14, 2), black_ridges=False)
    feature = stretch_to_255(ridge)

    mask_raw  = otsu_mask(feature)
    mask_core = pick_central_blob(mask_raw, close_k=13)
    e_core = fit_ellipse(mask_core)
    core_angle = e_core['angle'] if e_core else 90.0
    dil_w = max(5,  int(minor_est * 0.33))
    dil_h = max(15, int(minor_est * 1.20))
    if core_angle < 45 or core_angle > 135:
        dil_w, dil_h = dil_h, dil_w
    mask = cv2.dilate(mask_core, cv2.getStructuringElement(cv2.MORPH_RECT, (dil_w, dil_h)))

    e = fit_ellipse(mask)
    return feature, mask, e, minor_est, sigma_l

# ── Method C: Adaptive DoG ────────────────────────────────────────────────────

def _estimate_minor(red_str: np.ndarray, top_pct: float = 0.005) -> float:
    """Coarse estimate of minor axis from the very brightest pixels."""
    thresh = float(np.percentile(red_str, 100*(1-top_pct)))
    _, coarse = cv2.threshold(red_str, thresh, 255, cv2.THRESH_BINARY)
    blob = pick_central_blob(coarse, min_area=10, open_k=3, close_k=3)
    e = fit_ellipse(blob)
    return max(6.0, e['minor']) if e else 12.0


def run_adaptive_dog(red_str: np.ndarray):
    """
    2-step adaptive DoG with ratio-aware dilation:
      1. Estimate minor axis from tight percentile threshold.
      2. Set sigma_large = minor_est * 0.75 to remove halo wider than the reflex.
      3. Fit the raw DoG core to check its eccentricity (core_ratio).
         - core_ratio < 0.20 (elongated): apply tall dilation to recover dim tips.
           The DoG cuts off the faint ends → major is underestimated → needs extension.
         - core_ratio >= 0.20 (oval/round): the core major is already correct.
           Apply only a small symmetric closing to fill minor-axis holes.
    """
    minor_est  = _estimate_minor(red_str)
    sigma_l    = max(8.0, minor_est * 0.75)

    blur_s = cv2.GaussianBlur(red_str.astype(np.float32), (0,0), 1.5)
    blur_l = cv2.GaussianBlur(red_str.astype(np.float32), (0,0), sigma_l)
    dog    = stretch_to_255(np.clip(blur_s - blur_l, 0, None))

    mask_raw  = otsu_mask(dog)
    mask_core = pick_central_blob(mask_raw)

    e_core     = fit_ellipse(mask_core)
    core_ratio = e_core['ratio'] if e_core else 0.0
    core_angle = e_core['angle'] if e_core else 90.0

    if core_ratio < 0.20:
        # Elongated: dilation recovers dim tips cut off by DoG+Otsu
        dil_w = max(3,  int(minor_est * 0.33))
        dil_h = max(15, int(minor_est * 1.20))
        if core_angle < 45 or core_angle > 135:
            dil_w, dil_h = dil_h, dil_w
        mask  = cv2.dilate(mask_core, cv2.getStructuringElement(cv2.MORPH_RECT, (dil_w, dil_h)))
    else:
        # Oval/round: core major is already correct; just fill minor-axis gaps
        close_k = max(5, int(minor_est * 0.20)) | 1   # ensure odd
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask_core, cv2.MORPH_CLOSE, k)

    e = fit_ellipse(mask)
    return dog, mask, e, minor_est, sigma_l, core_ratio

# ── IQR-based quality filter ──────────────────────────────────────────────────

def iqr_filter(ellipses: list, k: float = 0.5) -> list[bool]:
    """
    Return a bool mask (True = keep) based on IQR outlier detection on major axis.

    Images whose major axis falls below  Q1 - k * IQR  are flagged as outliers
    (too small major = no reflex or extremely faint reflex).
    k=0.5 is the project default (more aggressive than the standard 1.5).

    Note: ratio is NOT filtered here — a high ratio (near-circular) is valid
    when refraction is near emmetropia.

    ellipses : list of ellipse dicts (or None) — one per image, from fit_ellipse()
    Returns  : list of bool, same length as ellipses
    """
    import numpy as np
    majors = np.array([e['major'] for e in ellipses if e and e.get('major')])
    if len(majors) < 4:
        return [True] * len(ellipses)   # too few images to compute IQR reliably
    q1    = float(np.percentile(majors, 25))
    iqr   = float(np.percentile(majors, 75) - q1)
    fence = q1 - k * iqr
    return [(e is not None and e.get('major', 0) >= fence) for e in ellipses]


def d_iqr_filter(records: list, k: float = 1.5) -> list[bool]:
    """
    Return a bool mask (True = keep) based on IQR outlier detection on D values,
    applied WITHIN each angle bin separately.

    With astigmatism, D varies sinusoidally with angle, so filtering globally
    would remove legitimate extremes of the cosine curve. Filtering per bin
    compares only images captured at the same angle condition.

    Bins with fewer than 4 images are kept as-is (IQR unreliable on tiny groups).
    k=1.5 is the standard outlier threshold (extreme outliers only).

    records : list of dicts with keys 'adopted_D' and 'angle_bin'
    Returns : list of bool, same length as records
    """
    import numpy as np
    from collections import defaultdict

    # group indices by bin
    bins = defaultdict(list)
    for i, r in enumerate(records):
        bins[r['angle_bin']].append(i)

    keep = [True] * len(records)
    for bin_name, idxs in bins.items():
        if len(idxs) < 4:
            continue
        vals = np.array([records[i]['adopted_D'] for i in idxs], dtype=float)
        q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        for i, v in zip(idxs, vals):
            if not (lo <= v <= hi):
                keep[i] = False
    return keep


# ── diagnostic grid ────────────────────────────────────────────────────────────

def build_grid(img, rows_data):
    """
    rows_data: list of (method_label, feature_img, mask_img, ellipse_dict)
    Output: header row + one row per method, 4 columns each.
    """
    W, H = 380, 280
    col_labels = ["original", "feature map", "binary mask", "ellipse fit"]

    header = np.zeros((28, W*4, 3), np.uint8)
    for i, h in enumerate(col_labels):
        cv2.putText(header, h, (i*W+8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

    grid_rows = [header]
    for method_label, feat, mask, e in rows_data:
        panels = [
            cv2.resize(to_bgr(img),                    (W, H)),
            cv2.resize(to_bgr(feat),                   (W, H)),
            cv2.resize(to_bgr(mask) if mask is not None else np.zeros((H,W,3),np.uint8), (W, H)),
            cv2.resize(draw_overlay(img, e, method_label), (W, H)),
        ]
        row = np.hstack(panels)
        grid_rows.append(row)

    return np.vstack(grid_rows)

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(INPUT_IMAGE))
    if img is None:
        raise FileNotFoundError(f"cannot read: {INPUT_IMAGE}")

    red_str = stretch_to_255(red_enhance(img))

    # ── run all three methods ──
    feat_a, mask_a, e_a, m_est_a                  = run_frangi(red_str)
    feat_b, mask_b, e_b, m_est_b, sig_b           = run_sato(red_str)
    feat_c, mask_c, e_c, m_est_c, sig_c, cr_c     = run_adaptive_dog(red_str)

    print(f"[minor estimates]  Frangi={m_est_a:.1f}  Sato(pre-dog={sig_b:.1f})={m_est_b:.1f}  AdaptDoG(s={sig_c:.1f})={m_est_c:.1f}")
    print(f"[AdaptDoG core_ratio={cr_c:.3f}]  {'elongated → tall dilation' if cr_c < 0.20 else 'oval/round → symmetric closing only'}")

    # ── grid + individual overlays ──
    grid = build_grid(img, [
        (f"Frangi+dil",               feat_a, mask_a, e_a),
        (f"DoG+Sato(s={sig_b:.0f})",  feat_b, mask_b, e_b),
        (f"AdaptDoG(s={sig_c:.0f})",  feat_c, mask_c, e_c),
    ])

    cv2.imwrite(str(OUTPUT_DIR / "00_comparison_grid.png"), grid)
    cv2.imwrite(str(OUTPUT_DIR / "01_frangi_overlay.png"),   draw_overlay(img, e_a, "Frangi+dil"))
    cv2.imwrite(str(OUTPUT_DIR / "02_sato_overlay.png"),     draw_overlay(img, e_b, f"DoG+Sato"))
    cv2.imwrite(str(OUTPUT_DIR / "03_adaptdog_overlay.png"), draw_overlay(img, e_c, f"AdaptDoG"))
    cv2.imwrite(str(OUTPUT_DIR / "04_frangi_feat.png"),  feat_a)
    cv2.imwrite(str(OUTPUT_DIR / "05_sato_feat.png"),    feat_b)
    cv2.imwrite(str(OUTPUT_DIR / "06_adaptdog_dog.png"), feat_c)

    # ── console report ──
    print()
    print(f"{'Method':<24}  {'major':>7}  {'minor':>7}  {'ratio':>6}  {'angle':>7}")
    print("-" * 62)
    ref = dict(major=140.7, minor=21.2, ratio=0.151, angle=88.2)
    for label, e in [("Frangi+dil", e_a), (f"DoG+Sato(s={sig_b:.0f})", e_b),
                     (f"AdaptDoG(s={sig_c:.0f})", e_c), ("--- ML target ---", ref)]:
        if e and e.get('major'):
            print(f"{label:<24}  {e['major']:7.1f}  {e['minor']:7.1f}  {e['ratio']:6.3f}  {e['angle']:7.1f}")
        else:
            print(f"{label:<24}  (no ellipse)")

    print(f"\noutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
