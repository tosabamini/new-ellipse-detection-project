"""
Standalone single-image ellipse tester.
Pipeline: red-enhance → histogram stretch [0,255] → Otsu → central blob → cv2.fitEllipse

No src.* imports – copy this file anywhere and run with:
    python otsu_ellipse_single.py
"""

import cv2
import numpy as np
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
INPUT_IMAGE = Path(r"C:\Users\issas\Desktop\new_ellipse_detection_project\data\processed\pipeline_runs\pipeline_run_101_106_v001\102_LEFT\roi\IMG_20260513_143554_905_roi.png")
OUTPUT_DIR  = Path(r"C:\Users\issas\Desktop\new_ellipse_detection_project\experiments\otsu_output")

# ── image-processing helpers ───────────────────────────────────────────────────

def red_enhance(bgr: np.ndarray) -> np.ndarray:
    """R − 0.5G − 0.5B  →  isolates red glow, suppresses blue/green artefacts."""
    f = bgr.astype(np.float32)
    red = np.clip(f[:, :, 2] - 0.5 * f[:, :, 1] - 0.5 * f[:, :, 0], 0, 255)
    return red.astype(np.uint8)


def stretch_to_255(gray: np.ndarray) -> np.ndarray:
    """Linear stretch so the darkest pixel → 0, brightest pixel → 255."""
    lo, hi = int(gray.min()), int(gray.max())
    if hi == lo:
        return gray.copy()
    return ((gray.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)


def dog_sharpen(gray: np.ndarray, sigma_small: float = 1.5, sigma_large: float = 15.0) -> np.ndarray:
    """
    Difference of Gaussians: subtract slow-varying glow from the sharp bright core.
    Small sigma preserves the thin streak; large sigma models the diffuse halo.
    Negative values (background) are clipped to 0, then re-stretched to [0,255].
    """
    blur_s = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma_small)
    blur_l = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma_large)
    dog = np.clip(blur_s - blur_l, 0, None)
    return stretch_to_255(dog)


def otsu_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask via Otsu threshold on the input grey image."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def dilate_along_major(mask: np.ndarray, kw: int = 7, kh: int = 25) -> np.ndarray:
    """
    Controlled rectangular dilation: kh pixels tall, kw pixels wide.
    Recovers the dim tips cut off by Otsu while keeping the minor axis controlled.
    Works well when the major axis is nearly vertical (~88-90 deg).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    return cv2.dilate(mask, kernel)


def pick_central_blob(binary: np.ndarray, min_area: int = 50) -> np.ndarray:
    """
    Morphological clean-up, then keep the single connected component
    whose score (area − 0.5 × distance² from image centre) is highest.
    This prefers a large region that is close to the centre over small
    peripheral artefacts.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    m = cv2.morphologyEx(m,      cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return m

    h, w  = m.shape
    cx0, cy0 = w / 2.0, h / 2.0

    best_cnt, best_score = None, None
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist2 = (cx - cx0) ** 2 + (cy - cy0) ** 2
        score = area - 0.5 * dist2
        if best_score is None or score > best_score:
            best_cnt, best_score = cnt, score

    if best_cnt is None:
        return m

    out = np.zeros_like(m)
    cv2.drawContours(out, [best_cnt], -1, 255, cv2.FILLED)
    return out


def fit_ellipse(single_mask: np.ndarray):
    """
    Fit an ellipse to the filled contour in single_mask.
    Returns a dict with cx, cy, major, minor, angle_deg, raw (cv2 tuple),
    or None if fitting is impossible.
    """
    cnts, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return None

    ellipse = cv2.fitEllipse(cnt)
    (cx, cy), (ax1, ax2), angle = ellipse
    major = max(ax1, ax2)
    minor = min(ax1, ax2)
    if ax2 > ax1:          # normalise so angle always refers to major axis
        angle += 90.0
    angle %= 180.0
    return {
        "cx": float(cx), "cy": float(cy),
        "major": float(major), "minor": float(minor),
        "angle": float(angle),
        "ratio": float(minor / major) if major > 0 else 0.0,
        "raw": ellipse,
    }


# ── diagnostic grid builder ────────────────────────────────────────────────────

def _to_bgr(g: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def build_grid(img, red_str, dog, mask_raw, mask_central, canvas, e) -> np.ndarray:
    W, H = 400, 300

    mask_vis = _to_bgr(mask_central)
    cnts, _ = cv2.findContours(mask_central, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask_vis, cnts, -1, (0, 255, 0), 1)

    panels = [img, _to_bgr(red_str), _to_bgr(dog), _to_bgr(mask_raw), mask_vis, canvas]
    titles = ["original", "red_stretched", "DoG (glow removed)",
              "otsu_on_DoG", "core + dilate(7x25)", "ellipse_fit"]

    row1 = np.hstack([cv2.resize(p, (W, H)) for p in panels[:3]])
    row2 = np.hstack([cv2.resize(p, (W, H)) for p in panels[3:]])
    grid = np.vstack([row1, row2])

    for i, title in enumerate(titles):
        col, row = i % 3, i // 3
        cv2.putText(grid, title, (col * W + 8, row * H + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 50), 2, cv2.LINE_AA)

    if e:
        info = [
            f"major = {e['major']:.1f} px",
            f"minor = {e['minor']:.1f} px",
            f"ratio = {e['ratio']:.3f}",
            f"angle = {e['angle']:.1f} deg",
        ]
        for k, line in enumerate(info):
            cv2.putText(grid, line, (2 * W + 8, H + 55 + k * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 255), 2, cv2.LINE_AA)

    return grid


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(INPUT_IMAGE))
    if img is None:
        raise FileNotFoundError(f"cannot read: {INPUT_IMAGE}")

    # ── pipeline ──
    red          = red_enhance(img)               # R − 0.5G − 0.5B
    red_str      = stretch_to_255(red)            # normalise [0, 255]
    dog          = dog_sharpen(red_str)           # remove diffuse halo via DoG
    mask_raw     = otsu_mask(dog)                 # Otsu on halo-free image
    mask_core    = pick_central_blob(mask_raw)    # keep central component
    mask_final   = dilate_along_major(mask_core)  # recover dim tips via (7×25) dilation
    e            = fit_ellipse(mask_final)

    # ── overlay ──
    canvas = img.copy()
    if e:
        cv2.ellipse(canvas, e["raw"], (255, 0, 255), 2)
        cv2.circle(canvas, (int(e["cx"]), int(e["cy"])), 4, (255, 0, 255), -1)

    # ── save ──
    grid = build_grid(img, red_str, dog, mask_raw, mask_final, canvas, e)
    cv2.imwrite(str(OUTPUT_DIR / "00_diagnostic_grid.png"), grid)
    cv2.imwrite(str(OUTPUT_DIR / "01_ellipse_overlay.png"), canvas)
    cv2.imwrite(str(OUTPUT_DIR / "02_mask_final.png"),      mask_final)
    cv2.imwrite(str(OUTPUT_DIR / "03_dog.png"),             dog)

    # ── report ──
    print("=== otsu ellipse result (DoG + Otsu) ===")
    if e:
        print(f"  center : ({e['cx']:.1f}, {e['cy']:.1f})")
        print(f"  major  : {e['major']:.1f} px   (target ~140.7)")
        print(f"  minor  : {e['minor']:.1f} px   (target ~21.2)")
        print(f"  ratio  : {e['ratio']:.3f}      (target ~0.151)")
        print(f"  angle  : {e['angle']:.1f} deg  (target ~88.2 deg)")
    else:
        print("  no ellipse found")
    print(f"outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
