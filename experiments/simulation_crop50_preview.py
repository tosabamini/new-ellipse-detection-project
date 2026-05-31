"""
Simulation data — crop preview (60% keep, shifted 10% left).

Reads all PNGs from data/Simulation/p*/  (RAS files ignored).
For each image:
  - Applies RedEnhance (R − 0.5G − 0.5B) then stretch to [0,255]
  - Crops 60% of the original image, center shifted 10% to the left
  - Saves side-by-side comparison: [original | crop_colour | crop_red]

Output: data/processed/simulation_crop60_left10_preview/<pupil_group>/<stem>_preview.png

Run:
    python experiments/simulation_crop50_preview.py
"""

import glob
from pathlib import Path

import cv2
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────

SIM_DIR    = Path(__file__).parent.parent / "data" / "Simulation"
OUT_DIR    = Path(__file__).parent.parent / "data" / "processed" / "simulation_crop60_left10_preview"

CROP_RATIO   = 0.60   # keep 60% of width and height
LEFT_SHIFT   = 0.10   # shift center 10% of width to the left

# ── helpers ───────────────────────────────────────────────────────────────────

def center_crop(img: np.ndarray, ratio: float, left_shift: float = 0.0) -> np.ndarray:
    h, w = img.shape[:2]
    crop_w = int(w * ratio)
    crop_h = int(h * ratio)
    cx = w // 2 - int(w * left_shift)
    cy = h // 2
    x1 = max(cx - crop_w // 2, 0)
    x2 = min(cx + crop_w // 2, w)
    y1 = max(cy - crop_h // 2, 0)
    y2 = min(cy + crop_h // 2, h)
    return img[y1:y2, x1:x2].copy()


def red_enhance(img_bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    red = np.clip(r - 0.5 * g - 0.5 * b, 0, 255)
    mn, mx = red.min(), red.max()
    if mx > mn:
        red = (red - mn) / (mx - mn) * 255.0
    return red.astype(np.uint8)


def make_preview(orig: np.ndarray, crop_bgr: np.ndarray,
                 crop_red_gray: np.ndarray,
                 target_h: int = 400) -> np.ndarray:
    """Stack [original | colour crop | red crop] side-by-side, all at target_h."""
    def resize_h(img, h):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    panels = [
        resize_h(orig, target_h),
        resize_h(crop_bgr, target_h),
        resize_h(cv2.cvtColor(crop_red_gray, cv2.COLOR_GRAY2BGR), target_h),
    ]
    # vertical separator (thin white line)
    sep = np.full((target_h, 3, 3), 220, dtype=np.uint8)
    out = np.hstack([panels[0], sep, panels[1], sep, panels[2]])
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    pupil_dirs = sorted(SIM_DIR.glob("p*"))
    if not pupil_dirs:
        print(f"No p* subdirectories found under {SIM_DIR}")
        return

    total = 0
    for pupil_dir in pupil_dirs:
        group = pupil_dir.name
        out_group = OUT_DIR / group
        out_group.mkdir(parents=True, exist_ok=True)

        png_files = sorted(pupil_dir.glob("*.png")) + sorted(pupil_dir.glob("*.PNG"))
        print(f"{group}: {len(png_files)} PNG files")

        for png_path in png_files:
            img_bgr = cv2.imread(str(png_path))
            if img_bgr is None:
                print(f"  WARNING: could not load {png_path.name}")
                continue

            crop_bgr  = center_crop(img_bgr, CROP_RATIO, LEFT_SHIFT)
            crop_red  = red_enhance(crop_bgr)

            preview = make_preview(img_bgr, crop_bgr, crop_red)

            out_path = out_group / f"{png_path.stem}_preview.png"
            cv2.imwrite(str(out_path), preview)
            total += 1

        print(f"  → saved to {out_group}")

    print(f"\nDone. {total} preview images saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
