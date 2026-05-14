"""
SCALE_FACTOR sweep for 104_RIGHT.
Run: python experiments/sf_sweep_104_RIGHT.py
"""
import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import ellipse_method_compare as M
from src.analysis.refraction_estimator import estimate_D_for_image, fit_sca

ROI_DIR = Path(r"C:\Users\issas\Desktop\new_ellipse_detection_project\data\processed\pipeline_runs\pipeline_run_101_106_v001\104_RIGHT\roi")
IQR_K = 0.5
EXCLUDE_PREFIXES = ("r_3D_", "samarth_3D_")

ANGLE_BINS = {
    "90deg": (70, 110),
    "45deg": (30, 60),
    "0deg":  None,
    "other": None,
}

def classify_angle(deg):
    a = float(deg) % 180
    if 70 <= a < 110: return "90deg"
    if 30 <= a < 60:  return "45deg"
    if a < 20 or a >= 160: return "0deg"
    return "other"


def main():
    all_paths = sorted(ROI_DIR.glob("*_roi.png"))
    roi_paths = [p for p in all_paths if not p.name.startswith(EXCLUDE_PREFIXES)]
    print(f"Files: {len(roi_paths)} (excluded {len(all_paths)-len(roi_paths)} 3D prefix)")

    # Run AdaptDoG once
    stems, ellipses = [], []
    for p in roi_paths:
        stem = p.stem.replace("_roi", "")
        img  = cv2.imread(str(p))
        if img is None:
            continue
        red = M.stretch_to_255(M.red_enhance(img))
        _, _, e, _, _, _ = M.run_adaptive_dog(red)
        stems.append(stem)
        ellipses.append(e)

    # IQR filter on major axis
    keep_mask = M.iqr_filter(ellipses, k=IQR_K)
    n_kept = sum(keep_mask)
    print(f"IQR filter (k={IQR_K}): kept={n_kept}/{len(keep_mask)}")

    kept_stems    = [s for s, k in zip(stems, keep_mask) if k]
    kept_ellipses = [e for e, k in zip(ellipses, keep_mask) if k and e]

    print(f"\n{'SF':>5}  {'S':>7}  {'C':>7}  {'A':>7}  {'SE':>7}  {'R2':>6}  {'n':>3}  {'p_med':>7}")
    print("-" * 60)

    for sf in [1.3, 1.4, 1.5, 1.6, 1.7]:
        per_image = []
        for s, e in zip(kept_stems, kept_ellipses):
            est = estimate_D_for_image(e["major"], e["minor"], sf)
            if est["valid"]:
                per_image.append({
                    "stem":      s,
                    "angle":     e["angle"],
                    "angle_bin": classify_angle(e["angle"]),
                    "p_est":     est["p_est"],
                    **est,
                })

        valid = [x for x in per_image if x["valid"]]

        # per-bin D IQR filter
        d_mask = M.d_iqr_filter(valid, k=1.5)
        valid  = [x for x, keep in zip(valid, d_mask) if keep]

        if len(valid) < 3:
            print(f"{sf:>5.1f}  {'<3 valid':>7}")
            continue

        alpha_arr = np.array([x["angle"]     for x in valid])
        D_arr     = np.array([x["adopted_D"] for x in valid])
        sca       = fit_sca(alpha_arr, D_arr)
        p_med     = float(np.median([x["p_est"] for x in valid]))

        print(f"{sf:>5.1f}  {sca['S']:>+7.2f}  {sca['C']:>+7.2f}  {sca['A']:>7.1f}  "
              f"{sca['SE']:>+7.2f}  {sca['R2']:>6.3f}  {sca['n']:>3}  {p_med:>6.2f}mm")

    print(f"\n  True: S=-1.00  C=-1.50  A=100deg  (SE=-1.75)")


if __name__ == "__main__":
    main()
