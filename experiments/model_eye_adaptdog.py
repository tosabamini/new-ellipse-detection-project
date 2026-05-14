"""
Apply AdaptDoG to all model eye ROI images and build calibration table.

For each (pupil_mm, refraction_D) combination:
  - Run AdaptDoG on all ROI images
  - Apply IQR filter (k=0.5) on major axis
  - Record median major_px and median ratio

Output:
  experiments/method_compare_output/model_eye_adaptdog/
    calibration_table.csv     per-(pupil, D) summary
    major_vs_D.png            major axis vs D, one curve per pupil size
    ratio_vs_D.png            ratio vs D, one curve per pupil size
    per_image.csv             raw per-image results

Run:
    python experiments/model_eye_adaptdog.py
"""

import csv
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import ellipse_method_compare as M

# ── run → pupil_mm mapping ─────────────────────────────────────────────────────
ME_RUNS = {
    7.0: "model_eye_v001",
    5.0: "model_eye_5mm_v001",
    3.0: "model_eye_3mm_v001",
}
ME_BASE = PROJECT_ROOT / "data/processed/model_eye_runs"

OUT_DIR = PROJECT_ROOT / "experiments/method_compare_output/model_eye_adaptdog"
IQR_K   = 0.5


def parse_D_from_folder(name: str) -> float | None:
    """
    '1200_M_04_00D' -> -4.0
    '1600_Z_00_00D' ->  0.0
    '1700_P_01_00D' -> +1.0
    """
    parts = name.split("_")
    if len(parts) < 4:
        return None
    sign_code = parts[1].upper()
    try:
        d_int  = int(parts[2])
        d_frac = int(parts[3].replace("D", "").replace("d", ""))
        val = d_int + d_frac / 100.0
    except ValueError:
        return None
    if sign_code == "M":
        return -val
    elif sign_code == "P":
        return +val
    elif sign_code == "Z":
        return 0.0
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_image_rows = []
    calib_rows     = []

    for pupil_mm, run_name in sorted(ME_RUNS.items()):
        run_path = ME_BASE / run_name
        if not run_path.exists():
            print(f"SKIP: {run_path} not found")
            continue

        print(f"\n{'='*50}")
        print(f"pupil={pupil_mm}mm  run={run_name}")

        ref_folders = sorted([f for f in run_path.iterdir()
                               if f.is_dir() and f.name != "analysis"])

        for ref_folder in ref_folders:
            D = parse_D_from_folder(ref_folder.name)
            if D is None:
                continue

            roi_dir = ref_folder / "roi"
            if not roi_dir.exists():
                continue

            roi_paths = sorted(roi_dir.glob("*.png"))
            if not roi_paths:
                continue

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

            keep_mask = M.iqr_filter(ellipses, k=IQR_K)
            kept = [(s, e) for s, e, k in zip(stems, ellipses, keep_mask) if k and e]

            if not kept:
                print(f"  {ref_folder.name:<25}  D={D:+5.2f}  NO valid images")
                continue

            majors = [e["major"] for _, e in kept]
            ratios = [e["ratio"] for _, e in kept]
            med_major = float(np.median(majors))
            med_ratio = float(np.median(ratios))

            print(f"  {ref_folder.name:<25}  D={D:+5.2f}  n={len(kept)}/{len(roi_paths)}"
                  f"  major={med_major:.1f}  ratio={med_ratio:.3f}")

            for stem, e in kept:
                per_image_rows.append({
                    "pupil_mm":   pupil_mm,
                    "D":          D,
                    "folder":     ref_folder.name,
                    "stem":       stem,
                    "major":      f"{e['major']:.2f}",
                    "minor":      f"{e['minor']:.2f}",
                    "ratio":      f"{e['ratio']:.4f}",
                    "angle":      f"{e['angle']:.1f}",
                })

            calib_rows.append({
                "pupil_mm":   pupil_mm,
                "D":          D,
                "folder":     ref_folder.name,
                "n_kept":     len(kept),
                "n_total":    len(roi_paths),
                "major_med":  f"{med_major:.2f}",
                "ratio_med":  f"{med_ratio:.4f}",
                "major_std":  f"{float(np.std(majors)):.2f}",
                "ratio_std":  f"{float(np.std(ratios)):.4f}",
            })

    # ── save CSVs ──────────────────────────────────────────────────────────────
    if calib_rows:
        with open(OUT_DIR / "calibration_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(calib_rows[0].keys()))
            w.writeheader(); w.writerows(calib_rows)
        print(f"\nSaved: calibration_table.csv  ({len(calib_rows)} rows)")

    if per_image_rows:
        with open(OUT_DIR / "per_image.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
            w.writeheader(); w.writerows(per_image_rows)
        print(f"Saved: per_image.csv  ({len(per_image_rows)} rows)")

    # ── plots ──────────────────────────────────────────────────────────────────
    colors = {7.0: "#2980b9", 5.0: "#f39c12", 3.0: "#e74c3c"}

    from collections import defaultdict
    by_pupil = defaultdict(list)
    for r in calib_rows:
        by_pupil[r["pupil_mm"]].append(r)

    # major vs D
    fig, ax = plt.subplots(figsize=(9, 5))
    for pupil_mm in sorted(by_pupil.keys(), reverse=True):
        rows = sorted(by_pupil[pupil_mm], key=lambda x: x["D"])
        Ds  = [r["D"]                    for r in rows]
        mj  = [float(r["major_med"])     for r in rows]
        std = [float(r["major_std"])     for r in rows]
        c   = colors.get(pupil_mm, "gray")
        ax.errorbar(Ds, mj, yerr=std, fmt="o-", color=c,
                    label=f"{pupil_mm}mm", capsize=3, linewidth=1.5)
    ax.set_xlabel("Refraction D (diopters)")
    ax.set_ylabel("Major axis (px)")
    ax.set_title("Model eye — Major axis vs Refraction (AdaptDoG)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "major_vs_D.png", dpi=120)
    plt.close()

    # ratio vs D
    fig, ax = plt.subplots(figsize=(9, 5))
    for pupil_mm in sorted(by_pupil.keys(), reverse=True):
        rows = sorted(by_pupil[pupil_mm], key=lambda x: x["D"])
        Ds  = [r["D"]                   for r in rows]
        rt  = [float(r["ratio_med"])    for r in rows]
        std = [float(r["ratio_std"])    for r in rows]
        c   = colors.get(pupil_mm, "gray")
        ax.errorbar(Ds, rt, yerr=std, fmt="o-", color=c,
                    label=f"{pupil_mm}mm", capsize=3, linewidth=1.5)
    ax.set_xlabel("Refraction D (diopters)")
    ax.set_ylabel("Ratio (minor/major)")
    ax.set_title("Model eye — Ratio vs Refraction (AdaptDoG)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ratio_vs_D.png", dpi=120)
    plt.close()

    print(f"Saved: major_vs_D.png  ratio_vs_D.png")
    print(f"\nAll outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
