"""
104_RIGHT: AdaptDoG → IQR filter → SCA estimation
r_3D_ / samarth_3D_ prefixed images are excluded before processing.

Run:
    python experiments/sca_104_RIGHT.py
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
from src.analysis.refraction_estimator import estimate_D_for_image, fit_sca, SCALE_FACTOR

ROI_DIR = Path(r"C:\Users\issas\Desktop\new_ellipse_detection_project\data\processed\pipeline_runs\pipeline_run_101_106_v001\104_RIGHT\roi")
OUT_DIR = Path(r"C:\Users\issas\Desktop\new_ellipse_detection_project\experiments\method_compare_output\104_RIGHT_sca")

IQR_K   = 0.5
EXCLUDE_PREFIXES = ("r_3D_", "samarth_3D_")

ANGLE_BINS = {
    "90deg": {"range": (70, 110), "color": "#2980b9", "label": "90deg cond"},
    "45deg": {"range": (30,  60), "color": "#f39c12", "label": "45deg cond"},
    "0deg":  {"range": None,      "color": "#e74c3c", "label": "0/180deg cond"},
    "other": {"range": None,      "color": "#95a5a6", "label": "other"},
}

def classify_angle(deg: float) -> str:
    a = float(deg) % 180
    if 70 <= a < 110: return "90deg"
    if 30 <= a < 60:  return "45deg"
    if a < 20 or a >= 160: return "0deg"
    return "other"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(ROI_DIR.glob("*_roi.png"))
    roi_paths = [p for p in all_paths if not p.name.startswith(EXCLUDE_PREFIXES)]
    skipped   = len(all_paths) - len(roi_paths)
    print(f"Total files: {len(all_paths)}  Skipped (3D prefixes): {skipped}  Processing: {len(roi_paths)}")

    # ── Step 1: AdaptDoG ──────────────────────────────────────────────────────
    stems, ellipses = [], []
    for idx, p in enumerate(roi_paths):
        stem = p.stem.replace("_roi", "")
        img  = cv2.imread(str(p))
        if img is None:
            print(f"  skip {stem}")
            continue
        red = M.stretch_to_255(M.red_enhance(img))
        _, _, e, _, _, _ = M.run_adaptive_dog(red)
        stems.append(stem)
        ellipses.append(e)
        if e:
            print(f"  [{idx+1:2d}/{len(roi_paths)}] {stem[-22:]}  major={e['major']:.1f}  ratio={e['ratio']:.3f}  angle={e['angle']:.1f}")
        else:
            print(f"  [{idx+1:2d}/{len(roi_paths)}] {stem[-22:]}  no ellipse")

    # ── Step 2: IQR filter ────────────────────────────────────────────────────
    keep_mask = M.iqr_filter(ellipses, k=IQR_K)
    n_kept = sum(keep_mask)
    print(f"\nIQR filter (k={IQR_K}): kept={n_kept}  excluded={len(keep_mask)-n_kept}")
    for s, e, k in zip(stems, ellipses, keep_mask):
        if not k:
            print(f"  EXCLUDED: {s[-22:]}  major={e['major']:.1f}" if e else f"  EXCLUDED: {s[-22:]}  no ellipse")

    # ── Step 3: estimate D ────────────────────────────────────────────────────
    per_image = []
    for s, e, k in zip(stems, ellipses, keep_mask):
        if not k or not e:
            continue
        est = estimate_D_for_image(e["major"], e["minor"], SCALE_FACTOR)
        per_image.append({
            "stem":      s,
            "major":     e["major"],
            "minor":     e["minor"],
            "ratio":     e["ratio"],
            "angle":     e["angle"],
            "angle_bin": classify_angle(e["angle"]),
            **est,
        })

    valid = [x for x in per_image if x["valid"]]
    print(f"\nValid for SCA fit: {len(valid)} / {len(per_image)}")

    # D-value IQR filter per angle bin (extreme outliers only, k=1.5)
    d_mask = M.d_iqr_filter(valid, k=1.5)
    excluded_d = [x for x, keep in zip(valid, d_mask) if not keep]
    valid = [x for x, keep in zip(valid, d_mask) if keep]
    if excluded_d:
        print(f"D-IQR filter (k=1.5, per bin): removed {len(excluded_d)} outlier(s)")
        for x in excluded_d:
            print(f"  D-EXCLUDED: {x['stem'][-22:]}  bin={x['angle_bin']}  D={x['adopted_D']:.3f}  ratio={x['ratio']:.3f}")
    print(f"After D filter: {len(valid)} images for SCA fit")

    # ── Step 4: SCA fit ───────────────────────────────────────────────────────
    if len(valid) < 3:
        print("ERROR: fewer than 3 valid images")
        return

    alpha_arr = np.array([x["angle"]     for x in valid])
    D_arr     = np.array([x["adopted_D"] for x in valid])
    sca       = fit_sca(alpha_arr, D_arr)

    print(f"\n{'='*40}")
    print(f"  S  = {sca['S']:+.2f} D")
    print(f"  C  = {sca['C']:+.2f} D")
    print(f"  A  = {sca['A']:.1f} deg")
    print(f"  SE = {sca['SE']:+.2f} D")
    print(f"  R2 = {sca['R2']:.3f}  (n={sca['n']})")
    print(f"{'='*40}")

    # ── Step 5: plots ─────────────────────────────────────────────────────────
    # angle distribution
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist([x["angle"] for x in per_image], bins=18, range=(0, 180),
            color="#3498db", edgecolor="white", linewidth=0.5)
    ax.axvspan(70, 110, alpha=0.15, color="#2980b9", label="90deg cond")
    ax.axvspan(30,  60, alpha=0.15, color="#f39c12", label="45deg cond")
    ax.set_xlabel("Major axis angle (deg)")
    ax.set_ylabel("Count")
    ax.set_title("104_RIGHT  angle distribution (after IQR filter)")
    ax.set_xlim(0, 180)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "angle_dist.png", dpi=120)
    plt.close()

    # cos curve
    a_fine = np.linspace(0, 180, 360)
    a_rad2 = np.deg2rad(alpha_arr)
    X = np.column_stack([np.ones(len(a_rad2)), np.cos(2*a_rad2), np.sin(2*a_rad2)])
    P, *_ = np.linalg.lstsq(X, D_arr, rcond=None)
    D_fit_fine = P[0] + P[1]*np.cos(2*np.deg2rad(a_fine)) + P[2]*np.sin(2*np.deg2rad(a_fine))

    fig, ax = plt.subplots(figsize=(9, 5))
    for bin_key, binfo in ANGLE_BINS.items():
        pts = [(x["angle"], x["adopted_D"]) for x in valid if x["angle_bin"] == bin_key]
        if pts:
            xa, ya = zip(*pts)
            ax.scatter(xa, ya, color=binfo["color"],
                       label=f"{binfo['label']} (n={len(pts)})",
                       s=60, zorder=3, edgecolors="white", linewidths=0.5)
    ax.plot(a_fine, D_fit_fine, color="black", linewidth=2, label="cos fit", zorder=2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Major axis angle (deg)")
    ax.set_ylabel("Adopted D (diopters)")
    ax.set_title(f"104_RIGHT  D(alpha) fit\nS={sca['S']:+.2f}D  C={sca['C']:+.2f}D  A={sca['A']:.1f}deg  R2={sca['R2']:.3f}")
    ax.set_xlim(0, 180)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cos_curve.png", dpi=120)
    plt.close()

    print(f"\nSaved: angle_dist.png  cos_curve.png")

    # CSV
    fields = ["stem", "major", "minor", "ratio", "angle", "angle_bin",
              "p_est", "d1", "d2", "adopted_D", "valid"]
    with open(OUT_DIR / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in per_image:
            w.writerow({k: (f"{row[k]:.4f}" if isinstance(row[k], float) else row[k])
                        for k in fields})

    with open(OUT_DIR / "sca.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "S_D", "C_D", "A_deg", "SE_D", "R2", "n_valid", "n_kept", "n_total", "iqr_k"])
        w.writerow(["104_RIGHT",
                    f"{sca['S']:.3f}", f"{sca['C']:.3f}", f"{sca['A']:.1f}",
                    f"{sca['SE']:.3f}", f"{sca['R2']:.4f}",
                    sca["n"], n_kept, len(roi_paths), IQR_K])

    print(f"Saved: per_image.csv  sca.csv")
    print(f"\nAll outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
