"""
101-106（103除く）の LEFT/RIGHT 全患者に新パイプラインを適用。

Pipeline:
  AdaptDoG → IQR filter (k=0.5) → ratio+area→pupil → D推定 → D-IQR → SCA fit

Run:
    python experiments/sca_batch_all.py
"""

import csv, sys, cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import ellipse_method_compare as M
from src.analysis.build_patient_model import estimate_D_from_ratio_and_p
from src.analysis.refraction_estimator import fit_sca

BASE    = PROJECT_ROOT / "data/processed/pipeline_runs/pipeline_run_101_106_v001"
OUT_DIR = PROJECT_ROOT / "experiments/method_compare_output/sca_batch_all"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCALE_FACTOR = 1.3
IQR_K        = 0.5
P_MIN, P_MAX = 2.0, 9.0

S2, S1, S0 =   928.28, 1780.95,  -872.10
I2, I1, I0 =  -462.23, 3344.24, -4477.24

TARGETS = [
    ("101_LEFT",  []),
    ("101_RIGHT", []),
    ("102_LEFT",  []),
    ("104_LEFT",  []),
    ("104_RIGHT", ("r_3D_", "samarth_3D_")),
    ("105_LEFT",  []),
    ("105_RIGHT", []),
    ("106_LEFT",  []),
    ("106_RIGHT", []),
]

def classify_angle(deg):
    a = float(deg) % 180
    if 70 <= a < 110: return "90deg"
    if 30 <= a < 60:  return "45deg"
    if a < 20 or a >= 160: return "0deg"
    return "other"

def estimate_pupil(ratio, area_scaled):
    a_c = S2 * ratio + I2
    b_c = S1 * ratio + I1
    c_c = S0 * ratio + I0 - area_scaled
    disc = b_c**2 - 4 * a_c * c_c
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    roots = [(-b_c + sq) / (2 * a_c), (-b_c - sq) / (2 * a_c)]
    valid = [r for r in roots if P_MIN <= r <= P_MAX]
    return float(max(valid)) if valid else None

def run_patient(patient_id, exclude_prefixes):
    roi_dir = BASE / patient_id / "roi"
    if not roi_dir.exists():
        return None

    all_paths = sorted(roi_dir.glob("*_roi.png"))
    roi_paths = [p for p in all_paths
                 if not (exclude_prefixes and p.name.startswith(tuple(exclude_prefixes)))]
    n_total = len(roi_paths)
    if n_total == 0:
        return None

    # AdaptDoG
    stems, ellipses = [], []
    for p in roi_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        red = M.stretch_to_255(M.red_enhance(img))
        _, _, e, _, _, _ = M.run_adaptive_dog(red)
        stems.append(p.stem.replace("_roi", ""))
        ellipses.append(e)

    # IQR filter
    keep_mask = M.iqr_filter(ellipses, k=IQR_K)
    n_kept = sum(keep_mask)

    # pupil + D estimation
    per_image, n_no_p = [], 0
    for stem, e, keep in zip(stems, ellipses, keep_mask):
        if not keep or not e:
            continue
        ratio       = e["minor"] / e["major"]
        area_scaled = e["major"] * e["minor"] * SCALE_FACTOR**2
        p_est = estimate_pupil(ratio, area_scaled)
        if p_est is None:
            n_no_p += 1
            continue
        d1, d2 = estimate_D_from_ratio_and_p(ratio, p_est)
        if d2 is None:
            continue
        per_image.append({
            "stem":      stem,
            "major":     e["major"],
            "minor":     e["minor"],
            "ratio":     ratio,
            "angle":     e["angle"],
            "angle_bin": classify_angle(e["angle"]),
            "p_est":     p_est,
            "adopted_D": float(d2),
        })

    valid = list(per_image)

    # D-IQR per angle bin
    d_mask   = M.d_iqr_filter(valid, k=1.5)
    valid    = [x for x, k in zip(valid, d_mask) if k]
    n_d_excl = sum(1 for k in d_mask if not k)

    if len(valid) < 3:
        return {"patient_id": patient_id, "error": "fewer than 3 valid",
                "n_total": n_total, "n_kept": n_kept}

    alpha_arr = np.array([x["angle"]     for x in valid])
    D_arr     = np.array([x["adopted_D"] for x in valid])
    sca       = fit_sca(alpha_arr, D_arr)
    p_med     = float(np.median([x["p_est"] for x in valid]))

    # cos curve plot
    pat_dir = OUT_DIR / patient_id
    pat_dir.mkdir(exist_ok=True)

    a_fine = np.linspace(0, 180, 360)
    a_rad2 = np.deg2rad(alpha_arr)
    X = np.column_stack([np.ones(len(a_rad2)), np.cos(2*a_rad2), np.sin(2*a_rad2)])
    P, *_ = np.linalg.lstsq(X, D_arr, rcond=None)
    D_fit  = P[0] + P[1]*np.cos(2*np.deg2rad(a_fine)) + P[2]*np.sin(2*np.deg2rad(a_fine))

    COLORS = {"90deg": "#2980b9", "45deg": "#f39c12", "0deg": "#e74c3c", "other": "#95a5a6"}
    fig, ax = plt.subplots(figsize=(8, 4))
    for bin_key, col in COLORS.items():
        pts = [(x["angle"], x["adopted_D"]) for x in valid if x["angle_bin"] == bin_key]
        if pts:
            xa, ya = zip(*pts)
            ax.scatter(xa, ya, color=col, label=f"{bin_key} (n={len(pts)})",
                       s=50, zorder=3, edgecolors="white", linewidths=0.4)
    ax.plot(a_fine, D_fit, color="black", linewidth=2, label="cos fit")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlim(0, 180)
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("D (diopters)")
    ax.set_title(f"{patient_id}  S={sca['S']:+.2f}  C={sca['C']:+.2f}  "
                 f"A={sca['A']:.0f}deg  R2={sca['R2']:.3f}  p_med={p_med:.1f}mm")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(pat_dir / "cos_curve.png", dpi=110)
    plt.close()

    return {
        "patient_id": patient_id,
        "S":  sca["S"],  "C":  sca["C"],  "A":  sca["A"],
        "SE": sca["SE"], "R2": sca["R2"], "n":  sca["n"],
        "p_med": p_med,
        "n_total": n_total, "n_kept": n_kept,
        "n_no_p": n_no_p, "n_d_excl": n_d_excl,
    }


def main():
    results = []
    for patient_id, excl in TARGETS:
        r = run_patient(patient_id, excl)
        if r is None:
            print(f"{patient_id}: SKIP (no ROI)")
            continue
        if "error" in r:
            print(f"{patient_id}: ERROR — {r['error']}")
            continue
        print(f"{patient_id}: S={r['S']:+.2f}  C={r['C']:+.2f}  A={r['A']:.0f}deg  "
              f"R2={r['R2']:.3f}  p_med={r['p_med']:.1f}mm  "
              f"(n_total={r['n_total']} kept={r['n_kept']} no_p={r['n_no_p']} d_excl={r['n_d_excl']})")
        results.append(r)

    # summary CSV
    fields = ["patient_id", "S", "C", "A", "SE", "R2", "n",
              "p_med", "n_total", "n_kept", "n_no_p", "n_d_excl"]
    with open(OUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{r[k]:.3f}" if isinstance(r[k], float) else r[k])
                        for k in fields if k in r})

    print(f"\nSaved: {OUT_DIR / 'summary.csv'}")
    print(f"Plots: {OUT_DIR}/<patient>/cos_curve.png")

    # summary table
    print(f"\n{'Patient':<12}  {'S':>7}  {'C':>7}  {'A':>6}  {'SE':>7}  {'R2':>6}  {'p_med':>6}")
    print("-" * 62)
    for r in results:
        print(f"{r['patient_id']:<12}  {r['S']:>+7.2f}  {r['C']:>+7.2f}  "
              f"{r['A']:>6.1f}  {r['SE']:>+7.2f}  {r['R2']:>6.3f}  {r['p_med']:>5.1f}mm")


if __name__ == "__main__":
    main()
