"""
104_LEFT: AdaptDoG → IQR filter → 新瞳孔径推定(ratio+area) → SCA推定

新しい瞳孔径推定:
  area_scaled = major * minor * SF^2
  area = slope(p)*ratio + intercept(p)
  slope(p)     = 928.28*p^2 + 1780.95*p -  872.10
  intercept(p) = -462.23*p^2 + 3344.24*p - 4477.24
  → pの2次方程式を解く

Run:
    python experiments/sca_104_LEFT_newpupil.py
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
from src.analysis.build_patient_model import estimate_D_from_ratio_and_p
from src.analysis.refraction_estimator import fit_sca

ROI_DIR = Path(r"C:\Users\issas\Desktop\new_ellipse_detection_project\data\processed\pipeline_runs\pipeline_run_101_106_v001\104_LEFT\roi")
OUT_DIR = Path(r"C:\Users\issas\Desktop\new_ellipse_detection_project\experiments\method_compare_output\104_LEFT_newpupil")

SCALE_FACTOR = 1.3
IQR_K        = 0.5
P_MIN, P_MAX = 2.0, 9.0   # 有効な瞳孔径範囲 (mm)

# ── 統合式の係数 ───────────────────────────────────────────────
# slope(p)     = S2*p^2 + S1*p + S0
# intercept(p) = I2*p^2 + I1*p + I0
# area = slope(p)*ratio + intercept(p)
S2, S1, S0 =   928.28, 1780.95,  -872.10
I2, I1, I0 =  -462.23, 3344.24, -4477.24

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


def estimate_pupil(ratio: float, area_scaled: float) -> float | None:
    """
    (ratio, area_scaled) から瞳孔径 p (mm) を推定する。
    area = slope(p)*ratio + intercept(p) をpの2次方程式として解く。
    有効範囲 [P_MIN, P_MAX] に入る実数解を返す。複数あれば正の判別式側を選ぶ。
    """
    # [S2*ratio + I2]*p^2 + [S1*ratio + I1]*p + [S0*ratio + I0 - area] = 0
    a_coef = S2 * ratio + I2
    b_coef = S1 * ratio + I1
    c_coef = S0 * ratio + I0 - area_scaled

    disc = b_coef**2 - 4 * a_coef * c_coef
    if disc < 0:
        return None

    sq = np.sqrt(disc)
    roots = [(-b_coef + sq) / (2 * a_coef),
             (-b_coef - sq) / (2 * a_coef)]

    valid = [r for r in roots if P_MIN <= r <= P_MAX]
    if not valid:
        return None
    # 2つ有効なら大きい方（瞳孔が大きい側）を選ぶ
    return float(max(valid))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    roi_paths = sorted(ROI_DIR.glob("*_roi.png"))
    print(f"Images: {len(roi_paths)}")

    # ── Step 1: AdaptDoG ──────────────────────────────────────────
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

    # ── Step 2: IQR filter ────────────────────────────────────────
    keep_mask = M.iqr_filter(ellipses, k=IQR_K)
    n_kept = sum(keep_mask)
    print(f"IQR filter (k={IQR_K}): kept={n_kept}/{len(keep_mask)}")

    # ── Step 3: 瞳孔径推定 + D推定 ───────────────────────────────
    per_image = []
    for stem, e, keep in zip(stems, ellipses, keep_mask):
        if not keep or not e:
            continue

        ratio       = e["minor"] / e["major"]
        area_scaled = e["major"] * e["minor"] * SCALE_FACTOR**2

        p_est = estimate_pupil(ratio, area_scaled)
        if p_est is None:
            print(f"  {stem[-22:]}  ratio={ratio:.3f}  area={area_scaled:.0f}  -> no valid p")
            continue

        d1, d2 = estimate_D_from_ratio_and_p(ratio, p_est)
        valid  = d2 is not None
        adopted_D = float(d2) if valid else None

        per_image.append({
            "stem":      stem,
            "major":     e["major"],
            "minor":     e["minor"],
            "ratio":     ratio,
            "angle":     e["angle"],
            "angle_bin": classify_angle(e["angle"]),
            "area_scaled": area_scaled,
            "p_est":     p_est,
            "d1":        float(d1) if d1 is not None else None,
            "d2":        float(d2) if d2 is not None else None,
            "adopted_D": adopted_D,
            "valid":     valid,
        })

    valid_imgs = [x for x in per_image if x["valid"]]
    print(f"\np_est stats: min={min(x['p_est'] for x in valid_imgs):.2f}  "
          f"max={max(x['p_est'] for x in valid_imgs):.2f}  "
          f"median={float(np.median([x['p_est'] for x in valid_imgs])):.2f} mm")
    print(f"Valid for SCA fit: {len(valid_imgs)} / {len(per_image)}")

    # ── Step 4: D IQR filter per angle bin ───────────────────────
    d_mask = M.d_iqr_filter(valid_imgs, k=1.5)
    excluded = [x for x, k in zip(valid_imgs, d_mask) if not k]
    valid_imgs = [x for x, k in zip(valid_imgs, d_mask) if k]
    if excluded:
        print(f"D-IQR filter: removed {len(excluded)}")
        for x in excluded:
            print(f"  EXCLUDED: {x['stem'][-22:]}  D={x['adopted_D']:.3f}  bin={x['angle_bin']}")
    print(f"After D filter: {len(valid_imgs)} images for SCA fit")

    # ── Step 5: SCA fit ───────────────────────────────────────────
    if len(valid_imgs) < 3:
        print("ERROR: fewer than 3 valid images")
        return

    alpha_arr = np.array([x["angle"]     for x in valid_imgs])
    D_arr     = np.array([x["adopted_D"] for x in valid_imgs])
    sca       = fit_sca(alpha_arr, D_arr)

    print(f"\n{'='*40}")
    print(f"  S  = {sca['S']:+.2f} D")
    print(f"  C  = {sca['C']:+.2f} D")
    print(f"  A  = {sca['A']:.1f} deg")
    print(f"  SE = {sca['SE']:+.2f} D")
    print(f"  R2 = {sca['R2']:.3f}  (n={sca['n']})")
    print(f"{'='*40}")
    print(f"  True: S=-0.75  C=-1.50  A=70deg")

    # ── plots ─────────────────────────────────────────────────────
    a_fine = np.linspace(0, 180, 360)
    a_rad2 = np.deg2rad(alpha_arr)
    X = np.column_stack([np.ones(len(a_rad2)), np.cos(2*a_rad2), np.sin(2*a_rad2)])
    P, *_ = np.linalg.lstsq(X, D_arr, rcond=None)
    D_fit_fine = P[0] + P[1]*np.cos(2*np.deg2rad(a_fine)) + P[2]*np.sin(2*np.deg2rad(a_fine))

    fig, ax = plt.subplots(figsize=(9, 5))
    for bin_key, binfo in ANGLE_BINS.items():
        pts = [(x["angle"], x["adopted_D"]) for x in valid_imgs if x["angle_bin"] == bin_key]
        if pts:
            xa, ya = zip(*pts)
            ax.scatter(xa, ya, color=binfo["color"],
                       label=f"{binfo['label']} (n={len(pts)})",
                       s=60, zorder=3, edgecolors="white", linewidths=0.5)
    ax.plot(a_fine, D_fit_fine, color="black", linewidth=2, label="cos fit", zorder=2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Major axis angle (deg)")
    ax.set_ylabel("Adopted D (diopters)")
    ax.set_title(f"104_LEFT (new pupil method)\nS={sca['S']:+.2f}D  C={sca['C']:+.2f}D  "
                 f"A={sca['A']:.1f}deg  R2={sca['R2']:.3f}")
    ax.set_xlim(0, 180)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cos_curve.png", dpi=120)
    plt.close()

    # p_est distribution
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist([x["p_est"] for x in valid_imgs], bins=20, color="#3498db",
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Estimated pupil diameter (mm)")
    ax.set_ylabel("Count")
    ax.set_title("104_LEFT — pupil diameter distribution (new method)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pupil_dist.png", dpi=120)
    plt.close()

    # CSV
    fields = ["stem", "major", "minor", "ratio", "angle", "angle_bin",
              "area_scaled", "p_est", "d1", "d2", "adopted_D", "valid"]
    with open(OUT_DIR / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in per_image:
            w.writerow({k: (f"{row[k]:.4f}" if isinstance(row[k], float) else row[k])
                        for k in fields})

    with open(OUT_DIR / "sca.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "S_D", "C_D", "A_deg", "SE_D", "R2", "n", "SF"])
        w.writerow(["104_LEFT",
                    f"{sca['S']:.3f}", f"{sca['C']:.3f}", f"{sca['A']:.1f}",
                    f"{sca['SE']:.3f}", f"{sca['R2']:.4f}", sca["n"], SCALE_FACTOR])

    print(f"\nAll outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
