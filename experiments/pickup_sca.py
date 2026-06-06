"""
PickUP フォルダの ellipse_results.csv から ratio → D → SCA を計算する。

フィッティングミスのある画像・被験者は除外。

Run:
  python -m experiments.pickup_sca
"""

import csv
import numpy as np
from pathlib import Path

from src.analysis.sim_ratio_model_0607 import estimate_D_from_ratio_sim
from src.analysis.refraction_estimator import fit_sca

PICKUP_DIR = Path("data/Repeatability/PickUP")

# フィッティングミスのある被験者フォルダを完全除外
SKIP_SUBJECTS = {
    "05_Dilsha", "06_Dilsha 0604",
    "07_Abhishek", "08_Abhishek 0604",
    "23_Niranjana", "24_Niranjana 0604",
}


def main():
    results = []

    for subj_dir in sorted(PICKUP_DIR.iterdir()):
        if not subj_dir.is_dir():
            continue
        if subj_dir.name in SKIP_SUBJECTS:
            continue

        for eye in ("LEFT", "RIGHT"):
            eye_dir = subj_dir / eye
            csv_path = eye_dir / "ellipse_results.csv"
            if not csv_path.exists():
                continue

            pairs = []
            with open(csv_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if not row["ratio"]:
                        continue
                    ratio = float(row["ratio"])
                    angle = float(row["angle"])
                    d_myo, _ = estimate_D_from_ratio_sim(ratio)
                    if d_myo is None:
                        continue
                    pairs.append((angle, float(d_myo)))

            if len(pairs) < 3:
                print(f"  SKIP {subj_dir.name}/{eye}: only {len(pairs)} valid points")
                continue

            alpha = np.array([p[0] for p in pairs])
            D     = np.array([p[1] for p in pairs])
            sca   = fit_sca(alpha, D)

            tag = f"{subj_dir.name}/{eye}"
            print(f"  {tag:<30s}  S={sca['S']:+.2f}  C={sca['C']:+.2f}"
                  f"  A={sca['A']:5.1f}deg  R2={sca['R2']:.3f}  n={sca['n']}")

            results.append({
                "subject": subj_dir.name,
                "eye":     eye,
                "n":       sca["n"],
                "S":       sca["S"],
                "C":       sca["C"],
                "A":       sca["A"],
                "SE":      sca["SE"],
                "R2":      sca["R2"],
            })

    out_csv = PICKUP_DIR / "pickup_sca_results_0607.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["subject", "eye", "n", "S", "C", "A", "SE", "R2"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    print(f"\nSaved: {out_csv}  ({len(results)} rows)")


if __name__ == "__main__":
    main()
