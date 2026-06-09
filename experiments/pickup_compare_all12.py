"""
全12名の0603 vs 0604 SCA比較 (S=mean D, C=-range D, A=cosフィット軸, 0607補正モデル)

Run:
  python -m experiments.pickup_compare_all12
"""

import csv
import numpy as np
from pathlib import Path
from src.analysis.sim_ratio_model_0607 import estimate_D_from_ratio_sim
from src.analysis.refraction_estimator import fit_sca

PICKUP_DIR = Path("data/Repeatability/PickUP")

PAIRS = [
    ("Kavya",      "01_Kavya",      "02_Kavya 0604"),
    ("Linsha",     "03_Linsha",     "04_Linsha 0604"),
    ("Dilsha",     "05_Dilsha",     "06_Dilsha 0604"),
    ("Abhishek",   "07_Abhishek",   "08_Abhishek 0604"),
    ("Prateeksha", "09_Prateeksha", "10_Prateeksha 0604"),
    ("Aslaha",     "11_Aslaha",     "12_Aslaha 0604"),
    ("Anagha",     "13_Anagha",     "14_Anagha 0604"),
    ("Aiswarya",   "15_Aiswarya",   "16_Aiswarya 0604"),
    ("Lubana",     "17_Lubana",     "18_Lubana 0604"),
    ("Rinsha",     "19_Rinsha",     "20_Rinsha 0604"),
    ("Fathima",    "21_Fathima",    "22_Fathima 0604"),
    ("Niranjana",  "23_Niranjana",  "24_Niranjana 0604"),
]


def get_sca(subj_dir: str, eye: str) -> dict | None:
    csv_path = PICKUP_DIR / subj_dir / eye / "ellipse_results.csv"
    if not csv_path.exists():
        return None
    pairs = []
    for row in csv.DictReader(open(csv_path, encoding="utf-8")):
        if not row["ratio"]:
            continue
        d, _ = estimate_D_from_ratio_sim(float(row["ratio"]))
        if d is not None:
            pairs.append((float(row["angle"]), d))
    if len(pairs) < 3:
        return None
    ds    = [p[1] for p in pairs]
    alpha = np.array([p[0] for p in pairs])
    S  = float(np.mean(ds))
    C  = -(max(ds) - min(ds))
    SE = S + C / 2
    A  = fit_sca(alpha, np.array(ds))["A"]
    return dict(S=S, C=C, A=A, SE=SE)


def main():
    rows = []

    for name, s03, s04 in PAIRS:
        for eye in ("LEFT", "RIGHT"):
            r03 = get_sca(s03, eye)
            r04 = get_sca(s04, eye)
            if not r03 or not r04:
                continue
            dS  = r04["S"]  - r03["S"]
            dC  = r04["C"]  - r03["C"]
            dSE = r04["SE"] - r03["SE"]
            raw = abs(r03["A"] - r04["A"])
            dA  = min(raw, 180 - raw)
            rows.append({
                "subject": name, "eye": eye,
                "S_0603":  r03["S"],  "C_0603":  r03["C"],
                "A_0603":  r03["A"],  "SE_0603": r03["SE"],
                "S_0604":  r04["S"],  "C_0604":  r04["C"],
                "A_0604":  r04["A"],  "SE_0604": r04["SE"],
                "dS": dS, "dC": dC, "dA": dA, "dSE": dSE,
            })

    # 表示
    hdr = (f"{'被験者/眼':<22}  "
           f"{'S_03':>6} {'S_04':>6} {'dS':>6}  "
           f"{'C_03':>6} {'C_04':>6} {'dC':>6}  "
           f"{'dA':>6}  "
           f"{'SE_03':>6} {'SE_04':>6} {'dSE':>6}")
    print(hdr)
    print("-" * 100)
    for r in rows:
        label = r["subject"] + "/" + r["eye"]
        print(f"{label:<22}  "
              f"{r['S_0603']:+6.2f} {r['S_0604']:+6.2f} {r['dS']:+6.2f}  "
              f"{r['C_0603']:+6.2f} {r['C_0604']:+6.2f} {r['dC']:+6.2f}  "
              f"{r['dA']:6.1f}  "
              f"{r['SE_0603']:+6.2f} {r['SE_0604']:+6.2f} {r['dSE']:+6.2f}")

    # 統計
    print("-" * 100)
    abs_dS  = [abs(r["dS"])  for r in rows]
    abs_dC  = [abs(r["dC"])  for r in rows]
    abs_dSE = [abs(r["dSE"]) for r in rows]
    abs_dA  = [r["dA"]       for r in rows]
    dA_sig  = [r["dA"] for r in rows
                if abs(r["C_0603"]) > 0.5 and abs(r["C_0604"]) > 0.5]
    n = len(rows)
    print(f"{'平均|Δ| (全24眼)':<22}  "
          f"{'':>6} {'':>6} {sum(abs_dS)/n:+6.2f}  "
          f"{'':>6} {'':>6} {sum(abs_dC)/n:+6.2f}  "
          f"{sum(abs_dA)/n:6.1f}  "
          f"{'':>6} {'':>6} {sum(abs_dSE)/n:+6.2f}")
    print(f"{'最大|Δ| (全24眼)':<22}  "
          f"{'':>6} {'':>6} {max(abs_dS):+6.2f}  "
          f"{'':>6} {'':>6} {max(abs_dC):+6.2f}  "
          f"{max(abs_dA):6.1f}  "
          f"{'':>6} {'':>6} {max(abs_dSE):+6.2f}")
    if dA_sig:
        print(f"AX平均 (C>0.5D両日, n={len(dA_sig)}眼): {sum(dA_sig)/len(dA_sig):.1f}°  "
              f"最大: {max(dA_sig):.1f}°")

    # CSV保存
    out = PICKUP_DIR / "comparison_all12_0603_vs_0604.csv"
    fields = ["subject", "eye",
              "S_0603", "C_0603", "A_0603", "SE_0603",
              "S_0604", "C_0604", "A_0604", "SE_0604",
              "dS", "dC", "dA", "dSE"]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
