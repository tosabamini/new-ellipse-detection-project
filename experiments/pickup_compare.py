"""0603 vs 0604 S/C/A/SE 差分集計"""
import csv
from pathlib import Path

PICKUP_DIR = Path("data/Repeatability/PickUP")

data = {}
for row in csv.DictReader(open(PICKUP_DIR / "pickup_sca_range_0607.csv", encoding="utf-8")):
    data[(row["subject"], row["eye"])] = {k: float(row[k]) for k in ("S","C","A","SE")}

pairs = [
    ("Kavya",      "01_Kavya",      "02_Kavya 0604"),
    ("Linsha",     "03_Linsha",     "04_Linsha 0604"),
    ("Prateeksha", "09_Prateeksha", "10_Prateeksha 0604"),
    ("Aslaha",     "11_Aslaha",     "12_Aslaha 0604"),
    ("Anagha",     "13_Anagha",     "14_Anagha 0604"),
    ("Aiswarya",   "15_Aiswarya",   "16_Aiswarya 0604"),
    ("Lubana",     "17_Lubana",     "18_Lubana 0604"),
    ("Rinsha",     "19_Rinsha",     "20_Rinsha 0604"),
    ("Fathima",    "21_Fathima",    "22_Fathima 0604"),
]

print(f"{'被験者/眼':<22}  {'|ΔS|':>6} {'|ΔC|':>6} {'|ΔA|':>6} {'|ΔSE|':>7}  C判定")
print("-" * 72)

dS_all, dC_all, dA_all, dSE_all = [], [], [], []
dA_sig = []

for name, s03, s04 in pairs:
    for eye in ("LEFT", "RIGHT"):
        r03 = data.get((s03, eye))
        r04 = data.get((s04, eye))
        if not r03 or not r04:
            continue

        dS  = abs(r03["S"]  - r04["S"])
        dC  = abs(r03["C"]  - r04["C"])
        dSE = abs(r03["SE"] - r04["SE"])
        raw = abs(r03["A"]  - r04["A"])
        dA  = min(raw, 180 - raw)

        c_sig = abs(r03["C"]) > 0.5 and abs(r04["C"]) > 0.5
        tag = "  ★" if c_sig else "  (C小)"

        label = name + "/" + eye
        print(f"{label:<22}  {dS:6.2f} {dC:6.2f} {dA:6.1f} {dSE:7.2f}{tag}")

        dS_all.append(dS); dC_all.append(dC)
        dSE_all.append(dSE); dA_all.append(dA)
        if c_sig:
            dA_sig.append(dA)

n = len(dS_all)
print("-" * 72)
print(f"{'平均 (全18眼)':<22}  "
      f"{sum(dS_all)/n:6.2f} {sum(dC_all)/n:6.2f} "
      f"{sum(dA_all)/n:6.1f} {sum(dSE_all)/n:7.2f}")
print(f"{'最大 (全18眼)':<22}  "
      f"{max(dS_all):6.2f} {max(dC_all):6.2f} "
      f"{max(dA_all):6.1f} {max(dSE_all):7.2f}")
print(f"{'最小 (全18眼)':<22}  "
      f"{min(dS_all):6.2f} {min(dC_all):6.2f} "
      f"{min(dA_all):6.1f} {min(dSE_all):7.2f}")

if dA_sig:
    ns = len(dA_sig)
    print()
    print(f"AX (C>0.5D 両日 ★のみ, n={ns}眼)")
    print(f"  平均: {sum(dA_sig)/ns:.1f}°")
    print(f"  最大: {max(dA_sig):.1f}°")
    print(f"  最小: {min(dA_sig):.1f}°")
