"""0603 vs 0604 比較CSV生成"""
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

fields = [
    "subject", "eye",
    "S_0603",  "C_0603",  "A_0603",  "SE_0603",
    "S_0604",  "C_0604",  "A_0604",  "SE_0604",
    "dS", "dC", "dA", "dSE",
]

rows = []
for name, s03, s04 in pairs:
    for eye in ("LEFT", "RIGHT"):
        r03 = data.get((s03, eye))
        r04 = data.get((s04, eye))
        if not r03 or not r04:
            continue

        dS  = r04["S"]  - r03["S"]
        dC  = r04["C"]  - r03["C"]
        dSE = r04["SE"] - r03["SE"]
        raw = abs(r03["A"] - r04["A"])
        dA  = min(raw, 180 - raw)

        rows.append({
            "subject": name, "eye": eye,
            "S_0603":  f"{r03['S']:.2f}",  "C_0603": f"{r03['C']:.2f}",
            "A_0603":  f"{r03['A']:.1f}",  "SE_0603": f"{r03['SE']:.2f}",
            "S_0604":  f"{r04['S']:.2f}",  "C_0604": f"{r04['C']:.2f}",
            "A_0604":  f"{r04['A']:.1f}",  "SE_0604": f"{r04['SE']:.2f}",
            "dS":  f"{dS:+.2f}",
            "dC":  f"{dC:+.2f}",
            "dA":  f"{dA:.1f}",
            "dSE": f"{dSE:+.2f}",
        })

out = PICKUP_DIR / "comparison_0603_vs_0604.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"Saved: {out}  ({len(rows)} rows)")
