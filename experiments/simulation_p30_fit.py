"""
p30 近視・遠視両側のフィッティング解析スクリプト。

対象: data/processed/simulation_runs/sim_run01/p30/per_image_label.csv
0D を両側共通の anchor として使用。近視・遠視それぞれ Logistic でフィット。

出力: data/processed/simulation_runs/sim_run01/p30/fitting/
       ratio_both_sides.png   近視・遠視の全体像（D軸）
       ratio_myopia.png       近視側フィット詳細
       ratio_hyperopia.png    遠視側フィット詳細
       fit_summary.csv        パラメータ・R² 一覧
"""

import csv
import re
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── paths ─────────────────────────────────────────────────────────────────────

CSV_PATH = Path(__file__).parent.parent / \
    "data/processed/simulation_runs/sim_run01/p30/per_image_label.csv"
OUT_DIR = CSV_PATH.parent / "fitting"
OUT_DIR.mkdir(exist_ok=True)

# ── data loading ──────────────────────────────────────────────────────────────

def parse_D(stem: str) -> float | None:
    m = re.search(r"_D(m?p?)(\d+)_roi", stem)
    if not m:
        return None
    sign_str = m.group(1)
    val = int(m.group(2)) / 100.0
    if sign_str == "m":
        return -val
    elif sign_str == "p":
        return +val
    else:
        return 0.0

rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
data = []
for r in rows:
    if r["status"] != "ok":
        continue
    D = parse_D(r["stem"])
    if D is None:
        continue
    data.append({
        "D":     D,
        "absD":  abs(D),
        "ratio": float(r["ratio"]),
        "major": float(r["major"]),
        "minor": float(r["minor"]),
    })

data.sort(key=lambda x: x["D"])

# 近視側: D <= 0（0D 含む）
myopia    = [d for d in data if d["D"] <= 0]
# 遠視側: D >= 0（0D 含む）
hyperopia = [d for d in data if d["D"] >= 0]

absD_m  = np.array([d["absD"]  for d in myopia])
ratio_m = np.array([d["ratio"] for d in myopia])
absD_h  = np.array([d["absD"]  for d in hyperopia])
ratio_h = np.array([d["ratio"] for d in hyperopia])

D_all    = np.array([d["D"]    for d in data])
ratio_all = np.array([d["ratio"] for d in data])

print(f"近視側: {len(myopia)} 点  遠視側: {len(hyperopia)} 点  (0D は両側共通)")

# ── model ─────────────────────────────────────────────────────────────────────

# 実測 0D 値（両側共通のアンカー）
ratio_0 = float(next(d["ratio"] for d in data if d["D"] == 0.0))
print(f"0D anchor (measured): ratio_0 = {ratio_0:.4f}")

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def logistic_anchored(x, a, k, x0):
    """f(0) = ratio_0 を保証するロジスティック（offset を内部導出）"""
    offset = ratio_0 - a / (1 + np.exp(k * x0))
    return a / (1 + np.exp(-k * (x - x0))) + offset

def fit_logistic(absD, ratio, label):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            popt, _ = curve_fit(
                logistic_anchored, absD, ratio,
                p0=[1.0, 0.8, 3.0],
                bounds=([0.3, 0.01, 0.1], [2.0, 10, 12]),
                maxfev=20000,
            )
            a, k, x0 = popt
            offset = ratio_0 - a / (1 + np.exp(k * x0))
            y_pred = logistic_anchored(absD, *popt)
            r2 = r_squared(ratio, y_pred)
            print(f"  [{label}] a={a:.4f}  k={k:.4f}  x0={x0:.4f}  "
                  f"offset={offset:.4f} (derived)  R2={r2:.5f}")
            return popt, offset, r2
        except Exception as e:
            print(f"  [{label}] fit failed: {e}")
            return None, None, None

print("\n=== Logistic fit (anchored at D=0) ===")
popt_m, offset_m, r2_m = fit_logistic(absD_m, ratio_m, "myopia")
popt_h, offset_h, r2_h = fit_logistic(absD_h, ratio_h, "hyperopia")

# ── plot: both sides on D axis ────────────────────────────────────────────────

x_fine_m = np.linspace(0, absD_m.max() * 1.05, 400)
x_fine_h = np.linspace(0, absD_h.max() * 1.05, 400)

fig, ax = plt.subplots(figsize=(12, 6))

# scatter
ax.scatter(-absD_m, ratio_m, color="#2980b9", s=45, zorder=5,
           edgecolors="white", linewidths=0.4, label="myopia (measured)")
ax.scatter( absD_h, ratio_h, color="#e74c3c", s=45, zorder=5,
           edgecolors="white", linewidths=0.4, label="hyperopia (measured)")

# 0D point (shared)
ax.scatter([0], [data[[d["D"] for d in data].index(0.0)]["ratio"]],
           color="black", s=80, zorder=6, marker="D", label="0D (shared)")

# fit curves
if popt_m is not None:
    y_fit_m = logistic_anchored(x_fine_m, *popt_m)
    ax.plot(-x_fine_m, y_fit_m, color="#2980b9", linewidth=2.5,
            label=f"myopia logistic  R²={r2_m:.4f}")
if popt_h is not None:
    y_fit_h = logistic_anchored(x_fine_h, *popt_h)
    ax.plot( x_fine_h, y_fit_h, color="#e74c3c", linewidth=2.5,
            label=f"hyperopia logistic  R²={r2_h:.4f}")

ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xlabel("D (diopters)  [negative = myopia, positive = hyperopia]", fontsize=11)
ax.set_ylabel("ratio = minor / major", fontsize=11)
ax.set_title("p30: ratio vs D  — myopia & hyperopia logistic fit", fontsize=13)
ax.set_ylim(-0.05, 1.10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "ratio_both_sides.png", dpi=150)
plt.close()
print("\nratio_both_sides.png saved")

# ── plot: myopia detail ───────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(absD_m, ratio_m, color="#2980b9", s=55, zorder=5,
           edgecolors="white", linewidths=0.4, label="myopia (measured)")
if popt_m is not None:
    a, k, x0 = popt_m
    ax.plot(x_fine_m, logistic_anchored(x_fine_m, *popt_m), color="#2980b9", linewidth=2.5,
            label=(f"ratio = {a:.4f}/(1+exp(-{k:.4f}*(x-{x0:.4f})))+{offset_m:.4f}\n"
                   f"  [offset derived: f(0)={ratio_0:.4f}]  R²={r2_m:.5f}"))
ax.set_xlabel("|D| (diopters)", fontsize=11)
ax.set_ylabel("ratio = minor / major", fontsize=11)
ax.set_title("p30 myopia: ratio vs |D|  logistic fit (anchored at D=0)", fontsize=12)
ax.set_ylim(-0.05, 1.10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "ratio_myopia.png", dpi=150)
plt.close()

# ── plot: hyperopia detail ────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(absD_h, ratio_h, color="#e74c3c", s=55, zorder=5,
           edgecolors="white", linewidths=0.4, label="hyperopia (measured)")
if popt_h is not None:
    a, k, x0 = popt_h
    ax.plot(x_fine_h, logistic_anchored(x_fine_h, *popt_h), color="#e74c3c", linewidth=2.5,
            label=(f"ratio = {a:.4f}/(1+exp(-{k:.4f}*(x-{x0:.4f})))+{offset_h:.4f}\n"
                   f"  [offset derived: f(0)={ratio_0:.4f}]  R²={r2_h:.5f}"))
ax.set_xlabel("|D| (diopters)", fontsize=11)
ax.set_ylabel("ratio = minor / major", fontsize=11)
ax.set_title("p30 hyperopia: ratio vs |D|  logistic fit", fontsize=12)
ax.set_ylim(-0.05, 1.10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "ratio_hyperopia.png", dpi=150)
plt.close()
print("ratio_myopia.png / ratio_hyperopia.png saved")

# ── CSV summary ───────────────────────────────────────────────────────────────

summary = []
for side, popt, offset, r2 in [("myopia",    popt_m, offset_m, r2_m),
                                ("hyperopia", popt_h, offset_h, r2_h)]:
    if popt is not None:
        a, k, x0 = popt
        summary.append({
            "side":         side,
            "model":        "logistic_anchored",
            "a":            f"{a:.5f}",
            "k":            f"{k:.5f}",
            "x0":           f"{x0:.5f}",
            "offset":       f"{offset:.5f}",
            "ratio_0":      f"{ratio_0:.5f}",
            "R2":           f"{r2:.6f}",
            "equation":     (f"ratio = {a:.4f}/(1+exp(-{k:.4f}*(|D|-{x0:.4f})))+{offset:.4f}"
                             f"  [f(0)={ratio_0:.4f}]"),
        })

fields = ["side", "model", "a", "k", "x0", "offset", "ratio_0", "R2", "equation"]
with open(OUT_DIR / "fit_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(summary)

print(f"\nAll outputs: {OUT_DIR}")
