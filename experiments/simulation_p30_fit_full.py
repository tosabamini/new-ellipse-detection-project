"""
p30 全変数フィッティング: ratio / major / minor / area (major×minor)
近視・遠視両側、C⁰ Logistic（D=0 アンカー）。

出力: data/processed/simulation_runs/sim_run01/p30/fitting/
  {var}_both_sides.png  x 4 変数
  fit_summary_full.csv
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

def parse_D(stem):
    m = re.search(r"_D(m?p?)(\d+)_roi", stem)
    if not m:
        return None
    s, v = m.group(1), int(m.group(2)) / 100.0
    return -v if s == "m" else (v if s == "p" else 0.0)

rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
data = []
for r in rows:
    if r["status"] != "ok":
        continue
    D = parse_D(r["stem"])
    if D is None:
        continue
    mj = float(r["major"])
    mn = float(r["minor"])
    data.append({"D": D, "absD": abs(D),
                 "ratio": float(r["ratio"]),
                 "major": mj,
                 "minor": mn,
                 "area":  mj * mn})

data.sort(key=lambda x: x["D"])

myopia    = [d for d in data if d["D"] <= 0]
hyperopia = [d for d in data if d["D"] >= 0]

d0 = next(d for d in data if d["D"] == 0.0)
anchors = {k: d0[k] for k in ("ratio", "major", "minor", "area")}

print(f"0D anchors: ratio={anchors['ratio']:.4f}  major={anchors['major']:.2f}"
      f"  minor={anchors['minor']:.2f}  area={anchors['area']:.1f}")
print(f"近視側: {len(myopia)}点  遠視側: {len(hyperopia)}点\n")

# ── モデル ────────────────────────────────────────────────────────────────────

def logistic_anchored(x, a, k, x0, v0):
    """f(0)=v0 を保証。a<0 で減少関数にもなる。"""
    offset = v0 - a / (1 + np.exp(k * x0))
    return a / (1 + np.exp(-k * (x - x0))) + offset

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


VAR_CONFIGS = {
    "ratio": {
        "label": "ratio = minor / major",
        "color_m": "#2980b9", "color_h": "#e74c3c",
        "p0_m": [1.0, 0.8, 3.4], "p0_h": [0.7, 0.5, 3.5],
        "bounds_m": ([-2, 0.01, 0.1], [2, 10, 12]),
        "bounds_h": ([-2, 0.01, 0.1], [2, 10, 12]),
    },
    "major": {
        "label": "major axis (px)",
        "color_m": "#27ae60", "color_h": "#e67e22",
        "p0_m": [-150, 0.5, 3.0], "p0_h": [-200, 0.4, 3.0],
        "bounds_m": ([-600, 0.01, 0.1], [600, 10, 12]),
        "bounds_h": ([-600, 0.01, 0.1], [600, 10, 12]),
    },
    "minor": {
        "label": "minor axis (px)",
        "color_m": "#8e44ad", "color_h": "#c0392b",
        "p0_m": [300, 0.7, 3.5], "p0_h": [200, 0.5, 4.0],
        "bounds_m": ([-500, 0.01, 0.1], [500, 10, 12]),
        "bounds_h": ([-500, 0.01, 0.1], [500, 10, 12]),
    },
    "area": {
        "label": "area = major × minor (px²)",
        "color_m": "#16a085", "color_h": "#d35400",
        "p0_m": [80000, 0.6, 3.5], "p0_h": [40000, 0.4, 4.0],
        "bounds_m": ([-200000, 0.01, 0.1], [200000, 10, 12]),
        "bounds_h": ([-200000, 0.01, 0.1], [200000, 10, 12]),
    },
}


def fit_side(absD, values, v0, p0, bounds, label):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            popt, _ = curve_fit(
                lambda x, a, k, x0: logistic_anchored(x, a, k, x0, v0),
                absD, values,
                p0=p0, bounds=bounds, maxfev=30000,
            )
            a, k, x0 = popt
            y_pred = logistic_anchored(absD, a, k, x0, v0)
            r2 = r_squared(values, y_pred)
            offset = v0 - a / (1 + np.exp(k * x0))
            print(f"    [{label}] a={a:.3f}  k={k:.4f}  x0={x0:.4f}"
                  f"  offset={offset:.3f}  R²={r2:.4f}")
            return (a, k, x0), r2
        except Exception as e:
            print(f"    [{label}] fit failed: {e}")
            return None, None


summary_rows = []

for var, cfg in VAR_CONFIGS.items():
    v0 = anchors[var]
    absD_m = np.array([d["absD"] for d in myopia])
    vals_m = np.array([d[var]    for d in myopia])
    absD_h = np.array([d["absD"] for d in hyperopia])
    vals_h = np.array([d[var]    for d in hyperopia])

    print(f"\n=== {var} (0D anchor={v0:.3f}) ===")
    popt_m, r2_m = fit_side(absD_m, vals_m, v0,
                             cfg["p0_m"], cfg["bounds_m"], "myopia")
    popt_h, r2_h = fit_side(absD_h, vals_h, v0,
                             cfg["p0_h"], cfg["bounds_h"], "hyperopia")

    # ── プロット ──────────────────────────────────────────────────────────────
    x_fine = np.linspace(0, max(absD_m.max(), absD_h.max()) * 1.05, 400)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(-absD_m, vals_m, color=cfg["color_m"], s=40, zorder=5,
               edgecolors="white", linewidths=0.3, label="myopia (measured)")
    ax.scatter( absD_h, vals_h, color=cfg["color_h"], s=40, zorder=5,
               edgecolors="white", linewidths=0.3, label="hyperopia (measured)")
    ax.scatter([0], [v0], color="black", s=80, zorder=6, marker="D",
               label=f"0D = {v0:.3f}")

    if popt_m is not None:
        a, k, x0 = popt_m
        y_fit = logistic_anchored(x_fine, a, k, x0, v0)
        ax.plot(-x_fine, y_fit, color=cfg["color_m"], linewidth=2.2,
                label=f"myopia logistic  R²={r2_m:.4f}")

    if popt_h is not None:
        a, k, x0 = popt_h
        y_fit = logistic_anchored(x_fine, a, k, x0, v0)
        ax.plot( x_fine, y_fit, color=cfg["color_h"], linewidth=2.2,
                label=f"hyperopia logistic  R²={r2_h:.4f}")

    ax.axvline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel("D (diopters)  [negative=myopia, positive=hyperopia]", fontsize=10)
    ax.set_ylabel(cfg["label"], fontsize=10)
    ax.set_title(f"p30: {var} vs D  — C⁰ logistic fit (anchored at D=0)", fontsize=12)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / f"{var}_both_sides.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  -> {out_path.name}")

    # サマリ行
    for side, popt, r2 in [("myopia", popt_m, r2_m),
                            ("hyperopia", popt_h, r2_h)]:
        if popt is not None:
            a, k, x0 = popt
            offset = v0 - a / (1 + np.exp(k * x0))
            summary_rows.append({
                "variable": var,
                "side":     side,
                "a":        f"{a:.4f}",
                "k":        f"{k:.5f}",
                "x0":       f"{x0:.5f}",
                "offset":   f"{offset:.4f}",
                "v0":       f"{v0:.4f}",
                "R2":       f"{r2:.5f}",
                "equation": (f"{var} = {a:.3f}/(1+exp(-{k:.4f}*(|D|-{x0:.4f})))"
                             f"+{offset:.3f}  [f(0)={v0:.4f}]"),
            })
        else:
            summary_rows.append({
                "variable": var, "side": side,
                "a": "", "k": "", "x0": "", "offset": "",
                "v0": f"{v0:.4f}", "R2": "FAILED", "equation": "fit failed",
            })

# ── CSV ───────────────────────────────────────────────────────────────────────
fields = ["variable", "side", "a", "k", "x0", "offset", "v0", "R2", "equation"]
with open(OUT_DIR / "fit_summary_full.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(summary_rows)

print(f"\n全出力: {OUT_DIR}")
