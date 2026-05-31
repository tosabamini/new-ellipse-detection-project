"""
p30 C² 版フィッティングスクリプト。

C⁰版 (simulation_p30_fit.py) の式・グラフはそのまま残し、
こちらで C² を保証する Hill 方程式バージョンを作成する。

C² 保証モデル（Hill 方程式 + anchor）:
  f(x) = a * x^n / (K^n + x^n) + ratio_0
    n > 2 → f'(0) = 0 (C¹), f''(0) = 0 (C²)
    f(0) = ratio_0                (C⁰)

出力: data/processed/simulation_runs/sim_run01/p30/fitting/
       ratio_both_sides_C2.png
       ratio_myopia_C2.png
       ratio_hyperopia_C2.png
       fit_summary_C2.csv
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
    data.append({"D": D, "absD": abs(D),
                 "ratio": float(r["ratio"]),
                 "major": float(r["major"]),
                 "minor": float(r["minor"])})

data.sort(key=lambda x: x["D"])

myopia    = [d for d in data if d["D"] <= 0]
hyperopia = [d for d in data if d["D"] >= 0]

absD_m   = np.array([d["absD"]  for d in myopia])
ratio_m  = np.array([d["ratio"] for d in myopia])
absD_h   = np.array([d["absD"]  for d in hyperopia])
ratio_h  = np.array([d["ratio"] for d in hyperopia])

ratio_0 = float(next(d["ratio"] for d in data if d["D"] == 0.0))
print(f"0D anchor: ratio_0 = {ratio_0:.4f}")
print(f"近視側: {len(myopia)} 点  遠視側: {len(hyperopia)} 点\n")

# ── C² モデル定義 ─────────────────────────────────────────────────────────────

def hill_C2(x, a, K, n):
    """
    Hill 方程式 + anchor。
    n > 2 のとき f'(0)=0, f''(0)=0 が保証される (C²)。
    f(0) = ratio_0 (C⁰)。
    """
    return a * x**n / (K**n + x**n) + ratio_0

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def fit_hill_C2(absD, ratio, label):
    # x=0 のデータ点を除外（f(0)=ratio_0 は拘束済みなので残差が 0 になるだけ）
    # → 含めても問題ないがフィット安定のため除外
    mask = absD > 0
    x, y = absD[mask], ratio[mask] - ratio_0   # ratio_0 を引いた残差をフィット
    # ratio_0 を引いた残差に対して: a*x^n/(K^n+x^n) をフィット
    def _hill(x, a, K, n):
        return a * x**n / (K**n + x**n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            popt, _ = curve_fit(
                _hill, x, y,
                p0=[0.9, 3.5, 2.5],
                bounds=([0.1, 0.1, 2.001], [2.0, 15.0, 10.0]),
                maxfev=30000,
            )
            a, K, n = popt
            y_pred_full = hill_C2(absD, a, K, n)
            r2 = r_squared(ratio, y_pred_full)
            print(f"  [{label}] a={a:.4f}  K={K:.4f}  n={n:.4f}  R²={r2:.5f}")
            print(f"    式: ratio = {a:.4f}*|D|^{n:.4f} / ({K:.4f}^{n:.4f} + |D|^{n:.4f}) + {ratio_0:.4f}")
            return popt, r2
        except Exception as e:
            print(f"  [{label}] fit failed: {e}")
            return None, None

print("=== Hill C² fit (n > 2 enforced) ===")
popt_m, r2_m = fit_hill_C2(absD_m, ratio_m, "myopia")
popt_h, r2_h = fit_hill_C2(absD_h, ratio_h, "hyperopia")

# ── 近似式の確認: D=0での連続・微分値 ────────────────────────────────────────

print("\n=== D=0 での値・微分値の確認 ===")
if popt_m is not None and popt_h is not None:
    for label, popt in [("myopia", popt_m), ("hyperopia", popt_h)]:
        a, K, n = popt
        # f(0) = ratio_0 (設計上)
        # f'(0) = a*n*K^n * 0^(n-1) / (K^n)^2 = 0 for n>1
        # f''(0) = 0 for n>2
        # 数値微分で確認
        eps = 1e-6
        f0   = hill_C2(np.array([0.0]),     a, K, n)[0]
        feps = hill_C2(np.array([eps]),     a, K, n)[0]
        f2eps= hill_C2(np.array([2*eps]),   a, K, n)[0]
        df   = (feps - f0) / eps
        d2f  = (f2eps - 2*feps + f0) / eps**2
        print(f"  [{label}]  f(0)={f0:.6f}  f'(0)≈{df:.2e}  f''(0)≈{d2f:.2e}")
    print(f"  → f'(0)≈0, f''(0)≈0 が両側で保証されていれば C² 達成")

# ── plot: both sides ──────────────────────────────────────────────────────────

x_fine_m = np.linspace(0, absD_m.max() * 1.05, 400)
x_fine_h = np.linspace(0, absD_h.max() * 1.05, 400)

# --- 近視・遠視 両側プロット ---
fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(-absD_m, ratio_m, color="#2980b9", s=45, zorder=5,
           edgecolors="white", linewidths=0.4, label="myopia (measured)")
ax.scatter( absD_h, ratio_h, color="#e74c3c", s=45, zorder=5,
           edgecolors="white", linewidths=0.4, label="hyperopia (measured)")
ax.scatter([0], [ratio_0], color="black", s=80, zorder=6,
           marker="D", label=f"0D anchor = {ratio_0:.4f}")

if popt_m is not None:
    y_m = hill_C2(x_fine_m, *popt_m)
    a, K, n = popt_m
    ax.plot(-x_fine_m, y_m, color="#2980b9", linewidth=2.5,
            label=f"myopia Hill-C2  R²={r2_m:.4f}\n"
                  f"  a={a:.3f}, K={K:.3f}, n={n:.3f}")

if popt_h is not None:
    y_h = hill_C2(x_fine_h, *popt_h)
    a, K, n = popt_h
    ax.plot( x_fine_h, y_h, color="#e74c3c", linewidth=2.5,
            label=f"hyperopia Hill-C2  R²={r2_h:.4f}\n"
                  f"  a={a:.3f}, K={K:.3f}, n={n:.3f}")

ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xlabel("D (diopters)  [negative = myopia, positive = hyperopia]", fontsize=11)
ax.set_ylabel("ratio = minor / major", fontsize=11)
ax.set_title("p30: ratio vs D  — Hill C² fit  [f(0)=ratio_0, f'(0)=0, f''(0)=0]",
             fontsize=13)
ax.set_ylim(-0.05, 1.10)
ax.legend(fontsize=8.5)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "ratio_both_sides_C2.png", dpi=150)
plt.close()
print("\nratio_both_sides_C2.png saved")

# --- 近視詳細 ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(absD_m, ratio_m, color="#2980b9", s=55, zorder=5,
           edgecolors="white", linewidths=0.4, label="myopia (measured)")
if popt_m is not None:
    a, K, n = popt_m
    ax.plot(x_fine_m, hill_C2(x_fine_m, a, K, n), color="#2980b9", linewidth=2.5,
            label=f"ratio = {a:.4f}*|D|^{n:.4f} / ({K:.4f}^{n:.4f}+|D|^{n:.4f}) + {ratio_0:.4f}\n"
                  f"R²={r2_m:.5f}   [C²: n={n:.3f}>2]")
ax.set_xlabel("|D| (diopters)", fontsize=11)
ax.set_ylabel("ratio = minor / major", fontsize=11)
ax.set_title("p30 myopia: Hill C² fit", fontsize=12)
ax.set_ylim(-0.05, 1.10); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "ratio_myopia_C2.png", dpi=150)
plt.close()

# --- 遠視詳細 ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(absD_h, ratio_h, color="#e74c3c", s=55, zorder=5,
           edgecolors="white", linewidths=0.4, label="hyperopia (measured)")
if popt_h is not None:
    a, K, n = popt_h
    ax.plot(x_fine_h, hill_C2(x_fine_h, a, K, n), color="#e74c3c", linewidth=2.5,
            label=f"ratio = {a:.4f}*|D|^{n:.4f} / ({K:.4f}^{n:.4f}+|D|^{n:.4f}) + {ratio_0:.4f}\n"
                  f"R²={r2_h:.5f}   [C²: n={n:.3f}>2]")
ax.set_xlabel("|D| (diopters)", fontsize=11)
ax.set_ylabel("ratio = minor / major", fontsize=11)
ax.set_title("p30 hyperopia: Hill C² fit", fontsize=12)
ax.set_ylim(-0.05, 1.10); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "ratio_hyperopia_C2.png", dpi=150)
plt.close()
print("ratio_myopia_C2.png / ratio_hyperopia_C2.png saved")

# ── CSV summary ───────────────────────────────────────────────────────────────

summary = []
for side, popt, r2 in [("myopia", popt_m, r2_m), ("hyperopia", popt_h, r2_h)]:
    if popt is not None:
        a, K, n = popt
        summary.append({
            "side":        side,
            "model":       "Hill_C2",
            "a":           f"{a:.5f}",
            "K":           f"{K:.5f}",
            "n":           f"{n:.5f}",
            "ratio_0":     f"{ratio_0:.5f}",
            "continuity":  "C2",
            "R2":          f"{r2:.6f}",
            "equation":    (f"ratio = {a:.4f}*|D|^{n:.4f}"
                            f" / ({K:.4f}^{n:.4f}+|D|^{n:.4f}) + {ratio_0:.4f}"),
        })

fields = ["side", "model", "a", "K", "n", "ratio_0", "continuity", "R2", "equation"]
with open(OUT_DIR / "fit_summary_C2.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(summary)

print(f"\nAll C2 outputs: {OUT_DIR}")
