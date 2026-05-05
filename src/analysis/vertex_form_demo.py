"""
頂点形式パラメータ化 + kクリップの具体的なデモ
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "data/processed/model_eye_runs/model_eye_v001"
OUT_DIR = RUN_ROOT / "analysis"
OUT_DIR.mkdir(exist_ok=True)

# ── データ読み込み（前のスクリプトと同じ） ──────────────────
def folder_to_diopter(name):
    parts = name.split("_")
    sign_char = parts[1]
    major_d   = int(parts[2])
    minor_d   = int(parts[3][:-1])
    value = major_d + minor_d / 100.0
    if sign_char == "M":
        value = -value
    elif sign_char == "Z":
        value = 0.0
    return value

def load_data():
    folders = sorted(
        d for d in RUN_ROOT.iterdir()
        if d.is_dir() and (d / "ellipse_results.csv").exists()
    )
    D_vals, R_means = [], []
    for folder_dir in folders:
        csv_path = folder_dir / "ellipse_results.csv"
        ratios = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["status"] != "ok":
                    continue
                ratios.append(float(row["minor_axis"]) / float(row["major_axis"]))
        if ratios:
            D_vals.append(folder_to_diopter(folder_dir.name))
            R_means.append(np.mean(ratios))
    idx = np.argsort(D_vals)
    return np.array(D_vals)[idx], np.array(R_means)[idx]

# ── 頂点形式の関数 ────────────────────────────────────────
def vertex_parabola(D, a, h, k):
    return a * (D - h) ** 2 + k

# ── 頂点形式で逆算（kクリップあり） ──────────────────────
def inverse_vertex(ratio, a, h, k):
    ratio_clipped = max(ratio, k)
    clipped = ratio_clipped > ratio  # クリップが発動したか
    disc = (ratio_clipped - k) / a
    delta = np.sqrt(disc)
    d1 = h + delta
    d2 = h - delta
    return d1, d2, clipped

def main():
    D_vals, R_means = load_data()

    # ── 1. 頂点形式で直接フィット ─────────────────────────
    popt, pcov = curve_fit(
        vertex_parabola, D_vals, R_means,
        p0=[0.025, 0.5, 0.12],
        bounds=([0, -2, 0], [1, 3, 0.5])   # a>0, h合理範囲, k>0
    )
    a, h, k = popt
    perr = np.sqrt(np.diag(pcov))

    R_pred = vertex_parabola(D_vals, a, h, k)
    ss_res = np.sum((R_means - R_pred) ** 2)
    ss_tot = np.sum((R_means - np.mean(R_means)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((R_means - R_pred) ** 2))

    print("=" * 60)
    print("  頂点形式フィット結果")
    print("=" * 60)
    print(f"  Ratio = a*(D - h)^2 + k")
    print(f"  a = {a:.6f}  (+/- {perr[0]:.6f})")
    print(f"  h = {h:.6f}  (+/- {perr[1]:.6f})  [頂点のD値]")
    print(f"  k = {k:.6f}  (+/- {perr[2]:.6f})  [Ratioの最小値]")
    print(f"  R^2  = {r2:.4f}")
    print(f"  RMSE = {rmse:.5f}")
    print()

    # ── 2. 逆算式の表示 ──────────────────────────────────
    print("  逆算式:")
    print(f"    D = {h:.4f} +/- sqrt( (Ratio - {k:.4f}) / {a:.4f} )")
    print(f"    ※ Ratio < {k:.4f} のときは {k:.4f} にクリップ → D = {h:.4f}")
    print()

    # ── 3. 全データへの逆算結果 ──────────────────────────
    print("-" * 60)
    print(f"{'Diopter':>10}  {'Ratio':>7}  {'D_sol1':>8}  {'D_sol2':>8}  {'Clipped':>8}  {'Err_best':>9}")
    print("-" * 60)
    loo_errors = []
    for D_true, R in zip(D_vals, R_means):
        d1, d2, clipped = inverse_vertex(R, a, h, k)
        err = min(abs(d1 - D_true), abs(d2 - D_true))
        loo_errors.append(err)
        clip_str = "YES" if clipped else "-"
        print(f"{D_true:>+10.2f}  {R:>7.4f}  {d1:>+8.2f}  {d2:>+8.2f}  {clip_str:>8}  {err:>9.3f} D")
    print("-" * 60)
    print(f"  MAE: {np.mean(loo_errors):.3f} D   max: {max(loo_errors):.3f} D")
    print()

    # ── 4. グラフ：データ・両フィット・頂点 ──────────────
    D_fine = np.linspace(-5.5, 4.5, 400)

    # 標準形フィット（前回の結果）
    std_coeffs = np.polyfit(D_vals, R_means, deg=2)
    std_poly = np.poly1d(std_coeffs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左: 曲線の比較
    ax = axes[0]
    ax.scatter(D_vals, R_means, color="#1f77b4", s=60, zorder=5, label="Data (mean)")
    ax.plot(D_fine, vertex_parabola(D_fine, a, h, k), "-", color="#d62728",
            linewidth=2, label=f"Vertex fit  a={a:.4f}, h={h:.3f}, k={k:.4f}\nR²={r2:.4f}")
    ax.plot(D_fine, std_poly(D_fine), "--", color="#ff7f0e",
            linewidth=1.5, label=f"Standard fit (prev)\nR²=0.9793")
    ax.axhline(k, color="#2ca02c", linestyle=":", linewidth=1.2,
               label=f"k = {k:.4f}  (clip threshold)")
    ax.axvline(h, color="#9467bd", linestyle=":", linewidth=1.2,
               label=f"h = {h:.3f} D  (vertex)")
    ax.set_xlabel("Refraction power (D)", fontsize=12)
    ax.set_ylabel("Minor / Major axis ratio", fontsize=12)
    ax.set_title("Vertex form fit vs Standard fit", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    # 右: 逆算精度（kクリップあり）
    ax2 = axes[1]
    sol1 = [inverse_vertex(R, a, h, k)[0] for R in R_means]
    sol2 = [inverse_vertex(R, a, h, k)[1] for R in R_means]
    ax2.plot([-6, 5], [-6, 5], "k--", linewidth=1, alpha=0.5, label="Perfect")
    ax2.scatter(D_vals, sol1, color="#d62728", s=60, marker="^", zorder=5, label="Solution 1 (D = h + delta)")
    ax2.scatter(D_vals, sol2, color="#1f77b4", s=60, marker="v", zorder=5, label="Solution 2 (D = h - delta)")
    ax2.set_xlabel("True refraction (D)", fontsize=12)
    ax2.set_ylabel("Predicted refraction (D)", fontsize=12)
    ax2.set_title("Inverse prediction (k-clipped)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = OUT_DIR / "vertex_form_demo.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  グラフ: {out_path}")

if __name__ == "__main__":
    main()
