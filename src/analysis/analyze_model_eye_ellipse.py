"""
模型眼楕円解析スクリプト

- 屈折力ごとの major/minor/ratio の mean・variance 集計
- Ratio vs 屈折力のグラフ
- Ratio → 屈折力の逆算式（多項式フィット）
- Leave-one-out 交差検証による精度評価
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_EYE_RUNS_DIR = PROJECT_ROOT / "data/processed/model_eye_runs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True, help="例: model_eye_3mm_v001")
    return parser.parse_args()

# ── フォルダ名 → 屈折力 (D) ──────────────────────────────
def folder_to_diopter(name: str) -> float:
    # 例: 1200_M_04_00D → -4.00,  2000_P_04_00D → +4.00
    parts = name.split("_")
    sign_char = parts[1]          # M / Z / P
    major_d   = int(parts[2])     # 04
    minor_d   = int(parts[3][:-1])  # 00  (末尾の D を除去)
    value = major_d + minor_d / 100.0
    if sign_char == "M":
        value = -value
    elif sign_char == "Z":
        value = 0.0
    return value


# ── 各フォルダの ellipse_results.csv を読み込んで集計 ─────
def load_folder(folder_dir: Path) -> dict | None:
    csv_path = folder_dir / "ellipse_results.csv"
    if not csv_path.exists():
        return None

    majors, minors, ratios = [], [], []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok":
                continue
            major = float(row["major_axis"])
            minor = float(row["minor_axis"])
            majors.append(major)
            minors.append(minor)
            ratios.append(minor / major)

    if not majors:
        return None

    return {
        "folder":    folder_dir.name,
        "diopter":   folder_to_diopter(folder_dir.name),
        "n":         len(majors),
        "major_mean": np.mean(majors),
        "major_var":  np.var(majors, ddof=1),
        "minor_mean": np.mean(minors),
        "minor_var":  np.var(minors, ddof=1),
        "ratio_mean": np.mean(ratios),
        "ratio_var":  np.var(ratios, ddof=1),
        "ratios":    ratios,
    }


# ── メイン ────────────────────────────────────────────────
def main():
    args = parse_args()
    run_root = MODEL_EYE_RUNS_DIR / args.run_name
    out_dir = run_root / "analysis"
    out_dir.mkdir(exist_ok=True)

    folders = sorted(
        d for d in run_root.iterdir()
        if d.is_dir() and (d / "ellipse_results.csv").exists()
    )
    records = [r for d in folders if (r := load_folder(d)) is not None]
    records.sort(key=lambda x: x["diopter"])

    # ── 1. サマリテーブル ─────────────────────────────────
    print("=" * 80)
    print(f"{'Refraction':>12}  {'N':>4}  "
          f"{'Major mean':>11} {'Major var':>10}  "
          f"{'Minor mean':>11} {'Minor var':>10}  "
          f"{'Ratio mean':>11} {'Ratio var':>10}")
    print("-" * 80)
    for r in records:
        print(f"{r['diopter']:>+12.2f}  {r['n']:>4}  "
              f"{r['major_mean']:>11.2f} {r['major_var']:>10.4f}  "
              f"{r['minor_mean']:>11.2f} {r['minor_var']:>10.4f}  "
              f"{r['ratio_mean']:>11.4f} {r['ratio_var']:>10.6f}")
    print("=" * 80)

    # CSV にも保存
    summary_csv = out_dir / "axis_ratio_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["diopter","n","major_mean","major_var","minor_mean","minor_var","ratio_mean","ratio_var"])
        for r in records:
            w.writerow([r["diopter"], r["n"],
                        f"{r['major_mean']:.4f}", f"{r['major_var']:.6f}",
                        f"{r['minor_mean']:.4f}", f"{r['minor_var']:.6f}",
                        f"{r['ratio_mean']:.6f}", f"{r['ratio_var']:.8f}"])
    print(f"\nSummary CSV: {summary_csv}")

    # ── 2. Ratio グラフ (-5D ～ +4D) ─────────────────────
    D_vals   = np.array([r["diopter"]   for r in records])
    R_means  = np.array([r["ratio_mean"] for r in records])
    R_stds   = np.array([np.sqrt(r["ratio_var"]) for r in records])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(D_vals, R_means, yerr=R_stds,
                fmt="o-", color="#1f77b4", capsize=5,
                label="Ratio mean ± SD")
    ax.set_xlabel("Refraction power (D)", fontsize=12)
    ax.set_ylabel("Minor / Major axis ratio", fontsize=12)
    ax.set_title("Red-reflex ellipse axis ratio vs Refraction power", fontsize=13)
    ax.set_xticks(D_vals)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plot_path = out_dir / "ratio_vs_diopter.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot: {plot_path}")

    # ── 3. 式導出: ratio = a·D² + b·D + c (最小二乗) ────
    coeffs = np.polyfit(D_vals, R_means, deg=2)
    a, b, c = coeffs
    poly = np.poly1d(coeffs)
    R_pred_fit = poly(D_vals)
    ss_res = np.sum((R_means - R_pred_fit) ** 2)
    ss_tot = np.sum((R_means - np.mean(R_means)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((R_means - R_pred_fit) ** 2))

    print("\n-- Fit (quadratic) -----------------------------------------")
    print(f"  Ratio = {a:.6f}*D^2 + ({b:.6f})*D + ({c:.6f})")
    print(f"  R^2 = {r2:.4f},  RMSE = {rmse:.5f}")

    print("\n-- Inverse formula -----------------------------------------")
    print(f"  a*D^2 + b*D + (c - Ratio) = 0  =>  solve quadratic for D")
    print(f"  D = [ -({b:.6f}) +/- sqrt(({b:.6f})^2 - 4*({a:.6f})*(c - Ratio)) ]")
    print(f"      / (2*{a:.6f})")
    print("  Note: two solutions exist; select based on clinical context.")

    # ── 4. Leave-one-out 交差検証 ────────────────────────
    print(f"\n-- Leave-one-out cross-validation --------------------------")
    print(f"{'Diopter':>10}  {'True':>6}  {'Pred1':>8}  {'Pred2':>8}  {'Err_best':>9}")
    loo_errors = []
    for i in range(len(D_vals)):
        D_tr = np.delete(D_vals, i)
        R_tr = np.delete(R_means, i)
        co = np.polyfit(D_tr, R_tr, deg=2)
        a_l, b_l, c_l = co

        ratio_test = R_means[i]
        disc = b_l**2 - 4*a_l*(c_l - ratio_test)
        if disc < 0:
            print(f"{D_vals[i]:>+10.2f}  {ratio_test:>6.4f}  {'(no real solution)':>18}")
            continue
        d1 = (-b_l + np.sqrt(disc)) / (2*a_l)
        d2 = (-b_l - np.sqrt(disc)) / (2*a_l)
        err1 = abs(d1 - D_vals[i])
        err2 = abs(d2 - D_vals[i])
        best_err = min(err1, err2)
        loo_errors.append(best_err)
        print(f"{D_vals[i]:>+10.2f}  {ratio_test:>6.4f}  {d1:>+8.2f}  {d2:>+8.2f}  {best_err:>9.3f} D")

    if loo_errors:
        mae = np.mean(loo_errors)
        print(f"\n  LOO MAE (best solution): {mae:.3f} D")
        print(f"  LOO max error          : {max(loo_errors):.3f} D")

    # ── 5. フィット曲線付きグラフを追加保存 ──────────────
    D_fine = np.linspace(D_vals.min() - 0.3, D_vals.max() + 0.3, 300)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.errorbar(D_vals, R_means, yerr=R_stds,
                 fmt="o", color="#1f77b4", capsize=5, label="Data (mean ± SD)", zorder=5)
    ax2.plot(D_fine, poly(D_fine), "-", color="#d62728", linewidth=1.8,
             label=f"Fit: {a:.4f}D² + ({b:.4f})D + {c:.4f}\nR²={r2:.4f}")
    ax2.set_xlabel("Refraction power (D)", fontsize=12)
    ax2.set_ylabel("Minor / Major axis ratio", fontsize=12)
    ax2.set_title("Ellipse ratio vs Refraction power (with quadratic fit)", fontsize=13)
    ax2.set_xticks(D_vals)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    plt.tight_layout()
    plot2_path = out_dir / "ratio_vs_diopter_fit.png"
    fig2.savefig(plot2_path, dpi=150)
    plt.close(fig2)
    print(f"\nFit plot: {plot2_path}")


if __name__ == "__main__":
    main()
