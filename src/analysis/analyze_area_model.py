"""
楕円面積 vs 屈折力 の 2次多項式フィット (3瞳孔径)

analyze_model_eye_ellipse.py の ratio を ellipse_area に置き換えたもの。
各瞳孔径ごとに:
  area = a*D^2 + b*D + c
を最小二乗フィットし、逆算式・LOO交差検証・グラフを出力する。

出力先: data/processed/model_eye_runs/combined_analysis/area_model/
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_EYE_RUNS_DIR = PROJECT_ROOT / "data/processed/model_eye_runs"
OUT_DIR = MODEL_EYE_RUNS_DIR / "combined_analysis" / "area_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PUPILS = [
    {"label": "3.0 mm", "run": "model_eye_3mm_v001", "color": "#1f77b4"},
    {"label": "5.0 mm", "run": "model_eye_5mm_v001", "color": "#ff7f0e"},
    {"label": "7.0 mm", "run": "model_eye_v001",     "color": "#2ca02c"},
]


def folder_to_diopter(name: str) -> float:
    parts = name.split("_")
    sign  = parts[1]
    major = int(parts[2])
    minor = int(parts[3][:-1])
    val   = major + minor / 100.0
    return -val if sign == "M" else (0.0 if sign == "Z" else val)


def load_run(run_name: str) -> list[dict]:
    """1つの run から屈折力ごとの ellipse_area 平均・分散を返す"""
    run_root = MODEL_EYE_RUNS_DIR / run_name
    records = []
    for folder_dir in sorted(run_root.iterdir()):
        csv_path = folder_dir / "ellipse_results.csv"
        if not csv_path.exists():
            continue
        areas = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["status"] != "ok":
                    continue
                if row.get("ellipse_area", "") == "":
                    continue
                areas.append(float(row["ellipse_area"]))
        if not areas:
            continue
        records.append({
            "folder":    folder_dir.name,
            "diopter":   folder_to_diopter(folder_dir.name),
            "n":         len(areas),
            "area_mean": float(np.mean(areas)),
            "area_std":  float(np.std(areas, ddof=1)) if len(areas) > 1 else 0.0,
            "areas":     areas,
        })
    records.sort(key=lambda r: r["diopter"])
    return records


def r2_rmse(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return float(r2), float(rmse)


def loo_cv(D_vals, A_means):
    """Leave-one-out で D推定の MAE を返す（2次式の逆算）"""
    errors = []
    rows = []
    for i in range(len(D_vals)):
        D_tr = np.delete(D_vals, i)
        A_tr = np.delete(A_means, i)
        a, b, c = np.polyfit(D_tr, A_tr, deg=2)

        area_test = A_means[i]
        # a*D^2 + b*D + (c - area_test) = 0
        disc = b**2 - 4 * a * (c - area_test)
        if disc < 0:
            rows.append((D_vals[i], area_test, None, None, None))
            continue
        d1 = (-b + np.sqrt(disc)) / (2 * a)
        d2 = (-b - np.sqrt(disc)) / (2 * a)
        err1 = abs(d1 - D_vals[i])
        err2 = abs(d2 - D_vals[i])
        best_err = min(err1, err2)
        errors.append(best_err)
        rows.append((D_vals[i], area_test, d1, d2, best_err))
    return errors, rows


def main():
    # ── サマリテーブル ───────────────────────────────────────
    print("=" * 70)
    print(f"{'Pupil':>8}  {'Diopter':>9}  {'N':>3}  "
          f"{'Area mean':>10}  {'Area std':>9}")
    print("-" * 70)

    all_data = {}
    for pupil in PUPILS:
        records = load_run(pupil["run"])
        all_data[pupil["label"]] = records
        for r in records:
            print(f"{pupil['label']:>8}  {r['diopter']:>+9.2f}  {r['n']:>3}  "
                  f"{r['area_mean']:>10.1f}  {r['area_std']:>9.2f}")
        print()
    print("=" * 70)

    # ── 各瞳孔径のフィット & LOO ─────────────────────────────
    D_fine = np.linspace(-5.5, 4.5, 300)
    fit_info = {}

    for pupil in PUPILS:
        label   = pupil["label"]
        records = all_data[label]
        D_vals  = np.array([r["diopter"]  for r in records])
        A_means = np.array([r["area_mean"] for r in records])
        A_stds  = np.array([r["area_std"]  for r in records])

        a, b, c = np.polyfit(D_vals, A_means, deg=2)
        poly     = np.poly1d([a, b, c])
        r2, rmse = r2_rmse(A_means, poly(D_vals))

        errors, loo_rows = loo_cv(D_vals, A_means)
        mae     = float(np.mean(errors))  if errors else float("nan")
        max_err = float(np.max(errors))   if errors else float("nan")

        fit_info[label] = {
            "a": a, "b": b, "c": c,
            "poly": poly,
            "D_vals": D_vals, "A_means": A_means, "A_stds": A_stds,
            "r2": r2, "rmse": rmse,
            "loo_mae": mae, "loo_max": max_err,
            "loo_rows": loo_rows,
        }

        print(f"\n-- {label} fit --")
        print(f"  Area = {a:.4f}*D^2 + ({b:.4f})*D + ({c:.4f})")
        print(f"  R2={r2:.4f},  RMSE={rmse:.2f} px^2")
        print(f"  LOO MAE={mae:.3f} D,  max={max_err:.3f} D")

        print(f"  {'Diopter':>9}  {'Area':>9}  {'D_sol1':>8}  {'D_sol2':>8}  {'best_err':>9}")
        for row in loo_rows:
            d_true, area, d1, d2, err = row
            if d1 is None:
                print(f"  {d_true:>+9.2f}  {area:>9.1f}  {'(no real)':>18}")
            else:
                print(f"  {d_true:>+9.2f}  {area:>9.1f}  {d1:>+8.2f}  {d2:>+8.2f}  {err:>9.3f}")

    # ── グラフ1: 3瞳孔径の Area vs D（データ + フィット）───
    fig, ax = plt.subplots(figsize=(10, 6))
    for pupil in PUPILS:
        label = pupil["label"]
        col   = pupil["color"]
        fi    = fit_info[label]
        ax.errorbar(fi["D_vals"], fi["A_means"], yerr=fi["A_stds"],
                    fmt="o", color=col, capsize=4, markersize=5, zorder=5)
        ax.plot(D_fine, fi["poly"](D_fine), "-", color=col, linewidth=2,
                label=(f"{label}:  {fi['a']:.4f}D^2 + ({fi['b']:.4f})D + {fi['c']:.1f}"
                       f"   R2={fi['r2']:.4f}"))

    ax.set_xlabel("Refraction power (D)", fontsize=12)
    ax.set_ylabel("Ellipse area (px^2)", fontsize=12)
    ax.set_title("Ellipse area vs Refraction power — all pupil diameters", fontsize=13)
    ax.set_xticks(np.arange(-5, 5))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "area_vs_diopter.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot: {OUT_DIR / 'area_vs_diopter.png'}")

    # ── グラフ2: LOO 推定精度（棒グラフ）───────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    labels_list = [p["label"] for p in PUPILS]
    maes = [fit_info[l]["loo_mae"] for l in labels_list]
    maxs = [fit_info[l]["loo_max"] for l in labels_list]
    x = np.arange(len(labels_list))
    w = 0.35
    bars1 = ax2.bar(x - w/2, maes, width=w, label="LOO MAE (D)")
    bars2 = ax2.bar(x + w/2, maxs, width=w, label="LOO max error (D)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_list)
    ax2.set_ylabel("Error (D)")
    ax2.set_title("LOO cross-validation: D estimation from area")
    ax2.legend()
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    for bar in list(bars1) + list(bars2):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}",
                 ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / "loo_accuracy.png", dpi=150)
    plt.close(fig2)
    print(f"LOO plot: {OUT_DIR / 'loo_accuracy.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
