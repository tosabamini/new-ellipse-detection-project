"""
瞳孔径を無視した統一 ratio–D フィッティングスクリプト。

対象範囲: D ∈ [−8.0, +2.0]  (遠視は +2D まで)
手法: p20/p30/p40 の ratio を D ごとに平均 → C⁰ Logistic (D=0 アンカー) で 1 本フィット。

【注意】
この式は発展途上の暫定モデルです。
  - 使用グループが p20/p30/p40 の 3 種のみ（p10/p15/p25/p35/p40/p45 未実装）
  - 瞳孔径依存性を意図的に無視して近似している
  - D > +2D の遠視側は範囲外のため精度保証なし
  - 今後グループ追加・再フィッティングが必要

Run:
  python experiments/simulation_unified_fit.py
  python experiments/simulation_unified_fit.py --run_name sim_run01 --d_max_hyper 2.0
"""

import argparse
import csv
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).parent.parent
GROUPS = ["p20", "p30", "p40"]


def parse_D(stem: str) -> float | None:
    m = re.search(r"_D(m?p?)(\d+)", stem)
    if not m:
        return None
    s, v = m.group(1), int(m.group(2)) / 100.0
    if s == "m":
        return -v
    if s == "p":
        return v
    return 0.0


def load_group(csv_path: Path) -> dict[float, float]:
    """D → ratio のマップを返す"""
    d2r = {}
    for row in csv.DictReader(open(csv_path, encoding="utf-8-sig")):
        if row.get("status", "ok") != "ok":
            continue
        d = parse_D(row["stem"])
        if d is None:
            continue
        d2r[d] = float(row["ratio"])
    return d2r


def logistic_c0(abs_d, a, k, x0, ratio_0):
    """C⁰ Logistic アンカー付き: offset は ratio_0 から導出"""
    offset = ratio_0 - a / (1 + np.exp(k * x0))
    return a / (1 + np.exp(-k * (abs_d - x0))) + offset


def fit_logistic(abs_d_arr, ratio_arr, ratio_0, label):
    """ratio_0 を固定して curve_fit"""
    def model(x, a, k, x0):
        return logistic_c0(x, a, k, x0, ratio_0)

    p0 = [0.9, 0.6, 3.5]
    bounds = ([0.01, 0.01, 0.5], [2.0, 5.0, 10.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(model, abs_d_arr, ratio_arr, p0=p0, bounds=bounds, maxfev=10000)
    a, k, x0 = popt
    pred = model(abs_d_arr, *popt)
    ss_res = np.sum((ratio_arr - pred) ** 2)
    ss_tot = np.sum((ratio_arr - np.mean(ratio_arr)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  [{label}] a={a:.4f}, k={k:.4f}, x0={x0:.4f}, ratio_0={ratio_0:.4f}  R2={r2:.4f}")
    return a, k, x0, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="sim_run01")
    parser.add_argument("--d_max_hyper", type=float, default=2.0,
                        help="遠視側の上限 D (default: 2.0)")
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "data" / "processed" / "simulation_runs" / args.run_name
    out_dir = run_dir / "unified_fit"
    out_dir.mkdir(exist_ok=True)

    # --- 各グループのデータ読み込み ---
    group_data: dict[str, dict[float, float]] = {}
    for g in GROUPS:
        csv_path = run_dir / g / "per_image_label.csv"
        if not csv_path.exists():
            print(f"[SKIP] {g}: per_image_label.csv not found")
            continue
        group_data[g] = load_group(csv_path)

    if len(group_data) < 2:
        raise RuntimeError("有効なグループが 2 つ未満です")

    # --- D値の共通集合 (対象範囲のみ) ---
    all_d_sets = [set(d for d in gd.keys() if -8.0 <= d <= args.d_max_hyper)
                  for gd in group_data.values()]
    common_ds = sorted(set.intersection(*all_d_sets))
    print(f"共通 D 値: {len(common_ds)} 点  ({min(common_ds):.2f} D ～ {max(common_ds):.2f} D)")
    print(f"使用グループ: {list(group_data.keys())}")

    # --- 平均 ratio 計算 ---
    avg_ratio = {}
    for d in common_ds:
        vals = [group_data[g][d] for g in group_data if d in group_data[g]]
        avg_ratio[d] = np.mean(vals)

    ds_arr = np.array(common_ds)
    ratio_arr = np.array([avg_ratio[d] for d in common_ds])

    # ratio_0 = D=0 の平均 ratio
    ratio_0 = avg_ratio[0.0]
    print(f"ratio_0 (D=0 平均) = {ratio_0:.4f}")

    # --- 近視側フィット (D ≤ 0) ---
    myo_mask = ds_arr <= 0.0
    myo_abs = np.abs(ds_arr[myo_mask])
    myo_ratio = ratio_arr[myo_mask]
    print("\n[近視側フィット]")
    a_m, k_m, x0_m, r2_m = fit_logistic(myo_abs, myo_ratio, ratio_0, "myopia")

    # --- 遠視側フィット (0 ≤ D ≤ d_max_hyper) ---
    hyp_mask = ds_arr >= 0.0
    hyp_abs = ds_arr[hyp_mask]
    hyp_ratio = ratio_arr[hyp_mask]
    print("\n[遠視側フィット]")
    a_h, k_h, x0_h, r2_h = fit_logistic(hyp_abs, hyp_ratio, ratio_0, "hyperopia")

    # --- プロット ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Unified ratio–D fit  (groups: {', '.join(group_data.keys())})\n"
        f"D range: [{min(common_ds):.1f}, {args.d_max_hyper:.1f}] D  |  "
        f"[暫定モデル: 瞳孔径依存性未考慮 / グループ数不足]",
        fontsize=10, color="red"
    )

    # 左: 各グループ生データ + 平均 + フィット
    ax = axes[0]
    colors = {"p20": "#f1c40f", "p30": "#3498db", "p40": "#1abc9c"}
    for g, gd in group_data.items():
        gd_filtered = {d: v for d, v in gd.items() if -8.0 <= d <= args.d_max_hyper}
        gds = sorted(gd_filtered.keys())
        ax.scatter(gds, [gd_filtered[d] for d in gds],
                   color=colors.get(g, "gray"), alpha=0.6, s=25, label=f"{g} raw")

    ax.plot(sorted(avg_ratio.keys()), [avg_ratio[d] for d in sorted(avg_ratio.keys())],
            "ko-", linewidth=2, markersize=4, label="平均")

    d_fine_m = np.linspace(0, 8, 300)
    d_fine_h = np.linspace(0, args.d_max_hyper, 100)
    ax.plot(-d_fine_m, logistic_c0(d_fine_m, a_m, k_m, x0_m, ratio_0),
            "r-", linewidth=2, label=f"fit myopia R²={r2_m:.4f}")
    ax.plot(d_fine_h, logistic_c0(d_fine_h, a_h, k_h, x0_h, ratio_0),
            "b-", linewidth=2, label=f"fit hyperopia R²={r2_h:.4f}")

    ax.set_xlabel("D (Diopter)")
    ax.set_ylabel("ratio (minor/major)")
    ax.set_title("各グループ + 平均 + フィット")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 右: 平均値のみ + フィット (見やすいビュー)
    ax2 = axes[1]
    ax2.plot(sorted(avg_ratio.keys()), [avg_ratio[d] for d in sorted(avg_ratio.keys())],
             "ko", markersize=5, label="平均 ratio")
    ax2.plot(-d_fine_m, logistic_c0(d_fine_m, a_m, k_m, x0_m, ratio_0),
             "r-", linewidth=2.5,
             label=f"近視: a={a_m:.3f}, k={k_m:.3f}, x0={x0_m:.3f}")
    ax2.plot(d_fine_h, logistic_c0(d_fine_h, a_h, k_h, x0_h, ratio_0),
             "b-", linewidth=2.5,
             label=f"遠視(≤+{args.d_max_hyper:.0f}D): a={a_h:.3f}, k={k_h:.3f}, x0={x0_h:.3f}")
    ax2.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(args.d_max_hyper, color="blue", linestyle=":", alpha=0.5, label=f"+{args.d_max_hyper:.0f}D 上限")

    formula_text = (
        f"【暫定モデル — 使用注意】\n"
        f"f(|D|) = a/(1+exp(−k·(|D|−x0))) + offset\n"
        f"offset = ratio_0 − a/(1+exp(k·x0))\n"
        f"ratio_0 = {ratio_0:.4f}  (D=0 アンカー)\n\n"
        f"近視側 (D≤0):\n  a={a_m:.4f}, k={k_m:.4f}, x0={x0_m:.4f}\n  R²={r2_m:.4f}\n\n"
        f"遠視側 (0≤D≤+{args.d_max_hyper:.0f}):\n  a={a_h:.4f}, k={k_h:.4f}, x0={x0_h:.4f}\n  R²={r2_h:.4f}\n\n"
        f"※ 不十分な点:\n"
        f"  - グループ: {list(group_data.keys())} のみ\n"
        f"  - 瞳孔径依存性を無視\n"
        f"  - 遠視 D>+{args.d_max_hyper:.0f} は範囲外"
    )
    ax2.text(0.02, 0.98, formula_text,
             transform=ax2.transAxes, fontsize=7.5,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax2.set_xlabel("D (Diopter)")
    ax2.set_ylabel("ratio (minor/major)")
    ax2.set_title("統一フィット曲線")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "unified_ratio_D_fit.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n保存: {out_path}")

    # --- CSV 保存 ---
    csv_path = out_dir / "unified_fit_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["side", "a", "k", "x0", "ratio_0", "R2",
                    "d_range", "groups", "note"])
        w.writerow(["myopia", f"{a_m:.6f}", f"{k_m:.6f}", f"{x0_m:.6f}",
                    f"{ratio_0:.6f}", f"{r2_m:.6f}",
                    "0 to -8.0", "+".join(group_data.keys()),
                    "PRELIMINARY: pupil-size dependence ignored"])
        w.writerow(["hyperopia", f"{a_h:.6f}", f"{k_h:.6f}", f"{x0_h:.6f}",
                    f"{ratio_0:.6f}", f"{r2_h:.6f}",
                    f"0 to +{args.d_max_hyper:.1f}", "+".join(group_data.keys()),
                    f"PRELIMINARY: pupil-size dependence ignored; valid only up to +{args.d_max_hyper:.1f}D"])
    print(f"保存: {csv_path}")

    # --- 平均データ CSV ---
    avg_csv = out_dir / "averaged_ratio_by_D.csv"
    with open(avg_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["D"] + [f"ratio_{g}" for g in group_data] + ["ratio_mean"])
        for d in sorted(avg_ratio.keys()):
            row = [f"{d:.2f}"]
            for g in group_data:
                row.append(f"{group_data[g].get(d, ''):.4f}" if d in group_data.get(g, {}) else "")
            row.append(f"{avg_ratio[d]:.4f}")
            w.writerow(row)
    print(f"保存: {avg_csv}")


if __name__ == "__main__":
    main()
