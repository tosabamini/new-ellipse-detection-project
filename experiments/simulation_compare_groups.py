"""
複数瞳孔グループの ratio vs D フィッティング曲線を重ねて比較するスクリプト。

Run:
  python experiments/simulation_compare_groups.py --groups p20 p30
  python experiments/simulation_compare_groups.py --groups p20 p30 p25
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

# グループごとの色（近視・遠視で濃淡）
GROUP_COLORS = {
    "p10": "#e74c3c",
    "p15": "#e67e22",
    "p20": "#f1c40f",
    "p25": "#2ecc71",
    "p30": "#3498db",
    "p35": "#9b59b6",
    "p40": "#1abc9c",
    "p45": "#e91e63",
}
DEFAULT_COLOR = "#888888"


def logistic_anchored(x, a, k, x0, v0):
    offset = v0 - a / (1 + np.exp(k * x0))
    return a / (1 + np.exp(-k * (x - x0))) + offset


def load_fit_params(group: str, run_name: str, variable: str) -> dict:
    """fit_summary_full.csv から指定変数の行を読み込む。"""
    csv_path = (PROJECT_ROOT / "data" / "processed" / "simulation_runs"
                / run_name / group / "fitting" / "fit_summary_full.csv")
    result = {}
    for row in csv.DictReader(open(csv_path, encoding="utf-8-sig")):
        if row["variable"] != variable:
            continue
        side = row["side"]
        if row["R2"] == "FAILED":
            continue
        result[side] = {
            "a":  float(row["a"]),
            "k":  float(row["k"]),
            "x0": float(row["x0"]),
            "v0": float(row["v0"]),
            "R2": float(row["R2"]),
        }
    return result


def load_measured(group: str, run_name: str) -> list[dict]:
    """per_image_label.csv から D・ratio を読む。"""
    csv_path = (PROJECT_ROOT / "data" / "processed" / "simulation_runs"
                / run_name / group / "per_image_label.csv")
    data = []
    for row in csv.DictReader(open(csv_path, encoding="utf-8-sig")):
        if row.get("status", "ok") != "ok":
            continue
        m = re.search(r"_D(m?p?)(\d+)(?:_roi)?", row["stem"])
        if not m:
            continue
        s, v = m.group(1), int(m.group(2)) / 100.0
        D = -v if s == "m" else (v if s == "p" else 0.0)
        mj = float(row["major"])
        mn = float(row["minor"])
        data.append({"D": D, "ratio": float(row["ratio"]),
                     "major": mj, "minor": mn, "area": mj * mn})
    return data


VAR_LABELS = {
    "ratio": "ratio = minor / major",
    "major": "major axis (px)",
    "minor": "minor axis (px)",
    "area":  "area = major x minor (px^2)",
}


def run(groups: list[str], run_name: str, show_scatter: bool, variable: str) -> None:
    out_dir = (PROJECT_ROOT / "data" / "processed" / "simulation_runs" / run_name)
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ax_m, ax_h = axes
    x_fine = np.linspace(0, 8.5, 400)

    for group in groups:
        color = GROUP_COLORS.get(group, DEFAULT_COLOR)
        params = load_fit_params(group, run_name, variable)
        measured = load_measured(group, run_name) if show_scatter else []

        for side, ax, sign, scatter_D_fn in [
            ("myopia",    ax_m, -1, lambda d: d["D"] <= 0),
            ("hyperopia", ax_h,  1, lambda d: d["D"] >= 0),
        ]:
            if side not in params:
                continue
            p = params[side]
            y = logistic_anchored(x_fine, p["a"], p["k"], p["x0"], p["v0"])
            ax.plot(sign * x_fine, y, color=color, linewidth=2.2,
                    label=f"{group}  R²={p['R2']:.4f}")

            if show_scatter:
                pts = [d for d in measured if scatter_D_fn(d)]
                if pts:
                    ax.scatter([d["D"] for d in pts],
                               [d[variable] for d in pts],
                               color=color, alpha=0.3, s=20, zorder=3)

    # 共通装飾
    for ax, title in [(ax_m, "Myopia side  (D < 0)"),
                      (ax_h, "Hyperopia side  (D > 0)")]:
        ax.axvline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_xlabel("D (diopters)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc="lower right" if ax is ax_m else "lower left")
        ax.grid(alpha=0.3)
        if variable == "ratio":
            ax.set_ylim(-0.05, 1.10)

    ylabel = VAR_LABELS.get(variable, variable)
    ax_m.set_ylabel(ylabel, fontsize=11)
    group_str = " vs ".join(groups)
    fig.suptitle(f"{variable} vs D: {group_str}  (C⁰ Logistic)", fontsize=13)
    plt.tight_layout()

    out_path = out_dir / f"compare_{variable}_{'_'.join(groups)}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", nargs="+", required=True,
                   help="比較するグループ (例: p20 p30)")
    p.add_argument("--run_name", default="sim_run01")
    p.add_argument("--variable", default="ratio",
                   choices=["ratio", "major", "minor", "area"],
                   help="比較する変数 (default: ratio)")
    p.add_argument("--scatter", action="store_true",
                   help="実測点も重ねて表示する")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.groups, args.run_name, args.scatter, args.variable)
