"""
Ratio と Area の 2 式から (D, p) を 1 点に絞る連立ソルバー。

各画像で観測した (ratio_obs, area_obs) に対し、
    ratio_real(D, p) = ratio_obs       ← 屈折のおおよその位置 (スケール不変)
    area_real(D, p)  = area_obs        ← 瞳孔径と屈折を実 px で拘束
を連続最適化 (scipy.minimize) で同時に最も満たす (D, p) を求める。

特別処理:
    ratio < RATIO_THRESH (=0.13) の場合は正視付近で D推定不能。
    D=0.00 固定で p のみを area から推定し、status="unmeasurable" とする。

Run:
  python experiments/refraction_from_ratio_area.py \
      --folder data/Repeatability/PickUP/01_Kavya/LEFT
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, brentq

from src.analysis.ratio_model import ratio_real
from src.analysis.area_model import area_real

# ── 定数 ──────────────────────────────────────────────────────────────────
RATIO_THRESH = 0.13   # これ以下は正視付近として測定不能
D_MIN, D_MAX = -8.0, 0.0   # 近視側のみ採用
P_MIN, P_MAX =  2.0, 9.0

# 等高線描画用グリッド (可視化専用)
D_GRID = np.linspace(-8.0, 8.0, 321)
P_GRID = np.linspace(2.0,  9.0, 141)


# ── ソルバー ───────────────────────────────────────────────────────────────

def _loss(x, ratio_obs, area_obs):
    D, p = x
    r_res = (ratio_real(D, p) - ratio_obs) / ratio_obs
    a_res = (area_real(D, p)  - area_obs)  / area_obs
    return r_res**2 + a_res**2


def solve_p_at_D0(area_obs):
    """D=0 固定で area から p を推定 (測定不能時用)。"""
    try:
        p = brentq(lambda p: area_real(0.0, p) - area_obs, P_MIN, P_MAX)
        return float(p)
    except Exception:
        return float("nan")


def solve_one(ratio_obs, area_obs):
    """
    連続最適化で (D, p) を求める。
    ratio < RATIO_THRESH の場合は D=0 固定、p のみ推定。
    """
    if ratio_obs < RATIO_THRESH:
        p_est = solve_p_at_D0(area_obs)
        return dict(
            D=0.0, p=p_est,
            ratio_fit=float(ratio_real(0.0, p_est)) if not np.isnan(p_est) else float("nan"),
            area_fit=float(area_real(0.0, p_est))   if not np.isnan(p_est) else float("nan"),
            loss=float("nan"),
            status="unmeasurable",
        )

    # グリッドで粗い初期値を探す（局所解回避）
    D_coarse = np.linspace(D_MIN, D_MAX, 65)
    P_coarse = np.linspace(P_MIN, P_MAX, 29)
    best_loss = np.inf; best_x0 = [-1.0, 4.0]
    for d0 in D_coarse:
        for p0 in P_coarse:
            l = _loss([d0, p0], ratio_obs, area_obs)
            if l < best_loss:
                best_loss = l; best_x0 = [d0, p0]

    # 連続最適化
    res = minimize(
        _loss, best_x0,
        args=(ratio_obs, area_obs),
        method="L-BFGS-B",
        bounds=[(D_MIN, D_MAX), (P_MIN, P_MAX)],
        options=dict(ftol=1e-12, gtol=1e-9, maxiter=500),
    )
    D_sol, p_sol = res.x
    return dict(
        D=float(D_sol), p=float(p_sol),
        ratio_fit=float(ratio_real(D_sol, p_sol)),
        area_fit=float(area_real(D_sol, p_sol)),
        loss=float(res.fun),
        status="ok",
    )


# ── 可視化 ─────────────────────────────────────────────────────────────────

def plot_solution(stem, ratio_obs, area_obs, sol, out_path):
    DD, PP = np.meshgrid(D_GRID, P_GRID)
    RR = np.array([[ratio_real(d, p) for d in D_GRID] for p in P_GRID])
    AA = np.array([[area_real(d, p)  for d in D_GRID] for p in P_GRID])

    fig, ax = plt.subplots(figsize=(9, 6.5))

    if sol["status"] == "unmeasurable":
        # area等高線のみ (D=0縦線で交点)
        ca = ax.contour(DD, PP, AA, levels=[area_obs], colors="#e67e22", linewidths=2.2)
        ax.clabel(ca, fmt=f"area={area_obs:.0f}", fontsize=8)
        ax.axvline(0.0, color="#2980b9", lw=2.2,
                   label=f"D=0 fixed (ratio={ratio_obs:.3f} < {RATIO_THRESH})")
        title_extra = f"[UNMEASURABLE: near emmetropia]  D=0.00, p={sol['p']:.2f}mm"
    else:
        cr = ax.contour(DD, PP, RR, levels=[ratio_obs], colors="#2980b9", linewidths=2.2)
        ca = ax.contour(DD, PP, AA, levels=[area_obs], colors="#e67e22", linewidths=2.2)
        ax.clabel(cr, fmt=f"ratio={ratio_obs:.3f}", fontsize=8)
        ax.clabel(ca, fmt=f"area={area_obs:.0f}",   fontsize=8)
        title_extra = f"D={sol['D']:+.3f}D, p={sol['p']:.3f}mm  (loss={sol['loss']:.5f})"

    if not np.isnan(sol["p"]):
        ax.plot(sol["D"], sol["p"], "r*", ms=18, zorder=6,
                label=f"solution  D={sol['D']:+.3f}  p={sol['p']:.3f}mm")

    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("D (Diopter)"); ax.set_ylabel("p (mm)")
    ax.set_title(f"{stem}\nratio={ratio_obs:.4f}, area={area_obs:.0f}  ->  {title_extra}")
    ax.legend(fontsize=9, loc="upper center")
    ax.grid(alpha=0.3)
    ax.set_xlim(-8, 8); ax.set_ylim(2, 9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ── メイン ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/Repeatability/PickUP/01_Kavya/LEFT")
    args = ap.parse_args()

    folder   = Path(args.folder)
    csv_path = folder / "ellipse_results.csv"
    out_dir  = folder / "ratio_area_solve"
    out_dir.mkdir(exist_ok=True)

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))

    print(f"{'stem':<34} {'ratio':>7} {'area':>8} | "
          f"{'D':>8} {'p(mm)':>7} {'r_fit':>7} {'a_fit':>8} {'status'}")
    results = []
    for r in rows:
        maj = float(r["major"]); mino = float(r["minor"])
        ratio_obs = mino / maj
        area_obs  = maj * mino
        sol = solve_one(ratio_obs, area_obs)
        results.append((r["stem"], ratio_obs, area_obs, sol))

        D_str = f"{sol['D']:+.3f}" if sol["status"] == "ok" else " 0.000*"
        p_str = f"{sol['p']:.3f}" if not np.isnan(sol["p"]) else "  nan"
        loss_str = f"{sol['loss']:.5f}" if not np.isnan(sol["loss"]) else "   ---"
        print(f"{r['stem']:<34} {ratio_obs:>7.4f} {area_obs:>8.0f} | "
              f"{D_str:>8} {p_str:>7} "
              f"{sol['ratio_fit']:>7.3f} {sol['area_fit']:>8.0f}  {sol['status']}")

        plot_solution(r["stem"], ratio_obs, area_obs, sol,
                      out_dir / f"{r['stem']}_solve.png")

    # CSV
    with open(out_dir / "solve_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "ratio_obs", "area_obs", "D", "p_mm",
                    "ratio_fit", "area_fit", "loss", "status"])
        for stem, ro, ao, s in results:
            w.writerow([stem, f"{ro:.4f}", f"{ao:.1f}",
                        f"{s['D']:.4f}", f"{s['p']:.4f}",
                        f"{s['ratio_fit']:.4f}", f"{s['area_fit']:.1f}",
                        f"{s['loss']:.6f}" if not np.isnan(s['loss']) else "nan",
                        s["status"]])
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
