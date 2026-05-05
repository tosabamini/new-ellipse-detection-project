"""
患者ごとの Sphere / Cylinder / Axis 推定

物理モデル:
  major axis angle α = cylinder の axis (sphere 方向)
  D from ratio      = minor axis 方向（α+90°）の屈折力

三角関数フィット:
  D = P0 + P1*cos(2α) + P2*sin(2α)

から
  SE = P0  (spherical equivalent)
  C  = -2*sqrt(P1^2 + P2^2)   (minus cylinder, C <= 0)
  S  = SE - C/2 = SE + |C|/2
  A  = 0.5 * atan2(-P2, -P1)  (cylinder axis, 0-180 deg)
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.build_patient_model import estimate_D

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
RUN_ROOT        = PROJECT_ROOT / "data/processed/pipeline_runs/pipeline_run_v001"
OUT_DIR         = RUN_ROOT / "refraction_summary"
OUT_DIR.mkdir(exist_ok=True)

PATIENTS    = ["52", "63", "66", "67"]
SCALE       = 1.3
P_EST_MAX   = 10.0


# ── データ読み込み ─────────────────────────────────────────────

def load_patient(pid: str) -> list[dict]:
    rows = list(csv.DictReader(
        open(RUN_ROOT / pid / "results.csv", encoding="utf-8-sig")))
    seen = set(); unique = []
    for r in rows:
        if r["filename"] not in seen:
            seen.add(r["filename"]); unique.append(r)

    valid = []
    for r in unique:
        if r["status"] != "ok":
            continue
        mj    = float(r["pred_major_axis_px"])
        mn    = float(r["pred_minor_axis_px"])
        ratio = mn / mj
        alpha = float(r["pred_angle_deg"])    # major axis angle [0,180)
        p_est, d1, d2 = estimate_D(mj * SCALE, ratio)
        if p_est >= P_EST_MAX or d2 is None:
            continue
        valid.append({
            "filename": r["filename"],
            "major":    mj,
            "ratio":    ratio,
            "alpha":    alpha,
            "p_est":    p_est,
            "d1":       d1,
            "d2":       d2,
        })
    return valid


# ── 三角関数フィット ───────────────────────────────────────────

def fit_sca(alphas_deg: np.ndarray, D_vals: np.ndarray) -> dict:
    """
    D = P0 + P1*cos(2α) + P2*sin(2α) を最小二乗フィット
    → S, C, A, SE を返す
    """
    a = np.deg2rad(alphas_deg)
    X = np.column_stack([np.ones(len(a)), np.cos(2*a), np.sin(2*a)])
    P, *_ = np.linalg.lstsq(X, D_vals, rcond=None)
    P0, P1, P2 = P

    SE  = float(P0)
    amp = float(np.sqrt(P1**2 + P2**2))
    C   = -2.0 * amp                        # minus cylinder (<=0)
    S   = SE - C / 2.0                      # = SE + amp
    A   = float(np.degrees(0.5 * np.arctan2(-P2, -P1)) % 180)

    D_fit  = X @ P
    ss_res = np.sum((D_vals - D_fit)**2)
    ss_tot = np.sum((D_vals - D_vals.mean())**2)
    r2     = float(1 - ss_res/ss_tot) if ss_tot > 0 else float("nan")

    return {"SE": SE, "S": S, "C": C, "A": A,
            "P0": P0, "P1": P1, "P2": P2, "R2": r2}


# ── グラフ ────────────────────────────────────────────────────

def plot_patient(pid: str, records: list[dict], fit: dict, ax):
    alphas = np.array([r["alpha"] for r in records])
    Ds     = np.array([r["d2"]    for r in records])

    a_fine = np.linspace(0, 180, 360)
    a_rad  = np.deg2rad(a_fine)
    D_fine = (fit["P0"]
              + fit["P1"] * np.cos(2*a_rad)
              + fit["P2"] * np.sin(2*a_rad))

    ax.scatter(alphas, Ds, s=50, zorder=5, label="measured D2")
    ax.plot(a_fine, D_fine, "r-", linewidth=1.8, label="fit")
    ax.axvline(fit["A"], color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Axis={fit['A']:.0f}deg")

    title = (f"Patient {pid}\n"
             f"S={fit['S']:+.2f}D  C={fit['C']:+.2f}D  A={fit['A']:.0f}deg  "
             f"SE={fit['SE']:+.2f}D  R²={fit['R2']:.3f}")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Major axis angle α (deg)", fontsize=8)
    ax.set_ylabel("D2 adopted (D)", fontsize=8)
    ax.set_xlim(0, 180)
    ax.set_xticks(range(0, 181, 30))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=7)


# ── メイン ────────────────────────────────────────────────────

def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    print(f"{'patient':>8}  {'S':>7}  {'C':>7}  {'A':>6}  {'SE':>7}  {'R2':>6}  n")
    print("-" * 55)

    for i, pid in enumerate(PATIENTS):
        records = load_patient(pid)
        if len(records) < 3:
            print(f"{pid:>8}  データ不足 (n={len(records)})")
            continue

        alphas = np.array([r["alpha"] for r in records])
        Ds     = np.array([r["d2"]    for r in records])

        fit = fit_sca(alphas, Ds)

        print(f"{pid:>8}  {fit['S']:>+7.2f}  {fit['C']:>+7.2f}  "
              f"{fit['A']:>5.1f}  {fit['SE']:>+7.2f}  {fit['R2']:>6.3f}  {len(records)}")

        plot_patient(pid, records, fit, axes[i])

    plt.suptitle("Sphere / Cylinder / Axis estimation per patient\n"
                 "(fitted from D2 vs major axis angle, scale x1.3)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_path = OUT_DIR / "patient_SCA.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot: {out_path}")


if __name__ == "__main__":
    main()
