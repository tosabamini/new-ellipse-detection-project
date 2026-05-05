"""
2次元モデル解析: ratio と major_axis を (D, 瞳孔径) の関数として同時フィット

測定値:  ratio = minor/major,  major_axis  の2変数
未知数:  屈折力 D (D),  瞳孔径 p (mm) の2変数

モデル:
  ratio(D, p) = c @ [1, D, D², p, D·p, D²·p]
  major(D, p) = d @ [1, D, D², p, D·p, D²·p]

推定:
  患者観測値 (ratio_obs, major_obs) から (D, p) を同時推定
  目的関数: (ratio_pred - ratio_obs)²/σ_r² + (major_pred - major_obs)²/σ_m²

出力:
  data/processed/model_eye_runs/combined_analysis/2d_model/
    coefficients.csv   : フィット係数
    loo_results.csv    : LOO交差検証結果
    fit_surfaces.png   : 2Dフィット曲面
    loo_scatter.png    : 推定 vs 真値散布図
"""

import csv
import sys
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_EYE_RUNS_DIR = PROJECT_ROOT / "data/processed/model_eye_runs"
OUT_DIR = MODEL_EYE_RUNS_DIR / "combined_analysis" / "2d_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PUPILS = [
    {"mm": 3.0, "run": "model_eye_3mm_v001"},
    {"mm": 5.0, "run": "model_eye_5mm_v001"},
    {"mm": 7.0, "run": "model_eye_v001"},
]


# ── データ読み込み ─────────────────────────────────────────

def folder_to_diopter(name: str) -> float:
    parts = name.split("_")
    sign  = parts[1]
    major = int(parts[2])
    minor = int(parts[3][:-1])
    val   = major + minor / 100.0
    return -val if sign == "M" else (0.0 if sign == "Z" else val)


def load_all_data() -> list[dict]:
    """全瞳孔径の屈折力別平均値を返す"""
    records = []
    for pupil in PUPILS:
        run_root = MODEL_EYE_RUNS_DIR / pupil["run"]
        for folder_dir in sorted(run_root.iterdir()):
            csv_path = folder_dir / "ellipse_results.csv"
            if not csv_path.exists():
                continue
            majors, minors = [], []
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    if row["status"] != "ok":
                        continue
                    majors.append(float(row["major_axis"]))
                    minors.append(float(row["minor_axis"]))
            if not majors:
                continue
            ratios = [mn / mj for mn, mj in zip(minors, majors)]
            records.append({
                "p":           pupil["mm"],
                "D":           folder_to_diopter(folder_dir.name),
                "major_mean":  float(np.mean(majors)),
                "ratio_mean":  float(np.mean(ratios)),
                "n":           len(majors),
            })
    return records


# ── 多項式フィーチャ ───────────────────────────────────────

def feat(D: float, p: float) -> np.ndarray:
    """[1, D, D², p, D·p, D²·p]"""
    return np.array([1.0, D, D**2, p, D * p, D**2 * p])


def feat_matrix(Ds, ps) -> np.ndarray:
    """shape (n, 6)"""
    return np.array([feat(d, p) for d, p in zip(Ds, ps)])


# ── フィット & 推定 ────────────────────────────────────────

def fit_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """最小二乗フィット → 係数ベクトル (6,)"""
    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeff


def predict(D: float, p: float, coeff: np.ndarray) -> float:
    return float(feat(D, p) @ coeff)


def estimate_Dp(
    ratio_obs: float,
    major_obs: float,
    coeff_r: np.ndarray,
    coeff_m: np.ndarray,
    sigma_r: float,
    sigma_m: float,
) -> tuple[float, float, float]:
    """
    (ratio_obs, major_obs) から (D, p) を同時推定。
    D は [-5, +4] の範囲、p は [3, 7] の範囲に制約。
    グリッドサーチ初期値 + Nelder-Mead 最適化。
    returns: (D_est, p_est, loss)
    """
    def loss(x):
        D_, p_ = x
        r_pred = predict(D_, p_, coeff_r)
        m_pred = predict(D_, p_, coeff_m)
        return ((r_pred - ratio_obs) / sigma_r) ** 2 + \
               ((m_pred - major_obs) / sigma_m) ** 2

    D_inits = np.linspace(-5, 4, 10)
    p_inits = [3.0, 5.0, 7.0]

    best_loss = np.inf
    best_x = None
    for D0, p0 in product(D_inits, p_inits):
        res = minimize(
            loss, [D0, p0],
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 2000},
        )
        if res.fun < best_loss:
            best_loss = res.fun
            best_x = res.x

    return float(best_x[0]), float(best_x[1]), float(best_loss)


# ── R² / RMSE ─────────────────────────────────────────────

def r2_rmse(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return r2, rmse


# ── メイン ────────────────────────────────────────────────

def main():
    records = load_all_data()
    records.sort(key=lambda r: (r["p"], r["D"]))
    n = len(records)
    print(f"データ点数: {n}  (3瞳孔径 × 10屈折力)")

    Ds     = np.array([r["D"]          for r in records])
    Ps     = np.array([r["p"]          for r in records])
    R_obs  = np.array([r["ratio_mean"] for r in records])
    M_obs  = np.array([r["major_mean"] for r in records])

    X = feat_matrix(Ds, Ps)

    # ── 全データでフィット ────────────────────────────────
    coeff_r = fit_model(X, R_obs)
    coeff_m = fit_model(X, M_obs)

    R_pred_all = X @ coeff_r
    M_pred_all = X @ coeff_m

    r2_r, rmse_r = r2_rmse(R_obs, R_pred_all)
    r2_m, rmse_m = r2_rmse(M_obs, M_pred_all)

    print("\n── 全データフィット精度 ──────────────────────────────")
    print(f"  ratio model:  R2={r2_r:.4f}, RMSE={rmse_r:.5f}")
    print(f"  major model:  R2={r2_m:.4f}, RMSE={rmse_m:.3f} px")

    # 正規化スケール（全データの std）
    sigma_r = float(np.std(R_obs))
    sigma_m = float(np.std(M_obs))

    # 係数を CSV に保存
    feat_names = ["1", "D", "D2", "p", "Dp", "D2p"]
    with open(OUT_DIR / "coefficients.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["feature", "coeff_ratio", "coeff_major"])
        for name, cr, cm in zip(feat_names, coeff_r, coeff_m):
            w.writerow([name, f"{cr:.8f}", f"{cm:.8f}"])
    print(f"\n係数 CSV: {OUT_DIR / 'coefficients.csv'}")

    # ── LOO 交差検証 ─────────────────────────────────────
    print("\n── LOO 交差検証 ─────────────────────────────────────")
    print(f"{'p(true)':>8}  {'D(true)':>8}  {'D_est':>8}  {'p_est':>7}  "
          f"{'D_err':>7}  {'p_err':>7}  {'loss':>8}")

    loo_rows = []
    D_errs, p_errs = [], []

    for i in range(n):
        idx = list(range(n))
        idx.pop(i)
        X_tr  = X[idx]
        Rtr   = R_obs[idx]
        Mtr   = M_obs[idx]

        cr_l = fit_model(X_tr, Rtr)
        cm_l = fit_model(X_tr, Mtr)

        # スケール（訓練データ基準）
        sr = float(np.std(Rtr)) or sigma_r
        sm = float(np.std(Mtr)) or sigma_m

        D_est, p_est, lv = estimate_Dp(R_obs[i], M_obs[i], cr_l, cm_l, sr, sm)
        D_err = abs(D_est - Ds[i])
        p_err = abs(p_est - Ps[i])
        D_errs.append(D_err)
        p_errs.append(p_err)

        print(f"{Ps[i]:>8.1f}  {Ds[i]:>+8.2f}  {D_est:>+8.2f}  {p_est:>7.2f}  "
              f"{D_err:>7.3f}  {p_err:>7.3f}  {lv:>8.4f}")

        loo_rows.append({
            "p_true": Ps[i], "D_true": Ds[i],
            "D_est": f"{D_est:.4f}", "p_est": f"{p_est:.4f}",
            "D_err": f"{D_err:.4f}", "p_err": f"{p_err:.4f}",
        })

    D_errs = np.array(D_errs)
    p_errs = np.array(p_errs)
    print(f"\n  D_est MAE: {np.mean(D_errs):.3f} D   max: {np.max(D_errs):.3f} D")
    print(f"  p_est MAE: {np.mean(p_errs):.3f} mm  max: {np.max(p_errs):.3f} mm")

    # 瞳孔径の分類精度（最近傍の 3/5/7 mm に丸める）
    P_est_rounded = np.array([
        min([3.0, 5.0, 7.0], key=lambda pv: abs(pv - float(r["p_est"])))
        for r in loo_rows
    ])
    correct = np.sum(P_est_rounded == Ps)
    print(f"  瞳孔径分類精度: {correct}/{n} ({100*correct/n:.1f}%)")

    with open(OUT_DIR / "loo_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(loo_rows[0].keys()))
        w.writeheader()
        w.writerows(loo_rows)
    print(f"\nLOO CSV: {OUT_DIR / 'loo_results.csv'}")

    # ── プロット1: フィット曲面 ───────────────────────────
    D_fine = np.linspace(-5.5, 4.5, 60)
    p_vals = [3.0, 5.0, 7.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_r, ax_m = axes

    for p_v, col in zip(p_vals, colors):
        idx_p = [i for i, r in enumerate(records) if r["p"] == p_v]
        D_pts = Ds[idx_p]
        R_pts = R_obs[idx_p]
        M_pts = M_obs[idx_p]

        R_line = [predict(d, p_v, coeff_r) for d in D_fine]
        M_line = [predict(d, p_v, coeff_m) for d in D_fine]

        ax_r.scatter(D_pts, R_pts, color=col, zorder=5, s=40)
        ax_r.plot(D_fine, R_line, color=col, linewidth=1.8, label=f"{p_v:.0f} mm")

        ax_m.scatter(D_pts, M_pts, color=col, zorder=5, s=40)
        ax_m.plot(D_fine, M_line, color=col, linewidth=1.8, label=f"{p_v:.0f} mm")

    for ax, title, ylabel in [
        (ax_r, f"ratio(D, p)  R²={r2_r:.4f}", "Minor / Major ratio"),
        (ax_m, f"major(D, p)  R²={r2_m:.4f}", "Major axis (px)"),
    ]:
        ax.set_xlabel("Refraction power (D)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(np.arange(-5, 5))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="Pupil", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fit_surfaces.png", dpi=150)
    plt.close(fig)
    print(f"フィット曲面: {OUT_DIR / 'fit_surfaces.png'}")

    # ── プロット2: LOO 散布図 ────────────────────────────
    D_true_arr = Ds
    D_est_arr  = np.array([float(r["D_est"]) for r in loo_rows])
    p_true_arr = Ps
    p_est_arr  = np.array([float(r["p_est"]) for r in loo_rows])

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for p_v, col in zip(p_vals, colors):
        mask = p_true_arr == p_v
        ax1.scatter(D_true_arr[mask], D_est_arr[mask], color=col,
                    label=f"{p_v:.0f} mm", s=60, zorder=5)
        ax2.scatter(p_true_arr[mask], p_est_arr[mask], color=col,
                    label=f"{p_v:.0f} mm", s=60, zorder=5)

    lim_D = (-5.5, 4.5)
    ax1.plot(lim_D, lim_D, "k--", linewidth=1, alpha=0.5)
    ax1.set_xlim(lim_D); ax1.set_ylim(lim_D)
    ax1.set_xlabel("True D (D)"); ax1.set_ylabel("Estimated D (D)")
    ax1.set_title(f"LOO: D estimation  MAE={np.mean(D_errs):.3f} D", fontsize=12)
    ax1.legend(title="Pupil (true)"); ax1.grid(True, linestyle="--", alpha=0.4)

    lim_p = (1.5, 8.5)
    ax2.plot(lim_p, lim_p, "k--", linewidth=1, alpha=0.5)
    ax2.set_xlim(lim_p); ax2.set_ylim(lim_p)
    ax2.set_xlabel("True pupil (mm)"); ax2.set_ylabel("Estimated pupil (mm)")
    ax2.set_title(f"LOO: pupil estimation  MAE={np.mean(p_errs):.3f} mm", fontsize=12)
    ax2.legend(title="Pupil (true)"); ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig2.savefig(OUT_DIR / "loo_scatter.png", dpi=150)
    plt.close(fig2)
    print(f"LOO 散布図: {OUT_DIR / 'loo_scatter.png'}")

    print("\n完了。")


if __name__ == "__main__":
    main()
