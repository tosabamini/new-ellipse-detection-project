"""
患者用屈折力推定モデル

入力: major_axis (px),  ratio (minor / major)
出力: 屈折力 D (D)

─── パイプライン ────────────────────────────────────────────────
 major ──[式1: M→p]──▶ 瞳孔径 p ──[式2: (p,ratio)→D]──▶ D
────────────────────────────────────────────────────────────────

式1  major_to_pupil(M):
  3参照点 (130px,3mm), (180px,5mm), (200px,7mm) を通る正確な2次式
  p = α·M² + β·M + γ

式2  estimate_D(ratio, p):
  各瞳孔径の ratio 2次フィット係数 a,b,c を p の2次式で補間し
  a(p)·D² + b(p)·D + c(p) = ratio  を逆算
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_EYE_RUNS_DIR = PROJECT_ROOT / "data/processed/model_eye_runs"
OUT_DIR = MODEL_EYE_RUNS_DIR / "combined_analysis" / "patient_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PUPIL_RUNS = [
    (3.0, "model_eye_3mm_v001"),
    (5.0, "model_eye_5mm_v001"),
    (7.0, "model_eye_v001"),
]

# 参照 major_axis (px) — 各瞳孔径の代表値
REFERENCE_MAJOR = {3.0: 130.0, 5.0: 180.0, 7.0: 200.0}


# ── データ読み込み ─────────────────────────────────────────────

def folder_to_diopter(name: str) -> float:
    parts = name.split("_")
    sign  = parts[1]
    major = int(parts[2])
    minor = int(parts[3][:-1])
    val   = major + minor / 100.0
    return -val if sign == "M" else (0.0 if sign == "Z" else val)


def load_run(run_name: str) -> list[dict]:
    run_root = MODEL_EYE_RUNS_DIR / run_name
    records  = []
    for folder in sorted(run_root.iterdir()):
        csv_path = folder / "ellipse_results.csv"
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
            "D":          folder_to_diopter(folder.name),
            "ratio_mean": float(np.mean(ratios)),
        })
    records.sort(key=lambda r: r["D"])
    return records


# ── 式1: major_axis → 瞳孔径 p ────────────────────────────────
#
# 3参照点を通る正確な2次式を np.polyfit で導出（3点=過決定なし）
# p = major_poly[0]·M² + major_poly[1]·M + major_poly[2]

_M_REF = np.array([130.0, 180.0, 200.0])
_P_REF = np.array([  3.0,   5.0,   7.0])
MAJOR_TO_P_COEFFS = np.polyfit(_M_REF, _P_REF, deg=2)   # 正確に3点を通る


def major_to_pupil(major: float) -> float:
    """major_axis (px) → 瞳孔径 p (mm)"""
    return float(np.polyval(MAJOR_TO_P_COEFFS, major))


# ── 式2: (ratio, p) → D の統合モデル ──────────────────────────
#
# 各瞳孔径の ratio 2次フィット: ratio = a_i·D² + b_i·D + c_i  (i=3,5,7mm)
# を fit して (a_i, b_i, c_i) を取得後、
# 係数をそれぞれ p の2次式で補間:
#   a(p), b(p), c(p)  ← 3点を通る正確な2次式
# 統合モデル: ratio = a(p)·D² + b(p)·D + c(p)
# 逆算: a(p)·D² + b(p)·D + (c(p) - ratio_obs) = 0


def _fit_ratio_per_pupil() -> tuple[list[float], list[float], list[float], list[dict]]:
    """各瞳孔径の ratio 2次フィット係数 (a,b,c) を返す"""
    A, B, C = [], [], []
    all_records = []
    for p_mm, run_name in PUPIL_RUNS:
        records = load_run(run_name)
        D_vals  = np.array([r["D"]          for r in records])
        R_vals  = np.array([r["ratio_mean"] for r in records])
        a, b, c = np.polyfit(D_vals, R_vals, deg=2)
        A.append(a); B.append(b); C.append(c)
        for r in records:
            all_records.append({**r, "p": p_mm})
    return A, B, C, all_records


_A_raw, _B_raw, _C_raw, _ALL_RECORDS = _fit_ratio_per_pupil()
_P_VALS = np.array([p for p, _ in PUPIL_RUNS])

# 係数の p への2次フィット (3点なので正確に通る)
A_COEFFS = np.polyfit(_P_VALS, _A_raw, deg=2)
B_COEFFS = np.polyfit(_P_VALS, _B_raw, deg=2)
C_COEFFS = np.polyfit(_P_VALS, _C_raw, deg=2)


def _abc(p: float) -> tuple[float, float, float]:
    return (
        float(np.polyval(A_COEFFS, p)),
        float(np.polyval(B_COEFFS, p)),
        float(np.polyval(C_COEFFS, p)),
    )


def estimate_D_from_ratio_and_p(
    ratio_obs: float, p: float
) -> tuple[float | None, float | None]:
    """ratio と 瞳孔径 p (mm) から D の2解を返す（実数解なければ None）"""
    a, b, c = _abc(p)
    disc = b**2 - 4 * a * (c - ratio_obs)
    if disc < 0:
        return None, None
    sq = np.sqrt(disc)
    d1 = (-b + sq) / (2 * a)
    d2 = (-b - sq) / (2 * a)
    return float(d1), float(d2)


def estimate_D(major: float, ratio_obs: float) -> tuple[float, float | None, float | None]:
    """
    統合推定: (major_axis, ratio) → (p_est, D1, D2)
    D1, D2 は2次式の2解。Noneは実数解なし。
    """
    p = major_to_pupil(major)
    d1, d2 = estimate_D_from_ratio_and_p(ratio_obs, p)
    return p, d1, d2


# ── メイン ────────────────────────────────────────────────────

def main():
    # ── 式1: major → p の係数を表示 ─────────────────────────
    alpha, beta, gamma = MAJOR_TO_P_COEFFS
    print("=" * 65)
    print("[式1] major_axis → 瞳孔径 p (mm)")
    print(f"  p = {alpha:.8f}*M^2 + ({beta:.8f})*M + ({gamma:.8f})")
    print("  検証:")
    for M_ref, p_ref in zip(_M_REF, _P_REF):
        p_est = major_to_pupil(M_ref)
        print(f"    M={M_ref:.0f}px -> p_est={p_est:.4f}mm  (true={p_ref:.1f}mm)")

    # ── 各瞳孔径の ratio フィット係数 ─────────────────────────
    print("\n[ratio 2次フィット係数 per pupil]")
    print(f"  {'p':>5}  {'a':>12}  {'b':>12}  {'c':>12}")
    for p_mm, a, b, c in zip(_P_VALS, _A_raw, _B_raw, _C_raw):
        print(f"  {p_mm:>5.1f}  {a:>12.8f}  {b:>12.8f}  {c:>12.8f}")

    # ── 係数を p の2次式で補間した結果 ───────────────────────
    print("\n[係数の p 補間式]  (p=3,5,7mm を正確に通る2次式)")
    for label, coeffs, vals in [("a", A_COEFFS, _A_raw),
                                  ("b", B_COEFFS, _B_raw),
                                  ("c", C_COEFFS, _C_raw)]:
        a2, a1, a0 = coeffs
        print(f"  {label}(p) = {a2:.8f}*p^2 + ({a1:.8f})*p + ({a0:.8f})")

    # ── 式2: 統合モデルの式 ──────────────────────────────────
    print("\n[式2] 統合 ratio モデル: ratio = a(p)*D^2 + b(p)*D + c(p)")
    print("  (a,b,c は上記の p の2次式で与えられる)")
    print("  逆算: a(p)*D^2 + b(p)*D + (c(p) - ratio) = 0")
    print("        -> D = [-b(p) +/- sqrt(b(p)^2 - 4*a(p)*(c(p)-ratio))] / (2*a(p))")

    # ── 全体パイプライン式 ───────────────────────────────────
    print("\n[最終パイプライン]  (major, ratio) -> D")
    print("  1. p   = alpha*M^2 + beta*M + gamma          [式1]")
    print("  2. a_p = A2*p^2 + A1*p + A0")
    print("     b_p = B2*p^2 + B1*p + B0")
    print("     c_p = C2*p^2 + C1*p + C0")
    print("  3. D   = [-b_p +/- sqrt(b_p^2 - 4*a_p*(c_p - ratio))] / (2*a_p)  [式2]")

    # ── LOO 検証 (per-pupil ratio モデルで) ──────────────────
    print("\n[LOO 検証: 各瞳孔径の ratio モデル (LOO per pupil)]")
    print(f"  {'p':>6}  {'D_true':>8}  {'D_sol1':>8}  {'D_sol2':>8}  {'err_best':>9}")
    loo_errs = []
    for p_mm, run_name in PUPIL_RUNS:
        records = load_run(run_name)
        D_arr = np.array([r["D"]          for r in records])
        R_arr = np.array([r["ratio_mean"] for r in records])
        for i in range(len(D_arr)):
            idx = list(range(len(D_arr)))
            idx.pop(i)
            a, b, c = np.polyfit(D_arr[idx], R_arr[idx], 2)
            ratio_i  = R_arr[i]
            disc     = b**2 - 4*a*(c - ratio_i)
            if disc < 0:
                print(f"  {p_mm:>6.1f}  {D_arr[i]:>+8.2f}  {'(no real)':>18}")
                continue
            d1 = (-b + np.sqrt(disc)) / (2*a)
            d2 = (-b - np.sqrt(disc)) / (2*a)
            err = min(abs(d1 - D_arr[i]), abs(d2 - D_arr[i]))
            loo_errs.append(err)
            print(f"  {p_mm:>6.1f}  {D_arr[i]:>+8.2f}  {d1:>+8.2f}  {d2:>+8.2f}  {err:>9.3f}")
    print(f"\n  LOO MAE (best sol): {np.mean(loo_errs):.3f} D")
    print(f"  LOO max error:      {np.max(loo_errs):.3f} D")

    # ── グラフ1: ratio vs D — データ + 統合モデル ─────────────
    D_fine = np.linspace(-5.5, 4.5, 300)
    p_colors = {"3.0 mm": "#1f77b4", "5.0 mm": "#ff7f0e", "7.0 mm": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for p_mm, run_name in PUPIL_RUNS:
        col   = list(p_colors.values())[PUPIL_RUNS.index((p_mm, run_name))]
        label = f"{p_mm:.1f} mm"
        records = load_run(run_name)
        D_pts = [r["D"]          for r in records]
        R_pts = [r["ratio_mean"] for r in records]
        R_model = [float(np.polyval([_A_raw[PUPIL_RUNS.index((p_mm, run_name))],
                                     _B_raw[PUPIL_RUNS.index((p_mm, run_name))],
                                     _C_raw[PUPIL_RUNS.index((p_mm, run_name))]], d))
                   for d in D_fine]
        ax.scatter(D_pts, R_pts, color=col, zorder=5, s=50)
        ax.plot(D_fine, R_model, color=col, linewidth=2, label=label)

    ax.set_xlabel("Refraction power (D)", fontsize=12)
    ax.set_ylabel("Ratio (minor/major)", fontsize=12)
    ax.set_title("Ratio vs D: per-pupil quadratic fits", fontsize=13)
    ax.set_xticks(np.arange(-5, 5))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Pupil diameter", fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "ratio_vs_D_per_pupil.png", dpi=150)
    plt.close(fig)

    # ── グラフ2: major → p 対応 ─────────────────────────────
    M_fine = np.linspace(110, 220, 200)
    p_fine = np.polyval(MAJOR_TO_P_COEFFS, M_fine)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(M_fine, p_fine, "-", color="#333", linewidth=2, label="Fitted curve")
    ax2.scatter(_M_REF, _P_REF, s=80, color="red", zorder=5, label="Reference points")
    for M_r, p_r in zip(_M_REF, _P_REF):
        ax2.annotate(f"({M_r:.0f}px, {p_r:.0f}mm)",
                     xy=(M_r, p_r), xytext=(M_r + 4, p_r - 0.3), fontsize=9)
    ax2.set_xlabel("Major axis (px)", fontsize=12)
    ax2.set_ylabel("Pupil diameter (mm)", fontsize=12)
    ax2.set_title("major_axis -> pupil diameter mapping", fontsize=13)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=10)
    plt.tight_layout()
    fig2.savefig(OUT_DIR / "major_to_pupil.png", dpi=150)
    plt.close(fig2)

    # ── グラフ3: 統合モデル — ratio surface (p を連続変化) ────
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    p_range = np.linspace(3.0, 7.0, 5)
    cmap    = plt.cm.viridis
    for i, p_v in enumerate(p_range):
        col = cmap(i / (len(p_range) - 1))
        a_p, b_p, c_p = _abc(p_v)
        R_line = [a_p*d**2 + b_p*d + c_p for d in D_fine]
        ax3.plot(D_fine, R_line, color=col, linewidth=1.5, label=f"p={p_v:.1f}mm")

    # 実データ点
    for j, (p_mm, run_name) in enumerate(PUPIL_RUNS):
        records = load_run(run_name)
        D_pts = [r["D"]          for r in records]
        R_pts = [r["ratio_mean"] for r in records]
        col = ["#1f77b4", "#ff7f0e", "#2ca02c"][j]
        ax3.scatter(D_pts, R_pts, color=col, zorder=5, s=60,
                    edgecolors="black", linewidths=0.5)

    ax3.set_xlabel("Refraction power (D)", fontsize=12)
    ax3.set_ylabel("Ratio (minor/major)", fontsize=12)
    ax3.set_title("Unified model: ratio = a(p)*D^2 + b(p)*D + c(p)", fontsize=13)
    ax3.set_xticks(np.arange(-5, 5))
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(title="Pupil", fontsize=9, loc="upper right")
    plt.tight_layout()
    fig3.savefig(OUT_DIR / "unified_ratio_model.png", dpi=150)
    plt.close(fig3)

    print(f"\nPlots saved to: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
