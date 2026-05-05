"""
患者画像の楕円パラメータから屈折力（S, C, A）を推定するモジュール

パイプライン:
  (major_px, minor_px, angle_deg) per image
       ↓
  ratio = minor / major
  major_scaled = major * SCALE_FACTOR          [式1: px スケール補正]
  p_est = major_to_pupil(major_scaled)         [式1: major → 瞳孔径]
  D = estimate_D_from_ratio_and_p(ratio, p_est)[式2: ratio → 屈折力]
       ↓
  D vs angle_deg の三角関数フィット
  D = P0 + P1*cos(2α) + P2*sin(2α)
       ↓
  S (sphere), C (cylinder, minus), A (axis deg)

現状の近似（要将来改善）:
  - SCALE_FACTOR = 1.3 は暫定値。患者画像とモデル眼画像のpxスケール比を
    実測して校正する必要がある。
  - 2解問題（D1/D2）は近視側（D2）を採用する方針で暫定対応。
  - 3未満の有効画像ではSCAフィットを行わない。
"""

import csv
from pathlib import Path

import numpy as np

# ── 定数（現状の暫定値）──────────────────────────────────────
SCALE_FACTOR = 1.3    # 患者画像 / モデル眼画像のpxスケール補正係数（暫定）
P_EST_MAX    = 10.0   # 瞳孔径推定がこの値以上の画像はノイズとして除外 (mm)
MIN_VALID    = 3      # SCAフィットに必要な最低有効画像数

# モジュールロード時にモデル眼参照データをフィット
from src.analysis.build_patient_model import (
    estimate_D as _estimate_D_full,
)


# ── 1画像の屈折力推定 ─────────────────────────────────────────

def estimate_D_for_image(
    major_px: float,
    minor_px: float,
    scale_factor: float = SCALE_FACTOR,
) -> dict:
    """
    楕円の major / minor (px) から屈折力を推定する。

    Returns dict:
        ratio      : minor / major
        p_est      : 推定瞳孔径 (mm)
        d1, d2     : 屈折力の2解 (D)。実数解なければ None
        adopted_D  : 採用解 (近視側 d2)。無効なら None
        valid      : bool — ノイズ除外後に有効かどうか
    """
    ratio = minor_px / major_px
    p_est, d1, d2 = _estimate_D_full(major_px * scale_factor, ratio)
    valid = (p_est < P_EST_MAX) and (d2 is not None)
    return {
        "ratio":     ratio,
        "p_est":     float(p_est),
        "d1":        float(d1) if d1 is not None else None,
        "d2":        float(d2) if d2 is not None else None,
        "adopted_D": float(d2) if valid else None,
        "valid":     valid,
    }


# ── SCA フィット ──────────────────────────────────────────────

def fit_sca(alpha_deg: np.ndarray, D_vals: np.ndarray) -> dict:
    """
    D = P0 + P1·cos(2α) + P2·sin(2α) を最小二乗フィットし
    処方箋形式の S / C / A を返す。

    物理的解釈:
      α (major axis angle) = cylinder axis A (sphere の向き)
      D = minor axis 方向の屈折力 (最大近視方向)

    Returns dict:
        S   : sphere (D)
        C   : cylinder, minus cylinder notation (D, <= 0)
        A   : cylinder axis (deg, 0-180)
        SE  : spherical equivalent = S + C/2 (D)
        R2  : フィットの決定係数
        n   : 使用画像数
    """
    a  = np.deg2rad(alpha_deg)
    X  = np.column_stack([np.ones(len(a)), np.cos(2*a), np.sin(2*a)])
    P, *_ = np.linalg.lstsq(X, D_vals, rcond=None)
    P0, P1, P2 = float(P[0]), float(P[1]), float(P[2])

    SE  = P0
    amp = np.sqrt(P1**2 + P2**2)
    C   = -2.0 * amp                                     # minus cylinder
    S   = SE - C / 2.0                                   # = SE + amp
    A   = float(np.degrees(0.5 * np.arctan2(-P2, -P1)) % 180)

    D_fit  = X @ P
    ss_res = float(np.sum((D_vals - D_fit)**2))
    ss_tot = float(np.sum((D_vals - D_vals.mean())**2))
    R2     = (1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "S": float(S), "C": float(C), "A": float(A),
        "SE": float(SE), "R2": float(R2), "n": len(D_vals),
    }


# ── 患者単位の一括解析 ────────────────────────────────────────

def run_refraction_analysis(
    results_csv_path: Path,
    patient_id: str,
    scale_factor: float = SCALE_FACTOR,
) -> dict:
    """
    process_patient() が出力した results.csv を読み込み、
    画像ごとの屈折力推定と SCA フィットを実行する。

    Returns:
        per_image : list[dict]  — 画像ごとの推定結果（ok画像のみ）
        sca       : dict | None — SCAフィット結果（有効画像 < MIN_VALID で None）
        n_total   : int  — ok画像の総数（重複除去後）
        n_valid   : int  — SCAフィットに使用した有効画像数
    """
    rows = list(csv.DictReader(open(results_csv_path, encoding="utf-8-sig")))

    # 拡張子大小文字の重複を除去
    seen = set(); unique = []
    for r in rows:
        if r["filename"] not in seen:
            seen.add(r["filename"]); unique.append(r)

    per_image = []
    for r in unique:
        if r["status"] != "ok":
            continue
        mj    = float(r["pred_major_axis_px"])
        mn    = float(r["pred_minor_axis_px"])
        alpha = float(r["pred_angle_deg"])
        est   = estimate_D_for_image(mj, mn, scale_factor)
        per_image.append({
            "patient_id": patient_id,
            "filename":   r["filename"],
            "major_px":   mj,
            "minor_px":   mn,
            "angle_deg":  alpha,
            **est,
        })

    valid   = [x for x in per_image if x["valid"]]
    n_valid = len(valid)

    sca = None
    if n_valid >= MIN_VALID:
        alphas = np.array([x["angle_deg"]  for x in valid])
        Ds     = np.array([x["adopted_D"]  for x in valid])
        sca    = fit_sca(alphas, Ds)

    return {
        "per_image": per_image,
        "sca":       sca,
        "n_total":   len([r for r in unique if r["status"] == "ok"]),
        "n_valid":   n_valid,
    }


# ── CSV 書き出しユーティリティ ────────────────────────────────

def write_per_image_csv(per_image: list[dict], out_path: Path) -> None:
    if not per_image:
        return
    fields = ["patient_id", "filename", "major_px", "minor_px",
              "angle_deg", "ratio", "p_est", "d1", "d2", "adopted_D", "valid"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in per_image:
            w.writerow({k: (f"{row[k]:.4f}" if isinstance(row[k], float) else row[k])
                        for k in fields})


def write_sca_csv(patient_id: str, result: dict, out_path: Path) -> None:
    sca = result["sca"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "S_D", "C_D", "A_deg", "SE_D",
                    "R2", "n_valid", "n_total", "scale_factor", "note"])
        if sca:
            w.writerow([
                patient_id,
                f"{sca['S']:.3f}", f"{sca['C']:.3f}",
                f"{sca['A']:.1f}", f"{sca['SE']:.3f}",
                f"{sca['R2']:.4f}", sca["n"],
                result["n_total"], SCALE_FACTOR,
                "scale_factor is approximate; 2-solution issue handled by adopting myopic (D2)",
            ])
        else:
            w.writerow([patient_id, "", "", "", "", "",
                        result["n_valid"], result["n_total"], SCALE_FACTOR,
                        f"insufficient valid images (need >= {MIN_VALID})"])
