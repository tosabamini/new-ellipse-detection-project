"""
古典CV（Otsu閾値 + 連結成分）によるセグメンテーション試験

Classify・Segmentation を両方 ML なしで代替する試み。
屈折力の計算は行わない。マスクオーバーレイ画像のみを出力する。

入力: data/processed/pipeline_runs/<run_name>/<patient_id>/red/ の画像
出力: experiments/classical_seg_out/<run_name>/<patient_id>/
       └ overlay/   マスク + 楕円（または失敗理由）を重ねた画像
       └ results.csv

使い方:
  python -m experiments.classical_seg_trial \\
    --run_name pipeline_run_101_106_v001 \\
    --patient_ids 101_LEFT 101_RIGHT 102_LEFT 104_LEFT 104_RIGHT \\
                  105_LEFT 105_RIGHT 106_LEFT 106_RIGHT

調整パラメータ（このファイルの上部）:
  OPEN_K          モルフォロジー・オープニングのカーネルサイズ（ノイズ除去）
  CLOSE_K         クロージングのカーネルサイズ（穴埋め）
  MIN_AREA_PX     有効マスクの最小面積（px²）
  MAX_AREA_FRAC   有効マスクの最大面積（画像全体に対する割合）
  CENTER_WEIGHT   中心距離スコアの重み
  AREA_BONUS      面積スコアのボーナス係数
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

# ── 調整パラメータ ──────────────────────────────────────────────
OPEN_K         = 3      # オープニング（小ノイズ除去）
CLOSE_K        = 7      # クロージング（穴埋め）
MIN_AREA_PX    = 300    # これ以下は無視
MAX_AREA_FRAC  = 0.50   # 画像面積のこの割合以上は無視（瞼など大物を除外）
CENTER_WEIGHT  = 1.0    # 中心距離スコア重み
AREA_BONUS     = 0.01   # 面積スコアボーナス

# ── パス設定 ────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
PIPELINE_RUNS = PROJECT_ROOT / "data" / "processed" / "pipeline_runs"
OUT_ROOT      = PROJECT_ROOT / "experiments" / "classical_seg_out"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",    required=True)
    p.add_argument("--patient_ids", nargs="+", required=True)
    return p.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ── 古典 CV セグメンテーション ──────────────────────────────────

def classical_segment(red_img: np.ndarray) -> dict:
    """
    グレースケール red_img から赤反帰光領域のマスクを推定する。

    Returns dict:
      status      : "ok" | "no_component" | "too_large"
      mask        : uint8 binary mask (same size as input), or None
      single_mask : 最良1連結成分のみのマスク、または None
      ellipse_info: cv2.fitEllipse 結果 dict、または None
      n_candidates: 連結成分の候補数
    """
    h, w = red_img.shape[:2]
    img_area = h * w
    cx, cy   = w / 2.0, h / 2.0

    # 1. Otsu 二値化
    _, binary = cv2.threshold(red_img, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. モルフォロジー処理
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K,  OPEN_K))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_open)
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    # 3. 連結成分解析
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    candidates = []
    for i in range(1, n_labels):  # 0 は背景
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA_PX:
            continue
        if area > MAX_AREA_FRAC * img_area:
            continue
        px, py = centroids[i]
        dist   = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        # 中心からの距離（画像対角の割合で正規化）
        diag   = np.sqrt(w ** 2 + h ** 2)
        score  = CENTER_WEIGHT / (1.0 + dist / diag) + AREA_BONUS * np.sqrt(area)
        candidates.append((score, i, area))

    if not candidates:
        return {"status": "no_component", "mask": binary,
                "single_mask": None, "ellipse_info": None, "n_candidates": 0}

    # スコア最大の成分を選択
    candidates.sort(key=lambda x: -x[0])
    best_i = candidates[0][1]

    single_mask = np.zeros_like(binary)
    single_mask[labels == best_i] = 255

    # 4. 楕円フィッティング
    contours, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    ellipse_info = None
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) >= 5:
            ell  = cv2.fitEllipse(cnt)
            cx_e, cy_e = ell[0]
            axes        = ell[1]
            angle       = ell[2]
            major = max(axes)
            minor = min(axes)
            ellipse_info = {
                "center_x":    float(cx_e),
                "center_y":    float(cy_e),
                "major_axis":  float(major),
                "minor_axis":  float(minor),
                "angle_deg":   float(angle % 180),
                "ellipse_area": float(np.pi * major/2 * minor/2),
                "ellipse_raw": ell,
            }

    return {
        "status":       "ok",
        "mask":         binary,
        "single_mask":  single_mask,
        "ellipse_info": ellipse_info,
        "n_candidates": len(candidates),
    }


# ── オーバーレイ描画 ────────────────────────────────────────────

def make_overlay(red_img: np.ndarray, result: dict,
                 patient_id: str, filename: str) -> np.ndarray:
    vis = cv2.cvtColor(red_img, cv2.COLOR_GRAY2BGR)

    if result["single_mask"] is not None:
        # マスク領域を半透明青で塗る
        overlay = vis.copy()
        overlay[result["single_mask"] > 0] = (200, 80, 0)
        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

        # 輪郭
        contours, _ = cv2.findContours(result["single_mask"],
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, contours, -1, (255, 100, 0), 1)

    if result["ellipse_info"] is not None:
        e   = result["ellipse_info"]
        ell = e["ellipse_raw"]
        cv2.ellipse(vis, ell, (0, 255, 100), 2)

    # テキスト情報
    status = result["status"]
    lines  = [f"{patient_id}  {filename}",
              f"status: {status}  n_cand: {result['n_candidates']}"]
    if result["ellipse_info"] is not None:
        e = result["ellipse_info"]
        lines.append(
            f"major={e['major_axis']:.1f}  minor={e['minor_axis']:.1f}"
            f"  angle={e['angle_deg']:.1f}"
        )

    # 半透明テキストバー
    bar_h = 18 * len(lines) + 10
    bar   = vis[:bar_h].copy()
    cv2.rectangle(vis, (0, 0), (vis.shape[1], bar_h), (0, 0, 0), -1)
    vis[:bar_h] = cv2.addWeighted(vis[:bar_h], 0.0, bar, 1.0, 0)
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (6, 16 + 18 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, (220, 220, 220), 1, cv2.LINE_AA)

    return vis


# ── メイン ──────────────────────────────────────────────────────

def process_patient(patient_id: str, run_name: str) -> None:
    red_dir = PIPELINE_RUNS / run_name / patient_id / "red"
    if not red_dir.exists():
        print(f"[{patient_id}] red/ not found: {red_dir}")
        return

    out_dir = OUT_ROOT / run_name / patient_id / "overlay"
    ensure_dir(out_dir)

    red_paths = sorted(red_dir.glob("*_red.png"))
    print(f"[{patient_id}] {len(red_paths)} red images")

    rows = []
    for red_path in red_paths:
        # stem 例: "IMG_xxx_red" → 元ファイル名は "IMG_xxx"
        orig_stem = red_path.stem.removesuffix("_red")
        filename  = orig_stem  # 拡張子なし（元画像と対応）

        red_img = cv2.imread(str(red_path), cv2.IMREAD_GRAYSCALE)
        if red_img is None:
            print(f"  read failed: {red_path.name}")
            continue

        result  = classical_segment(red_img)
        overlay = make_overlay(red_img, result, patient_id, filename)

        out_path = out_dir / f"{orig_stem}_mask.png"
        cv2.imwrite(str(out_path), overlay)

        e = result["ellipse_info"]
        rows.append({
            "patient_id":  patient_id,
            "filename":    filename,
            "status":      result["status"],
            "n_candidates":result["n_candidates"],
            "major_axis":  f"{e['major_axis']:.2f}"  if e else "",
            "minor_axis":  f"{e['minor_axis']:.2f}"  if e else "",
            "angle_deg":   f"{e['angle_deg']:.2f}"   if e else "",
        })

    # results.csv
    csv_path = OUT_ROOT / run_name / patient_id / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "patient_id", "filename", "status", "n_candidates",
            "major_axis", "minor_axis", "angle_deg"
        ])
        w.writeheader()
        w.writerows(rows)

    ok  = sum(1 for r in rows if r["status"] == "ok")
    print(f"[{patient_id}] done  ok={ok}/{len(rows)}  -> {out_dir}")


def main():
    args = parse_args()
    for pid in args.patient_ids:
        process_patient(pid, args.run_name)
    print("all done.")


if __name__ == "__main__":
    main()
