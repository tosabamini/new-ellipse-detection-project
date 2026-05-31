"""
Simulation pipeline — Labelme JSON → ellipse fitting.

Labelme で roi/ フォルダをアノテーション後に実行する。
label = "red_reflex" のポリゴンをマスクに変換し、
fit_ellipse_from_mask で楕円パラメータを取得する。

入力:
  data/processed/simulation_runs/<run_name>/<pupil_group>/roi/*.json
  （Labelme が roi/*.png と同じ場所に生成する JSON）

出力 (各 pupil_group フォルダ内):
  ellipse_label/           楕円オーバーレイ画像
  per_image_label.csv      stem, major, minor, ratio, angle, mask_area, status

Run:
  # p10 だけ（動作確認用）
  python -m src.pipeline.simulation_ellipse_from_json --run_name sim_run01 --pupil_group p10

  # 全グループ一括
  python -m src.pipeline.simulation_ellipse_from_json --run_name sim_run01
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from src.common.paths import SIMULATION_RUNS_DIR
from src.ellipse.ellipse_utils import fit_ellipse_from_mask, make_pred_overlay


# ── JSON → mask ───────────────────────────────────────────────────────────────

def json_to_mask(json_path: Path) -> np.ndarray | None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h = data.get("imageHeight")
    w = data.get("imageWidth")
    if h is None or w is None:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in data["shapes"]:
        if shape["label"] != "red_reflex":
            continue
        points = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

    return mask


# ── per-group processing ──────────────────────────────────────────────────────

def process_group(run_name: str, pupil_group: str) -> list[dict]:
    group_dir = SIMULATION_RUNS_DIR / run_name / pupil_group
    roi_dir   = group_dir / "roi"
    out_dir   = group_dir / "ellipse_label"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(roi_dir.glob("*.json"))
    if not json_files:
        print(f"  [{pupil_group}] JSON なし — Labelme でアノテーション未完了")
        return []

    print(f"  [{pupil_group}] {len(json_files)} JSON")

    rows = []
    for json_path in json_files:
        stem     = json_path.stem          # e.g. "camera_p10_D000_roi"
        img_path = roi_dir / (stem + ".png")

        base_img = cv2.imread(str(img_path))
        if base_img is None:
            print(f"    WARNING: 画像読み込み失敗 {img_path.name}")
            rows.append(_row(stem, "read_failed", None, None))
            continue

        mask   = json_to_mask(json_path)
        result = fit_ellipse_from_mask(mask)

        if result["status"] == "ok":
            overlay = make_pred_overlay(base_img, result)
            cv2.imwrite(str(out_dir / (stem + "_ellipse.png")), overlay)

        rows.append(_row(stem, result["status"], result.get("ellipse_info"),
                         result.get("mask_area")))

    # per_image_label.csv
    csv_path = group_dir / "per_image_label.csv"
    _write_csv(csv_path, rows)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"  [{pupil_group}] 楕円OK: {n_ok}/{len(rows)}  → {csv_path.name}")
    return rows


def _row(stem: str, status: str, info: dict | None, mask_area) -> dict:
    base = {
        "stem":      stem,
        "status":    status,
        "mask_area": f"{mask_area:.1f}" if mask_area is not None else "",
    }
    if info is not None:
        base.update({
            "major":  f"{info['major_axis']:.2f}",
            "minor":  f"{info['minor_axis']:.2f}",
            "ratio":  f"{info['minor_axis'] / info['major_axis']:.4f}",
            "angle":  f"{info['angle_deg']:.2f}",
        })
    else:
        base.update({"major": "", "minor": "", "ratio": "", "angle": ""})
    return base


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = ["stem", "status", "major", "minor", "ratio", "angle", "mask_area"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Simulation: Labelme JSON → ellipse fitting"
    )
    p.add_argument("--run_name",     required=True, help="例: sim_run01")
    p.add_argument("--pupil_group",  default=None,
                   help="単一グループ指定 (例: p10)。省略時は全グループ")
    return p.parse_args()


def main():
    args    = parse_args()
    run_dir = SIMULATION_RUNS_DIR / args.run_name

    if not run_dir.exists():
        raise FileNotFoundError(f"run フォルダが見つかりません: {run_dir}")

    if args.pupil_group:
        groups = [args.pupil_group]
    else:
        groups = sorted(d.name for d in run_dir.iterdir()
                        if d.is_dir() and d.name.startswith("p"))

    print(f"Run  : {args.run_name}")
    print(f"Groups: {groups}\n")

    all_rows = {}
    for group in groups:
        all_rows[group] = process_group(args.run_name, group)

    # サマリ表示
    print(f"\n{'='*40}")
    print(f"{'Group':<10}  {'ok':>4}  {'total':>5}")
    print("-" * 25)
    for group, rows in all_rows.items():
        n_ok = sum(1 for r in rows if r["status"] == "ok")
        print(f"{group:<10}  {n_ok:>4}  {len(rows):>5}")


if __name__ == "__main__":
    main()
