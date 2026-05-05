"""
模型眼データの Labelme アノテーション（JSON）から楕円フィッティングを実行するバッチスクリプト。

入力: data/processed/model_eye_runs/<run_name>/*/red/*.json
出力:
  - <run_root>/<folder>/ellipse_results.csv  : 画像ごとの楕円パラメータ
  - <run_root>/<folder>/ellipse_overlay/     : 楕円描画オーバーレイ画像
  - <run_root>/ellipse_summary.csv           : 屈折度別の平均値サマリ
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from src.common.paths import PROCESSED_DIR
from src.ellipse.ellipse_utils import fit_ellipse_from_mask, make_pred_overlay
from src.pipeline.main import ensure_dir

MODEL_EYE_RUNS_DIR = PROCESSED_DIR / "model_eye_runs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="模型眼アノテーション JSON から楕円フィッティングを実行"
    )
    parser.add_argument("--run_name", type=str, required=True, help="例: model_eye_v001")
    parser.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help="処理対象フォルダ名を指定（省略時は全フォルダ）",
    )
    return parser.parse_args()


def json_to_mask(json_path: Path) -> tuple[np.ndarray | None, int, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h = data["imageHeight"]
    w = data["imageWidth"]
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data["shapes"]:
        if shape["label"] != "red_reflex":
            continue
        points = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

    return mask, h, w


def process_folder(folder_run_dir: Path) -> list[dict]:
    red_dir = folder_run_dir / "red"
    overlay_dir = folder_run_dir / "ellipse_overlay"
    ensure_dir(overlay_dir)

    json_files = sorted(red_dir.glob("*.json"))
    folder_name = folder_run_dir.name
    print(f"[{folder_name}] JSON数: {len(json_files)}")

    rows = []

    for json_path in json_files:
        stem = json_path.stem
        img_path = red_dir / (stem + ".png")

        base_img = cv2.imread(str(img_path))
        if base_img is None:
            print(f"  画像読み込み失敗: {img_path.name}")
            rows.append(_row(folder_name, json_path.name, "read_failed", None, None))
            continue

        mask, _, _ = json_to_mask(json_path)

        result = fit_ellipse_from_mask(mask)

        if result["status"] == "ok":
            info = result["ellipse_info"]
            overlay = make_pred_overlay(base_img, result)
            cv2.imwrite(str(overlay_dir / (stem + "_ellipse.png")), overlay)
            print(f"  {json_path.name} -> ok")
        else:
            info = None
            print(f"  {json_path.name} -> {result['status']}")

        rows.append(_row(folder_name, json_path.name, result["status"], info, result.get("mask_area")))

    csv_path = folder_run_dir / "ellipse_results.csv"
    _write_csv(csv_path, rows)
    print(f"  -> {csv_path}")

    return rows


def _row(folder, filename, status, info, mask_area) -> dict:
    base = {
        "folder": folder,
        "filename": filename,
        "status": status,
        "mask_area": f"{mask_area:.1f}" if mask_area is not None else "",
    }
    if info is not None:
        base.update({
            "center_x":    f"{info['center_x']:.2f}",
            "center_y":    f"{info['center_y']:.2f}",
            "major_axis":  f"{info['major_axis']:.2f}",
            "minor_axis":  f"{info['minor_axis']:.2f}",
            "angle_deg":   f"{info['angle_deg']:.2f}",
            "ellipse_area": f"{info['ellipse_area']:.1f}",
        })
    else:
        base.update({k: "" for k in ["center_x", "center_y", "major_axis", "minor_axis", "angle_deg", "ellipse_area"]})
    return base


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(folder_name: str, rows: list[dict]) -> dict:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    n_ok = len(ok_rows)
    n_total = len(rows)

    def mean(key):
        vals = [float(r[key]) for r in ok_rows if r[key] != ""]
        return f"{sum(vals) / len(vals):.2f}" if vals else ""

    return {
        "folder":          folder_name,
        "n_total":         n_total,
        "n_ok":            n_ok,
        "mean_center_x":   mean("center_x"),
        "mean_center_y":   mean("center_y"),
        "mean_major_axis": mean("major_axis"),
        "mean_minor_axis": mean("minor_axis"),
        "mean_angle_deg":  mean("angle_deg"),
        "mean_ellipse_area": mean("ellipse_area"),
        "mean_mask_area":  mean("mask_area"),
    }


def main():
    args = parse_args()
    run_root = MODEL_EYE_RUNS_DIR / args.run_name

    if not run_root.exists():
        raise RuntimeError(f"ランフォルダが見つかりません: {run_root}")

    print("========== MODEL EYE ELLIPSE ==========")
    print(f"RUN_ROOT : {run_root}")
    print("=======================================")

    all_folders = sorted(d for d in run_root.iterdir() if d.is_dir())

    if args.folders:
        target_names = set(args.folders)
        target_folders = [f for f in all_folders if f.name in target_names]
        missing = target_names - {f.name for f in target_folders}
        if missing:
            print(f"[WARNING] 指定フォルダが見つかりません: {missing}")
    else:
        target_folders = [f for f in all_folders if (f / "red").exists()]

    print(f"処理フォルダ数: {len(target_folders)}")

    summary_rows = []
    for folder_dir in target_folders:
        rows = process_folder(folder_dir)
        summary_rows.append(_summary_row(folder_dir.name, rows))

    summary_path = run_root / "ellipse_summary.csv"
    _write_csv(summary_path, summary_rows)

    total_ok = sum(int(r["n_ok"]) for r in summary_rows)
    total_all = sum(int(r["n_total"]) for r in summary_rows)

    print("========== 完了 ==========")
    print(f"総画像数: {total_all}  (楕円OK: {total_ok})")
    print(f"サマリ CSV: {summary_path}")


if __name__ == "__main__":
    main()
