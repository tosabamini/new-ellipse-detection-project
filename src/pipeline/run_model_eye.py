"""
模型眼データに対して RedEnhance + Image quality classification を実行するバッチスクリプト。

処理ロジックは src/pipeline/main.py の関数をそのまま import して使用しているため、
main.py 側の変更は自動的にこのスクリプトにも反映される。

出力先: data/processed/model_eye_runs/<run_name>/
"""

import argparse
import csv
import glob
from datetime import datetime
from pathlib import Path

import cv2

from src.common.paths import DATA_DIR, PROCESSED_DIR
from src.preprocessing.preprocess_utils import (
    center_crop,
    classify_brightness,
    get_mean_brightness,
    process_red_by_mode,
)
from src.pipeline.main import (
    CLASSIFIER_THRESHOLD,
    DEVICE,
    EXTENSIONS,
    ensure_dir,
    load_classifier,
    make_classify_overlay,
    run_classifier_on_red,
)

MODEL_EYE_DIR = DATA_DIR / "model_eye"
MODEL_EYE_RUNS_DIR = PROCESSED_DIR / "model_eye_runs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="模型眼データへの RedEnhance + 分類バッチ処理"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="出力フォルダ名（省略時は実行日時から自動生成）",
    )
    parser.add_argument(
        "--pupil_mm",
        type=str,
        default="7.0mm",
        help="瞳孔径フォルダ名（例: 7.0mm, 5.0mm, 3.0mm）",
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help="処理対象フォルダ名を指定（省略時は全フォルダ）。例: --folders 1600_Z_00_00D 1625_P_00_25D",
    )
    return parser.parse_args()


def find_images(folder: Path) -> list[Path]:
    paths = []
    for ext in EXTENSIONS:
        paths.extend(glob.glob(str(folder / ext)))
    return sorted(Path(p) for p in paths)


def process_folder(folder: Path, run_root: Path, classifier_model, pupil_mm: str) -> list[dict]:
    images = find_images(folder / pupil_mm)
    folder_name = folder.name
    print(f"[{folder_name}] 画像数: {len(images)}")

    if not images:
        return []

    out_root = run_root / folder_name
    roi_dir = out_root / "roi"
    red_dir = out_root / "red"
    classify_overlay_dir = out_root / "classify_overlay"
    for d in [roi_dir, red_dir, classify_overlay_dir]:
        ensure_dir(d)

    rows = []

    for path in images:
        filename = path.name
        stem = path.stem
        print(f"  {filename}")

        img = cv2.imread(str(path))
        if img is None:
            print(f"  -> read failed")
            rows.append(
                {
                    "folder": folder_name,
                    "filename": filename,
                    "brightness": "",
                    "mode": "",
                    "class_positive_prob": "",
                    "class_pred_label": "",
                    "status": "read_failed",
                }
            )
            continue

        # ---- RedEnhance ----
        roi = center_crop(img)
        brightness = get_mean_brightness(roi)
        mode = classify_brightness(brightness)
        red_img = process_red_by_mode(roi, mode)

        cv2.imwrite(str(roi_dir / f"{stem}_roi.png"), roi)
        cv2.imwrite(str(red_dir / f"{stem}_red.png"), red_img)

        # ---- Image quality classification ----
        class_prob, class_pred = run_classifier_on_red(red_img, classifier_model)

        overlay = make_classify_overlay(
            red_img=red_img,
            patient_id=folder_name,
            filename=filename,
            brightness=brightness,
            mode=mode,
            class_prob=class_prob,
            class_pred=class_pred,
        )
        cv2.imwrite(str(classify_overlay_dir / f"{stem}_classify.png"), overlay)

        rows.append(
            {
                "folder": folder_name,
                "filename": filename,
                "brightness": f"{brightness:.2f}",
                "mode": mode,
                "class_positive_prob": f"{class_prob:.6f}",
                "class_pred_label": class_pred,
                "status": "class_positive" if class_pred == 1 else "class_negative",
            }
        )

    # フォルダ単位の CSV
    csv_path = out_root / "results.csv"
    _write_csv(csv_path, rows)
    print(f"  -> {csv_path}")

    return rows


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_root = MODEL_EYE_RUNS_DIR / run_name
    ensure_dir(run_root)

    print("========== MODEL EYE BATCH ==========")
    print(f"MODEL_EYE_DIR : {MODEL_EYE_DIR}")
    print(f"PUPIL         : {args.pupil_mm}")
    print(f"RUN_ROOT      : {run_root}")
    print(f"DEVICE        : {DEVICE}")
    print(f"THRESHOLD     : {CLASSIFIER_THRESHOLD}")
    print("=====================================")

    if not MODEL_EYE_DIR.exists():
        raise RuntimeError(f"模型眼フォルダが見つかりません: {MODEL_EYE_DIR}")

    all_folders = sorted(d for d in MODEL_EYE_DIR.iterdir() if d.is_dir())
    if not all_folders:
        print("処理対象フォルダが存在しません。")
        return

    if args.folders:
        target_names = set(args.folders)
        target_folders = [f for f in all_folders if f.name in target_names]
        missing = target_names - {f.name for f in target_folders}
        if missing:
            print(f"[WARNING] 指定フォルダが見つかりません: {missing}")
    else:
        target_folders = all_folders

    print(f"処理フォルダ数: {len(target_folders)}")

    classifier_model = load_classifier()

    all_rows = []
    for folder in target_folders:
        rows = process_folder(folder, run_root, classifier_model, args.pupil_mm)
        all_rows.extend(rows)

    # 全体集計 CSV
    all_csv_path = run_root / "all_results.csv"
    _write_csv(all_csv_path, all_rows)

    total = len(all_rows)
    positive = sum(1 for r in all_rows if r["status"] == "class_positive")
    negative = sum(1 for r in all_rows if r["status"] == "class_negative")
    failed = sum(1 for r in all_rows if r["status"] == "read_failed")

    print("========== 完了 ==========")
    print(f"総画像数  : {total}")
    print(f"  positive: {positive}")
    print(f"  negative: {negative}")
    print(f"  failed  : {failed}")
    print(f"出力先    : {run_root}")
    print(f"集計CSV   : {all_csv_path}")


if __name__ == "__main__":
    main()
