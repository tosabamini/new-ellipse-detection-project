import csv
from pathlib import Path

import cv2

from src.common.paths import PATIENT_DATA_DIR, REDENHANCE_DIR
from src.preprocessing.preprocess_utils import (
    center_crop,
    get_mean_brightness,
    classify_brightness,
    process_red_by_mode,
    add_debug_overlay,
)

# =========================
# Settings
# =========================
VERSION_NAME = "default"
OUTPUT_ROOT = REDENHANCE_DIR / VERSION_NAME

EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
SAVE_DEBUG_IMAGE = True


def find_patient_image_paths(input_root: Path):
    results = []

    if not input_root.exists():
        print(f"[ERROR] input folder not found: {input_root}")
        return results

    patient_ids = sorted([p.name for p in input_root.iterdir() if p.is_dir()])

    for patient_id in patient_ids:
        patient_dir = input_root / patient_id

        image_paths = sorted([
            p for p in patient_dir.iterdir()
            if p.is_file() and p.suffix in EXTENSIONS
        ])

        if len(image_paths) == 0:
            print(f"[SKIP] no images: {patient_dir}")
            continue

        for path in image_paths:
            results.append((patient_id, path))

    return results


def main():
    items = find_patient_image_paths(PATIENT_DATA_DIR)

    print(f"total images: {len(items)}")

    if len(items) == 0:
        print("no images found.")
        print(f"expected: {PATIENT_DATA_DIR}/<patient_id>/*.jpg")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "red_enhance_log.csv"

    csv_rows = []

    for patient_id, path in items:
        print(f"processing: patient={patient_id}, file={path}")

        img = cv2.imread(str(path))
        if img is None:
            print(f"[WARN] failed to read: {path}")
            continue

        filename = path.name
        stem = path.stem

        patient_out_dir = OUTPUT_ROOT / patient_id
        roi_dir = patient_out_dir / "roi"
        red_dir = patient_out_dir / "red"
        debug_dir = patient_out_dir / "debug"

        roi_dir.mkdir(parents=True, exist_ok=True)
        red_dir.mkdir(parents=True, exist_ok=True)
        if SAVE_DEBUG_IMAGE:
            debug_dir.mkdir(parents=True, exist_ok=True)

        # 1. crop
        roi = center_crop(img)

        # 2. brightness
        brightness = get_mean_brightness(roi)
        mode = classify_brightness(brightness)

        # 3. red enhance
        red_img = process_red_by_mode(roi, mode)

        # 4. save
        roi_save_path = roi_dir / f"{stem}_roi.png"
        red_save_path = red_dir / f"{stem}_red.png"

        cv2.imwrite(str(roi_save_path), roi)
        cv2.imwrite(str(red_save_path), red_img)

        if SAVE_DEBUG_IMAGE:
            debug_lines = [
                f"patient_id: {patient_id}",
                f"file: {filename}",
                f"brightness: {brightness:.2f}",
                f"mode: {mode}"
            ]
            debug_img = add_debug_overlay(red_img, debug_lines)
            debug_save_path = debug_dir / f"{stem}_debug.png"
            cv2.imwrite(str(debug_save_path), debug_img)

        csv_rows.append([
            patient_id,
            filename,
            str(path),
            f"{brightness:.4f}",
            mode,
            str(roi_save_path),
            str(red_save_path)
        ])

        print(f" -> brightness={brightness:.2f}, mode={mode}")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patient_id",
            "filename",
            "original_path",
            "brightness",
            "mode",
            "roi_path",
            "red_path"
        ])
        writer.writerows(csv_rows)

    print("done.")
    print(f"output: {OUTPUT_ROOT}")
    print(f"log csv: {csv_path}")


if __name__ == "__main__":
    main()