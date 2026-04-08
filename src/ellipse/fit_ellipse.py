import csv
import glob
import shutil
import argparse
from pathlib import Path

import cv2

from src.common.paths import (
    SEGMENTATION_INFERENCE_DIR,
    ELLIPSE_OUTPUTS_DIR,
    REDENHANCE_DIR,
)
from src.ellipse.ellipse_utils import (
    fit_ellipse_from_mask,
    make_pred_overlay,
    add_text_block,
)

EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segmentation_run_name",
        type=str,
        required=True,
        help="example: seginf_v001_on_clf_v001_on_default"
    )
    parser.add_argument(
        "--redenhance_version",
        type=str,
        required=True,
        help="example: default"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="example: ellipse_v001_on_seginf_v001"
    )
    return parser.parse_args()


def parse_original_stem_from_pred_filename(filename: str):
    """
    例:
      IMG_20260326_174724_red_pred.png
      -> IMG_20260326_174724
    """
    stem = Path(filename).stem  # IMG_xxx_red_pred
    if stem.endswith("_red_pred"):
        return stem[:-9]
    if stem.endswith("_pred"):
        return stem[:-5]
    return stem


def get_roi_path(redenhance_root: Path, patient_id: str, pred_filename: str):
    original_stem = parse_original_stem_from_pred_filename(pred_filename)
    roi_filename = f"{original_stem}_roi.png"
    return redenhance_root / patient_id / "roi" / roi_filename


def find_pred_masks(input_root: Path):
    """
    data/processed/segmentation_inference/<run>/pred_binary/<patient_id>/*.png
    """
    results = []

    if not input_root.exists():
        print(f"[ERROR] input folder not found: {input_root}")
        return results

    patient_ids = sorted([p.name for p in input_root.iterdir() if p.is_dir()])

    for patient_id in patient_ids:
        patient_dir = input_root / patient_id

        image_paths = []
        for ext in EXTENSIONS:
            image_paths.extend(glob.glob(str(patient_dir / ext)))

        image_paths = sorted(image_paths)

        if len(image_paths) == 0:
            print(f"[SKIP] no pred masks: {patient_dir}")
            continue

        for path in image_paths:
            results.append((patient_id, Path(path)))

    return results


def main():
    args = parse_args()

    seg_root = SEGMENTATION_INFERENCE_DIR / args.segmentation_run_name
    pred_root = seg_root / "pred_binary"
    input_image_root = seg_root / "input_images"
    redenhance_root = REDENHANCE_DIR / args.redenhance_version

    output_root = ELLIPSE_OUTPUTS_DIR / args.run_name
    single_mask_dir = output_root / "single_component_masks"
    overlay_dir = output_root / "overlay"
    copied_pred_dir = output_root / "copied_pred_masks"

    for d in [single_mask_dir, overlay_dir, copied_pred_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("========== PATH CHECK ==========")
    print("PRED_ROOT       :", pred_root)
    print("INPUT_IMAGE_ROOT:", input_image_root)
    print("REDENHANCE_ROOT :", redenhance_root)
    print("OUTPUT_ROOT     :", output_root)
    print("================================")

    if not pred_root.exists():
        raise RuntimeError(f"pred_binary folder not found: {pred_root}")

    items = find_pred_masks(pred_root)
    print("target pred masks:", len(items))

    if len(items) == 0:
        raise RuntimeError("no pred masks found.")

    rows = []

    for patient_id, pred_path in items:
        print("processing:", pred_path)

        filename = pred_path.name
        base_name = pred_path.stem

        patient_single_dir = single_mask_dir / patient_id
        patient_overlay_dir = overlay_dir / patient_id
        patient_copied_pred_dir = copied_pred_dir / patient_id

        for d in [patient_single_dir, patient_overlay_dir, patient_copied_pred_dir]:
            d.mkdir(parents=True, exist_ok=True)

        shutil.copy2(str(pred_path), str(patient_copied_pred_dir / filename))

        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred_mask is None:
            print("failed to read:", pred_path)
            rows.append([
                patient_id, filename, "read_failed",
                "", "", "", "", "",
                "", "", ""
            ])
            continue

        pred_result = fit_ellipse_from_mask(pred_mask)

        if pred_result["single_mask"] is not None:
            single_save_path = patient_single_dir / f"{base_name}_single.png"
            cv2.imwrite(str(single_save_path), pred_result["single_mask"])
        else:
            single_save_path = None

        roi_path = get_roi_path(redenhance_root, patient_id, filename)
        roi_img = None

        if roi_path.exists():
            roi_img = cv2.imread(str(roi_path))
        else:
            # fallback: segmentation inference input image
            fallback_input_path = input_image_root / patient_id / filename.replace("_pred.png", ".png")
            if fallback_input_path.exists():
                roi_img = cv2.imread(str(fallback_input_path), cv2.IMREAD_GRAYSCALE)

        if roi_img is None:
            h, w = pred_mask.shape[:2]
            roi_img = cv2.cvtColor(
                cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST),
                cv2.COLOR_GRAY2BGR
            )

        overlay = make_pred_overlay(roi_img, pred_result)

        if pred_result["ellipse_info"] is not None:
            e = pred_result["ellipse_info"]
            lines = [
                f"patient_id: {patient_id}",
                f"file: {filename}",
                f"status: {pred_result['status']}",
                f"single_mask_area: {pred_result['mask_area']:.1f}",
                f"ellipse_area: {e['ellipse_area']:.1f}",
                f"major_axis: {e['major_axis']:.1f}",
                f"minor_axis: {e['minor_axis']:.1f}",
                f"angle_deg: {e['angle_deg']:.1f}",
            ]
            overlay = add_text_block(overlay, lines)

            rows.append([
                patient_id,
                filename,
                pred_result["status"],
                f"{pred_result['mask_area']:.2f}",
                f"{e['ellipse_area']:.2f}",
                f"{e['major_axis']:.2f}",
                f"{e['minor_axis']:.2f}",
                f"{e['angle_deg']:.2f}",
                str(pred_path),
                str(single_save_path) if single_save_path else "",
                str(roi_path) if roi_path.exists() else ""
            ])
        else:
            lines = [
                f"patient_id: {patient_id}",
                f"file: {filename}",
                f"status: {pred_result['status']}",
                "ellipse unavailable"
            ]
            overlay = add_text_block(overlay, lines)

            rows.append([
                patient_id,
                filename,
                pred_result["status"],
                f"{pred_result['mask_area']:.2f}" if pred_result["mask_area"] is not None else "",
                "", "", "", "",
                str(pred_path),
                str(single_save_path) if single_save_path else "",
                str(roi_path) if roi_path.exists() else ""
            ])

        overlay_save_path = patient_overlay_dir / f"{base_name}_overlay.png"
        cv2.imwrite(str(overlay_save_path), overlay)

    csv_path = output_root / "ellipse_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patient_id",
            "filename",
            "status",
            "single_mask_area_px",
            "ellipse_area_px",
            "major_axis_px",
            "minor_axis_px",
            "angle_deg",
            "pred_mask_path",
            "single_mask_path",
            "roi_path"
        ])
        writer.writerows(rows)

    print("done.")
    print("csv:", csv_path)
    print("single_component_masks:", single_mask_dir)
    print("overlay:", overlay_dir)


if __name__ == "__main__":
    main()