import csv
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np

from src.common.paths import (
    SEGMENTATION_MODELS_DIR,
    ELLIPSE_OUTPUTS_DIR,
    REDENHANCE_DIR,
)
from src.ellipse.ellipse_utils import (
    fit_ellipse_from_mask,
    angle_diff_deg,
    add_text_block,
)

TARGET_SPLITS = ["val", "test"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segmentation_train_run_name",
        type=str,
        required=True,
        help="example: seg_v001_on_seg_dataset_v001"
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
        help="example: ellipse_compare_v001"
    )
    return parser.parse_args()


def parse_base_name_from_pred_or_gt(filename: str):
    """
    例:
      01__IMG_20260326_174724_red_gt.png
      01__IMG_20260326_174724_red_pred.png
    -> 01__IMG_20260326_174724_red
    """
    if filename.endswith("_gt.png"):
        return filename[:-7]
    if filename.endswith("_pred.png"):
        return filename[:-9]
    return Path(filename).stem


def get_roi_path(redenhance_root: Path, base_name: str):
    """
    base_name = 01__IMG_20260326_174724_red
    -> patient_id = 01
    -> original_stem = IMG_20260326_174724
    -> ROI path = .../01/roi/IMG_20260326_174724_roi.png
    """
    if "__" not in base_name:
        return None

    patient_id, rest = base_name.split("__", 1)

    if not rest.endswith("_red"):
        return None

    original_stem = rest[:-4]  # remove "_red"
    roi_filename = f"{original_stem}_roi.png"

    return redenhance_root / patient_id / "roi" / roi_filename


def make_color_overlay(base_img, gt_result, pred_result):
    """
    GT  : 緑
    Pred: 赤マスク + マゼンタ楕円
    """
    if len(base_img.shape) == 2:
        canvas = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        canvas = base_img.copy()

    overlay = canvas.copy()

    if gt_result["single_mask"] is not None:
        gt_mask = gt_result["single_mask"] > 0
        overlay[gt_mask] = (0, 180, 0)

    if pred_result["single_mask"] is not None:
        pred_mask = pred_result["single_mask"] > 0
        overlay[pred_mask] = (0, 0, 220)

    canvas = cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0)

    if gt_result["contour"] is not None:
        cv2.drawContours(canvas, [gt_result["contour"]], -1, (0, 255, 0), 1)

    if pred_result["contour"] is not None:
        cv2.drawContours(canvas, [pred_result["contour"]], -1, (0, 0, 255), 1)

    if gt_result["ellipse_info"] is not None:
        cv2.ellipse(canvas, gt_result["ellipse_info"]["raw_ellipse"], (0, 255, 0), 2)
        cx = int(gt_result["ellipse_info"]["center_x"])
        cy = int(gt_result["ellipse_info"]["center_y"])
        cv2.circle(canvas, (cx, cy), 3, (0, 255, 0), -1)

    if pred_result["ellipse_info"] is not None:
        cv2.ellipse(canvas, pred_result["ellipse_info"]["raw_ellipse"], (255, 0, 255), 2)
        cx = int(pred_result["ellipse_info"]["center_x"])
        cy = int(pred_result["ellipse_info"]["center_y"])
        cv2.circle(canvas, (cx, cy), 3, (255, 0, 255), -1)

    return canvas


def make_mask_debug_panel(gt_result, pred_result):
    gt_img = gt_result["single_mask"]
    pred_img = pred_result["single_mask"]

    if gt_img is None and pred_img is None:
        return None

    if gt_img is None:
        gt_img = np.zeros_like(pred_img)
    if pred_img is None:
        pred_img = np.zeros_like(gt_img)

    gt_bgr = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)

    cv2.putText(gt_bgr, "GT single", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(pred_bgr, "PRED single", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return np.hstack([gt_bgr, pred_bgr])


def main():
    args = parse_args()

    seg_run_root = SEGMENTATION_MODELS_DIR / args.segmentation_train_run_name
    redenhance_root = REDENHANCE_DIR / args.redenhance_version
    output_root = ELLIPSE_OUTPUTS_DIR / args.run_name

    overlay_root = output_root / "overlay"
    debug_root = output_root / "mask_debug"
    single_root = output_root / "single_component_masks"

    overlay_root.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)
    single_root.mkdir(parents=True, exist_ok=True)

    print("========== PATH CHECK ==========")
    print("SEG_RUN_ROOT    :", seg_run_root)
    print("REDENHANCE_ROOT :", redenhance_root)
    print("OUTPUT_ROOT     :", output_root)
    print("================================")

    if not seg_run_root.exists():
        raise RuntimeError(f"segmentation run folder not found: {seg_run_root}")

    rows = []

    for split in TARGET_SPLITS:
        input_dir = seg_run_root / f"{split}_preds"
        if not input_dir.exists():
            print(f"[SKIP] split folder not found: {input_dir}")
            continue

        gt_paths = sorted(glob.glob(str(input_dir / "*_gt.png")))
        print(f"[{split}] gt images: {len(gt_paths)}")

        if len(gt_paths) == 0:
            continue

        split_overlay_dir = overlay_root / split
        split_debug_dir = debug_root / split
        split_single_dir = single_root / split

        split_overlay_dir.mkdir(parents=True, exist_ok=True)
        split_debug_dir.mkdir(parents=True, exist_ok=True)
        split_single_dir.mkdir(parents=True, exist_ok=True)

        for gt_path_str in gt_paths:
            gt_path = Path(gt_path_str)
            gt_filename = gt_path.name
            base_name = parse_base_name_from_pred_or_gt(gt_filename)
            pred_path = input_dir / f"{base_name}_pred.png"

            print(f"[{split}] processing: {base_name}")

            if not pred_path.exists():
                print("  pred not found:", pred_path)
                rows.append([
                    split, base_name, "missing_pred",
                    "", "", "", "",
                    "", "", "", "",
                    "", "", "", "",
                    "", "", "", "", ""
                ])
                continue

            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)

            gt_result = fit_ellipse_from_mask(gt_mask)
            pred_result = fit_ellipse_from_mask(pred_mask)

            # save single masks
            if gt_result["single_mask"] is not None:
                cv2.imwrite(str(split_single_dir / f"{base_name}_gt_single.png"), gt_result["single_mask"])
            if pred_result["single_mask"] is not None:
                cv2.imwrite(str(split_single_dir / f"{base_name}_pred_single.png"), pred_result["single_mask"])

            status = "ok"
            if gt_result["status"] != "ok":
                status = f"gt_{gt_result['status']}"
            elif pred_result["status"] != "ok":
                status = f"pred_{pred_result['status']}"

            roi_path = get_roi_path(redenhance_root, base_name)
            if roi_path is not None and roi_path.exists():
                raw_roi = cv2.imread(str(roi_path))
            else:
                if gt_result["single_mask"] is not None:
                    h, w = gt_result["single_mask"].shape[:2]
                elif pred_result["single_mask"] is not None:
                    h, w = pred_result["single_mask"].shape[:2]
                else:
                    h, w = 200, 200
                raw_roi = np.full((h, w, 3), 60, dtype=np.uint8)

            debug_panel = make_mask_debug_panel(gt_result, pred_result)
            if debug_panel is not None:
                cv2.imwrite(str(split_debug_dir / f"{base_name}_mask_debug.png"), debug_panel)

            if gt_result["ellipse_info"] is not None and pred_result["ellipse_info"] is not None:
                gt_e = gt_result["ellipse_info"]
                pred_e = pred_result["ellipse_info"]

                area_abs_err = abs(pred_e["ellipse_area"] - gt_e["ellipse_area"])
                area_rel_err_pct = 100.0 * area_abs_err / max(gt_e["ellipse_area"], 1e-6)

                major_abs_err = abs(pred_e["major_axis"] - gt_e["major_axis"])
                major_rel_err_pct = 100.0 * major_abs_err / max(gt_e["major_axis"], 1e-6)

                minor_abs_err = abs(pred_e["minor_axis"] - gt_e["minor_axis"])
                minor_rel_err_pct = 100.0 * minor_abs_err / max(gt_e["minor_axis"], 1e-6)

                angle_abs_err = angle_diff_deg(pred_e["angle_deg"], gt_e["angle_deg"])

                overlay = make_color_overlay(raw_roi, gt_result, pred_result)
                lines = [
                    f"split={split}  status={status}",
                    f"GT   area={gt_e['ellipse_area']:.1f}  major={gt_e['major_axis']:.1f}  minor={gt_e['minor_axis']:.1f}  angle={gt_e['angle_deg']:.1f}",
                    f"PRED area={pred_e['ellipse_area']:.1f}  major={pred_e['major_axis']:.1f}  minor={pred_e['minor_axis']:.1f}  angle={pred_e['angle_deg']:.1f}",
                    f"ERR  area={area_abs_err:.1f}px^2 ({area_rel_err_pct:.1f}%)  major={major_abs_err:.1f}px ({major_rel_err_pct:.1f}%)",
                    f"ERR  minor={minor_abs_err:.1f}px ({minor_rel_err_pct:.1f}%)  angle={angle_abs_err:.1f}deg",
                    "Green=GT single contour/ellipse, Red mask=Pred single, Magenta ellipse=Pred"
                ]
                overlay = add_text_block(overlay, lines)
                cv2.imwrite(str(split_overlay_dir / f"{base_name}_overlay.png"), overlay)

                rows.append([
                    split,
                    base_name,
                    status,

                    f"{gt_result['mask_area']:.2f}",
                    f"{gt_e['ellipse_area']:.2f}",
                    f"{gt_e['major_axis']:.2f}",
                    f"{gt_e['minor_axis']:.2f}",
                    f"{gt_e['angle_deg']:.2f}",

                    f"{pred_result['mask_area']:.2f}",
                    f"{pred_e['ellipse_area']:.2f}",
                    f"{pred_e['major_axis']:.2f}",
                    f"{pred_e['minor_axis']:.2f}",
                    f"{pred_e['angle_deg']:.2f}",

                    f"{area_abs_err:.2f}",
                    f"{area_rel_err_pct:.2f}",
                    f"{major_abs_err:.2f}",
                    f"{major_rel_err_pct:.2f}",
                    f"{minor_abs_err:.2f}",
                    f"{minor_rel_err_pct:.2f}",
                    f"{angle_abs_err:.2f}",
                ])
            else:
                overlay = make_color_overlay(raw_roi, gt_result, pred_result)
                lines = [
                    f"split={split}  status={status}",
                    f"GT status={gt_result['status']}",
                    f"PRED status={pred_result['status']}",
                    "ellipse compare unavailable"
                ]
                overlay = add_text_block(overlay, lines)
                cv2.imwrite(str(split_overlay_dir / f"{base_name}_overlay.png"), overlay)

                rows.append([
                    split,
                    base_name,
                    status,

                    f"{gt_result['mask_area']:.2f}" if gt_result["mask_area"] is not None else "",
                    "", "", "", "",

                    f"{pred_result['mask_area']:.2f}" if pred_result["mask_area"] is not None else "",
                    "", "", "", "",

                    "", "", "", "", "", "", ""
                ])

    csv_path = output_root / "ellipse_compare_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split",
            "base_name",
            "status",

            "gt_single_mask_area_px",
            "gt_ellipse_area_px",
            "gt_major_axis_px",
            "gt_minor_axis_px",
            "gt_angle_deg",

            "pred_single_mask_area_px",
            "pred_ellipse_area_px",
            "pred_major_axis_px",
            "pred_minor_axis_px",
            "pred_angle_deg",

            "err_area_abs_px2",
            "err_area_rel_pct",
            "err_major_abs_px",
            "err_major_rel_pct",
            "err_minor_abs_px",
            "err_minor_rel_pct",
            "err_angle_abs_deg",
        ])
        writer.writerows(rows)

    print("done.")
    print("csv:", csv_path)
    print("overlay:", overlay_root)
    print("mask_debug:", debug_root)
    print("single_component_masks:", single_root)


if __name__ == "__main__":
    main()