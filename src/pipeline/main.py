import csv
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.common.paths import (
    PATIENT_DATA_DIR,
    BEST_CLASSIFIER_MODEL_PATH,
    BEST_SEGMENTATION_MODEL_PATH,
    PROCESSED_DIR,
)
from src.preprocessing.preprocess_utils import (
    center_crop,
    get_mean_brightness,
    classify_brightness,
    process_red_by_mode,
)
from src.classify.classifier_model import SmallClassifier
from src.segmentation.segmentation_model import UNetSmall
from src.ellipse.ellipse_utils import (
    fit_ellipse_from_mask,
    make_pred_overlay,
    add_text_block,
)

# =========================
# Settings
# =========================
EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

CLASSIFIER_IMG_W = 160
CLASSIFIER_IMG_H = 72
CLASSIFIER_THRESHOLD = 0.9

SEG_THRESHOLD = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patient_ids",
        nargs="+",
        required=True,
        help="example: --patient_ids 01 02 14"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="example: pipeline_run_v001"
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def find_patient_image_paths(patient_id: str):
    patient_dir = PATIENT_DATA_DIR / patient_id
    if not patient_dir.exists():
        raise RuntimeError(f"patient folder not found: {patient_dir}")

    image_paths = []
    for ext in EXTENSIONS:
        image_paths.extend(glob.glob(str(patient_dir / ext)))

    image_paths = sorted(image_paths)
    return [Path(p) for p in image_paths]


def load_classifier():
    if not BEST_CLASSIFIER_MODEL_PATH.exists():
        raise RuntimeError(f"classifier model not found: {BEST_CLASSIFIER_MODEL_PATH}")

    model = SmallClassifier().to(DEVICE)
    model.load_state_dict(torch.load(BEST_CLASSIFIER_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def load_segmentation_model():
    if not BEST_SEGMENTATION_MODEL_PATH.exists():
        raise RuntimeError(f"segmentation model not found: {BEST_SEGMENTATION_MODEL_PATH}")

    model = UNetSmall().to(DEVICE)
    model.load_state_dict(torch.load(BEST_SEGMENTATION_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def run_classifier_on_red(red_img, classifier_model):
    inp = cv2.resize(red_img, (CLASSIFIER_IMG_W, CLASSIFIER_IMG_H), interpolation=cv2.INTER_AREA)
    inp = inp.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)  # channel
    inp = np.expand_dims(inp, axis=0)  # batch
    inp_tensor = torch.tensor(inp, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logit = classifier_model(inp_tensor)
        prob = torch.sigmoid(logit).item()

    pred_label = 1 if prob >= CLASSIFIER_THRESHOLD else 0
    return prob, pred_label


def run_segmentation_on_red(red_img, segmentation_model):
    inp = red_img.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)  # channel
    inp = np.expand_dims(inp, axis=0)  # batch
    inp_tensor = torch.tensor(inp, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred = segmentation_model(inp_tensor)[0, 0].cpu().numpy()

    pred_bin = (pred > SEG_THRESHOLD).astype(np.uint8) * 255
    return pred, pred_bin


def make_classify_overlay(red_img, patient_id, filename, brightness, mode, class_prob, class_pred):
    cls_vis = cv2.cvtColor(red_img, cv2.COLOR_GRAY2BGR)
    lines = [
        f"patient_id: {patient_id}",
        f"file: {filename}",
        f"brightness: {brightness:.2f}",
        f"mode: {mode}",
        f"class_prob: {class_prob:.4f}",
        f"class_pred: {'positive' if class_pred == 1 else 'negative'}",
    ]
    cls_vis = add_text_block(cls_vis, lines)
    return cls_vis


def make_seg_overlay(red_img, patient_id, filename, pred_bin):
    base = cv2.cvtColor(red_img, cv2.COLOR_GRAY2BGR)

    overlay = base.copy()
    pred_mask = pred_bin > 0
    overlay[pred_mask] = (0, 0, 220)
    vis = cv2.addWeighted(overlay, 0.28, base, 0.72, 0)

    contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        cv2.drawContours(vis, contours, -1, (255, 0, 255), 1)

    pred_area_px = int((pred_bin > 0).sum())
    lines = [
        f"patient_id: {patient_id}",
        f"file: {filename}",
        f"seg_threshold: {SEG_THRESHOLD:.2f}",
        f"pred_area_px: {pred_area_px}",
    ]
    vis = add_text_block(vis, lines)
    return vis


def make_ellipse_overlay(roi_img, patient_id, filename, class_prob, pred_result):
    overlay = make_pred_overlay(roi_img, pred_result)

    if pred_result["ellipse_info"] is not None:
        e = pred_result["ellipse_info"]
        lines = [
            f"patient_id: {patient_id}",
            f"file: {filename}",
            f"class_prob: {class_prob:.4f}",
            f"ellipse_status: {pred_result['status']}",
            f"single_mask_area: {pred_result['mask_area']:.1f}",
            f"ellipse_area: {e['ellipse_area']:.1f}",
            f"major_axis: {e['major_axis']:.1f}",
            f"minor_axis: {e['minor_axis']:.1f}",
            f"angle_deg: {e['angle_deg']:.1f}",
        ]
    else:
        lines = [
            f"patient_id: {patient_id}",
            f"file: {filename}",
            f"class_prob: {class_prob:.4f}",
            f"ellipse_status: {pred_result['status']}",
            "ellipse unavailable",
        ]

    overlay = add_text_block(overlay, lines)
    return overlay


def process_patient(patient_id, run_root, classifier_model, segmentation_model):
    image_paths = find_patient_image_paths(patient_id)
    print(f"[{patient_id}] image count: {len(image_paths)}")

    if len(image_paths) == 0:
        print(f"[{patient_id}] no images found")
        return

    patient_root = run_root / patient_id
    roi_dir = patient_root / "roi"
    red_dir = patient_root / "red"
    classify_overlay_dir = patient_root / "classify_overlay"
    seg_prob_dir = patient_root / "seg_prob"
    seg_pred_dir = patient_root / "seg_pred"
    seg_overlay_dir = patient_root / "seg_overlay"
    ellipse_single_dir = patient_root / "ellipse_single_mask"
    ellipse_overlay_dir = patient_root / "ellipse_overlay"

    for d in [
        roi_dir,
        red_dir,
        classify_overlay_dir,
        seg_prob_dir,
        seg_pred_dir,
        seg_overlay_dir,
        ellipse_single_dir,
        ellipse_overlay_dir,
    ]:
        ensure_dir(d)

    rows = []

    for path in image_paths:
        filename = path.name
        stem = path.stem

        print(f"[{patient_id}] processing: {filename}")

        img = cv2.imread(str(path))
        if img is None:
            print("  read failed")
            rows.append([
                patient_id, filename, "", "", "", "", "", "", "", "", "", "read_failed"
            ])
            continue

        # -------------------------------------------------
        # 1. preprocess
        # -------------------------------------------------
        roi = center_crop(img)
        brightness = get_mean_brightness(roi)
        mode = classify_brightness(brightness)
        red_img = process_red_by_mode(roi, mode)

        roi_path = roi_dir / f"{stem}_roi.png"
        red_path = red_dir / f"{stem}_red.png"

        cv2.imwrite(str(roi_path), roi)
        cv2.imwrite(str(red_path), red_img)

        # -------------------------------------------------
        # 2. classify
        # -------------------------------------------------
        class_prob, class_pred = run_classifier_on_red(red_img, classifier_model)

        classify_overlay = make_classify_overlay(
            red_img=red_img,
            patient_id=patient_id,
            filename=filename,
            brightness=brightness,
            mode=mode,
            class_prob=class_prob,
            class_pred=class_pred
        )
        classify_overlay_path = classify_overlay_dir / f"{stem}_classify.png"
        cv2.imwrite(str(classify_overlay_path), classify_overlay)

        # negative の場合はここで終了
        if class_pred == 0:
            rows.append([
                patient_id,
                filename,
                f"{brightness:.2f}",
                mode,
                f"{class_prob:.6f}",
                class_pred,
                "",
                "",
                "",
                "",
                "",
                "class_negative"
            ])
            continue

        # -------------------------------------------------
        # 3. segmentation
        # -------------------------------------------------
        seg_prob, seg_pred_bin = run_segmentation_on_red(red_img, segmentation_model)

        seg_prob_u8 = np.clip(seg_prob * 255.0, 0, 255).astype(np.uint8)

        seg_prob_path = seg_prob_dir / f"{stem}_prob.png"
        seg_pred_path = seg_pred_dir / f"{stem}_pred.png"

        cv2.imwrite(str(seg_prob_path), seg_prob_u8)
        cv2.imwrite(str(seg_pred_path), seg_pred_bin)

        seg_overlay = make_seg_overlay(
            red_img=red_img,
            patient_id=patient_id,
            filename=filename,
            pred_bin=seg_pred_bin
        )
        seg_overlay_path = seg_overlay_dir / f"{stem}_seg.png"
        cv2.imwrite(str(seg_overlay_path), seg_overlay)

        # -------------------------------------------------
        # 4. ellipse fitting
        # -------------------------------------------------
        pred_result = fit_ellipse_from_mask(seg_pred_bin)

        if pred_result["single_mask"] is not None:
            ellipse_single_path = ellipse_single_dir / f"{stem}_single.png"
            cv2.imwrite(str(ellipse_single_path), pred_result["single_mask"])
        else:
            ellipse_single_path = None

        ellipse_overlay = make_ellipse_overlay(
            roi_img=roi,
            patient_id=patient_id,
            filename=filename,
            class_prob=class_prob,
            pred_result=pred_result
        )
        ellipse_overlay_path = ellipse_overlay_dir / f"{stem}_ellipse.png"
        cv2.imwrite(str(ellipse_overlay_path), ellipse_overlay)

        if pred_result["ellipse_info"] is not None:
            e = pred_result["ellipse_info"]
            rows.append([
                patient_id,
                filename,
                f"{brightness:.2f}",
                mode,
                f"{class_prob:.6f}",
                class_pred,
                f"{pred_result['mask_area']:.2f}",
                f"{e['ellipse_area']:.2f}",
                f"{e['major_axis']:.2f}",
                f"{e['minor_axis']:.2f}",
                f"{e['angle_deg']:.2f}",
                pred_result["status"]
            ])
        else:
            rows.append([
                patient_id,
                filename,
                f"{brightness:.2f}",
                mode,
                f"{class_prob:.6f}",
                class_pred,
                f"{pred_result['mask_area']:.2f}" if pred_result["mask_area"] is not None else "",
                "",
                "",
                "",
                "",
                pred_result["status"]
            ])

    csv_path = patient_root / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patient_id",
            "filename",
            "brightness",
            "mode",
            "class_positive_prob",
            "class_pred_label",
            "pred_single_mask_area_px",
            "pred_ellipse_area_px",
            "pred_major_axis_px",
            "pred_minor_axis_px",
            "pred_angle_deg",
            "status"
        ])
        writer.writerows(rows)

    print(f"[{patient_id}] done -> {csv_path}")

    # -------------------------------------------------
    # 5. refraction estimation (S, C, A)
    # -------------------------------------------------
    from src.analysis.refraction_estimator import (
        run_refraction_analysis,
        write_per_image_csv,
        write_sca_csv,
    )
    refraction_result = run_refraction_analysis(csv_path, patient_id)
    write_per_image_csv(
        refraction_result["per_image"],
        patient_root / "refraction_per_image.csv",
    )
    write_sca_csv(patient_id, refraction_result, patient_root / "refraction_sca.csv")

    sca = refraction_result["sca"]
    if sca:
        print(
            f"[{patient_id}] SCA: S={sca['S']:+.2f}D  C={sca['C']:+.2f}D  "
            f"A={sca['A']:.1f}deg  SE={sca['SE']:+.2f}D  "
            f"R2={sca['R2']:.3f}  n={sca['n']}/{refraction_result['n_total']}"
        )
    else:
        print(
            f"[{patient_id}] SCA: insufficient valid images "
            f"(n_valid={refraction_result['n_valid']}/{refraction_result['n_total']})"
        )


def main():
    args = parse_args()

    run_root = PROCESSED_DIR / "pipeline_runs" / args.run_name
    ensure_dir(run_root)

    print("========== PATH CHECK ==========")
    print("PATIENT_DATA_DIR            :", PATIENT_DATA_DIR)
    print("BEST_CLASSIFIER_MODEL_PATH  :", BEST_CLASSIFIER_MODEL_PATH)
    print("BEST_SEGMENTATION_MODEL_PATH:", BEST_SEGMENTATION_MODEL_PATH)
    print("RUN_ROOT                    :", run_root)
    print("DEVICE                      :", DEVICE)
    print("================================")

    classifier_model = load_classifier()
    segmentation_model = load_segmentation_model()

    for patient_id in args.patient_ids:
        process_patient(
            patient_id=patient_id,
            run_root=run_root,
            classifier_model=classifier_model,
            segmentation_model=segmentation_model
        )

    print("all done.")
    print("run output:", run_root)


if __name__ == "__main__":
    main()