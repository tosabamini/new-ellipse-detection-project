import csv
import glob
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.common.paths import (
    CLASSIFY_OUTPUTS_DIR,
    SEGMENTATION_INFERENCE_DIR,
    BEST_SEGMENTATION_MODEL_PATH,
)
from src.segmentation.segmentation_model import UNetSmall

# =========================
# Settings
# =========================
EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]

THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classify_run_name",
        type=str,
        required=True,
        help="example: clf_v001_on_default"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="example: seginf_v001_on_clf_v001_on_default"
    )
    return parser.parse_args()


def find_positive_images(input_root: Path):
    """
    data/processed/classify_outputs/<run_name>/positive_for_mask/<patient_id>/*.png
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
            print(f"[SKIP] no images: {patient_dir}")
            continue

        for path in image_paths:
            results.append((patient_id, Path(path)))

    return results


def make_overlay(gray_img, prob_map, pred_bin):
    """
    gray_img : 1ch input image
    prob_map : float [0,1]
    pred_bin : uint8 0/255
    """
    base = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    overlay = base.copy()

    # 予測領域を赤系で重ねる
    pred_mask = pred_bin > 0
    overlay[pred_mask] = (0, 0, 220)

    vis = cv2.addWeighted(overlay, 0.28, base, 0.72, 0)

    # 輪郭描画
    contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        cv2.drawContours(vis, contours, -1, (255, 0, 255), 1)

    # 右上にprobヒートマップも小さく出したい場合は後で拡張可
    return vis


def add_text_block(img, lines, start=(10, 20), line_h=18):
    out = img.copy()
    x, y = start

    bg = out.copy()
    block_h = line_h * len(lines) + 10
    block_w = 520
    cv2.rectangle(bg, (5, 5), (5 + block_w, 5 + block_h), (0, 0, 0), -1)
    out = cv2.addWeighted(bg, 0.45, out, 0.55, 0)

    yy = y
    for line in lines:
        cv2.putText(
            out,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        yy += line_h

    return out


def main():
    args = parse_args()

    input_root = CLASSIFY_OUTPUTS_DIR / args.classify_run_name / "positive_for_mask"
    output_root = SEGMENTATION_INFERENCE_DIR / args.run_name

    raw_prob_dir = output_root / "raw_prob"
    pred_binary_dir = output_root / "pred_binary"
    overlay_dir = output_root / "overlay"
    copied_input_dir = output_root / "input_images"

    for d in [raw_prob_dir, pred_binary_dir, overlay_dir, copied_input_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("========== PATH CHECK ==========")
    print("INPUT_ROOT   :", input_root)
    print("MODEL_PATH   :", BEST_SEGMENTATION_MODEL_PATH)
    print("OUTPUT_ROOT  :", output_root)
    print("MODEL_EXISTS :", BEST_SEGMENTATION_MODEL_PATH.exists())
    print("================================")

    if not BEST_SEGMENTATION_MODEL_PATH.exists():
        raise RuntimeError(f"segmentation model not found: {BEST_SEGMENTATION_MODEL_PATH}")

    model = UNetSmall().to(DEVICE)
    model.load_state_dict(torch.load(BEST_SEGMENTATION_MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("model loaded:", BEST_SEGMENTATION_MODEL_PATH)
    print("device:", DEVICE)

    items = find_positive_images(input_root)
    print("target images:", len(items))

    if len(items) == 0:
        raise RuntimeError("no positive input images found.")

    rows = []

    with torch.no_grad():
        for patient_id, path in items:
            print("processing:", path)

            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("read failed:", path)
                continue

            filename = path.name
            base_name = path.stem

            # 保存用 patient dir
            patient_prob_dir = raw_prob_dir / patient_id
            patient_pred_dir = pred_binary_dir / patient_id
            patient_overlay_dir = overlay_dir / patient_id
            patient_input_dir = copied_input_dir / patient_id

            for d in [patient_prob_dir, patient_pred_dir, patient_overlay_dir, patient_input_dir]:
                d.mkdir(parents=True, exist_ok=True)

            # 入力画像コピー
            copied_input_path = patient_input_dir / filename
            shutil.copy2(str(path), str(copied_input_path))

            # 推論
            inp = img.astype(np.float32) / 255.0
            inp = np.expand_dims(inp, axis=0)   # channel
            inp = np.expand_dims(inp, axis=0)   # batch
            inp_tensor = torch.tensor(inp, dtype=torch.float32).to(DEVICE)

            pred = model(inp_tensor)[0, 0].cpu().numpy()
            pred_bin = (pred > THRESHOLD).astype(np.uint8) * 255

            # 予測面積
            pred_area_px = int((pred_bin > 0).sum())

            # raw probability map を 0-255 保存
            pred_prob_u8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)

            prob_save_path = patient_prob_dir / f"{base_name}_prob.png"
            pred_save_path = patient_pred_dir / f"{base_name}_pred.png"

            cv2.imwrite(str(prob_save_path), pred_prob_u8)
            cv2.imwrite(str(pred_save_path), pred_bin)

            overlay = make_overlay(img, pred, pred_bin)
            overlay = add_text_block(
                overlay,
                [
                    f"patient_id: {patient_id}",
                    f"file: {filename}",
                    f"threshold: {THRESHOLD:.2f}",
                    f"pred_area_px: {pred_area_px}"
                ]
            )

            overlay_save_path = patient_overlay_dir / f"{base_name}_overlay.png"
            cv2.imwrite(str(overlay_save_path), overlay)

            rows.append([
                patient_id,
                filename,
                str(path),
                str(copied_input_path),
                str(prob_save_path),
                str(pred_save_path),
                str(overlay_save_path),
                THRESHOLD,
                pred_area_px
            ])

    csv_path = output_root / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patient_id",
            "filename",
            "input_path",
            "copied_input_path",
            "prob_map_path",
            "pred_binary_path",
            "overlay_path",
            "threshold",
            "pred_area_px"
        ])
        writer.writerows(rows)

    print("done.")
    print("csv:", csv_path)
    print("pred_binary:", pred_binary_dir)
    print("overlay:", overlay_dir)


if __name__ == "__main__":
    main()