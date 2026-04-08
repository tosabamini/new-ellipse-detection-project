import os
import csv
import glob
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.common.paths import (
    REDENHANCE_DIR,
    CLASSIFY_OUTPUTS_DIR,
    BEST_CLASSIFIER_MODEL_PATH,
)
from src.classify.classifier_model import SmallClassifier

# =========================
# Settings
# =========================
EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]

IMG_W = 160
IMG_H = 72

THRESHOLD = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--redenhance_version",
        type=str,
        required=True,
        help="example: default or red_v002"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="example: clf_v001_on_default"
    )
    return parser.parse_args()


def find_red_images(input_root: Path):
    """
    data/processed/redenhance/<version>/<patient_id>/red/*.png
    """
    results = []

    if not input_root.exists():
        print(f"[ERROR] input folder not found: {input_root}")
        return results

    patient_ids = sorted([p.name for p in input_root.iterdir() if p.is_dir()])

    for patient_id in patient_ids:
        red_dir = input_root / patient_id / "red"

        if not red_dir.exists():
            print(f"[SKIP] red folder not found: {red_dir}")
            continue

        image_paths = []
        for ext in EXTENSIONS:
            image_paths.extend(glob.glob(str(red_dir / ext)))

        image_paths = sorted(image_paths)

        if len(image_paths) == 0:
            print(f"[SKIP] no red images: {red_dir}")
            continue

        for path in image_paths:
            results.append((patient_id, Path(path)))

    return results


def add_result_overlay(gray_img, lines, color):
    out = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    overlay = out.copy()
    cv2.rectangle(overlay, (5, 5), (430, 130), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)

    y = 25
    for line in lines:
        cv2.putText(
            out,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA
        )
        y += 20

    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, 2)

    return out


def main():
    args = parse_args()

    input_root = REDENHANCE_DIR / args.redenhance_version
    output_root = CLASSIFY_OUTPUTS_DIR / args.run_name

    all_results_dir = output_root / "all_results"
    positive_dir = output_root / "positive_for_mask"
    negative_dir = output_root / "negative"

    all_results_dir.mkdir(parents=True, exist_ok=True)
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    print("========== PATH CHECK ==========")
    print("INPUT_ROOT   :", input_root)
    print("MODEL_PATH   :", BEST_CLASSIFIER_MODEL_PATH)
    print("OUTPUT_ROOT  :", output_root)
    print("MODEL_EXISTS :", BEST_CLASSIFIER_MODEL_PATH.exists())
    print("================================")

    if not BEST_CLASSIFIER_MODEL_PATH.exists():
        raise RuntimeError(f"classifier model not found: {BEST_CLASSIFIER_MODEL_PATH}")

    model = SmallClassifier().to(DEVICE)
    model.load_state_dict(torch.load(BEST_CLASSIFIER_MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("model loaded:", BEST_CLASSIFIER_MODEL_PATH)
    print("device:", DEVICE)

    items = find_red_images(input_root)
    print("target images:", len(items))

    if len(items) == 0:
        raise RuntimeError("no input images found.")

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

            inp = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
            inp = inp.astype(np.float32) / 255.0
            inp = np.expand_dims(inp, axis=0)
            inp = np.expand_dims(inp, axis=0)
            inp_tensor = torch.tensor(inp, dtype=torch.float32).to(DEVICE)

            logit = model(inp_tensor)
            prob = torch.sigmoid(logit).item()
            pred = 1 if prob >= THRESHOLD else 0

            if pred == 1:
                result_text = "PRED: POSITIVE"
                color = (0, 255, 0)
                pred_str = "positive"
            else:
                result_text = "PRED: NEGATIVE"
                color = (0, 0, 255)
                pred_str = "negative"

            result_img = add_result_overlay(
                img,
                [
                    f"patient_id: {patient_id}",
                    result_text,
                    f"Prob(positive): {prob:.4f}"
                ],
                color
            )

            patient_all_dir = all_results_dir / patient_id
            patient_pos_dir = positive_dir / patient_id
            patient_neg_dir = negative_dir / patient_id

            patient_all_dir.mkdir(parents=True, exist_ok=True)
            patient_pos_dir.mkdir(parents=True, exist_ok=True)
            patient_neg_dir.mkdir(parents=True, exist_ok=True)

            result_save_path = patient_all_dir / f"{base_name}_result.png"
            cv2.imwrite(str(result_save_path), result_img)

            if pred == 1:
                copy_save_path = patient_pos_dir / filename
            else:
                copy_save_path = patient_neg_dir / filename

            shutil.copy2(str(path), str(copy_save_path))

            rows.append([
                patient_id,
                filename,
                str(path),
                f"{prob:.6f}",
                pred,
                pred_str,
                str(result_save_path),
                str(copy_save_path)
            ])

    csv_path = output_root / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patient_id",
            "filename",
            "input_path",
            "positive_prob",
            "pred_label",
            "pred_name",
            "result_image_path",
            "copied_image_path"
        ])
        writer.writerows(rows)

    print("done.")
    print("csv:", csv_path)
    print("positive:", positive_dir)
    print("negative:", negative_dir)


if __name__ == "__main__":
    main()