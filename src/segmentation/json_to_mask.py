import json
import argparse
from pathlib import Path

import numpy as np
import cv2

from src.common.paths import SEGMENTATION_DATASET_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="example: seg_dataset_v001"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = SEGMENTATION_DATASET_DIR / args.dataset_name
    image_root = dataset_root / "images"
    mask_root = dataset_root / "masks"

    print("========== PATH CHECK ==========")
    print("DATASET_ROOT:", dataset_root)
    print("IMAGE_ROOT  :", image_root)
    print("MASK_ROOT   :", mask_root)
    print("================================")

    if not dataset_root.exists():
        raise RuntimeError(f"dataset folder not found: {dataset_root}")

    for split in ["train", "val", "test"]:
        input_dir = image_root / split
        output_dir = mask_root / split
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_dir.exists():
            print(f"[SKIP] input dir not found: {input_dir}")
            continue

        json_files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix == ".json"])
        print(f"[{split}] json count:", len(json_files))

        for json_path in json_files:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            image_path = input_dir / data["imagePath"]

            if not image_path.exists():
                print("image not found:", image_path)
                continue

            img = cv2.imread(str(image_path))
            if img is None:
                print("failed to read image:", image_path)
                continue

            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

            for shape in data["shapes"]:
                if shape["label"] != "red_reflex":
                    continue

                points = np.array(shape["points"], dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)

            out_name = json_path.stem + ".png"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), mask)

    print("mask generation done.")


if __name__ == "__main__":
    main()