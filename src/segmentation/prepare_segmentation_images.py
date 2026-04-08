import csv
import shutil
import random
import argparse
from pathlib import Path

from src.common.paths import CLASSIFY_OUTPUTS_DIR, SEGMENTATION_DATASET_DIR

# =========================
# Settings
# =========================
EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classify_run_name",
        type=str,
        required=True,
        help="example: clf_v001_on_default"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="example: seg_dataset_v001"
    )
    parser.add_argument(
        "--max_train_images",
        type=int,
        default=120,
        help="max number of train images for first annotation"
    )
    parser.add_argument(
        "--max_val_images",
        type=int,
        default=40,
        help="max number of val images for first annotation"
    )
    parser.add_argument(
        "--max_test_images",
        type=int,
        default=40,
        help="max number of test images for first annotation"
    )
    return parser.parse_args()


def reset_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for p in path.iterdir():
        if p.is_file():
            p.unlink()


def is_image_file(path: Path):
    return path.suffix in EXTENSIONS


def round_robin_select(patient_to_files, max_images):
    """
    患者の広がりを優先して、各患者から1枚ずつ順に選ぶ
    """
    selected = []

    work = {}
    for pid, files in patient_to_files.items():
        copied = files[:]
        random.shuffle(copied)
        work[pid] = copied

    patient_ids = sorted(work.keys())
    still_has_files = True

    while len(selected) < max_images and still_has_files:
        still_has_files = False
        for pid in patient_ids:
            if len(selected) >= max_images:
                break
            if len(work[pid]) > 0:
                selected.append((pid, work[pid].pop(0)))
                still_has_files = True

    return selected


def main():
    args = parse_args()

    input_root = CLASSIFY_OUTPUTS_DIR / args.classify_run_name / "positive_for_mask"
    output_root = SEGMENTATION_DATASET_DIR / args.dataset_name

    output_image_dir = output_root / "images"
    output_mask_dir = output_root / "masks"
    output_meta_csv = output_root / "split_metadata.csv"

    print("========== PATH CHECK ==========")
    print("INPUT_ROOT  :", input_root)
    print("OUTPUT_ROOT :", output_root)
    print("================================")

    if not input_root.exists():
        raise RuntimeError(f"input folder not found: {input_root}")

    # 出力フォルダ作成
    for split in ["train", "val", "test"]:
        reset_dir(output_image_dir / split)
        (output_mask_dir / split).mkdir(parents=True, exist_ok=True)

    # patient 一覧取得
    patient_ids = sorted([p.name for p in input_root.iterdir() if p.is_dir()])

    if len(patient_ids) == 0:
        raise RuntimeError(f"no patient folders found: {input_root}")

    # 各患者の画像一覧
    patient_to_files_all = {}
    for patient_id in patient_ids:
        patient_dir = input_root / patient_id
        filenames = sorted([
            p.name for p in patient_dir.iterdir()
            if p.is_file() and is_image_file(p)
        ])

        if len(filenames) == 0:
            print(f"[SKIP] no images: {patient_dir}")
            continue

        patient_to_files_all[patient_id] = filenames

    patient_ids = sorted(patient_to_files_all.keys())

    if len(patient_ids) == 0:
        raise RuntimeError("no valid positive images found.")

    # 患者単位 split
    shuffled_patients = patient_ids[:]
    random.shuffle(shuffled_patients)

    n_patients = len(shuffled_patients)

    train_n = max(1, int(n_patients * TRAIN_RATIO))
    val_n = max(1, int(n_patients * VAL_RATIO))
    test_n = n_patients - train_n - val_n

    if test_n <= 0:
        test_n = 1
        if train_n >= val_n and train_n > 1:
            train_n -= 1
        elif val_n > 1:
            val_n -= 1

    train_patients = shuffled_patients[:train_n]
    val_patients = shuffled_patients[train_n:train_n + val_n]
    test_patients = shuffled_patients[train_n + val_n:]

    split_to_patients = {
        "train": sorted(train_patients),
        "val": sorted(val_patients),
        "test": sorted(test_patients),
    }

    print("=== Patient Split ===")
    print("train patients:", len(split_to_patients["train"]), split_to_patients["train"])
    print("val patients  :", len(split_to_patients["val"]), split_to_patients["val"])
    print("test patients :", len(split_to_patients["test"]), split_to_patients["test"])

    split_to_patient_files = {
        "train": {},
        "val": {},
        "test": {},
    }

    for split, pids in split_to_patients.items():
        for pid in pids:
            split_to_patient_files[split][pid] = patient_to_files_all[pid]

    selected_train = round_robin_select(split_to_patient_files["train"], args.max_train_images)
    selected_val = round_robin_select(split_to_patient_files["val"], args.max_val_images)
    selected_test = round_robin_select(split_to_patient_files["test"], args.max_test_images)

    split_to_selected = {
        "train": selected_train,
        "val": selected_val,
        "test": selected_test,
    }

    print("\n=== Selected Images For First Annotation ===")
    print("train:", len(selected_train))
    print("val  :", len(selected_val))
    print("test :", len(selected_test))

    rows = []

    for split in ["train", "val", "test"]:
        dst_dir = output_image_dir / split

        for patient_id, filename in split_to_selected[split]:
            src_path = input_root / patient_id / filename

            dst_filename = f"{patient_id}__{filename}"
            dst_path = dst_dir / dst_filename

            shutil.copy2(str(src_path), str(dst_path))

            rows.append([
                patient_id,
                split,
                filename,
                dst_filename,
                str(src_path),
                str(dst_path)
            ])

    with open(output_meta_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patient_id",
            "split",
            "original_filename",
            "copied_filename",
            "src_path",
            "dst_path"
        ])
        writer.writerows(rows)

    print("\ndone.")
    print("images:", output_image_dir)
    print("masks :", output_mask_dir)
    print("meta  :", output_meta_csv)


if __name__ == "__main__":
    main()