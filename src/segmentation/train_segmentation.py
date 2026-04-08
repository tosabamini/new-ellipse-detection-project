import csv
import json
import argparse
from pathlib import Path
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.common.paths import (
    SEGMENTATION_DATASET_DIR,
    SEGMENTATION_MODELS_DIR,
    SEGMENTATION_CHECKPOINTS_DIR,
    BEST_SEGMENTATION_MODEL_PATH,
)
from src.segmentation.segmentation_model import UNetSmall

# =========================
# Settings
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="example: seg_dataset_v001"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="example: seg_v001_on_seg_dataset_v001"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )
    return parser.parse_args()


# =========================
# Utility
# =========================
def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_image_mask_pairs(image_dir: Path, mask_dir: Path):
    pairs = []

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]:
        image_paths.extend(glob(str(image_dir / ext)))
    image_paths = sorted(image_paths)

    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        base = img_path.stem
        mask_path = mask_dir / f"{base}.png"

        if not mask_path.exists():
            print(f"[SKIP] mask not found: {img_path}")
            continue

        pairs.append((img_path, mask_path))

    return pairs


def verify_pairs_and_get_size(all_pairs):
    sample_img = cv2.imread(str(all_pairs[0][0]), cv2.IMREAD_GRAYSCALE)
    if sample_img is None:
        raise RuntimeError(f"failed to read first image: {all_pairs[0][0]}")

    h, w = sample_img.shape[:2]
    print(f"reference image size: {w} x {h}")

    for img_path, mask_path in all_pairs:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"failed to read image: {img_path}")
        if mask is None:
            raise RuntimeError(f"failed to read mask: {mask_path}")

        if img.shape[:2] != (h, w):
            raise RuntimeError(
                f"image size mismatch: {img_path} -> {img.shape[:2]} != {(h, w)}"
            )
        if mask.shape[:2] != (h, w):
            raise RuntimeError(
                f"mask size mismatch: {mask_path} -> {mask.shape[:2]} != {(h, w)}"
            )

    return h, w


# =========================
# Dataset
# =========================
class ReflexSegDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"failed to read image: {img_path}")
        if mask is None:
            raise ValueError(f"failed to read mask: {mask_path}")

        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # 念のため2値化
        mask = (mask > 0.5).astype(np.float32)

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            img_path.name
        )


# =========================
# Loss
# =========================
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, preds, targets, smooth=1.0):
        bce = self.bce(preds, targets)

        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        intersection = (preds_flat * targets_flat).sum()
        dice = (2.0 * intersection + smooth) / (
            preds_flat.sum() + targets_flat.sum() + smooth
        )

        dice_loss = 1.0 - dice
        return bce + dice_loss


# =========================
# Metrics
# =========================
def calc_iou(pred_mask, true_mask, threshold=0.5):
    pred_bin = (pred_mask > threshold).astype(np.uint8)
    true_bin = (true_mask > 0.5).astype(np.uint8)

    intersection = np.logical_and(pred_bin, true_bin).sum()
    union = np.logical_or(pred_bin, true_bin).sum()

    if union == 0:
        return 1.0
    return intersection / union


def calc_dice(pred_mask, true_mask, threshold=0.5, smooth=1.0):
    pred_bin = (pred_mask > threshold).astype(np.uint8)
    true_bin = (true_mask > 0.5).astype(np.uint8)

    intersection = (pred_bin * true_bin).sum()
    return (2.0 * intersection + smooth) / (pred_bin.sum() + true_bin.sum() + smooth)


# =========================
# Main
# =========================
def main():
    args = parse_args()
    set_seed()

    dataset_root = SEGMENTATION_DATASET_DIR / args.dataset_name

    image_root = dataset_root / "images"
    mask_root = dataset_root / "masks"

    run_output_dir = SEGMENTATION_MODELS_DIR / args.run_name
    val_pred_dir = run_output_dir / "val_preds"
    test_pred_dir = run_output_dir / "test_preds"
    checkpoints_dir = SEGMENTATION_CHECKPOINTS_DIR / args.run_name

    run_output_dir.mkdir(parents=True, exist_ok=True)
    val_pred_dir.mkdir(parents=True, exist_ok=True)
    test_pred_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    SEGMENTATION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTATION_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print("========== PATH CHECK ==========")
    print("DATASET_ROOT :", dataset_root)
    print("RUN_OUTPUT   :", run_output_dir)
    print("BEST_MODEL   :", BEST_SEGMENTATION_MODEL_PATH)
    print("DEVICE       :", DEVICE)
    print("================================")

    if not dataset_root.exists():
        raise RuntimeError(f"dataset folder not found: {dataset_root}")

    train_pairs = get_image_mask_pairs(
        image_root / "train",
        mask_root / "train"
    )
    val_pairs = get_image_mask_pairs(
        image_root / "val",
        mask_root / "val"
    )
    test_pairs = get_image_mask_pairs(
        image_root / "test",
        mask_root / "test"
    )

    print("train pairs:", len(train_pairs))
    print("val pairs  :", len(val_pairs))
    print("test pairs :", len(test_pairs))

    if len(train_pairs) == 0:
        raise RuntimeError("no train pairs found.")
    if len(val_pairs) == 0:
        raise RuntimeError("no val pairs found.")
    if len(test_pairs) == 0:
        print("[WARN] no test pairs found. test evaluation will be skipped.")

    all_pairs = train_pairs + val_pairs + test_pairs
    H, W = verify_pairs_and_get_size(all_pairs)

    train_dataset = ReflexSegDataset(train_pairs)
    val_dataset = ReflexSegDataset(val_pairs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    if len(test_pairs) > 0:
        test_dataset = ReflexSegDataset(test_pairs)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
    else:
        test_loader = None

    model = UNetSmall().to(DEVICE)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    run_best_model_path = run_output_dir / "best_segmentation_model.pth"
    train_log_csv = run_output_dir / "train_log.csv"
    config_json = run_output_dir / "config.json"

    # config 保存
    config = {
        "dataset_name": args.dataset_name,
        "run_name": args.run_name,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "device": DEVICE,
        "image_height": H,
        "image_width": W,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
    }
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    log_rows = []

    print("training on:", DEVICE)

    for epoch in range(args.epochs):
        # -------------------------
        # train
        # -------------------------
        model.train()
        train_loss = 0.0

        for imgs, masks, _ in train_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # -------------------------
        # val
        # -------------------------
        model.eval()
        val_loss = 0.0
        val_ious = []
        val_dices = []

        with torch.no_grad():
            for imgs, masks, names in val_loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                preds = model(imgs)
                loss = criterion(preds, masks)
                val_loss += loss.item()

                pred_np = preds[0, 0].cpu().numpy()
                mask_np = masks[0, 0].cpu().numpy()

                iou = calc_iou(pred_np, mask_np)
                dice = calc_dice(pred_np, mask_np)

                val_ious.append(iou)
                val_dices.append(dice)

                pred_bin = (pred_np > 0.5).astype(np.uint8) * 255
                gt_bin = (mask_np > 0.5).astype(np.uint8) * 255

                base = Path(names[0]).stem
                cv2.imwrite(str(val_pred_dir / f"{base}_pred.png"), pred_bin)
                cv2.imwrite(str(val_pred_dir / f"{base}_gt.png"), gt_bin)

        val_loss /= max(1, len(val_loader))
        mean_val_iou = float(np.mean(val_ious)) if len(val_ious) > 0 else 0.0
        mean_val_dice = float(np.mean(val_dices)) if len(val_dices) > 0 else 0.0

        print(
            f"Epoch {epoch+1:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_iou={mean_val_iou:.4f} "
            f"val_dice={mean_val_dice:.4f}"
        )

        log_rows.append([
            epoch + 1,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{mean_val_iou:.6f}",
            f"{mean_val_dice:.6f}",
        ])

        # checkpoint
        checkpoint_path = checkpoints_dir / f"epoch_{epoch+1:03d}.pth"
        torch.save(model.state_dict(), checkpoint_path)

        # best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), run_best_model_path)
            torch.save(model.state_dict(), BEST_SEGMENTATION_MODEL_PATH)
            print("  best model saved.")

    # train log 保存
    with open(train_log_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_dice"])
        writer.writerows(log_rows)

    print("\ntraining done.")
    print("run best model:", run_best_model_path)
    print("global best   :", BEST_SEGMENTATION_MODEL_PATH)
    print("val preds     :", val_pred_dir)

    # -------------------------
    # test evaluation
    # -------------------------
    if test_loader is not None and run_best_model_path.exists():
        print("\n=== Test Evaluation ===")

        model.load_state_dict(torch.load(run_best_model_path, map_location=DEVICE))
        model.eval()

        test_ious = []
        test_dices = []

        with torch.no_grad():
            for imgs, masks, names in test_loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                preds = model(imgs)

                pred_np = preds[0, 0].cpu().numpy()
                mask_np = masks[0, 0].cpu().numpy()

                iou = calc_iou(pred_np, mask_np)
                dice = calc_dice(pred_np, mask_np)

                test_ious.append(iou)
                test_dices.append(dice)

                pred_bin = (pred_np > 0.5).astype(np.uint8) * 255
                gt_bin = (mask_np > 0.5).astype(np.uint8) * 255

                base = Path(names[0]).stem
                cv2.imwrite(str(test_pred_dir / f"{base}_pred.png"), pred_bin)
                cv2.imwrite(str(test_pred_dir / f"{base}_gt.png"), gt_bin)

        mean_test_iou = float(np.mean(test_ious)) if len(test_ious) > 0 else 0.0
        mean_test_dice = float(np.mean(test_dices)) if len(test_dices) > 0 else 0.0

        print(f"test_iou  = {mean_test_iou:.4f}")
        print(f"test_dice = {mean_test_dice:.4f}")
        print("test preds:", test_pred_dir)

        # test summary 保存
        with open(run_output_dir / "test_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["test_iou", "test_dice"])
            writer.writerow([f"{mean_test_iou:.6f}", f"{mean_test_dice:.6f}"])


if __name__ == "__main__":
    main()