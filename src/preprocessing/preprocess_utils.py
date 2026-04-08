import cv2
import numpy as np

# =========================
# Parameters
# =========================
CROP_RATIO = 0.2

DARK_THRESHOLD = 30
VERY_DARK_THRESHOLD = 7

NORMAL_CLAHE_CLIP = 0.8
NORMAL_CONTRAST_ALPHA = 1.0
NORMAL_BETA = 0

DARK_CLAHE_CLIP = 0.8
DARK_CONTRAST_ALPHA = 1.0
DARK_BETA = 0

VERY_DARK_CONTRAST_ALPHA = 2.0
VERY_DARK_BETA = 20
VERY_DARK_CLAHE_CLIP = 1.0

CLAHE_GRID = (8, 8)


def center_crop(img, crop_ratio=CROP_RATIO):
    h, w = img.shape[:2]

    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)

    center_x = w // 2
    center_y = h // 2

    x1 = max(center_x - crop_w // 2, 0)
    x2 = min(center_x + crop_w // 2, w)
    y1 = max(center_y - crop_h // 2, 0)
    y2 = min(center_y + crop_h // 2, h)

    return img[y1:y2, x1:x2].copy()


def get_mean_brightness(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def classify_brightness(mean_brightness):
    if mean_brightness < VERY_DARK_THRESHOLD:
        return "VERY_DARK"
    elif mean_brightness < DARK_THRESHOLD:
        return "DARK"
    else:
        return "NORMAL"


def apply_contrast_brightness(img, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def apply_clahe_color(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    l2 = clahe.apply(l)

    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def make_red_enhanced(img_bgr):
    """
    赤強調画像（1チャンネル）
    red = R - 0.5G - 0.5B
    """
    b, g, r = cv2.split(img_bgr)

    red = (
        r.astype(np.float32)
        - 0.5 * g.astype(np.float32)
        - 0.5 * b.astype(np.float32)
    )

    red = np.clip(red, 0, 255).astype(np.uint8)
    return red


def process_red_by_mode(roi, mode):
    if mode == "NORMAL":
        img1 = apply_contrast_brightness(
            roi,
            alpha=NORMAL_CONTRAST_ALPHA,
            beta=NORMAL_BETA
        )
        img2 = apply_clahe_color(
            img1,
            clip_limit=NORMAL_CLAHE_CLIP,
            tile_grid_size=CLAHE_GRID
        )
        red = make_red_enhanced(img2)

    elif mode == "DARK":
        img1 = apply_contrast_brightness(
            roi,
            alpha=DARK_CONTRAST_ALPHA,
            beta=DARK_BETA
        )
        img2 = apply_clahe_color(
            img1,
            clip_limit=DARK_CLAHE_CLIP,
            tile_grid_size=CLAHE_GRID
        )
        red = make_red_enhanced(img2)

    else:  # VERY_DARK
        img1 = apply_contrast_brightness(
            roi,
            alpha=VERY_DARK_CONTRAST_ALPHA,
            beta=VERY_DARK_BETA
        )
        img2 = apply_clahe_color(
            img1,
            clip_limit=VERY_DARK_CLAHE_CLIP,
            tile_grid_size=CLAHE_GRID
        )
        red = make_red_enhanced(img2)

    return red


def add_debug_overlay(gray_img, text_lines):
    out = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    y = 25
    for line in text_lines:
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )
        y += 22

    return out