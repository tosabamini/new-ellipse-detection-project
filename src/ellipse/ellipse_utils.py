import cv2
import numpy as np

# =========================
# Default parameters
# =========================
BIN_THRESHOLD = 127

OPEN_KERNEL = 3
CLOSE_KERNEL = 5

MIN_COMPONENT_AREA = 20
CENTER_WEIGHT = 1.0
AREA_BONUS = 0.02


def contour_center(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return cx, cy


def component_score(
    mask_bin,
    min_component_area=MIN_COMPONENT_AREA,
    center_weight=CENTER_WEIGHT,
    area_bonus=AREA_BONUS
):
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cnts) == 0:
        return None, None, None

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < min_component_area:
        return None, None, None

    center = contour_center(cnt)
    if center is None:
        return None, None, None

    h, w = mask_bin.shape[:2]
    cx, cy = center
    dist2 = (cx - w / 2.0) ** 2 + (cy - h / 2.0) ** 2

    score = center_weight * dist2 - area_bonus * area
    return score, cnt, area


def make_single_component_mask(
    mask_gray,
    bin_threshold=BIN_THRESHOLD,
    open_kernel=OPEN_KERNEL,
    close_kernel=CLOSE_KERNEL,
    min_component_area=MIN_COMPONENT_AREA,
    center_weight=CENTER_WEIGHT,
    area_bonus=AREA_BONUS
):
    """
    mask を単一領域化する
    1. threshold
    2. open / close
    3. connected components
    4. 中心に近く、かつ大きい成分を1つ選ぶ
    """
    if mask_gray is None:
        return None, "read_failed"

    _, bin_mask = cv2.threshold(mask_gray, bin_threshold, 255, cv2.THRESH_BINARY)

    if open_kernel > 1:
        kernel_open = np.ones((open_kernel, open_kernel), np.uint8)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel_open)

    if close_kernel > 1:
        kernel_close = np.ones((close_kernel, close_kernel), np.uint8)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    candidates = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_component_area:
            continue

        component_mask = np.zeros_like(bin_mask)
        component_mask[labels == label_id] = 255

        score, cnt, contour_area = component_score(
            component_mask,
            min_component_area=min_component_area,
            center_weight=center_weight,
            area_bonus=area_bonus
        )
        if score is None or cnt is None:
            continue

        candidates.append((score, component_mask, cnt, contour_area))

    if len(candidates) == 0:
        return np.zeros_like(bin_mask), "no_component"

    candidates.sort(key=lambda x: x[0])
    best_mask = candidates[0][1]
    return best_mask, "ok"


def normalize_ellipse(ellipse):
    """
    cv2.fitEllipse の出力を
    major/minor/angle に正規化する
    """
    (cx, cy), (axis1, axis2), angle = ellipse

    major_axis = max(axis1, axis2)
    minor_axis = min(axis1, axis2)

    if axis2 > axis1:
        angle = angle + 90.0

    angle = angle % 180.0
    ellipse_area = np.pi * (major_axis / 2.0) * (minor_axis / 2.0)

    return {
        "center_x": float(cx),
        "center_y": float(cy),
        "major_axis": float(major_axis),
        "minor_axis": float(minor_axis),
        "angle_deg": float(angle),
        "ellipse_area": float(ellipse_area),
        "raw_ellipse": ellipse,
    }


def fit_ellipse_from_mask(
    mask_gray,
    bin_threshold=BIN_THRESHOLD,
    open_kernel=OPEN_KERNEL,
    close_kernel=CLOSE_KERNEL,
    min_component_area=MIN_COMPONENT_AREA,
    center_weight=CENTER_WEIGHT,
    area_bonus=AREA_BONUS
):
    """
    mask_gray -> 単一領域化 -> 輪郭 -> 楕円
    """
    single_mask, status = make_single_component_mask(
        mask_gray=mask_gray,
        bin_threshold=bin_threshold,
        open_kernel=open_kernel,
        close_kernel=close_kernel,
        min_component_area=min_component_area,
        center_weight=center_weight,
        area_bonus=area_bonus
    )

    if single_mask is None:
        return {
            "status": "read_failed",
            "single_mask": None,
            "contour": None,
            "mask_area": None,
            "ellipse_info": None,
        }

    if status != "ok":
        return {
            "status": status,
            "single_mask": single_mask,
            "contour": None,
            "mask_area": None,
            "ellipse_info": None,
        }

    contours, _ = cv2.findContours(
        single_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        return {
            "status": "no_contour_after_single",
            "single_mask": single_mask,
            "contour": None,
            "mask_area": None,
            "ellipse_info": None,
        }

    main_contour = max(contours, key=cv2.contourArea)
    mask_area = float(cv2.contourArea(main_contour))

    if len(main_contour) < 5:
        return {
            "status": "too_few_points",
            "single_mask": single_mask,
            "contour": main_contour,
            "mask_area": mask_area,
            "ellipse_info": None,
        }

    ellipse = cv2.fitEllipse(main_contour)
    ellipse_info = normalize_ellipse(ellipse)

    return {
        "status": "ok",
        "single_mask": single_mask,
        "contour": main_contour,
        "mask_area": mask_area,
        "ellipse_info": ellipse_info,
    }


def angle_diff_deg(a, b):
    """
    楕円角度差（180度周期）
    """
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def add_text_block(img, lines, start=(10, 20), line_h=18, block_width=620):
    out = img.copy()
    x, y = start

    bg = out.copy()
    block_h = line_h * len(lines) + 10
    cv2.rectangle(bg, (5, 5), (5 + block_width, 5 + block_h), (0, 0, 0), -1)
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


def make_pred_overlay(base_img, pred_result):
    """
    pred の単一領域マスク・輪郭・楕円を描画
    """
    if len(base_img.shape) == 2:
        canvas = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        canvas = base_img.copy()

    overlay = canvas.copy()

    if pred_result["single_mask"] is not None:
        pred_mask = pred_result["single_mask"] > 0
        overlay[pred_mask] = (0, 0, 220)

    canvas = cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0)

    if pred_result["contour"] is not None:
        cv2.drawContours(canvas, [pred_result["contour"]], -1, (0, 0, 255), 1)

    if pred_result["ellipse_info"] is not None:
        cv2.ellipse(canvas, pred_result["ellipse_info"]["raw_ellipse"], (255, 0, 255), 2)
        cx = int(pred_result["ellipse_info"]["center_x"])
        cy = int(pred_result["ellipse_info"]["center_y"])
        cv2.circle(canvas, (cx, cy), 3, (255, 0, 255), -1)

    return canvas