import argparse
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.preprocess_utils import make_red_enhanced
from src.ellipse.ellipse_utils import fit_ellipse_from_mask, make_pred_overlay, add_text_block


def stretch_to_255(img_gray: np.ndarray) -> np.ndarray:
    if img_gray is None:
        return None
    min_val = int(np.min(img_gray))
    max_val = int(np.max(img_gray))
    if max_val <= min_val:
        return img_gray.copy()
    stretched = ((img_gray.astype(np.float32) - min_val) / (max_val - min_val) * 255.0)
    return stretched.astype(np.uint8)


def threshold_otsu(gray_img: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def threshold_percentile(gray_img: np.ndarray, foreground_ratio: float = 0.1) -> np.ndarray:
    if gray_img is None:
        return None
    percentile = 100.0 - foreground_ratio * 100.0
    thresh = float(np.percentile(gray_img, percentile))
    _, mask = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


def extract_central_convex_hull_blob(binary_mask: np.ndarray, apply_morph: bool = True) -> np.ndarray:
    """
    Extract the central bright blob using convex hull.
    Finds all contours, selects the one closest to center,
    computes its convex hull, and returns the filled hull mask.
    """
    if apply_morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        mask = binary_mask.copy()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    h, w = mask.shape[:2]
    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    best_score = None
    best_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist2 = float((cx - center[0]) ** 2 + (cy - center[1]) ** 2)
        score = area - dist2 * 0.2
        if best_score is None or score > best_score:
            best_score = score
            best_contour = cnt

    if best_contour is None:
        return mask

    hull = cv2.convexHull(best_contour)
    hull_mask = np.zeros_like(mask)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)
    return hull_mask


def clean_mask(binary_mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    h, w = mask.shape[:2]
    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    best_score = None
    best_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist2 = float((cx - center[0]) ** 2 + (cy - center[1]) ** 2)
        score = area - dist2 * 0.2
        if best_score is None or score > best_score:
            best_score = score
            best_contour = cnt

    if best_contour is None:
        return mask

    best_mask = np.zeros_like(mask)
    cv2.drawContours(best_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
    return best_mask


def choose_best_mask(red_eq: np.ndarray, foreground_ratio: float = 0.1) -> tuple[np.ndarray, str]:
    masks = {
        "otsu": clean_mask(threshold_otsu(red_eq)),
        "percentile": clean_mask(threshold_percentile(red_eq, foreground_ratio)),
        "center_hull": extract_central_convex_hull_blob(threshold_percentile(red_eq, foreground_ratio), apply_morph=True),
    }

    best_method = None
    best_score = None
    best_mask = None

    for method_name, mask in masks.items():
        result = fit_ellipse_from_mask(mask)
        if result["status"] != "ok":
            score = -1e6
        else:
            h, w = mask.shape[:2]
            center_dist = np.hypot(
                result["ellipse_info"]["center_x"] - w / 2.0,
                result["ellipse_info"]["center_y"] - h / 2.0,
            )
            score = float(result["mask_area"] - center_dist * 0.4)

        if best_score is None or score > best_score:
            best_score = score
            best_method = method_name
            best_mask = mask

    return best_mask, best_method


def prepare_red_mask(roi_img: np.ndarray, foreground_ratio: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    red_img = make_red_enhanced(roi_img)
    red_norm = stretch_to_255(red_img)
    red_eq = cv2.equalizeHist(red_norm)

    mask_otsu = clean_mask(threshold_otsu(red_eq))
    mask_pct = clean_mask(threshold_percentile(red_eq, foreground_ratio))
    mask, selected_method = choose_best_mask(red_eq, foreground_ratio)

    return red_img, red_eq, mask, mask_otsu, mask_pct, selected_method


def make_visualization(roi_img: np.ndarray, red_img: np.ndarray, red_eq: np.ndarray, binary_mask: np.ndarray, mask_otsu: np.ndarray, mask_pct: np.ndarray, selected_method: str, pred_result: dict) -> np.ndarray:
    if len(roi_img.shape) == 2:
        base = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
    else:
        base = roi_img.copy()

    red_vis = cv2.cvtColor(red_img, cv2.COLOR_GRAY2BGR)
    eq_vis = cv2.cvtColor(red_eq, cv2.COLOR_GRAY2BGR)
    mask_vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    overlay = make_pred_overlay(base, pred_result)
    overlay = add_text_block(
        overlay,
        [
            f"Ellipse extraction: selected={selected_method}",
            f"mask status: {pred_result['status']}",
            f"major: {pred_result['ellipse_info']['major_axis']:.1f}" if pred_result['ellipse_info'] else "major: -",
            f"minor: {pred_result['ellipse_info']['minor_axis']:.1f}" if pred_result['ellipse_info'] else "minor: -",
        ],
        start=(10, 25),
        line_h=24,
        block_width=520,
    )

    top = np.hstack([cv2.resize(red_vis, (400, 400)), cv2.resize(eq_vis, (400, 400))])
    bottom = np.hstack([cv2.resize(mask_vis, (400, 400)), cv2.resize(overlay, (400, 400))])
    vis = np.vstack([top, bottom])
    return vis


def parse_args():
    parser = argparse.ArgumentParser(description="Single-image ellipse tester using red-enhance + Otsu")
    parser.add_argument("--input", type=str, required=True, help="input image path (ROI or raw image)")
    parser.add_argument("--output_dir", type=str, default=None, help="folder to write outputs")
    parser.add_argument("--foreground_ratio", type=float, default=0.1, help="foreground ratio for percentile thresholding (default 0.1 = top 10%)")
    parser.add_argument("--method", choices=["auto", "otsu", "percentile", "center_hull", "sweep", "top10_hull", "thin_hull"], default="auto", help="threshold method selection")
    parser.add_argument("--sweep_min", type=float, default=0.001, help="sweep minimum foreground ratio (default 0.1%)")
    parser.add_argument("--sweep_max", type=float, default=0.015, help="sweep maximum foreground ratio (default 1.5%)")
    parser.add_argument("--sweep_steps", type=int, default=51, help="number of sweep steps (default 51)")
    return parser.parse_args()


def sweep_foreground_ratios(red_eq: np.ndarray, min_ratio: float = 0.001, max_ratio: float = 0.015, num_steps: int = 51):
    """
    Sweep foreground ratios and find elbow point using derivative of mask area.
    """
    ratios = np.linspace(min_ratio, max_ratio, num_steps)
    results = []

    for ratio in ratios:
        mask_pct = clean_mask(threshold_percentile(red_eq, ratio))
        mask_area = float((mask_pct > 0).sum())
        pred_result = fit_ellipse_from_mask(mask_pct)

        if pred_result["status"] == "ok":
            e = pred_result["ellipse_info"]
            major = e["major_axis"]
            minor = e["minor_axis"]
            eccent = major / minor if minor > 0 else 0
        else:
            major = minor = eccent = 0.0

        results.append({
            "ratio": ratio,
            "mask_area": mask_area,
            "major": major,
            "minor": minor,
            "eccent": eccent,
            "status": pred_result["status"],
        })

    areas = np.array([r["mask_area"] for r in results])
    
    if len(areas) < 3:
        print("too few sweep points to detect elbow")
        return results[len(results) // 2], results

    first_deriv = np.diff(areas)
    
    print("  [ratio]      [area]   [Δarea]   [major]  [minor]  [eccent]")
    for i, r in enumerate(results):
        delta_area = first_deriv[i] if i < len(first_deriv) else 0
        print(f"  {r['ratio']:.5f}: area={r['mask_area']:7.0f}, Δ={delta_area:7.1f}, major={r['major']:7.1f}, minor={r['minor']:7.1f}, ecc={r['eccent']:6.2f}")
    
    if len(first_deriv) < 2:
        print("too few sweep points for second derivative")
        return results[len(results) // 2], results
    
    second_deriv = np.diff(first_deriv)
    elbow_idx = np.argmax(np.abs(second_deriv)) + 1
    
    print(f"\n[elbow detected at index {elbow_idx}, ratio={results[elbow_idx]['ratio']:.5f}]")
    
    return results[elbow_idx], results


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input image not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_img = cv2.imread(str(input_path))
    if roi_img is None:
        raise RuntimeError(f"failed to read image: {input_path}")

    red_img = make_red_enhanced(roi_img)
    red_norm = stretch_to_255(red_img)
    red_eq = cv2.equalizeHist(red_norm)

    if args.method == "sweep":
        print(f"[sweep mode] testing foreground ratios from {args.sweep_min*100:.2f}% to {args.sweep_max*100:.2f}% ({args.sweep_steps} steps)...")
        best_result, all_results = sweep_foreground_ratios(red_eq, min_ratio=args.sweep_min, max_ratio=args.sweep_max, num_steps=args.sweep_steps)
        selected_ratio = best_result["ratio"]
        selected_method = f"sweep (ratio={selected_ratio:.5f})"
        mask = clean_mask(threshold_percentile(red_eq, selected_ratio))
    elif args.method == "top10_hull":
        print("[top10_hull mode] extracting top 10% brightest pixels and central convex hull...")
        mask_top10 = threshold_percentile(red_eq, 0.1)
        mask = extract_central_convex_hull_blob(mask_top10, apply_morph=False)
        selected_method = "top10_hull"
    elif args.method == "thin_hull":
        print("[thin_hull mode] extracting top 0.5% brightest pixels and central convex hull...")
        mask_thin = threshold_percentile(red_eq, 0.005)
        mask = extract_central_convex_hull_blob(mask_thin, apply_morph=False)
        selected_method = "thin_hull"
    else:
        mask_otsu = clean_mask(threshold_otsu(red_eq))
        mask_pct = clean_mask(threshold_percentile(red_eq, args.foreground_ratio))
        mask_center_hull_regular = extract_central_convex_hull_blob(threshold_percentile(red_eq, args.foreground_ratio))

        if args.method == "otsu":
            mask = mask_otsu
            selected_method = "otsu"
        elif args.method == "percentile":
            mask = mask_pct
            selected_method = "percentile"
        elif args.method == "center_hull":
            mask = mask_center_hull_regular
            selected_method = "center_hull"
        else:
            mask, selected_method = choose_best_mask(red_eq, args.foreground_ratio)

    pred_result = fit_ellipse_from_mask(mask)
    
    if args.method in ["sweep", "top10_hull"]:
        overlay = make_visualization(roi_img, red_img, red_eq, mask, mask, mask, selected_method, pred_result)
    else:
        overlay = make_visualization(roi_img, red_img, red_eq, mask, mask_otsu, mask_pct, selected_method, pred_result)

    stem = input_path.stem
    cv2.imwrite(str(output_dir / f"{stem}_red.png"), red_img)
    cv2.imwrite(str(output_dir / f"{stem}_red_eq.png"), red_eq)
    if args.method not in ["sweep", "top10_hull"]:
        cv2.imwrite(str(output_dir / f"{stem}_mask_otsu.png"), mask_otsu)
        cv2.imwrite(str(output_dir / f"{stem}_mask_pct.png"), mask_pct)
    cv2.imwrite(str(output_dir / f"{stem}_mask_selected.png"), mask)
    cv2.imwrite(str(output_dir / f"{stem}_ellipse_overlay.png"), overlay)

    print(f"selected threshold method: {selected_method}")
    if pred_result["ellipse_info"] is not None:
        e = pred_result["ellipse_info"]
        print("ellipse found:")
        print(f"  center=({e['center_x']:.1f}, {e['center_y']:.1f})")
        print(f"  major={e['major_axis']:.1f}")
        print(f"  minor={e['minor_axis']:.1f}")
        print(f"  angle={e['angle_deg']:.1f}")
    else:
        print("ellipse not found, status=", pred_result["status"])

    print("outputs written to:", output_dir)


if __name__ == "__main__":
    main()
