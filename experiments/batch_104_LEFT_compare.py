"""
Compare 4 AdaptDoG thresholding variants on all 104_LEFT ROI images.

Variants:
  current  : DoG -> Otsu -> pick_central_blob -> ratio-adaptive dilation
  A        : DoG -> Otsu -> erode -> pick_central_blob -> ...
  B        : DoG -> Otsu+30%offset -> pick_central_blob -> ...
  C        : DoG -> CLAHE -> Otsu -> pick_central_blob -> ...

Output:
  experiments/method_compare_output/104_LEFT_variants/
    gallery_compare.png   <- all images x 5 columns (orig + 4 variants), masks + overlays
    results_variants.csv
    per_image/<stem>.png  <- individual 2-row strip per image

Run:
    python experiments/batch_104_LEFT_compare.py
"""

import csv
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ellipse_method_compare as M

# ── paths ──────────────────────────────────────────────────────────────────────
ROI_DIR   = Path(r"C:\Users\issas\Desktop\new_ellipse_detection_project\data\processed\pipeline_runs\pipeline_run_101_106_v001\104_LEFT\roi")
OUT_ROOT  = Path(r"C:\Users\issas\Desktop\new_ellipse_detection_project\experiments\method_compare_output\104_LEFT_variants")

VARIANTS  = ["current", "A", "B", "C"]
LABELS    = {
    "current": "Current\n(Otsu)",
    "A":       "A: Otsu\n+erode",
    "B":       "B: Otsu\n+offset30%",
    "C":       "C: CLAHE\n+Otsu",
}

# ── 4 AdaptDoG variants ───────────────────────────────────────────────────────

def run_variant(red_str: np.ndarray, variant: str):
    minor_est = M._estimate_minor(red_str)
    sigma_l   = max(8.0, minor_est * 0.75)

    # shared DoG
    blur_s = cv2.GaussianBlur(red_str.astype(np.float32), (0, 0), 1.5)
    blur_l = cv2.GaussianBlur(red_str.astype(np.float32), (0, 0), sigma_l)
    dog    = M.stretch_to_255(np.clip(blur_s - blur_l, 0, None))

    if variant == "current":
        mask_raw = M.otsu_mask(dog)

    elif variant == "A":
        # erode after Otsu to remove shoulders, then close fills gaps later
        mask_raw = M.otsu_mask(dog)
        ek = max(3, int(minor_est * 0.12)) | 1
        k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
        mask_raw = cv2.erode(mask_raw, k)

    elif variant == "B":
        # raise threshold: Otsu + 30% of remaining headroom
        otsu_val, _ = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        higher = min(254, int(otsu_val) + int((255 - otsu_val) * 0.30))
        _, mask_raw = cv2.threshold(dog, higher, 255, cv2.THRESH_BINARY)

    elif variant == "C":
        # local contrast boost before Otsu
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        dog_enh  = clahe.apply(dog)
        mask_raw = M.otsu_mask(dog_enh)

    else:
        raise ValueError(f"unknown variant: {variant}")

    mask_core  = M.pick_central_blob(mask_raw)
    e_core     = M.fit_ellipse(mask_core)
    core_ratio = e_core["ratio"] if e_core else 0.0
    core_angle = e_core["angle"] if e_core else 90.0

    if core_ratio < 0.20:
        dil_w = max(3,  int(minor_est * 0.33))
        dil_h = max(15, int(minor_est * 1.20))
        if core_angle < 45 or core_angle > 135:
            dil_w, dil_h = dil_h, dil_w
        mask  = cv2.dilate(mask_core, cv2.getStructuringElement(cv2.MORPH_RECT, (dil_w, dil_h)))
    else:
        ck   = max(5, int(minor_est * 0.20)) | 1
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        mask = cv2.morphologyEx(mask_core, cv2.MORPH_CLOSE, k)

    e = M.fit_ellipse(mask)
    return mask_raw, mask, e, minor_est, core_ratio


# ── build per-image 2-row strip ───────────────────────────────────────────────
MW, MH = 160, 120   # mask / overlay thumbnail size
OW, OH = 160, 120

def make_strip(img: np.ndarray, results: dict) -> np.ndarray:
    """
    Row 0: orig | mask_curr  | mask_A  | mask_B  | mask_C
    Row 1: orig | ovly_curr  | ovly_A  | ovly_B  | ovly_C
    """
    def t(x, w=MW, h=MH):
        return cv2.resize(M.to_bgr(x), (w, h))

    orig = t(img)

    row0 = [orig]
    row1 = [orig.copy()]
    for v in VARIANTS:
        mask_raw, mask, e, _, _ = results[v]
        row0.append(t(mask))
        row1.append(t(M.draw_overlay(img, e, "")))

    r0 = np.hstack(row0)
    r1 = np.hstack(row1)

    # add tiny header labels on row0
    col_labels = ["original"] + [LABELS[v].replace("\n", " ") for v in VARIANTS]
    for i, lbl in enumerate(col_labels):
        cv2.putText(r0, lbl, (i * MW + 4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (50, 255, 50), 1, cv2.LINE_AA)

    return np.vstack([r0, r1])


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "per_image").mkdir(exist_ok=True)

    roi_paths = sorted(ROI_DIR.glob("*_roi.png"))
    total     = len(roi_paths)

    strips    = []
    csv_rows  = []

    for idx, roi_path in enumerate(roi_paths):
        stem = roi_path.stem.replace("_roi", "")
        print(f"[{idx+1:3d}/{total}] {stem}")

        img = cv2.imread(str(roi_path))
        if img is None:
            print("  skip"); continue

        red_str = M.stretch_to_255(M.red_enhance(img))

        results = {}
        for v in VARIANTS:
            try:
                results[v] = run_variant(red_str, v)
            except Exception as ex:
                print(f"  {v} error: {ex}")
                results[v] = (np.zeros_like(red_str),
                              np.zeros_like(red_str), None, 0, 0)

        # per-image strip
        strip = make_strip(img, results)
        cv2.imwrite(str(OUT_ROOT / "per_image" / f"{stem}.png"), strip)
        strips.append((stem, strip))

        # CSV row
        def efields(e, prefix):
            if e and e.get("major"):
                return {f"{prefix}_major": f"{e['major']:.1f}",
                        f"{prefix}_minor": f"{e['minor']:.1f}",
                        f"{prefix}_ratio": f"{e['ratio']:.3f}",
                        f"{prefix}_angle": f"{e['angle']:.1f}"}
            return {f"{prefix}_major":"",f"{prefix}_minor":"",
                    f"{prefix}_ratio":"",f"{prefix}_angle":""}

        row = {"stem": stem,
               "core_ratio": f"{results['current'][4]:.3f}"}
        for v in VARIANTS:
            row.update(efields(results[v][2], v))
        csv_rows.append(row)

    # ── gallery: all images stacked vertically ────────────────────────────────
    strip_imgs = [s for _, s in strips]

    # add stem label to the left of each strip
    labeled = []
    for (stem, _), strip in zip(strips, strip_imgs):
        lbl = np.zeros((strip.shape[0], 180, 3), np.uint8)
        short = stem[-18:]           # last 18 chars fit in the label column
        cv2.putText(lbl, short, (4, MH // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)
        labeled.append(np.hstack([lbl, strip]))

    # header
    gallery_w = labeled[0].shape[1] if labeled else 1000
    header = np.zeros((40, gallery_w, 3), np.uint8)
    for i, v in enumerate(["orig"] + VARIANTS):
        label = v if v == "orig" else LABELS[v].replace("\n", " ")
        cv2.putText(header, label, (180 + i * MW + 4, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    gallery = np.vstack([header] + labeled)
    cv2.imwrite(str(OUT_ROOT / "gallery_compare.png"), gallery)
    print(f"Gallery saved: {OUT_ROOT}/gallery_compare.png")

    # ── CSV ───────────────────────────────────────────────────────────────────
    if csv_rows:
        fields = list(csv_rows[0].keys())
        with open(OUT_ROOT / "results_variants.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(csv_rows)
        print(f"CSV saved:     {OUT_ROOT}/results_variants.csv")

    print(f"\nAll outputs: {OUT_ROOT}")


if __name__ == "__main__":
    main()
