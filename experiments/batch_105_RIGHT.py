"""
Batch ellipse extraction for 105_RIGHT — all ROI images, 3 methods.

Output layout:
  experiments/method_compare_output/105_RIGHT_batch/
    results.csv                   <- per-image numbers for all 3 methods + ML if available
    gallery_adaptdog.png          <- all AdaptDoG overlays in one grid
    gallery_all.png               <- 3-method comparison strip for every image
    comparisons/<stem>/           <- per-image comparison grid + overlays

Run:
    python experiments/batch_105_RIGHT.py
"""

import csv
import cv2
import numpy as np
from pathlib import Path

# ── import shared logic from the compare script ────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
import ellipse_method_compare as M

# ── paths ──────────────────────────────────────────────────────────────────────
ROI_DIR       = Path(r"C:\Users\issas\Desktop\new_ellipse_detection_project\data\processed\pipeline_runs\pipeline_run_101_106_v001\105_RIGHT\roi")
ELLIPSE_DIR   = Path(r"C:\Users\issas\Desktop\new_ellipse_detection_project\data\processed\pipeline_runs\pipeline_run_101_106_v001\105_RIGHT\ellipse_overlay")
RESULTS_CSV   = Path(r"C:\Users\issas\Desktop\new_ellipse_detection_project\data\processed\pipeline_runs\pipeline_run_101_106_v001\105_RIGHT\results.csv")
OUT_ROOT      = Path(r"C:\Users\issas\Desktop\new_ellipse_detection_project\experiments\method_compare_output\105_RIGHT_batch")

# ── read ML results CSV (major/minor/angle per image) ─────────────────────────
def load_ml_results(csv_path: Path) -> dict:
    """Returns {stem: {major, minor, ratio, angle}} from pipeline results.csv."""
    ml = {}
    if not csv_path.exists():
        return ml
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = Path(row.get("file", "")).stem
            try:
                major = float(row["major_axis"])
                minor = float(row["minor_axis"])
                angle = float(row["angle_deg"])
                ml[stem] = dict(major=major, minor=minor,
                                ratio=minor/major if major > 0 else 0,
                                angle=angle)
            except (KeyError, ValueError):
                pass
    return ml

# ── thumbnail helper ───────────────────────────────────────────────────────────
THUMB_W, THUMB_H = 320, 240

def make_thumb(img, e, label=""):
    c = M.draw_overlay(img, e, label)
    return cv2.resize(c, (THUMB_W, THUMB_H))

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "comparisons").mkdir(exist_ok=True)

    roi_paths = sorted(ROI_DIR.glob("*_roi.png"))
    ml_data   = load_ml_results(RESULTS_CSV)

    csv_rows  = []
    thumbs_frangi  = []
    thumbs_sato    = []
    thumbs_adaptdog = []

    total = len(roi_paths)
    for idx, roi_path in enumerate(roi_paths):
        stem = roi_path.stem.replace("_roi", "")
        print(f"[{idx+1:3d}/{total}] {stem}")

        img = cv2.imread(str(roi_path))
        if img is None:
            print(f"  skip (cannot read)")
            continue

        red_str = M.stretch_to_255(M.red_enhance(img))

        # run 3 methods
        try:
            feat_a, mask_a, e_a, _            = M.run_frangi(red_str)
        except Exception as ex:
            print(f"  Frangi error: {ex}"); feat_a=mask_a=e_a=None

        try:
            feat_b, mask_b, e_b, _, _         = M.run_sato(red_str)
        except Exception as ex:
            print(f"  Sato error: {ex}"); feat_b=mask_b=e_b=None

        try:
            feat_c, mask_c, e_c, m_est, sig_c, cr = M.run_adaptive_dog(red_str)
        except Exception as ex:
            print(f"  AdaptDoG error: {ex}"); feat_c=mask_c=e_c=None; cr=0

        # per-image comparison grid
        out_dir = OUT_ROOT / "comparisons" / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        blank = np.zeros_like(red_str)
        grid = M.build_grid(img, [
            ("Frangi+dil",
             feat_a if feat_a is not None else blank,
             mask_a if mask_a is not None else blank, e_a),
            (f"DoG+Sato(s={sig_c:.0f})" if sig_c else "DoG+Sato",
             feat_b if feat_b is not None else blank,
             mask_b if mask_b is not None else blank, e_b),
            (f"AdaptDoG(cr={cr:.2f})",
             feat_c if feat_c is not None else blank,
             mask_c if mask_c is not None else blank, e_c),
        ])
        cv2.imwrite(str(out_dir / "00_comparison_grid.png"), grid)
        cv2.imwrite(str(out_dir / "01_adaptdog_overlay.png"), M.draw_overlay(img, e_c, "AdaptDoG"))
        cv2.imwrite(str(out_dir / "02_frangi_overlay.png"),   M.draw_overlay(img, e_a, "Frangi"))
        cv2.imwrite(str(out_dir / "03_sato_overlay.png"),     M.draw_overlay(img, e_b, "DoG+Sato"))

        # collect thumbnails for gallery
        thumbs_frangi.append(make_thumb(img, e_a, stem[-12:]))
        thumbs_sato.append(make_thumb(img, e_b, stem[-12:]))
        thumbs_adaptdog.append(make_thumb(img, e_c, f"cr={cr:.2f}"))

        # ML ground truth (if available)
        ml = ml_data.get(stem, {})

        def erow(e):
            if e and e.get("major"):
                return e["major"], e["minor"], e["ratio"], e["angle"]
            return "", "", "", ""

        csv_rows.append({
            "stem":          stem,
            "has_ml":        "yes" if ml else "no",
            "core_ratio":    f"{cr:.3f}",
            "frangi_major":  erow(e_a)[0], "frangi_minor": erow(e_a)[1],
            "frangi_ratio":  erow(e_a)[2], "frangi_angle": erow(e_a)[3],
            "sato_major":    erow(e_b)[0], "sato_minor":   erow(e_b)[1],
            "sato_ratio":    erow(e_b)[2], "sato_angle":   erow(e_b)[3],
            "adog_major":    erow(e_c)[0], "adog_minor":   erow(e_c)[1],
            "adog_ratio":    erow(e_c)[2], "adog_angle":   erow(e_c)[3],
            "ml_major":      ml.get("major",""), "ml_minor": ml.get("minor",""),
            "ml_ratio":      ml.get("ratio",""), "ml_angle": ml.get("angle",""),
        })

    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_out = OUT_ROOT / "results.csv"
    fields = list(csv_rows[0].keys()) if csv_rows else []
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(csv_rows)
    print(f"\nCSV saved: {csv_out}")

    # ── gallery grids ─────────────────────────────────────────────────────────
    def make_gallery(thumbs, title, path):
        cols = 6
        rows = (len(thumbs) + cols - 1) // cols
        blank = np.zeros((THUMB_H, THUMB_W, 3), np.uint8)
        padded = thumbs + [blank] * (rows * cols - len(thumbs))
        rows_img = [np.hstack(padded[r*cols:(r+1)*cols]) for r in range(rows)]
        gallery = np.vstack(rows_img)
        header = np.zeros((32, gallery.shape[1], 3), np.uint8)
        cv2.putText(header, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imwrite(str(path), np.vstack([header, gallery]))
        print(f"Gallery saved: {path}")

    make_gallery(thumbs_adaptdog, "AdaptDoG — 105_RIGHT all images",
                 OUT_ROOT / "gallery_adaptdog.png")
    make_gallery(thumbs_frangi,   "Frangi+dil — 105_RIGHT all images",
                 OUT_ROOT / "gallery_frangi.png")
    make_gallery(thumbs_sato,     "DoG+Sato — 105_RIGHT all images",
                 OUT_ROOT / "gallery_sato.png")

    print(f"\nAll outputs: {OUT_ROOT}")


if __name__ == "__main__":
    main()
