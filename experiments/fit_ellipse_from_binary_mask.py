"""
外部バイナリマスクに対して fitEllipse を実行し、オーバーレイ画像と CSV を保存。

対象: data/simu_masked/binary_mask_flat75/<group>/*_mask.png

Run:
  python experiments/fit_ellipse_from_binary_mask.py
  python experiments/fit_ellipse_from_binary_mask.py --groups p30
"""

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

from src.ellipse.adaptdog import _fit_ellipse_on_mask

MASK_DIR = Path("data/simu_masked/binary_mask_flat75")
OUT_DIR  = Path("data/simu_masked/ellipse_flat75")


def parse_D(stem: str) -> float | None:
    m = re.search(r"_D(m?p?)(\d+)", stem)
    if not m:
        return None
    s, v = m.group(1), int(m.group(2)) / 100.0
    if s == "m": return -v
    if s == "p": return  v
    return 0.0


def process_group(group: str):
    in_dir  = MASK_DIR / group
    img_dir = OUT_DIR / group / "ellipse_overlay"
    img_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*_mask.png"))
    print(f"  [{group}] {len(files)} masks")

    rows = []
    for mask_path in files:
        stem  = mask_path.stem.replace("_mask", "")
        d_val = parse_D(stem)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        # 念のため二値化（既にバイナリのはずだが）
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        e = _fit_ellipse_on_mask(mask_bin)

        # オーバーレイ
        overlay = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
        if e:
            cv2.ellipse(overlay, e['raw'], (0, 255, 80), 2)
            cv2.drawMarker(overlay,
                           (int(e['cx']), int(e['cy'])),
                           (0, 255, 80), cv2.MARKER_CROSS, 10, 1)
            line1 = f"D={d_val:+.2f}D  angle={e['angle']:.1f}deg" if d_val is not None else f"angle={e['angle']:.1f}deg"
            line2 = f"maj={e['major']:.1f}  min={e['minor']:.1f}  r={e['minor']/e['major']:.3f}"
            status = "ok"
        else:
            line1 = f"D={d_val:+.2f}D  [no fit]" if d_val is not None else "[no fit]"
            line2 = ""
            status = "no_fit"

        h = overlay.shape[0]
        cv2.putText(overlay, line1, (4, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, line2, (4, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 80),  1, cv2.LINE_AA)

        cv2.imwrite(str(img_dir / f"{stem}_ellipse.png"), overlay)

        rows.append({
            "stem":   stem,
            "D":      f"{d_val:.2f}" if d_val is not None else "",
            "status": status,
            "major":  f"{e['major']:.2f}"  if e else "",
            "minor":  f"{e['minor']:.2f}"  if e else "",
            "ratio":  f"{e['minor']/e['major']:.4f}" if e else "",
            "angle":  f"{e['angle']:.2f}"  if e else "",
        })

    # CSV
    csv_path = OUT_DIR / group / "per_image_ellipse.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem","D","status","major","minor","ratio","angle"])
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"    fit ok: {n_ok}/{len(rows)}  -> {OUT_DIR / group}")


def main():
    parser = argparse.ArgumentParser()
    all_groups = sorted([d.name for d in MASK_DIR.iterdir() if d.is_dir()])
    parser.add_argument("--groups", nargs="+", default=all_groups)
    args = parser.parse_args()

    print(f"Groups: {args.groups}")
    for g in args.groups:
        process_group(g)
    print("Done.")


if __name__ == "__main__":
    main()
