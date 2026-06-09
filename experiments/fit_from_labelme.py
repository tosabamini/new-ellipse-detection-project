"""
LabelMe JSON → マスク → fitEllipse して PickUP の ellipse/ と ellipse_results.csv を更新する。

Run:
  python -m experiments.fit_from_labelme
"""

import csv
import json
import numpy as np
import cv2
from pathlib import Path

from src.ellipse.adaptdog import _fit_ellipse_on_mask

# 対象
TARGETS = [
    {
        "json":  Path("data/Repeatability/PickUP/07_Abhishek/LEFT/IMG_20260603_131348_712_red_roi.json"),
        "image": Path("data/Repeatability/PickUP/07_Abhishek/LEFT/IMG_20260603_131348_712_red_roi.png"),
        "csv":   Path("data/Repeatability/PickUP/07_Abhishek/LEFT/ellipse_results.csv"),
        "out":   Path("data/Repeatability/PickUP/07_Abhishek/LEFT/ellipse/IMG_20260603_131348_712_red_roi.png"),
    },
]


def fit_from_json(target: dict) -> None:
    jp = target["json"]
    j  = json.loads(jp.read_text(encoding="utf-8"))
    h, w = j["imageHeight"], j["imageWidth"]

    # ポリゴン → マスク
    mask = np.zeros((h, w), dtype=np.uint8)
    for shape in j.get("shapes", []):
        if shape["label"] == "red_reflex":
            pts = np.array(shape["points"], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

    # マスク → 楕円フィット
    e = _fit_ellipse_on_mask(mask)
    if e is None:
        print(f"  [FAIL] no ellipse: {jp.name}")
        return

    ratio = e["minor"] / e["major"]
    print(f"  major={e['major']:.2f}  minor={e['minor']:.2f}  "
          f"ratio={ratio:.4f}  angle={e['angle']:.2f}")

    # オーバーレイ画像を保存
    gray = cv2.imread(str(target["image"]), cv2.IMREAD_GRAYSCALE)
    out_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = (0, 255, 80)
    cv2.ellipse(out_img, e["raw"], color, 2)
    cv2.drawMarker(out_img, (int(e["cx"]), int(e["cy"])),
                   color, cv2.MARKER_CROSS, 10, 1)
    lbl = f"maj={e['major']:.1f} min={e['minor']:.1f} r={ratio:.3f}"
    cv2.putText(out_img, lbl, (4, out_img.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    target["out"].parent.mkdir(exist_ok=True)
    cv2.imwrite(str(target["out"]), out_img)
    print(f"  Overlay saved: {target['out'].name}")

    # ellipse_results.csv を更新
    stem = jp.stem  # JSON stem = 画像stem
    csv_path = target["csv"]
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    updated = False
    for row in rows:
        if row["stem"] == stem:
            row["major"] = f"{e['major']:.2f}"
            row["minor"] = f"{e['minor']:.2f}"
            row["ratio"] = f"{ratio:.4f}"
            row["angle"] = f"{e['angle']:.2f}"
            updated = True
    if not updated:
        print(f"  [WARN] stem not found in CSV, appending: {stem}")
        rows.append({"stem": stem,
                     "major": f"{e['major']:.2f}",
                     "minor": f"{e['minor']:.2f}",
                     "ratio": f"{ratio:.4f}",
                     "angle": f"{e['angle']:.2f}"})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wt = csv.DictWriter(f, fieldnames=["stem", "major", "minor", "ratio", "angle"])
        wt.writeheader()
        wt.writerows(rows)
    print(f"  CSV saved: {csv_path.name}")


def main():
    for t in TARGETS:
        print(f"\n{t['json'].parent.parent.name}/{t['json'].parent.name}: {t['json'].name}")
        fit_from_json(t)
    print("\nDone.")


if __name__ == "__main__":
    main()
