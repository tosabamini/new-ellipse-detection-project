"""
パイプライン結果のレポート画像を生成する。

生成物 (run_root/<patient_id>/report/ 以下):
  cos_curve.png         : D vs angle の三角関数フィット曲線（角度ビン色分け）
  ellipse_grid.png      : 楕円近似オーバーレイの全体グリッド
  classify_grid.png     : 分類ステップ後の全画像一覧（[2]終了時点）

生成物 (run_root/<patient_id>/ellipse_by_angle/ 以下):
  angle_0/              : 0/180°条件 (0-20° または 160-180°)
  angle_45/             : 45°条件  (30-60°)
  angle_90/             : 90°条件  (70-110°)
  angle_other/          : 上記以外

生成物 (run_root/ 直下):
  angle_bin_summary.csv : 全患者・全眼の角度ビン枚数まとめ

使い方:
  python -m src.pipeline.make_report --run_name pipeline_run_v001 \\
      --patient_ids 101_LEFT 101_RIGHT
"""

import argparse
import csv
import math
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.paths import PROCESSED_DIR


# ── 角度ビン定義 ────────────────────────────────────────────────
# 撮影条件: 0°(=180°)、45°、90° の 3 方向
ANGLE_BINS = {
    "angle_0":     {"label": "0/180deg cond",  "color": "#e74c3c"},  # 0-20° or 160-180°
    "angle_45":    {"label": "45deg cond",     "color": "#f39c12"},  # 30-60°
    "angle_90":    {"label": "90deg cond",     "color": "#2980b9"},  # 70-110°
    "angle_other": {"label": "other",           "color": "#95a5a6"},  # above range
}


def classify_angle(deg: float) -> str:
    a = float(deg) % 180
    if a < 20 or a >= 160:
        return "angle_0"
    if 30 <= a < 60:
        return "angle_45"
    if 70 <= a < 110:
        return "angle_90"
    return "angle_other"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--patient_ids", nargs="+", required=True)
    return p.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ── CSV 読み込み ────────────────────────────────────────────────

def load_per_image(per_image_csv: Path) -> list[dict]:
    if not per_image_csv.exists():
        return []
    with open(per_image_csv, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_sca(sca_csv: Path) -> dict | None:
    if not sca_csv.exists():
        return None
    rows = list(csv.DictReader(open(sca_csv, encoding="utf-8-sig")))
    if not rows or not rows[0].get("S_D"):
        return None
    r = rows[0]
    return {
        "S":  float(r["S_D"]),
        "C":  float(r["C_D"]),
        "A":  float(r["A_deg"]),
        "SE": float(r["SE_D"]),
        "R2": float(r["R2"]),
        "n":  int(r["n_valid"]),
    }


# ── 角度ビンごとの楕円画像振り分け ─────────────────────────────

def sort_ellipse_by_angle(patient_id: str, patient_root: Path) -> dict[str, list[str]]:
    """
    per_image.csv の angle_deg を読んで、楕円オーバーレイ画像を
    ellipse_by_angle/<bin>/ にコピーする。

    Returns: bin_name -> [filename, ...] (有効画像のみ)
    """
    per_image = load_per_image(patient_root / "refraction_per_image.csv")
    ellipse_dir = patient_root / "ellipse_overlay"
    out_root = patient_root / "ellipse_by_angle"

    # ビンフォルダを作成・クリア
    for bin_name in ANGLE_BINS:
        bin_dir = out_root / bin_name
        if bin_dir.exists():
            shutil.rmtree(bin_dir)
        ensure_dir(bin_dir)

    bin_map: dict[str, list[str]] = {k: [] for k in ANGLE_BINS}

    for row in per_image:
        if row.get("valid") != "True":
            continue
        fname    = row["filename"]
        stem     = Path(fname).stem
        angle    = float(row["angle_deg"])
        bin_name = classify_angle(angle)

        src = ellipse_dir / f"{stem}_ellipse.png"
        if not src.exists():
            continue

        dst = out_root / bin_name / f"{stem}_ellipse.png"
        shutil.copy2(str(src), str(dst))
        bin_map[bin_name].append(fname)

    # ビン枚数を CSV に保存
    summary_path = out_root / "bin_counts.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "bin", "label", "n_images"])
        for bin_name, fnames in bin_map.items():
            w.writerow([patient_id, bin_name, ANGLE_BINS[bin_name]["label"], len(fnames)])

    return bin_map


# ── cos カーブ（角度ビン色分け） ────────────────────────────────

def make_cos_curve(patient_id: str, patient_root: Path, out_path: Path,
                   bin_map: dict[str, list[str]] | None = None) -> None:
    rows = load_per_image(patient_root / "refraction_per_image.csv")
    sca  = load_sca(patient_root / "refraction_sca.csv")

    valid_rows = [r for r in rows if r.get("valid") == "True" and r.get("adopted_D")]
    if not valid_rows:
        print(f"  cos_curve: no valid images, skipped")
        return

    angles = np.array([float(r["angle_deg"])  for r in valid_rows])
    Ds     = np.array([float(r["adopted_D"])   for r in valid_rows])

    fig, ax = plt.subplots(figsize=(9, 5))

    # 撮影条件帯の背景塗り
    ax.axvspan(0,   20,  alpha=0.07, color=ANGLE_BINS["angle_0"]["color"],   zorder=0)
    ax.axvspan(160, 180, alpha=0.07, color=ANGLE_BINS["angle_0"]["color"],   zorder=0)
    ax.axvspan(30,  60,  alpha=0.07, color=ANGLE_BINS["angle_45"]["color"],  zorder=0)
    ax.axvspan(70,  110, alpha=0.07, color=ANGLE_BINS["angle_90"]["color"],  zorder=0)

    # ビンごとに色分けして散布
    fname_to_bin: dict[str, str] = {}
    if bin_map:
        for bin_name, fnames in bin_map.items():
            for fn in fnames:
                fname_to_bin[fn] = bin_name

    for bin_name, info in ANGLE_BINS.items():
        mask = np.array([fname_to_bin.get(r["filename"]) == bin_name for r in valid_rows])
        if mask.sum() == 0:
            continue
        n = int(mask.sum())
        ax.scatter(angles[mask], Ds[mask],
                   color=info["color"], s=55, zorder=4,
                   label=f"{info['label']}  n={n}")

    # フィット曲線
    if sca is not None:
        a_rad = np.deg2rad(angles)
        X     = np.column_stack([np.ones(len(a_rad)), np.cos(2 * a_rad), np.sin(2 * a_rad)])
        P, *_ = np.linalg.lstsq(X, Ds, rcond=None)
        a_fine = np.linspace(0, 180, 360)
        D_fit  = P[0] + P[1] * np.cos(2 * np.deg2rad(a_fine)) + P[2] * np.sin(2 * np.deg2rad(a_fine))
        ax.plot(a_fine, D_fit, color="black", linewidth=1.8, linestyle="--",
                label="cos fit", zorder=3)
        sca_text = (
            f"S={sca['S']:+.2f}D  C={sca['C']:+.2f}D  A={sca['A']:.1f}°  "
            f"SE={sca['SE']:+.2f}D  R²={sca['R2']:.3f}  n={sca['n']}"
        )
    else:
        sca_text = "SCA fit: unavailable"

    # ビン枚数の副題
    if bin_map:
        bin_counts_str = "  |  ".join(
            f"{ANGLE_BINS[k]['label']}: {len(v)}枚"
            for k, v in bin_map.items()
        )
    else:
        bin_counts_str = ""

    ax.set_xlabel("angle_deg (major axis)", fontsize=11)
    ax.set_ylabel("adopted D (D)", fontsize=11)
    ax.set_title(f"{patient_id}\n{sca_text}\n{bin_counts_str}", fontsize=9.5)
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 20, 30, 45, 60, 70, 90, 110, 160, 180])
    ax.tick_params(axis="x", labelsize=8)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  cos_curve saved: {out_path}")


# ── 楕円グリッド ────────────────────────────────────────────────

def make_ellipse_grid(patient_id: str, patient_root: Path, out_path: Path,
                      cols: int = 6, thumb_w: int = 200) -> None:
    ellipse_dir = patient_root / "ellipse_overlay"
    if not ellipse_dir.exists():
        return
    img_paths = sorted(ellipse_dir.glob("*.png"))
    if not img_paths:
        return
    _write_grid(img_paths, patient_id, cols, thumb_w, out_path)
    print(f"  ellipse_grid saved: {out_path} ({len(img_paths)} images)")


# ── 分類グリッド ────────────────────────────────────────────────
# results.csv を参照し positive=緑枠 / negative=赤枠 で色分け。
# positive / negative を別ファイルにも書き出す。

def make_classify_grid(patient_id: str, patient_root: Path, report_dir: Path,
                       cols: int = 6, thumb_w: int = 260, border: int = 6) -> None:
    classify_dir = patient_root / "classify_overlay"
    results_csv  = patient_root / "results.csv"
    if not classify_dir.exists():
        return

    # filename → (label, prob) マップ（重複除去：先頭のみ）
    label_map: dict[str, tuple[int, float]] = {}
    if results_csv.exists():
        seen: set[str] = set()
        with open(results_csv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                fn = row["filename"]
                if fn in seen:
                    continue
                seen.add(fn)
                try:
                    label_map[fn] = (
                        int(row["class_pred_label"]),
                        float(row["class_positive_prob"]),
                    )
                except (KeyError, ValueError):
                    pass

    img_paths = sorted(classify_dir.glob("*.png"))
    if not img_paths:
        return

    pos_thumbs: list[np.ndarray] = []
    neg_thumbs: list[np.ndarray] = []
    all_thumbs: list[np.ndarray] = []

    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        th = max(1, int(thumb_w * h / w))
        thumb = cv2.resize(img, (thumb_w, th), interpolation=cv2.INTER_AREA)

        # オリジナルファイル名を stem から逆引き（stem = "{orig_stem}_classify"）
        orig_stem = p.stem.removesuffix("_classify")
        # results.csv の filename は拡張子付きなので複数候補を探す
        pred_label, prob = 0, 0.0
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            fn = orig_stem + ext
            if fn in label_map:
                pred_label, prob = label_map[fn]
                break

        # 枠色: positive=緑, negative=赤
        color = (0, 200, 0) if pred_label == 1 else (0, 0, 220)
        thumb = cv2.copyMakeBorder(thumb, border, border, border, border,
                                   cv2.BORDER_CONSTANT, value=color)
        # prob を右下に書く
        label_txt = f"{prob:.2f}"
        (tw, _), bl = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        tx = thumb.shape[1] - tw - border - 2
        ty = thumb.shape[0] - border - 3
        cv2.putText(thumb, label_txt, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(thumb, label_txt, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),     1, cv2.LINE_AA)

        all_thumbs.append(thumb)
        (pos_thumbs if pred_label == 1 else neg_thumbs).append(thumb)

    def _save(thumbs, path, title):
        if not thumbs:
            return
        thumb_h = thumbs[0].shape[0]
        thumb_w_ = thumbs[0].shape[1]
        blank = np.zeros((thumb_h, thumb_w_, 3), dtype=np.uint8)
        rows_n = math.ceil(len(thumbs) / cols)
        grid_rows = []
        for r in range(rows_n):
            row_imgs = thumbs[r * cols: (r + 1) * cols]
            while len(row_imgs) < cols:
                row_imgs.append(blank)
            grid_rows.append(np.hstack(row_imgs))
        grid = np.vstack(grid_rows)
        cv2.putText(grid, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(path), grid)
        print(f"  classify_grid saved: {path} ({len(thumbs)} images)")

    _save(all_thumbs, report_dir / "classify_grid_all.png",
          f"{patient_id}  pos={len(pos_thumbs)} neg={len(neg_thumbs)}")
    _save(pos_thumbs, report_dir / "classify_grid_positive.png",
          f"{patient_id} POSITIVE ({len(pos_thumbs)})")
    _save(neg_thumbs, report_dir / "classify_grid_negative.png",
          f"{patient_id} NEGATIVE ({len(neg_thumbs)})")


def _write_grid(img_paths, label, cols, thumb_w, out_path):
    thumbs = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        th = max(1, int(thumb_w * h / w))
        thumbs.append(cv2.resize(img, (thumb_w, th), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    thumb_h = thumbs[0].shape[0]
    blank   = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
    rows_n  = math.ceil(len(thumbs) / cols)
    grid_rows = []
    for r in range(rows_n):
        row_imgs = thumbs[r * cols: (r + 1) * cols]
        while len(row_imgs) < cols:
            row_imgs.append(blank)
        grid_rows.append(np.hstack(row_imgs))
    grid = np.vstack(grid_rows)
    cv2.putText(grid, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), grid)


# ── ランレベル全患者まとめ CSV ──────────────────────────────────

def write_run_summary(run_root: Path, patient_ids: list[str]) -> None:
    out_path = run_root / "angle_bin_summary.csv"
    rows = []
    for pid in patient_ids:
        bin_csv = run_root / pid / "ellipse_by_angle" / "bin_counts.csv"
        if not bin_csv.exists():
            continue
        with open(bin_csv, encoding="utf-8-sig") as f:
            rows.extend(list(csv.DictReader(f)))
    if not rows:
        return
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "bin", "label", "n_images"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nRun-level summary saved: {out_path}")


# ── エントリポイント ────────────────────────────────────────────

def main():
    args = parse_args()
    run_root = PROCESSED_DIR / "pipeline_runs" / args.run_name

    for patient_id in args.patient_ids:
        patient_root = run_root / patient_id
        if not patient_root.exists():
            print(f"[{patient_id}] output folder not found, skipping")
            continue

        report_dir = patient_root / "report"
        ensure_dir(report_dir)
        print(f"[{patient_id}] generating report -> {report_dir}")

        bin_map = sort_ellipse_by_angle(patient_id, patient_root)
        counts_str = "  ".join(
            f"{ANGLE_BINS[k]['label']}:{len(v)}枚" for k, v in bin_map.items()
        )
        print(f"  angle bins: {counts_str}")

        make_cos_curve(patient_id, patient_root, report_dir / "cos_curve.png", bin_map)
        make_ellipse_grid(patient_id, patient_root, report_dir / "ellipse_grid.png")
        make_classify_grid(patient_id, patient_root, report_dir)

    write_run_summary(run_root, args.patient_ids)
    print("report generation done.")


if __name__ == "__main__":
    main()
