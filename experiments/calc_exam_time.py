"""
Repeatability データの片目あたり検査時間を算出。
ファイル名 IMG_YYYYMMDD_HHMMSS_* から最初・最後の撮影時刻を取得し、差分を計算。

特記事項:
  narmadha/RIGHT には LEFT と同一ファイル18枚が混入している。
  そのため narmadha RIGHT は LEFT と重複しないファイルのみを使用して算出する。

Run:
  python experiments/calc_exam_time.py
"""

import csv
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data/Repeatability/0603")
OUT_CSV  = Path("data/processed/pipeline_runs/repeatability_0603_sim_ratio/exam_time_summary.csv")


def get_times(eye_dir: Path, exclude_names: set = None) -> list[datetime]:
    times = []
    for f in eye_dir.glob("*.jpg"):
        if exclude_names and f.name in exclude_names:
            continue
        m = re.search(r"IMG_(\d{8})_(\d{6})", f.name)
        if m:
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            times.append(dt)
    return sorted(times)


def main():
    results = []

    for subject_dir in sorted(DATA_DIR.iterdir()):
        if not subject_dir.is_dir():
            continue

        # narmadha RIGHT: LEFT と同一ファイル名を除外
        left_names = set()
        if subject_dir.name == "narmadha":
            left_dir = subject_dir / "LEFT"
            if left_dir.exists():
                left_names = {f.name for f in left_dir.glob("*.jpg")}

        for eye in ["LEFT", "RIGHT"]:
            eye_dir = subject_dir / eye
            if not eye_dir.exists():
                continue

            exclude = left_names if (subject_dir.name == "narmadha" and eye == "RIGHT") else None
            times = get_times(eye_dir, exclude_names=exclude)

            if len(times) < 2:
                continue

            elapsed = int((times[-1] - times[0]).total_seconds())
            note = ""
            if subject_dir.name == "narmadha" and eye == "RIGHT":
                note = "LEFT重複18枚除外済み"

            results.append({
                "subject":      subject_dir.name,
                "eye":          eye,
                "n_images":     len(times),
                "start":        times[0].strftime("%H:%M:%S"),
                "end":          times[-1].strftime("%H:%M:%S"),
                "elapsed_sec":  elapsed,
                "elapsed_mmss": f"{elapsed // 60}m{elapsed % 60:02d}s",
                "note":         note,
            })

    # --- 表示 ---
    header = f"{'Subject':<14} {'Eye':<6} {'N':>4}  {'Start':>8}  {'End':>8}  {'Time':>8}  Note"
    print(header)
    print("-" * 70)
    for r in results:
        print(
            f"{r['subject']:<14} {r['eye']:<6} {r['n_images']:>4}"
            f"  {r['start']:>8}  {r['end']:>8}  {r['elapsed_mmss']:>8}"
            f"  {r['note']}"
        )

    total = len(results)
    avg   = sum(r["elapsed_sec"] for r in results) / total
    left  = [r for r in results if r["eye"] == "LEFT"]
    right = [r for r in results if r["eye"] == "RIGHT"]
    avg_l = sum(r["elapsed_sec"] for r in left)  / len(left)
    avg_r = sum(r["elapsed_sec"] for r in right) / len(right)

    print()
    print(f"全 {total} 眼  平均: {int(avg)//60}m{int(avg)%60:02d}s")
    print(f"LEFT  平均: {int(avg_l)//60}m{int(avg_l)%60:02d}s  ({len(left)} 眼)")
    print(f"RIGHT 平均: {int(avg_r)//60}m{int(avg_r)%60:02d}s  ({len(right)} 眼)")

    # --- 全体時間 (LEFT start → RIGHT end) per subject ---
    narmadha_left_names = {
        f.name for f in (DATA_DIR / "narmadha/LEFT").glob("*.jpg")
    }

    total_rows = []
    for subject_dir in sorted(DATA_DIR.iterdir()):
        if not subject_dir.is_dir():
            continue
        name = subject_dir.name
        excl = narmadha_left_names if name == "narmadha" else None

        l_times = get_times(subject_dir / "LEFT")
        r_times = get_times(subject_dir / "RIGHT", exclude_names=excl)
        if not l_times or not r_times:
            continue

        total_sec   = int((r_times[-1] - l_times[0]).total_seconds())
        between_sec = int((r_times[0]  - l_times[-1]).total_seconds())
        total_rows.append({
            "subject":        name,
            "l_start":        l_times[0].strftime("%H:%M:%S"),
            "l_end":          l_times[-1].strftime("%H:%M:%S"),
            "r_start":        r_times[0].strftime("%H:%M:%S"),
            "r_end":          r_times[-1].strftime("%H:%M:%S"),
            "l_elapsed_sec":  int((l_times[-1] - l_times[0]).total_seconds()),
            "l_elapsed_mmss": f"{int((l_times[-1]-l_times[0]).total_seconds())//60}m"
                              f"{int((l_times[-1]-l_times[0]).total_seconds())%60:02d}s",
            "r_elapsed_sec":  int((r_times[-1] - r_times[0]).total_seconds()),
            "r_elapsed_mmss": f"{int((r_times[-1]-r_times[0]).total_seconds())//60}m"
                              f"{int((r_times[-1]-r_times[0]).total_seconds())%60:02d}s",
            "between_sec":    between_sec,
            "between_mmss":   f"{between_sec//60}m{between_sec%60:02d}s",
            "total_sec":      total_sec,
            "total_mmss":     f"{total_sec//60}m{total_sec%60:02d}s",
        })

    # --- CSV (per-eye) ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["subject", "eye", "n_images", "start", "end", "elapsed_sec", "elapsed_mmss", "note"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\n保存: {OUT_CSV}")

    # --- CSV (total per subject) ---
    total_csv = OUT_CSV.parent / "exam_total_time_summary.csv"
    total_fields = ["subject", "l_start", "l_end", "r_start", "r_end",
                    "l_elapsed_sec", "l_elapsed_mmss",
                    "r_elapsed_sec", "r_elapsed_mmss",
                    "between_sec", "between_mmss",
                    "total_sec", "total_mmss"]
    with open(total_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=total_fields)
        w.writeheader()
        w.writerows(total_rows)
    print(f"保存: {total_csv}")


if __name__ == "__main__":
    main()
