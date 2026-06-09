"""
0603・0604 両日の検査時間を算出。

ファイル名 IMG_YYYYMMDD_HHMMSS_* から最初・最後の撮影時刻を取得。
LEFT測定時間・RIGHT測定時間・インターバル・合計時間を出力。

特記事項 (0603):
  narmadha/RIGHT には LEFT と同一ファイル18枚が混入しているため除外済み。

Run:
  python experiments/calc_exam_time_all.py
"""

import csv
import re
from datetime import datetime
from pathlib import Path

DATES = {
    "0603": Path("data/Repeatability/0603"),
    "0604": Path("data/Repeatability/0604"),
}

OUT_DIR = Path("data/Repeatability")


def get_times(eye_dir: Path, exclude_names: set = None) -> list[datetime]:
    times = []
    for f in sorted(eye_dir.glob("*.jpg")) + sorted(eye_dir.glob("*.JPG")):
        if exclude_names and f.name in exclude_names:
            continue
        m = re.search(r"IMG_(\d{8})_(\d{6})", f.name)
        if m:
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            times.append(dt)
    return sorted(set(times))  # 重複除去


def calc_subject(subject_dir: Path, date_label: str) -> dict | None:
    name = subject_dir.name

    # narmadha (0603) RIGHT の重複除外
    excl_right = set()
    if date_label == "0603" and name == "narmadha":
        left_dir = subject_dir / "LEFT"
        if left_dir.exists():
            excl_right = {f.name for f in left_dir.glob("*.jpg")}

    l_dir = subject_dir / "LEFT"
    r_dir = subject_dir / "RIGHT"
    if not l_dir.exists() or not r_dir.exists():
        return None

    l_times = get_times(l_dir)
    r_times = get_times(r_dir, exclude_names=excl_right if excl_right else None)

    if len(l_times) < 2 or len(r_times) < 2:
        return None

    l_sec = int((l_times[-1] - l_times[0]).total_seconds())
    r_sec = int((r_times[-1] - r_times[0]).total_seconds())
    interval_sec = int((r_times[0] - l_times[-1]).total_seconds())
    total_sec = int((r_times[-1] - l_times[0]).total_seconds())

    def mmss(s): return f"{s//60}m{s%60:02d}s"

    return {
        "date":           date_label,
        "subject":        name,
        "l_n":            len(l_times),
        "l_start":        l_times[0].strftime("%H:%M:%S"),
        "l_end":          l_times[-1].strftime("%H:%M:%S"),
        "l_sec":          l_sec,
        "l_time":         mmss(l_sec),
        "r_n":            len(r_times),
        "r_start":        r_times[0].strftime("%H:%M:%S"),
        "r_end":          r_times[-1].strftime("%H:%M:%S"),
        "r_sec":          r_sec,
        "r_time":         mmss(r_sec),
        "interval_sec":   interval_sec,
        "interval_time":  mmss(interval_sec),
        "total_sec":      total_sec,
        "total_time":     mmss(total_sec),
    }


def main():
    all_rows = []

    for date_label, data_dir in DATES.items():
        if not data_dir.exists():
            print(f"[SKIP] {data_dir} not found")
            continue
        for subject_dir in sorted(data_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            row = calc_subject(subject_dir, date_label)
            if row:
                all_rows.append(row)

    # --- 表示 ---
    print(f"{'Date':<6} {'Subject':<16} {'L_n':>4} {'L_time':>8}  {'R_n':>4} {'R_time':>8}  {'Interval':>9}  {'Total':>8}")
    print("-" * 80)
    cur_date = None
    for r in all_rows:
        if r["date"] != cur_date:
            if cur_date is not None:
                print()
            cur_date = r["date"]
        print(
            f"{r['date']:<6} {r['subject']:<16} {r['l_n']:>4} {r['l_time']:>8}"
            f"  {r['r_n']:>4} {r['r_time']:>8}  {r['interval_time']:>9}  {r['total_time']:>8}"
        )

    # --- 統計 ---
    for date_label in DATES:
        rows = [r for r in all_rows if r["date"] == date_label]
        if not rows:
            continue
        avg_l   = sum(r["l_sec"]        for r in rows) / len(rows)
        avg_r   = sum(r["r_sec"]        for r in rows) / len(rows)
        avg_iv  = sum(r["interval_sec"] for r in rows) / len(rows)
        avg_tot = sum(r["total_sec"]    for r in rows) / len(rows)
        def mmss(s): return f"{int(s)//60}m{int(s)%60:02d}s"
        print(f"\n[{date_label}] n={len(rows)}人  "
              f"LEFT平均:{mmss(avg_l)}  RIGHT平均:{mmss(avg_r)}  "
              f"インターバル平均:{mmss(avg_iv)}  合計平均:{mmss(avg_tot)}")

    # --- CSV保存 ---
    fields = [
        "date", "subject",
        "l_n", "l_start", "l_end", "l_sec", "l_time",
        "r_n", "r_start", "r_end", "r_sec", "r_time",
        "interval_sec", "interval_time",
        "total_sec", "total_time",
    ]
    out_csv = OUT_DIR / "exam_time_all.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n保存: {out_csv}")


if __name__ == "__main__":
    main()
