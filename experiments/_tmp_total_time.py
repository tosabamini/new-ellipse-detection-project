import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data/Repeatability/0603")

# narmadha RIGHT の LEFT混入ファイル名
narmadha_left_names = {f.name for f in (DATA_DIR / "narmadha/LEFT").glob("*.jpg")}

def get_times(eye_dir, exclude=None):
    times = []
    for f in eye_dir.glob("*.jpg"):
        if exclude and f.name in exclude:
            continue
        m = re.search(r"IMG_(\d{8})_(\d{6})", f.name)
        if m:
            times.append(datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"))
    return sorted(times)

print(f"{'Subject':<14} {'L_start':>8}  {'R_end':>8}  {'Total':>8}  {'L_time':>7}  {'R_time':>7}  {'Between':>8}")
print("-" * 80)

totals = []
for subject_dir in sorted(DATA_DIR.iterdir()):
    if not subject_dir.is_dir():
        continue
    name = subject_dir.name

    l_times = get_times(subject_dir / "LEFT")
    excl = narmadha_left_names if name == "narmadha" else None
    r_times = get_times(subject_dir / "RIGHT", exclude=excl)

    if not l_times or not r_times:
        continue

    l_start, l_end = l_times[0], l_times[-1]
    r_start, r_end = r_times[0], r_times[-1]
    total_sec   = int((r_end - l_start).total_seconds())
    l_sec       = int((l_end - l_start).total_seconds())
    r_sec       = int((r_end - r_start).total_seconds())
    between_sec = int((r_start - l_end).total_seconds())

    def fmt(s): return f"{s//60}m{s%60:02d}s"

    print(f"{name:<14} {l_start.strftime('%H:%M:%S'):>8}  {r_end.strftime('%H:%M:%S'):>8}"
          f"  {fmt(total_sec):>8}  {fmt(l_sec):>7}  {fmt(r_sec):>7}  {fmt(between_sec):>8}")
    totals.append(total_sec)

avg = sum(totals) / len(totals)
print()
print(f"全{len(totals)}名  全体測定時間 平均: {int(avg)//60}m{int(avg)%60:02d}s")
print(f"  最短: {min(totals)//60}m{min(totals)%60:02d}s  最長: {max(totals)//60}m{max(totals)%60:02d}s")
