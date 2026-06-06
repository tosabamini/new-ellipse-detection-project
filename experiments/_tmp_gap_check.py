import re
from datetime import datetime
from pathlib import Path

base = Path("data/Repeatability/0603/narmadha")

for eye in ["LEFT", "RIGHT"]:
    eye_dir = base / eye
    times = []
    for f in sorted(eye_dir.glob("*.jpg")):
        m = re.search(r"IMG_(\d{8})_(\d{6})", f.name)
        if m:
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            times.append((dt, f.name))
    times.sort()
    print(f"=== {eye} ({len(times)}枚) ===")
    for dt, name in times:
        print(f"  {dt.strftime('%H:%M:%S')}  {name}")
    print()
