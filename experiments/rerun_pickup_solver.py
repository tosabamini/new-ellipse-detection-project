"""
PickUP 全患者の solve_summary.csv を poly10 モデルで再生成。

ellipse_results.csv から ratio/area を読み、poly10 joint solver で (D,p) を再計算。

Run:
  python experiments/rerun_pickup_solver.py
"""

import sys, io, csv
import numpy as np
from pathlib import Path
from scipy.optimize import minimize, brentq

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.ratio_model import ratio_real
from src.analysis.area_model import area_real

PICKUP_DIR   = Path("data/Repeatability/PickUP")
RATIO_THRESH = 0.13
D_MIN, D_MAX = -8.0, 0.0
P_MIN, P_MAX =  2.0, 9.0


def _loss(x, ro, ao):
    D, p = x
    return ((ratio_real(D, p) - ro) / ro)**2 + ((area_real(D, p) - ao) / ao)**2


def solve_one(ro, ao):
    if ro < RATIO_THRESH:
        try:
            p = float(brentq(lambda pp: area_real(0.0, pp) - ao, P_MIN, P_MAX))
        except Exception:
            p = float("nan")
        return dict(D=0.0, p=p,
                    ratio_fit=ratio_real(0.0, p) if not np.isnan(p) else float("nan"),
                    area_fit =area_real(0.0, p)  if not np.isnan(p) else float("nan"),
                    status="unmeasurable")

    D_c = np.linspace(D_MIN, D_MAX, 65)
    P_c = np.linspace(P_MIN, P_MAX, 29)
    best = np.inf; x0 = [-1.0, 4.0]
    for d0 in D_c:
        for p0 in P_c:
            l = _loss([d0, p0], ro, ao)
            if l < best:
                best = l; x0 = [d0, p0]

    res = minimize(_loss, x0, args=(ro, ao), method="L-BFGS-B",
                   bounds=[(D_MIN, D_MAX), (P_MIN, P_MAX)],
                   options=dict(ftol=1e-12, gtol=1e-9, maxiter=500))
    D_sol, p_sol = res.x
    return dict(D=float(D_sol), p=float(p_sol),
                ratio_fit=ratio_real(D_sol, p_sol),
                area_fit =area_real(D_sol, p_sol),
                status="ok")


def run_folder(eye_dir: Path):
    csv_path = eye_dir / "ellipse_results.csv"
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    results = []
    for r in rows:
        if not r.get("major") or not r.get("minor"):
            continue
        maj = float(r["major"]); mino = float(r["minor"])
        ro = mino / maj; ao = maj * mino
        sol = solve_one(ro, ao)
        results.append((r["stem"], ro, ao, sol))

    out_dir = eye_dir / "ratio_area_solve"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "solve_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "ratio_obs", "area_obs", "D", "p_mm",
                    "ratio_fit", "area_fit", "loss", "status"])
        for stem, ro, ao, s in results:
            w.writerow([stem, f"{ro:.4f}", f"{ao:.1f}",
                        f"{s['D']:.4f}", f"{s['p']:.4f}",
                        f"{s['ratio_fit']:.4f}", f"{s['area_fit']:.1f}",
                        f"{s['loss']:.6f}" if not np.isnan(s.get('loss', float('nan'))) else "nan",
                        s["status"]])
    print(f"  {eye_dir.parent.name}/{eye_dir.name}: {len(results)} images")


def main():
    for subj_dir in sorted(PICKUP_DIR.iterdir()):
        if not subj_dir.is_dir():
            continue
        for eye in ("LEFT", "RIGHT"):
            eye_dir = subj_dir / eye
            if eye_dir.is_dir():
                run_folder(eye_dir)
    print("完了: 全 PickUP solve_summary.csv を poly10 モデルで更新")


if __name__ == "__main__":
    main()
