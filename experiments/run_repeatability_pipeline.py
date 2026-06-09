"""
Repeatability データ (0603/0604) の全患者に対して
AdaptDoG → IQR → poly10 joint solver → D-IQR → SCA を適用し CSV 出力。

Run:
  python experiments/run_repeatability_pipeline.py
"""

import sys, io, csv, cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize, brentq

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ellipse.adaptdog import run_adaptive_dog, iqr_filter, d_iqr_filter
from src.preprocessing.preprocess_utils import process_red_by_mode
from src.analysis.ratio_model import ratio_real
from src.analysis.area_model import area_real
from src.analysis.refraction_estimator import fit_sca

RATIO_THRESH = 0.13
D_MIN, D_MAX = -8.0, 0.0
P_MIN, P_MAX =  2.0, 9.0
IQR_K = 0.5
D_IQR_K = 1.5
CROP_RATIO = 0.2

REPEAT_DIR = Path("data/Repeatability")
OUT_CSV    = REPEAT_DIR / "sca_all_patients.csv"
COMP_CSV   = REPEAT_DIR / "sca_comparison_0603_0604.csv"

PAIRS_12 = [
    ("Kavya",      "kavya",      "kavya2"),
    ("Linsha",     "linsha",     "linsha2"),
    ("Dilsha",     "dilsha",     "dilsha2"),
    ("Abhishek",   "Abhishek",   "Abhishek 2"),
    ("Prateeksha", "Prateeksha", "prateeksha2"),
    ("Aslaha",     "aslaha",     "aslaha 2"),
    ("Anagha",     "anagha",     "anagha2"),
    ("Aiswarya",   "aiswarya",   "aiswarya 2"),
    ("Lubana",     "lubana",     "lubana2"),
    ("Rinsha",     "rinsha",     "rinsha2"),
    ("Fathima",    "fathima",    "Fathima2"),
    ("Niranjana",  "niranjana",  "Niranjana 2"),
]


def red_enhance(img):
    r = img[:, :, 2].astype(float)
    g = img[:, :, 1].astype(float)
    b = img[:, :, 0].astype(float)
    red = r - 0.5 * g - 0.5 * b
    mn, mx = red.min(), red.max()
    if mx > mn:
        red = (red - mn) / (mx - mn) * 255
    return red.clip(0, 255).astype(np.uint8)


def center_crop(img, crop_ratio=0.2):
    h, w = img.shape[:2]
    new_w = int(w * crop_ratio)
    cx = w // 2
    x0 = max(0, cx - new_w // 2)
    x1 = x0 + new_w
    return img[:, x0:x1]


def _loss(x, ratio_obs, area_obs):
    D, p = x
    r_res = (ratio_real(D, p) - ratio_obs) / ratio_obs
    a_res = (area_real(D, p)  - area_obs)  / area_obs
    return r_res**2 + a_res**2


def solve_one(ratio_obs, area_obs):
    if ratio_obs < RATIO_THRESH:
        try:
            p = float(brentq(lambda pp: area_real(0.0, pp) - area_obs, P_MIN, P_MAX))
        except Exception:
            p = float("nan")
        return dict(D=0.0, p=p,
                    ratio_fit=ratio_real(0.0, p) if not np.isnan(p) else float("nan"),
                    area_fit =area_real(0.0, p)  if not np.isnan(p) else float("nan"),
                    status="unmeasurable")

    D_coarse = np.linspace(D_MIN, D_MAX, 65)
    P_coarse = np.linspace(P_MIN, P_MAX, 29)
    best_loss = np.inf; best_x0 = [-1.0, 4.0]
    for d0 in D_coarse:
        for p0 in P_coarse:
            l = _loss([d0, p0], ratio_obs, area_obs)
            if l < best_loss:
                best_loss = l; best_x0 = [d0, p0]

    res = minimize(_loss, best_x0, args=(ratio_obs, area_obs),
                   method="L-BFGS-B",
                   bounds=[(D_MIN, D_MAX), (P_MIN, P_MAX)],
                   options=dict(ftol=1e-12, gtol=1e-9, maxiter=500))
    D_sol, p_sol = res.x
    return dict(D=float(D_sol), p=float(p_sol),
                ratio_fit=ratio_real(D_sol, p_sol),
                area_fit =area_real(D_sol, p_sol),
                status="ok")


def process_eye(img_dir: Path):
    """画像フォルダ1つ分を処理して SCA を返す。"""
    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.JPG"))
    if not img_paths:
        return None

    ellipses, stems = [], []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        roi = center_crop(img, CROP_RATIO)
        red = red_enhance(roi)
        e = run_adaptive_dog(red)
        ellipses.append(e)
        stems.append(p.stem)

    keep_mask = iqr_filter(ellipses, k=IQR_K)

    per_image = []
    for stem, e, keep in zip(stems, ellipses, keep_mask):
        if not keep or not e:
            continue
        ratio_obs = e["minor"] / e["major"]
        area_obs  = e["major"] * e["minor"]
        sol = solve_one(ratio_obs, area_obs)
        if np.isnan(sol["D"]) or np.isnan(sol["p"]):
            continue
        per_image.append({
            "stem":  stem,
            "ratio": ratio_obs,
            "area":  area_obs,
            "angle": e["angle"],
            "D":     sol["D"],
            "p":     sol["p"],
            "status": sol["status"],
        })

    if len(per_image) < 3:
        return None

    def _angle_bin(deg):
        a = float(deg) % 180
        if 70 <= a < 110: return "90deg"
        if 30 <= a < 60:  return "45deg"
        if a < 20 or a >= 160: return "0deg"
        return "other"

    d_mask = d_iqr_filter(
        [{"angle_bin": _angle_bin(x["angle"]), "adopted_D": x["D"]} for x in per_image],
        k=D_IQR_K)
    per_image = [x for x, k in zip(per_image, d_mask) if k]

    if len(per_image) < 3:
        return None

    alpha_arr = np.array([x["angle"] for x in per_image])
    D_arr     = np.array([x["D"]     for x in per_image])
    sca = fit_sca(alpha_arr, D_arr)
    sca["n_img"] = len(per_image)
    return sca


def run_date(date_str: str) -> dict:
    """date_str: '0603' or '0604'. 患者名 -> {eye -> sca} を返す。"""
    base = REPEAT_DIR / date_str
    results = {}
    for patient_dir in sorted(base.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient = patient_dir.name
        results[patient] = {}
        for eye in ("LEFT", "RIGHT"):
            eye_dir = patient_dir / eye
            if not eye_dir.is_dir():
                continue
            sca = process_eye(eye_dir)
            if sca:
                results[patient][eye] = sca
    return results


def main():
    print("=== 0603 処理中 ===")
    r03 = run_date("0603")
    print("=== 0604 処理中 ===")
    r04 = run_date("0604")

    # ── sca_all_patients.csv ─────────────────────────────────────────────
    rows_all = []
    for date_str, rdict in [("0603", r03), ("0604", r04)]:
        for patient in sorted(rdict):
            for eye in ("LEFT", "RIGHT"):
                sca = rdict[patient].get(eye)
                if sca:
                    rows_all.append({
                        "date": date_str, "patient": patient, "eye": eye,
                        "S": sca["S"], "C": sca["C"], "A": sca["A"],
                        "SE": sca["SE"], "R2": sca["R2"],
                    })

    fields_all = ["date", "patient", "eye", "S", "C", "A", "SE", "R2"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields_all)
        w.writeheader()
        for r in rows_all:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\n保存: {OUT_CSV}  ({len(rows_all)} 行)")

    # ── sca_comparison_0603_0604.csv ─────────────────────────────────────
    comp_rows = []
    for name, f03, f04 in PAIRS_12:
        for eye in ("LEFT", "RIGHT"):
            s03 = r03.get(f03, {}).get(eye)
            s04 = r04.get(f04, {}).get(eye)
            if not s03 or not s04:
                continue
            dSE = s04["SE"] - s03["SE"]
            dS  = s04["S"]  - s03["S"]
            dC  = s04["C"]  - s03["C"]
            raw = abs(s03["A"] - s04["A"])
            dA  = min(raw, 180 - raw)
            comp_rows.append({
                "subject": name, "eye": eye,
                "SE_0603": s03["SE"], "S_0603": s03["S"],
                "C_0603":  s03["C"],  "A_0603": s03["A"], "R2_0603": s03["R2"],
                "SE_0604": s04["SE"], "S_0604": s04["S"],
                "C_0604":  s04["C"],  "A_0604": s04["A"], "R2_0604": s04["R2"],
                "dSE": dSE, "dS": dS, "dC": dC, "dA": dA,
            })

    fields_comp = ["subject", "eye",
                   "SE_0603", "S_0603", "C_0603", "A_0603", "R2_0603",
                   "SE_0604", "S_0604", "C_0604", "A_0604", "R2_0604",
                   "dSE", "dS", "dC", "dA"]
    with open(COMP_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields_comp)
        w.writeheader()
        for r in comp_rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"保存: {COMP_CSV}  ({len(comp_rows)} 行)")

    # ── 表示 ─────────────────────────────────────────────────────────────
    print(f"\n{'被験者/眼':<22} {'SE_03':>7} {'SE_04':>7} {'dSE':>6} | {'S_03':>6} {'S_04':>6} {'dS':>6} | {'C_03':>6} {'C_04':>6} {'dC':>6}")
    print("-" * 100)
    for r in comp_rows:
        label = r["subject"] + "/" + r["eye"]
        print(f"{label:<22} {r['SE_0603']:>+7.2f} {r['SE_0604']:>+7.2f} {r['dSE']:>+6.2f} | "
              f"{r['S_0603']:>+6.2f} {r['S_0604']:>+6.2f} {r['dS']:>+6.2f} | "
              f"{r['C_0603']:>+6.2f} {r['C_0604']:>+6.2f} {r['dC']:>+6.2f}")

    if comp_rows:
        print("-" * 100)
        mae_se = np.mean([abs(r["dSE"]) for r in comp_rows])
        mae_s  = np.mean([abs(r["dS"])  for r in comp_rows])
        mae_c  = np.mean([abs(r["dC"])  for r in comp_rows])
        print(f"{'MAE':22} {'':>7} {'':>7} {mae_se:>+6.2f} | {'':>6} {'':>6} {mae_s:>+6.2f} | {'':>6} {'':>6} {mae_c:>+6.2f}")


if __name__ == "__main__":
    main()
