"""
全患者の cosine fit グラフを生成して1枚のグリッド画像に並べる。

Run:
  python experiments/plot_cos_curves_all.py
"""

import sys, io, cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize, brentq

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.preprocess_utils import center_crop
from src.ellipse.adaptdog import (
    red_channel, stretch_to_255,
    _otsu_mask, _pick_central_blob, _fit_ellipse_on_mask, _estimate_minor,
    iqr_filter, d_iqr_filter,
)
from src.analysis.ratio_model import ratio_real
from src.analysis.area_model import area_real
from src.analysis.refraction_estimator import fit_sca

RATIO_THRESH = 0.13
D_MIN, D_MAX = -8.0, 0.0
P_MIN, P_MAX =  2.0, 9.0
IQR_K, D_IQR_K = 0.5, 1.5
CROP_RATIO = 0.2

REPEAT_DIR = Path("data/Repeatability")
OUT_DIR    = REPEAT_DIR / "cos_curves"
OUT_DIR.mkdir(exist_ok=True)

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


def core_fit(red_str):
    """現行手法: mask_core から直接 fitEllipse (dilation なし)。"""
    minor_est=_estimate_minor(red_str); sigma_l=max(8.0,minor_est*0.75)
    bs=cv2.GaussianBlur(red_str.astype(np.float32),(0,0),1.5)
    bl=cv2.GaussianBlur(red_str.astype(np.float32),(0,0),sigma_l)
    dog=stretch_to_255(np.clip(bs-bl,0,None))
    return _fit_ellipse_on_mask(_pick_central_blob(_otsu_mask(dog)))

def _loss(x,ro,ao):
    D,p=x
    return ((ratio_real(D,p)-ro)/ro)**2+((area_real(D,p)-ao)/ao)**2

def solve_one(ro,ao):
    if ro<RATIO_THRESH:
        try: p=float(brentq(lambda pp:area_real(0.,pp)-ao,P_MIN,P_MAX))
        except: p=float("nan")
        return 0.0, p
    D_c=np.linspace(D_MIN,D_MAX,65); P_c=np.linspace(P_MIN,P_MAX,29)
    best=np.inf; x0=[-1.,4.]
    for d0 in D_c:
        for p0 in P_c:
            l=_loss([d0,p0],ro,ao)
            if l<best: best=l; x0=[d0,p0]
    res=minimize(_loss,x0,args=(ro,ao),method="L-BFGS-B",
                 bounds=[(D_MIN,D_MAX),(P_MIN,P_MAX)],
                 options=dict(ftol=1e-12,gtol=1e-9,maxiter=500))
    return float(res.x[0]), float(res.x[1])

def angle_bin(deg):
    a=float(deg)%180
    if 70<=a<110: return "90deg"
    if 30<=a<60:  return "45deg"
    if a<20 or a>=160: return "0deg"
    return "other"

def process_eye(img_dir):
    img_paths=sorted(img_dir.glob("*.jpg"))+sorted(img_dir.glob("*.JPG"))
    if not img_paths: return None
    ellipses=[]; stems=[]
    for p in img_paths:
        img=cv2.imread(str(p))
        if img is None: continue
        e=core_fit(stretch_to_255(red_channel(center_crop(img, CROP_RATIO))))
        ellipses.append(e); stems.append(p.stem)
    keep=iqr_filter(ellipses,k=IQR_K)
    per_image=[]
    for stem,e,k in zip(stems,ellipses,keep):
        if not k or not e: continue
        ro=e["minor"]/e["major"]; ao=e["major"]*e["minor"]
        D,p=solve_one(ro,ao)
        if np.isnan(D) or np.isnan(p): continue
        per_image.append({"stem":stem,"angle":e["angle"],"D":D,
                          "angle_bin":angle_bin(e["angle"]),"adopted_D":D})
    if len(per_image)<3: return None
    d_mask=d_iqr_filter(per_image,k=D_IQR_K)
    valid=[x for x,m in zip(per_image,d_mask) if m]
    if len(valid)<3: return None
    return valid


def plot_eye(ax, valid, label, color):
    alpha_arr=np.array([x["angle"] for x in valid])
    D_arr    =np.array([x["D"]     for x in valid])
    sca=fit_sca(alpha_arr,D_arr)

    a_fine=np.linspace(0,180,360)
    a_rad=np.deg2rad(alpha_arr)
    X=np.column_stack([np.ones(len(a_rad)),np.cos(2*a_rad),np.sin(2*a_rad)])
    P,*_=np.linalg.lstsq(X,D_arr,rcond=None)
    D_fit=P[0]+P[1]*np.cos(2*np.deg2rad(a_fine))+P[2]*np.sin(2*np.deg2rad(a_fine))

    ax.scatter(alpha_arr,D_arr,color=color,s=20,zorder=3,alpha=0.8)
    ax.plot(a_fine,D_fit,color=color,lw=1.5,
            label=f"{label}: SE={sca['SE']:+.2f} C={sca['C']:+.2f} R²={sca['R2']:.2f} n={sca['n']}")
    return sca


def main():
    # ── 各眼を1回だけ処理してキャッシュ ───────────────────────────────────
    cache = {}   # (name, eye) -> (v03, v04)
    for name, f03, f04 in PAIRS_12:
        print(f"Processing {name}...")
        for eye in ("LEFT", "RIGHT"):
            v03 = process_eye(REPEAT_DIR/"0603"/f03/eye)
            v04 = process_eye(REPEAT_DIR/"0604"/f04/eye)
            cache[(name, eye)] = (v03, v04)

    # ── 各患者ペアごとに LEFT/RIGHT を1枚に ────────────────────────────────
    for name, f03, f04 in PAIRS_12:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        fig.suptitle(f"{name}  (blue=0603, red=0604)", fontsize=12, fontweight="bold")
        for ax, eye in zip(axes, ("LEFT", "RIGHT")):
            ax.set_title(f"{eye}", fontsize=10)
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.set_xlim(0, 180); ax.set_xlabel("angle (deg)"); ax.set_ylabel("D (D)")
            ax.grid(alpha=0.25)
            v03, v04 = cache[(name, eye)]
            if v03: plot_eye(ax, v03, "0603", "#2980b9")
            if v04: plot_eye(ax, v04, "0604", "#e74c3c")
            if v03 or v04:
                ax.legend(fontsize=7.5, loc="upper right")
        plt.tight_layout()
        out=OUT_DIR/f"{name}.png"
        plt.savefig(str(out), dpi=130)
        plt.close(fig)
        print(f"  -> {out}")

    # ── 全患者グリッド (12×2 = 24サブプロット) ───────────────────────────
    fig, axes = plt.subplots(12, 2, figsize=(14, 4.5*12))
    fig.suptitle("Cos-fit results — all 12 patients (blue=0603, red=0604)", fontsize=14, fontweight="bold")
    for row, (name, f03, f04) in enumerate(PAIRS_12):
        for col, eye in enumerate(("LEFT", "RIGHT")):
            ax=axes[row][col]
            ax.set_title(f"{name} / {eye}", fontsize=9)
            ax.axhline(0,color="gray",lw=0.5,ls="--")
            ax.set_xlim(0,180); ax.set_xlabel("angle (deg)",fontsize=7); ax.set_ylabel("D (D)",fontsize=7)
            ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
            v03, v04 = cache[(name, eye)]
            if v03: plot_eye(ax,v03,"0603","#2980b9")
            if v04: plot_eye(ax,v04,"0604","#e74c3c")
            if v03 or v04: ax.legend(fontsize=6.5,loc="upper right")
    plt.tight_layout(rect=[0,0,1,0.98])
    out_grid=OUT_DIR/"all_patients_grid.png"
    plt.savefig(str(out_grid), dpi=110)
    plt.close(fig)
    print(f"\n全体グリッド -> {out_grid}")


if __name__ == "__main__":
    main()
