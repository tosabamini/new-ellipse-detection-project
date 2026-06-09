"""Compare linear interpolation vs 10th-degree polynomial SCA results."""
import sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

lin = {
    ("0603","Kavya","L"):      dict(SE=-1.02,S=-0.92,C=-0.20),
    ("0603","Kavya","R"):      dict(SE=-2.07,S=-1.28,C=-1.58),
    ("0604","Kavya","L"):      dict(SE=-1.08,S=-0.08,C=-2.00),
    ("0604","Kavya","R"):      dict(SE=-1.24,S=-1.17,C=-0.13),
    ("0603","Linsha","L"):     dict(SE=-1.00,S=-0.86,C=-0.28),
    ("0603","Linsha","R"):     dict(SE=-1.19,S=-0.84,C=-0.70),
    ("0604","Linsha","L"):     dict(SE=-0.53,S= 0.05,C=-1.16),
    ("0604","Linsha","R"):     dict(SE=-1.10,S=-0.96,C=-0.28),
    ("0603","Dilsha","L"):     dict(SE=-0.47,S= 0.43,C=-1.81),
    ("0603","Dilsha","R"):     dict(SE=-1.20,S=-0.38,C=-1.64),
    ("0604","Dilsha","L"):     dict(SE=-1.15,S=-1.03,C=-0.24),
    ("0604","Dilsha","R"):     dict(SE=-1.03,S=-0.93,C=-0.20),
    ("0603","Abhishek","L"):   dict(SE=-3.24,S=-2.54,C=-1.39),
    ("0603","Abhishek","R"):   dict(SE=-4.77,S=-2.68,C=-4.19),
    ("0604","Abhishek","L"):   dict(SE=-5.68,S=-4.66,C=-2.05),
    ("0604","Abhishek","R"):   dict(SE=-4.35,S=-2.84,C=-3.01),
    ("0603","Prateeksha","L"): dict(SE=-1.05,S=-0.63,C=-0.83),
    ("0603","Prateeksha","R"): dict(SE=-1.16,S=-0.81,C=-0.70),
    ("0604","Prateeksha","L"): dict(SE=-0.99,S= 0.08,C=-2.14),
    ("0604","Prateeksha","R"): dict(SE=-1.16,S=-0.58,C=-1.16),
    ("0603","Aslaha","L"):     dict(SE= 0.04,S= 2.26,C=-4.45),
    ("0603","Aslaha","R"):     dict(SE=-1.83,S=-0.28,C=-3.10),
    ("0604","Aslaha","L"):     dict(SE=-0.50,S= 0.37,C=-1.74),
    ("0604","Aslaha","R"):     dict(SE=-0.24,S= 0.71,C=-1.90),
    ("0603","Anagha","L"):     dict(SE=-1.16,S=-0.63,C=-1.06),
    ("0603","Anagha","R"):     dict(SE=-0.98,S=-0.71,C=-0.53),
    ("0604","Anagha","L"):     dict(SE=-0.63,S=-0.41,C=-0.45),
    ("0604","Anagha","R"):     dict(SE=-0.83,S=-0.59,C=-0.48),
    ("0603","Aiswarya","L"):   dict(SE=-1.00,S=-0.04,C=-1.91),
    ("0603","Aiswarya","R"):   dict(SE=-0.71,S= 0.42,C=-2.25),
    ("0604","Aiswarya","L"):   dict(SE=-0.82,S=-0.25,C=-1.15),
    ("0604","Aiswarya","R"):   dict(SE=-0.73,S= 0.37,C=-2.21),
    ("0603","Lubana","L"):     dict(SE=-0.95,S=-0.87,C=-0.15),
    ("0603","Lubana","R"):     dict(SE=-1.00,S= 0.11,C=-2.23),
    ("0604","Lubana","L"):     dict(SE=-0.71,S= 0.27,C=-1.95),
    ("0604","Lubana","R"):     dict(SE=-0.63,S= 0.07,C=-1.41),
    ("0603","Rinsha","L"):     dict(SE=-0.84,S= 0.23,C=-2.14),
    ("0603","Rinsha","R"):     dict(SE=-1.19,S=-0.41,C=-1.57),
    ("0604","Rinsha","L"):     dict(SE=-0.45,S= 1.40,C=-3.69),
    ("0604","Rinsha","R"):     dict(SE=-0.91,S= 0.34,C=-2.49),
    ("0603","Fathima","L"):    dict(SE=-4.72,S=-3.75,C=-1.93),
    ("0603","Fathima","R"):    dict(SE=-5.05,S=-4.17,C=-1.76),
    ("0604","Fathima","L"):    dict(SE=-5.60,S=-4.59,C=-2.02),
    ("0604","Fathima","R"):    dict(SE=-5.55,S=-3.85,C=-3.41),
    ("0603","Niranjana","L"):  dict(SE=-0.98,S=-0.63,C=-0.69),
    ("0603","Niranjana","R"):  dict(SE=-1.26,S=-0.92,C=-0.68),
    ("0604","Niranjana","L"):  dict(SE=-1.56,S=-1.17,C=-0.78),
    ("0604","Niranjana","R"):  dict(SE=-2.57,S=-2.40,C=-0.34),
}

poly = {
    ("0603","Kavya","L"):      dict(SE=-1.03,S=-0.89,C=-0.27),
    ("0603","Kavya","R"):      dict(SE=-2.08,S=-1.26,C=-1.64),
    ("0604","Kavya","L"):      dict(SE=-1.10,S=-0.09,C=-2.03),
    ("0604","Kavya","R"):      dict(SE=-1.35,S=-1.17,C=-0.35),
    ("0603","Linsha","L"):     dict(SE=-0.99,S=-0.66,C=-0.66),
    ("0603","Linsha","R"):     dict(SE=-1.28,S=-0.84,C=-0.89),
    ("0604","Linsha","L"):     dict(SE=-0.42,S= 0.33,C=-1.51),
    ("0604","Linsha","R"):     dict(SE=-0.94,S=-0.67,C=-0.54),
    ("0603","Dilsha","L"):     dict(SE=-0.46,S= 0.52,C=-1.97),
    ("0603","Dilsha","R"):     dict(SE=-1.21,S=-0.39,C=-1.66),
    ("0604","Dilsha","L"):     dict(SE=-1.05,S=-1.00,C=-0.09),
    ("0604","Dilsha","R"):     dict(SE=-1.04,S=-0.85,C=-0.38),
    ("0603","Abhishek","L"):   dict(SE=-3.19,S=-2.45,C=-1.48),
    ("0603","Abhishek","R"):   dict(SE=-4.54,S=-2.60,C=-3.88),
    ("0604","Abhishek","L"):   dict(SE=-5.66,S=-4.56,C=-2.21),
    ("0604","Abhishek","R"):   dict(SE=-4.25,S=-2.73,C=-3.04),
    ("0603","Prateeksha","L"): dict(SE=-1.07,S=-0.63,C=-0.89),
    ("0603","Prateeksha","R"): dict(SE=-1.20,S=-0.88,C=-0.63),
    ("0604","Prateeksha","L"): dict(SE=-0.99,S= 0.14,C=-2.25),
    ("0604","Prateeksha","R"): dict(SE=-1.22,S=-0.68,C=-1.09),
    ("0603","Aslaha","L"):     dict(SE= 0.09,S= 2.42,C=-4.65),
    ("0603","Aslaha","R"):     dict(SE=-1.82,S=-0.26,C=-3.11),
    ("0604","Aslaha","L"):     dict(SE=-0.46,S= 0.41,C=-1.73),
    ("0604","Aslaha","R"):     dict(SE=-0.20,S= 0.84,C=-2.09),
    ("0603","Anagha","L"):     dict(SE=-1.20,S=-0.71,C=-0.99),
    ("0603","Anagha","R"):     dict(SE=-0.90,S=-0.65,C=-0.51),
    ("0604","Anagha","L"):     dict(SE=-0.57,S=-0.30,C=-0.53),
    ("0604","Anagha","R"):     dict(SE=-0.80,S=-0.59,C=-0.41),
    ("0603","Aiswarya","L"):   dict(SE=-0.95,S= 0.07,C=-2.02),
    ("0603","Aiswarya","R"):   dict(SE=-0.56,S= 0.73,C=-2.58),
    ("0604","Aiswarya","L"):   dict(SE=-0.74,S=-0.03,C=-1.40),
    ("0604","Aiswarya","R"):   dict(SE=-0.67,S= 0.62,C=-2.57),
    ("0603","Lubana","L"):     dict(SE=-0.91,S=-0.79,C=-0.23),
    ("0603","Lubana","R"):     dict(SE=-0.97,S= 0.20,C=-2.34),
    ("0604","Lubana","L"):     dict(SE=-0.65,S= 0.40,C=-2.11),
    ("0604","Lubana","R"):     dict(SE=-0.56,S= 0.17,C=-1.47),
    ("0603","Rinsha","L"):     dict(SE=-0.76,S= 0.36,C=-2.25),
    ("0603","Rinsha","R"):     dict(SE=-1.19,S=-0.38,C=-1.62),
    ("0604","Rinsha","L"):     dict(SE=-0.47,S= 1.39,C=-3.70),
    ("0604","Rinsha","R"):     dict(SE=-0.83,S= 0.51,C=-2.69),
    ("0603","Fathima","L"):    dict(SE=-4.65,S=-3.69,C=-1.92),
    ("0603","Fathima","R"):    dict(SE=-5.03,S=-4.08,C=-1.91),
    ("0604","Fathima","L"):    dict(SE=-5.56,S=-4.44,C=-2.22),
    ("0604","Fathima","R"):    dict(SE=-5.51,S=-3.78,C=-3.47),
    ("0603","Niranjana","L"):  dict(SE=-1.06,S=-0.54,C=-1.05),
    ("0603","Niranjana","R"):  dict(SE=-1.33,S=-0.93,C=-0.81),
    ("0604","Niranjana","L"):  dict(SE=-1.63,S=-1.25,C=-0.77),
    ("0604","Niranjana","R"):  dict(SE=-2.64,S=-2.45,C=-0.38),
}

keys = sorted(lin.keys())
dSE, dS, dC = [], [], []
hdr = f"{'Date':4} {'Patient':12} {'Eye':3} | {'linSE':>7} {'polSE':>7} {'dSE':>6} | {'linS':>6} {'polS':>6} {'dS':>6} | {'linC':>6} {'polC':>6} {'dC':>6}"
print(hdr)
print("-" * len(hdr))
for k in keys:
    l = lin[k]; p = poly[k]
    dse = p['SE']-l['SE']; ds = p['S']-l['S']; dc = p['C']-l['C']
    dSE.append(abs(dse)); dS.append(abs(ds)); dC.append(abs(dc))
    print(f"{k[0]:4} {k[1]:12} {k[2]:3} | {l['SE']:+7.2f} {p['SE']:+7.2f} {dse:+6.2f} | {l['S']:+6.2f} {p['S']:+6.2f} {ds:+6.2f} | {l['C']:+6.2f} {p['C']:+6.2f} {dc:+6.2f}")

print("=" * len(hdr))
print(f"MAE  |dSE|={np.mean(dSE):.3f}D  |dS|={np.mean(dS):.3f}D  |dC|={np.mean(dC):.3f}D")
print(f"最大 |dSE|={np.max(dSE):.3f}D  |dS|={np.max(dS):.3f}D  |dC|={np.max(dC):.3f}D")
print(f"中央 |dSE|={np.median(dSE):.3f}D |dS|={np.median(dS):.3f}D |dC|={np.median(dC):.3f}D")
