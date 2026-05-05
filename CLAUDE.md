# CLAUDE.md — Project Context for Claude Code

## What this project is

End-to-end red reflex analysis pipeline for ophthalmic images.  
Processes patient images through: RedEnhance → Classification → Segmentation → Ellipse fitting → Refraction estimation (S, C, A).  
A secondary workflow generates reference ellipse data from a model eye at known refraction powers, which is used as calibration data for refraction estimation.

---

## Architecture overview

All processing logic lives in `src/`. Paths are centralized in `src/common/paths.py` — always import from there, never hardcode paths.

Key modules:
- `src/preprocessing/preprocess_utils.py` — RedEnhance core (`center_crop`, `process_red_by_mode`, etc.)
- `src/classify/classifier_model.py` — SmallClassifier (1-ch CNN, input 160×72, threshold 0.9)
- `src/segmentation/segmentation_model.py` — UNetSmall (1-ch input/output, threshold 0.5)
- `src/ellipse/ellipse_utils.py` — `fit_ellipse_from_mask`, `make_pred_overlay`, `add_text_block`
- `src/pipeline/main.py` — end-to-end runner for patient data; also exports reusable functions
- `src/pipeline/run_model_eye.py` — model eye batch runner; imports directly from `main.py`
- `src/analysis/build_patient_model.py` — model eye reference calibration; `estimate_D(major, ratio)` → (p_est, D1, D2)
- `src/analysis/refraction_estimator.py` — refraction pipeline module; per-image D estimation + SCA trigonometric fit

---

## Shared-logic convention

`run_model_eye.py` intentionally imports `run_classifier_on_red`, `make_classify_overlay`, `load_classifier`, `EXTENSIONS`, `CLASSIFIER_THRESHOLD`, `DEVICE`, and `ensure_dir` **from `src.pipeline.main`** — not copied.  
When modifying classification or preprocessing logic, change it in `main.py` / `preprocess_utils.py` only; downstream scripts update automatically.

---

## Data layout

```
data/
├── raw/patient_data/         # source patient images
├── model_eye/                # model eye images
│   └── <refraction>/         # 49 folders, -6.00D to +6.00D, 0.25D steps
│       └── <pupil_mm>/       # e.g. 7.0mm, 5.0mm, 3.0mm
│           └── *.jpg
├── processed/
│   ├── pipeline_runs/        # output of src/pipeline/main.py
│   └── model_eye_runs/       # output of src/pipeline/run_model_eye.py
└── annotations/
```

### Model eye folder naming

Format: `CODE_M/Z/P_DD_DDd`  
Code = 1600 + refraction × 100. M = minus, Z = zero, P = plus.  
Examples: `1000_M_06_00D`, `1600_Z_00_00D`, `2200_P_06_00D`

Pupil diameter folders: `7.0mm`, `5.0mm`, `3.0mm`

---

## Model eye reference data workflow (current focus)

```
STEP 1  run_model_eye.py          RedEnhance + classification (done)
STEP 2  Labelme (manual)          annotate red/ images, label = "red_reflex"
STEP 3  run_model_eye_ellipse.py  JSON→mask + ellipse fitting  (not yet implemented)
STEP 4                            output ellipse_summary.csv per refraction power
```

STEP 3 script (`src/pipeline/run_model_eye_ellipse.py`) will:
- Read Labelme JSONs from `model_eye_runs/<run>/*/red/*.json`
- Convert to masks using same logic as `src/segmentation/json_to_mask.py`
- Run `fit_ellipse_from_mask` from `src/ellipse/ellipse_utils.py`
- Write per-folder results + `ellipse_summary.csv`

---

## Labelme annotation conventions

- Label name: `red_reflex` (must match `json_to_mask.py` filter)
- Annotate on the `red/` images (RedEnhance output), not raw images

---

## Key constants

| Constant | Value | Location |
|---|---|---|
| Classifier input size | 160 × 72 px | `main.py` |
| Classifier threshold | 0.9 | `main.py` |
| Segmentation threshold | 0.5 | `main.py` |
| Brightness VERY_DARK | < 7 | `preprocess_utils.py` |
| Brightness DARK | < 30 | `preprocess_utils.py` |
| CLAHE grid | 8 × 8 | `preprocess_utils.py` |
| SCALE_FACTOR (patient px correction) | 1.3 (暫定) | `refraction_estimator.py` |
| P_EST_MAX (noise filter) | 10.0 mm | `refraction_estimator.py` |
| MIN_VALID (SCA fit minimum) | 3 images | `refraction_estimator.py` |

---

## Run commands

```bash
# Patient data end-to-end
python -m src.pipeline.main --patient_ids 01 02 --run_name pipeline_run_v001

# Model eye: RedEnhance + classification (pupil_mm defaults to 7.0mm)
python -m src.pipeline.run_model_eye --run_name model_eye_7mm_v001 --pupil_mm 7.0mm

# Model eye: specific folders only
python -m src.pipeline.run_model_eye --run_name test --pupil_mm 7.0mm --folders 1600_Z_00_00D 1625_P_00_25D

# Standalone RedEnhance (patient data)
python -m src.preprocessing.redenhance

# Segmentation training
python -m src.segmentation.train_segmentation \
  --dataset_name seg_dataset_v001 --run_name seg_v001 --epochs 30

# Streamlit UI
streamlit run src/ui/app.py
```

---

## Refraction estimation model (src/analysis/)

### Physical model

```
Ellipse (major_px, minor_px, angle_deg)
    ↓
ratio = minor / major
major_scaled = major_px × SCALE_FACTOR          [px scale correction]
p_est = major_to_pupil(major_scaled)            [major → pupil diameter (mm)]
D1, D2 = estimate_D_from_ratio_and_p(ratio, p_est)  [ratio → refraction (D), 2 solutions]
adopted_D = D2                                  [myopic side adopted]
    ↓  (per patient, across all images with angle α)
D = P0 + P1·cos(2α) + P2·sin(2α)              [trigonometric fit]
    ↓
SE = P0
C  = -2·sqrt(P1^2 + P2^2)   (cylinder, minus notation)
S  = SE - C/2
A  = 0.5·atan2(-P2, -P1) % 180  (cylinder axis, deg)
```

### Calibration data (build_patient_model.py)

Model eye reference: 3 pupil sizes × multiple refraction powers.  
Reference major axis values: 3 mm → 130 px, 5 mm → 180 px, 7 mm → 200 px (model eye scale).

Fitted relationships (all quadratic):
- `p = 0.000857·M^2 − 0.22571·M + 17.857` (major → pupil mm)
- `ratio = a(p)·D^2 + b(p)·D + c(p)` with a, b, c interpolated as quadratics in p

### Pipeline output files (per patient)

| File | Contents |
|---|---|
| `results.csv` | ellipse parameters per image |
| `refraction_per_image.csv` | D1, D2, adopted_D, p_est per image |
| `refraction_sca.csv` | S, C, A, SE, R2, n for the patient |

---

## Known open problems (deferred)

### Major axis instability near emmetropia
When ratio < ~0.2 (near 0 D to +1 D), the major axis unexpectedly increases rather than
remaining stable with pupil diameter. Root cause unknown. The current patient model uses
fixed reference major values per pupil size (3 mm→130 px, 5 mm→180 px, 7 mm→200 px)
derived from the myopic range as an approximation. Do not try to "fix" this until the cause
is understood.

### 2-solution problem (D1 / D2)
Inverting the quadratic ratio formula yields two refraction solutions. Currently D2 (myopic
side) is adopted unconditionally. The user is working on a separate resolution strategy —
do not propose solutions.

---

## What NOT to do

- Do not hardcode paths outside of `src/common/paths.py`
- Do not copy processing logic — import from the source module
- Do not use `argparse` at module level (only inside `parse_args()` / `main()`)
- Do not add `data/`, `models/`, `reports/` content to git (large files / patient data)
