# Ellipse Detection Project (Red Reflex Pipeline)

End-to-end pipeline for **red reflex analysis** from ophthalmic images.  
Covers preprocessing, image quality classification, segmentation, ellipse fitting, and refraction estimation (Sphere / Cylinder / Axis).

---

## Project Structure

```
project/
├── data/
│   ├── raw/
│   │   └── patient_data/         # original patient images
│   ├── processed/
│   │   ├── redenhance/           # RedEnhance outputs (standalone runner)
│   │   ├── classify_outputs/     # classification inference outputs
│   │   ├── segmentation_dataset/ # annotated dataset for training
│   │   ├── segmentation_inference/
│   │   ├── ellipse_outputs/
│   │   ├── pipeline_runs/        # end-to-end pipeline outputs (patient data)
│   │   └── model_eye_runs/       # pipeline outputs (model eye reference data)
│   ├── annotations/
│   │   ├── classify/
│   │   └── segmentation/
│   └── model_eye/                # model eye reference data (49 refraction folders)
│       ├── 1000_M_06_00D/        # -6.00 D
│       ├── 1025_M_05_75D/        # -5.75 D
│       ├── ...
│       ├── 1600_Z_00_00D/        # 0.00 D
│       ├── ...
│       └── 2200_P_06_00D/        # +6.00 D
│
├── models/
│   ├── classify/
│   │   └── best_classifier.pth
│   └── segmentation/
│       └── best_segmentation_model.pth
│
├── src/
│   ├── common/
│   │   └── paths.py              # all path constants
│   ├── preprocessing/
│   │   ├── preprocess_utils.py   # RedEnhance core functions
│   │   └── redenhance.py         # standalone batch runner
│   ├── classify/
│   │   ├── classifier_model.py   # SmallClassifier (CNN)
│   │   └── infer_classifier.py   # standalone inference runner
│   ├── segmentation/
│   │   ├── segmentation_model.py # UNetSmall
│   │   ├── prepare_segmentation_images.py
│   │   ├── json_to_mask.py       # Labelme JSON → binary mask
│   │   ├── train_segmentation.py
│   │   └── infer_segmentation.py
│   ├── ellipse/
│   │   ├── ellipse_utils.py          # ellipse fitting core functions
│   │   ├── fit_ellipse.py            # standalone ellipse runner
│   │   ├── ellipse_otsu_tester.py    # classical (DoG+Otsu) CLI tester — multi-method
│   │   └── compare_ellipse_gt_pred.py
│   ├── analysis/
│   │   ├── build_patient_model.py   # model eye calibration → estimate_D(major, ratio)
│   │   └── refraction_estimator.py  # per-image D + SCA trigonometric fit
│   ├── pipeline/
│   │   ├── main.py               # end-to-end pipeline (patient data) + SCA output
│   │   ├── run_model_eye.py      # RedEnhance + classification (model eye)
│   │   └── make_report.py        # generate report images (cos-curve, ellipse grid, classify grid)
│   └── ui/
│       └── app.py                # Streamlit UI
│
├── experiments/
│   ├── otsu_ellipse_single.py  # standalone single-image ellipse tester (no src.* imports)
│   └── classical_seg_trial.py  # earlier classical segmentation experiments
├── reports/
└── requirements.txt
```

---

## Patient Data Pipeline

Full pipeline for processing patient images end-to-end.

### 1. Red Enhancement

```bash
python -m src.preprocessing.redenhance
```

Output: `data/processed/redenhance/<version>/`

### 2. Classification Inference

```bash
python -m src.classify.infer_classifier \
  --redenhance_version default \
  --run_name clf_v001_on_default
```

Output: `data/processed/classify_outputs/<run_name>/`

### 3. Segmentation Dataset Preparation

```bash
python -m src.segmentation.prepare_segmentation_images \
  --classify_run_name clf_v001_on_default \
  --dataset_name seg_dataset_v001 \
  --max_train_images 120 \
  --max_val_images 40 \
  --max_test_images 40
```

Then annotate with **Labelme** (label name: `red_reflex`).

### 4. JSON → Mask Conversion

```bash
python -m src.segmentation.json_to_mask \
  --dataset_name seg_dataset_v001
```

### 5. Train Segmentation Model

```bash
python -m src.segmentation.train_segmentation \
  --dataset_name seg_dataset_v001 \
  --run_name seg_v001_on_seg_dataset_v001 \
  --epochs 30 --batch_size 4 --lr 1e-3
```

### 6. Segmentation Inference

```bash
python -m src.segmentation.infer_segmentation \
  --classify_run_name clf_v001_on_default \
  --run_name seginf_v001_on_clf_v001_on_default
```

### 7. Ellipse Fitting

```bash
python -m src.ellipse.fit_ellipse \
  --segmentation_run_name seginf_v001_on_clf_v001_on_default \
  --redenhance_version default \
  --run_name ellipse_v001
```

### 8. GT vs Prediction Evaluation

```bash
python -m src.ellipse.compare_ellipse_gt_pred \
  --segmentation_train_run_name seg_v001_on_seg_dataset_v001 \
  --redenhance_version default \
  --run_name ellipse_compare_v001
```

### 9. End-to-End (Production)

```bash
python -m src.pipeline.main \
  --patient_ids 01 02 \
  --run_name pipeline_run_v001
```

Output: `data/processed/pipeline_runs/<run_name>/<patient_id>/`

Each patient folder contains:

```
<patient_id>/
├── roi/                      # center-cropped original
├── red/                      # RedEnhance output
├── classify_overlay/         # classification visualization
├── seg_prob/                 # segmentation probability map
├── seg_pred/                 # binary segmentation mask
├── seg_overlay/              # segmentation visualization
├── ellipse_single_mask/      # largest connected mask region
├── ellipse_overlay/          # ellipse fit visualization
├── results.csv               # ellipse parameters per image
├── refraction_per_image.csv  # D1, D2, adopted_D, p_est per image
└── refraction_sca.csv        # S (sphere), C (cylinder), A (axis) for the patient
```

---

## Refraction Estimation (S / C / A)

The pipeline automatically estimates **Sphere, Cylinder, and Axis** for each patient using model eye reference data as calibration.

### Physical model

```
Ellipse (major_px, minor_px, angle_deg)  — per image
    ↓
ratio = minor / major
major_scaled = major_px × SCALE_FACTOR      (px scale correction,暫定 1.3)
p_est  = major_to_pupil(major_scaled)       (major → pupil diameter in mm)
D1, D2 = solve quadratic in ratio           (2 refraction solutions)
adopted_D = D2                              (myopic side adopted)
    ↓  across all valid images for one patient
D(α) = P0 + P1·cos(2α) + P2·sin(2α)       (trigonometric fit, α = major axis angle)
    ↓
SE = P0
C  = −2·√(P1² + P2²)    (cylinder, minus notation)
S  = SE − C/2
A  = ½·atan2(−P2, −P1) mod 180°            (cylinder axis)
```

### Calibration (src/analysis/build_patient_model.py)

Fitted from model eye measurements at 3 pupil sizes (3, 5, 7 mm) × multiple refraction powers:

- **Major → pupil**: `p = 0.000857·M² − 0.22571·M + 17.857`  
  Reference points: 3 mm → 130 px, 5 mm → 180 px, 7 mm → 200 px (model eye scale)
- **Ratio model**: `ratio = a(p)·D² + b(p)·D + c(p)` with a, b, c quadratic in p

### Output files

| File | Contents |
|---|---|
| `refraction_per_image.csv` | ratio, p_est, D1, D2, adopted_D per image |
| `refraction_sca.csv` | S, C, A, SE, R², n\_valid, n\_total per patient |

---

## Model Eye Reference Data Pipeline

Pipeline for generating ellipse reference data from a model eye at known refraction powers.  
Refraction range: **-6.00 D to +6.00 D in 0.25 D steps** (49 folders total).

Folder naming convention: `CODE_M/Z/P_DD_DDd`  
where code = 1600 + refraction × 100, M = minus, Z = zero, P = plus.  
Example: `1000_M_06_00D` (-6.00 D), `1600_Z_00_00D` (0.00 D), `2200_P_06_00D` (+6.00 D).

### STEP 1 — RedEnhance + Image Quality Classification

```bash
python -m src.pipeline.run_model_eye --run_name model_eye_v001
```

Options:
- `--run_name`  : output folder name (auto-generated from timestamp if omitted)
- `--folders`   : process specific refraction folders only

Output: `data/processed/model_eye_runs/<run_name>/<refraction_folder>/`

```
<refraction_folder>/
├── roi/              # center-cropped original
├── red/              # RedEnhance output (annotate these in Labelme)
├── classify_overlay/ # classification result visualization
└── results.csv
```

### STEP 2 — Manual Segmentation (Labelme)

Open images in `red/` with Labelme and draw polygons around the red reflex region.

- Label name: **`red_reflex`**
- Labelme saves JSON alongside each image automatically

### STEP 3-4 — JSON → Mask + Ellipse Fitting (planned)

Script `src/pipeline/run_model_eye_ellipse.py` (to be implemented):
- Converts Labelme JSONs to binary masks (same logic as `json_to_mask.py`)
- Runs ellipse fitting (same logic as `ellipse_utils.py`)
- Outputs per-refraction summary CSV; feeds into `build_patient_model.py` calibration

```
model_eye_runs/<run_name>/ellipse_results/
├── 1600_Z_00_00D/
│   ├── masks/
│   └── overlay/
├── ...
└── ellipse_summary.csv
```

---

## Classical (Image-Processing) Ellipse Extraction

When the ML segmentation model produces poor masks (e.g., distorted or multi-blob results),
a classical image-processing pipeline can recover accurate ellipses without retraining.

### Pipeline

```
ROI image
    ↓  red_enhance        R − 0.5G − 0.5B  (suppresses blue/green artefacts)
    ↓  stretch_to_255     linear remap: darkest→0, brightest→255
    ↓  dog_sharpen        Difference of Gaussians (σ=1.5, σ=15)
                          subtracts diffuse background glow, keeps sharp bright core
    ↓  otsu_mask          Otsu threshold on the DoG image
    ↓  pick_central_blob  keep the connected component closest to image centre
    ↓  dilate_along_major rect dilation (7 wide × 25 tall)
                          recovers dim tips that Otsu cuts off
    ↓  cv2.fitEllipse     final ellipse parameters
```

**Why DoG?** A plain Otsu on the red-enhanced image includes the wide diffuse halo
around the reflex, inflating the minor axis (~28 px vs. true ~21 px). The DoG filter
suppresses slow-varying background while preserving the thin bright stripe.

**Why dilation?** After DoG + Otsu the dim tips of the ellipse are cut off (major
underestimated by ~20%). A fixed 7×25 rectangular dilation recovers them. The kernel
is taller than wide because the major axis of a red reflex is nearly vertical (~88–90°).

### Standalone tester (no `src.*` imports required)

```bash
python experiments/otsu_ellipse_single.py
```

Edit `INPUT_IMAGE` and `OUTPUT_DIR` at the top of the file.  
Outputs saved to `experiments/otsu_output/`: diagnostic grid, ellipse overlay, mask, DoG image.

### CLI tester (full-featured, requires project root in `PYTHONPATH`)

```bash
# auto mode (picks best of otsu / percentile / center_hull)
python -m src.ellipse.ellipse_otsu_tester --input path/to/roi.png --method auto

# sweep foreground ratios and detect elbow point
python -m src.ellipse.ellipse_otsu_tester --input path/to/roi.png --method sweep

# top-10% hull
python -m src.ellipse.ellipse_otsu_tester --input path/to/roi.png --method top10_hull
```

### Performance on reference image (`102_LEFT`)

| | Classical pipeline | ML segmentation |
|---|---|---|
| center | (377.7, 290.0) | (378.3, 289.7) |
| major | 148.0 px | 140.7 px |
| minor | 21.5 px | 21.2 px |
| ratio | 0.145 | 0.151 |
| angle | 88.4° | 88.2° |

---

## Report Generation

Generate visualization reports from a completed pipeline run.

```bash
python -m src.pipeline.make_report \
    --run_name pipeline_run_v001 \
    --patient_ids 101_LEFT 101_RIGHT
```

Outputs per patient under `<run_root>/<patient_id>/report/`:
- `cos_curve.png` — D vs angle trigonometric fit curve (colour-coded by angle bin)
- `ellipse_grid.png` — all ellipse overlays in a grid
- `classify_grid.png` — all classification overlay images

Also outputs `<run_root>/angle_bin_summary.csv` — image count per angle bin across all patients.

---

## Key Design Principles

1. **Versioned outputs** — never overwrite results; use named run folders
2. **Modular stages** — each stage reads from and writes to named directories
3. **Shared processing logic** — `run_model_eye.py` imports directly from `main.py` so any changes to core functions propagate automatically
4. **Patient-level data split** — train/val/test must be patient-separated to avoid leakage

---

## Common Issues

| Symptom | Check |
|---|---|
| Model not found | `models/classify/best_classifier.pth` and `models/segmentation/best_segmentation_model.pth` |
| Empty segmentation | Classifier threshold too strict, or preprocessing mismatch |
| Ellipse fitting fails | Multiple disconnected mask regions, or fewer than 5 contour points |

---

## Known Limitations / Open Problems

### Major axis instability near emmetropia (TODO)

When the ratio (minor/major) falls below approximately 0.2 (corresponding to refraction powers
near 0 D to +1 D), the measured major axis length increases abnormally rather than remaining
stable. Ideally, for a fixed pupil diameter the major axis should be roughly constant regardless
of refraction power. The cause is not yet identified.

**Impact:** The current patient model (`src/analysis/build_patient_model.py`) assumes fixed
reference major-axis values per pupil size (3 mm → 130 px, 5 mm → 180 px, 7 mm → 200 px).
These representative values are empirically derived from the mid-to-high myopia range and may
be unreliable near emmetropia.

**Status:** Deferred. The ratio-based refraction formula and the major-axis-to-pupil-diameter
mapping are intentional approximations until the root cause is understood.

---

## Notes

Designed for ophthalmic imaging / red reflex analysis in low-cost smartphone-based diagnostics.
