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
│   │   ├── compare_ellipse_gt_pred.py
│   │   └── adaptdog.py               # AdaptDoG ellipse fitting + IQR/D-IQR filters [v150526]
│   ├── analysis/
│   │   ├── build_patient_model.py   # model eye calibration → estimate_D(ratio, p)
│   │   ├── refraction_estimator.py  # per-image D + SCA trigonometric fit
│   │   └── pupil_estimator.py       # pupil diameter from (ratio, area_scaled) [v150526]
│   ├── pipeline/
│   │   ├── main.py                  # end-to-end pipeline (patient data, ML-based)
│   │   ├── run_model_eye.py         # RedEnhance + classification (model eye)
│   │   ├── make_report.py           # report images (cos-curve, ellipse grid, classify grid)
│   │   └── pipeline_v150526.py      # geometry-only pipeline Raw→SCA (no ML) [v150526]
│   └── ui/
│       └── app.py                # Streamlit UI
│
├── experiments/
│   ├── otsu_ellipse_single.py      # standalone single-image ellipse tester (no src.* imports)
│   ├── classical_seg_trial.py      # earlier classical segmentation experiments
│   ├── sca_batch_all.py            # batch SCA estimation for all patients (101-106 excl. 103)
│   ├── sf_sweep_104_RIGHT.py       # Scale Factor sensitivity sweep
│   └── sca_104_LEFT_newpupil.py    # new pupil estimation method validation
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

### Physical model (v150526)

```
Raw image
    ↓  RedEnhance  :  R − 0.5G − 0.5B
    ↓  center_crop :  ROI
    ↓  AdaptDoG    :  adaptive DoG → Otsu → blob → ellipse (cx, cy, major, minor, angle)
    ↓  IQR filter  :  exclude images with no/weak red reflex (fence = Q1 − 0.5·IQR on major)
    ↓
ratio = minor / major                        (scale-invariant)
area_scaled = major × minor × SCALE_FACTOR²  (暫定 SF=1.3)
p_est = solve quadratic in p:
    [S2·ratio + I2]·p² + [S1·ratio + I1]·p + [S0·ratio + I0 − area_scaled] = 0
    → keep root in [2, 9] mm
D1, D2 = estimate_D_from_ratio_and_p(ratio, p_est)
adopted_D = D2                               (myopic side adopted)
    ↓  D-IQR filter per angle bin (k=1.5)
    ↓  across all valid images for one patient
D(α) = P0 + P1·cos(2α) + P2·sin(2α)        (trigonometric fit, α = major axis angle)
    ↓
SE = P0
C  = −2·√(P1² + P2²)     (cylinder, minus notation)
S  = SE − C/2
A  = ½·atan2(−P2, −P1) mod 180°             (cylinder axis)
```

### Pupil estimation calibration (src/analysis/pupil_estimator.py)

Model:  `area = slope(p) · ratio + intercept(p)`,  inverted as a quadratic in p.

```
slope(p)     =  928.28·p² + 1780.95·p −  872.10
intercept(p) = −462.23·p² + 3344.24·p − 4477.24
```

Derived from hand-labeled model eye data at p = 3, 5, 7 mm; area = major × minor × SF².  
(R² > 0.95 for all three pupil sizes.)

### Calibration (src/analysis/build_patient_model.py)

Fitted from model eye measurements at 3 pupil sizes (3, 5, 7 mm) × multiple refraction powers:

- **Ratio model**: `ratio = a(p)·D² + b(p)·D + c(p)` with a, b, c quadratic in p

### Output files (per patient, pipeline_v150526)

| File | Contents |
|---|---|
| `per_image.csv` | ratio, p_est, adopted_D, angle_bin per image |
| `sca.csv` | S, C, A, SE, R², n per patient |
| `cos_curve.png` | D vs angle trigonometric fit |
| `angle_dist.png` | angle distribution histogram |
| `ellipse_grid.png` | grid of ellipse overlays |

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

## Pipeline v150526 — Geometry-Only (No ML)

End-to-end pipeline from raw images to S/C/A.  
**No neural networks** — all steps are classical image processing (OpenCV only).  
Suitable for on-device deployment on Android/iOS.

### Run

```bash
# Single or multiple patients
python -m src.pipeline.pipeline_v150526 \
    --patient_ids 101_LEFT 101_RIGHT \
    --run_name pipeline_v150526_run01

# With image exclusion (e.g. 104_RIGHT has 3D rig images)
python -m src.pipeline.pipeline_v150526 \
    --patient_ids 104_LEFT 104_RIGHT \
    --run_name pipeline_v150526_run01 \
    --exclude_prefixes r_3D_ samarth_3D_
```

Input: `data/raw/patient_data/<N>/<SIDE>/` (JPEG/PNG images)  
Output: `data/processed/pipeline_runs/<run_name>/<patient_id>/`

```
<patient_id>/
├── red/              # RedEnhance output (R−0.5G−0.5B)
├── roi/              # Center-cropped ROI images
├── ellipse/          # Ellipse overlay visualizations
├── per_image.csv     # Per-image: ratio, area, p_est, D, angle_bin
├── sca.csv           # Patient-level: S, C, A, SE, R², n
├── cos_curve.png     # D vs angle cosine fit plot
├── angle_dist.png    # Angle distribution histogram
└── ellipse_grid.png  # Grid of representative ellipse overlays
```

### Batch results (patients 101–106, excl. 103, 2026-05-15)

| Patient | S | C | A | R² | p_med |
|---|---|---|---|---|---|
| 101_LEFT | — | — | — | — | — |
| 101_RIGHT | — | — | — | 0.866 | — |
| 102_LEFT | * | * | * | * | * |
| 104_LEFT | −1.24 | −2.26 | 111° | 0.645 | 2.8mm |
| 104_RIGHT | −2.27 | −1.82 | 106° | 0.167 | — |
| 105_LEFT | — | — | — | 0.039 | — |
| 105_RIGHT | — | — | — | 0.898 | — |
| 106_LEFT | — | — | — | — | — |
| 106_RIGHT | — | — | — | — | — |

\* 102_LEFT: very narrow angle distribution → unreliable C estimate (artifact).

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

## Android Application

A clinical Android app under [`SmartphoneApplication/SmaKIArt_Camera_RemoCon/`](./SmartphoneApplication/) ports `pipeline_v150526` end-to-end on-device (Kotlin + Jetpack Compose + OpenCV for Android, no ML).

- **Live preview ellipse overlay** — real-time `EllipseAnalyzer` on TextureView frames at ~5 Hz; D estimate displayed continuously.
- **Capture** — manual ISO / Exposure / Focus, R/L eye toggle (orientation-driven), 3D/10D focus-pair shutter, JPEG saved to `Pictures/SmaKIArtClinical/{patientId}/{eye}/`.
- **Gallery** — Patient list (newest first) → Eye selector → image list with per-image **Analyze** and **All Analyze** buttons.  All Analyze loads every photo for that patient × eye, runs `EllipseAnalyzer`, fits the cosine curve (`SCAEstimator.kt`), and shows S / C / A / SE / R² / n with the cos plot rendered in Compose Canvas.
- **Min samples** — `SCAEstimator.MIN_VALID = 3` valid (α, D) pairs, same as `pipeline_v150526`.

See [`SmartphoneApplication/README.md`](./SmartphoneApplication/README.md) for the dual-device protocol and [`CLAUDE.md`](./CLAUDE.md) for the `getBitmap` / `setTransform` gotcha and the module layout.

---

## Simulation Data Pipeline

Optical simulation images (`data/Simulation/`) provide ground-truth ellipses at known refraction powers for building ratio–D fitting models independent of physical model eye measurements.

### Data layout

```
data/Simulation/
├── p10/  camera_p10_D000.png  camera_p10_Dm25.png ... camera_p10_Dp800.png
├── p15/  ...
└── p45/
```

`p<N>` = pupil radius; `D000` = 0.00 D, `Dm<N>` = −N/100 D, `Dp<N>` = +N/100 D. Ignore `.ras` files.

### Crop settings

Simulation images require a different crop from patient data:  
**60% keep, centre shifted 10% left** (`CROP_RATIO=0.60, LEFT_SHIFT=0.10`).

### Workflow

```bash
# STEP 1: generate roi/ crops (also runs AdaptDoG, but auto-fitting is unreliable on Simulation)
python -m src.pipeline.pipeline_simulation --run_name sim_run01

# STEP 2: annotate roi/ in Labelme (label = "red_reflex"), then:
# STEP 3: JSON → ellipse fitting per pupil group
python -m src.pipeline.simulation_ellipse_from_json --run_name sim_run01 --pupil_group p30
python -m src.pipeline.simulation_ellipse_from_json --run_name sim_run01  # all groups

# STEP 4: fitting analysis
python experiments/simulation_p30_fit_full.py
```

### Fitting model (p30, 2026-06-01)

Adopted: **C⁰ Logistic anchored at D=0**, fitted separately for myopia (D≤0) and hyperopia (D≥0).

```
ratio(|D|) = a / (1 + exp(-k·(|D| - x0))) + offset
             where offset = ratio_0 - a/(1+exp(k·x0))   [ensures f(0) = ratio_0]
```

| Side | a | k | x₀ | R² |
|---|---|---|---|---|
| Myopia | 1.0062 | 0.7814 | 3.4325 | 0.9962 |
| Hyperopia | 0.7336 | 0.5323 | 3.5265 | 0.9916 |

ratio_0 = 0.0204 (measured 0D value, shared anchor).  
Both curves meet at D=0 (C⁰ cusp); the asymmetry (myopia saturates at ~0.97, hyperopia at ~0.65 for ±8D) reflects the physical vignetting difference.

**Status (2026-06-01):** p30 fully annotated and fitted. Other pupil groups (p10–p45) pending.

---

## Next Session Roadmap (as of 2026-06-01)

1. **Simulation — extend to all pupil groups** — annotate p10, p15, p20, p25, p35, p40, p45 with Labelme → run `simulation_ellipse_from_json` → fit and integrate across pupil groups.

2. **Reference data retake** — recollect model eye images to recalibrate SCALE_FACTOR and the (ratio, area) → pupil model.  
   Current SF = 1.3 is provisional; axis error (~40° off) and C overestimation (~2×) may be partly explained by stale reference data.  When new constants are ready, update `src/analysis/*.py` **and** Android `analysis/EllipseConstants.kt` together.

3. **More patient data** — run pipeline_v150526 on additional patients and compare estimated S/C/A to ground-truth refraction records.

4. **Android polishing** — remove the live-preview debug overlay (`CameraScreen.kt:237`) once accuracy is verified in the field.

---

## Notes

Designed for ophthalmic imaging / red reflex analysis in low-cost smartphone-based diagnostics.
