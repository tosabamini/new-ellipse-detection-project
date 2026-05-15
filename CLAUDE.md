# CLAUDE.md — Project Context for Claude Code

## What this project is

End-to-end red reflex analysis pipeline for ophthalmic images.  
Two parallel pipelines:
- **ML-based** (legacy): RedEnhance → Classification → Segmentation → Ellipse fitting → Refraction estimation (S, C, A).
- **Geometry-only v150526** (current): RedEnhance → AdaptDoG → IQR filter → Pupil estimation → D estimation → D-IQR → SCA fit. No neural networks; suitable for Android/iOS deployment.

A secondary workflow generates reference ellipse data from a model eye at known refraction powers, used as calibration data for refraction estimation.

---

## Architecture overview

All processing logic lives in `src/`. Paths are centralized in `src/common/paths.py` — always import from there, never hardcode paths.

Key modules:
- `src/preprocessing/preprocess_utils.py` — RedEnhance core (`center_crop`, `process_red_by_mode`, etc.)
- `src/classify/classifier_model.py` — SmallClassifier (1-ch CNN, input 160×72, threshold 0.9)
- `src/segmentation/segmentation_model.py` — UNetSmall (1-ch input/output, threshold 0.5)
- `src/ellipse/ellipse_utils.py` — `fit_ellipse_from_mask`, `make_pred_overlay`, `add_text_block`
- `src/ellipse/ellipse_otsu_tester.py` — classical (DoG+Otsu) ellipse CLI tester; methods: auto/otsu/percentile/center_hull/sweep/top10_hull/thin_hull; imports from `src.*`
- `src/ellipse/adaptdog.py` — **[v150526]** AdaptDoG ellipse fitting (adaptive sigma, angle-aware dilation) + `iqr_filter` + `d_iqr_filter`
- `src/pipeline/main.py` — end-to-end runner for patient data (ML-based); also exports reusable functions; supports `101_LEFT`/`101_RIGHT` patient ID format (→ `data/101/LEFT/`)
- `src/pipeline/run_model_eye.py` — model eye batch runner; imports directly from `main.py`
- `src/pipeline/make_report.py` — report image generator: cos-curve, ellipse grid, classify grid per patient; angle-bin summary CSV
- `src/pipeline/pipeline_v150526.py` — **[v150526]** geometry-only end-to-end pipeline (Raw → SCA, no ML); CLI: `--patient_ids`, `--run_name`, `--exclude_prefixes`
- `src/analysis/build_patient_model.py` — model eye reference calibration; `estimate_D_from_ratio_and_p(ratio, p)` → (D1, D2)
- `src/analysis/refraction_estimator.py` — refraction pipeline module; per-image D estimation + SCA trigonometric fit
- `src/analysis/pupil_estimator.py` — **[v150526]** pupil diameter estimation from (ratio, area_scaled) via quadratic formula; `SCALE_FACTOR=1.3` (暫定)
- `experiments/otsu_ellipse_single.py` — **standalone** single-image classical ellipse tester; no `src.*` imports; edit `INPUT_IMAGE` at the top and run directly

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

## Classical ellipse pipeline (experiments/otsu_ellipse_single.py)

Used when ML segmentation produces distorted results.

```
red_enhance (R−0.5G−0.5B) → stretch_to_255 → DoG(σ=1.5, σ=15)
    → Otsu → pick_central_blob → dilate_along_major(7×25) → cv2.fitEllipse
```

- **DoG**: removes diffuse background halo (low-freq), keeps sharp bright core (high-freq).  
  Without DoG, Otsu includes the halo → minor axis ~28 px instead of ~21 px.
- **dilate_along_major(7×25)**: rect kernel, taller than wide. Recovers dim tips that  
  Otsu cuts off after DoG. Assumes major axis is nearly vertical (~88–90°).
- Kernel size (7×25) was tuned on `102_LEFT`; may need adjustment for other image sets.

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
| DoG sigma_small | 1.5 | `otsu_ellipse_single.py` / `ellipse_otsu_tester.py` |
| DoG sigma_large | 15.0 | `otsu_ellipse_single.py` / `ellipse_otsu_tester.py` |
| Dilation kernel (classical) | 7 × 25 px | `otsu_ellipse_single.py` |
| SCALE_FACTOR (patient px correction) | 1.3 (暫定) | `pupil_estimator.py` |
| P_MIN / P_MAX (pupil range) | 2.0 / 9.0 mm | `pupil_estimator.py` |
| IQR_K (major axis IQR filter) | 0.5 | `adaptdog.py` / `pipeline_v150526.py` |
| D_IQR_K (D outlier filter per bin) | 1.5 | `adaptdog.py` / `pipeline_v150526.py` |
| MIN_VALID (SCA fit minimum) | 3 images | `pipeline_v150526.py` |
| Pupil slope coefficients | S2=928.28, S1=1780.95, S0=−872.10 | `pupil_estimator.py` |
| Pupil intercept coefficients | I2=−462.23, I1=3344.24, I0=−4477.24 | `pupil_estimator.py` |

---

## Run commands

```bash
# [v150526] Geometry-only pipeline: Raw → SCA (no ML)
python -m src.pipeline.pipeline_v150526 \
  --patient_ids 101_LEFT 101_RIGHT \
  --run_name pipeline_v150526_run01

# [v150526] With image exclusion (e.g. 104_RIGHT has 3D rig images)
python -m src.pipeline.pipeline_v150526 \
  --patient_ids 104_LEFT 104_RIGHT \
  --run_name pipeline_v150526_run01 \
  --exclude_prefixes r_3D_ samarth_3D_

# Patient data end-to-end ML pipeline (supports 101_LEFT / 101_RIGHT style IDs)
python -m src.pipeline.main --patient_ids 101_LEFT 101_RIGHT --run_name pipeline_run_v001

# Model eye: RedEnhance + classification (pupil_mm defaults to 7.0mm)
python -m src.pipeline.run_model_eye --run_name model_eye_7mm_v001 --pupil_mm 7.0mm

# Model eye: specific folders only
python -m src.pipeline.run_model_eye --run_name test --pupil_mm 7.0mm --folders 1600_Z_00_00D 1625_P_00_25D

# Standalone RedEnhance (patient data)
python -m src.preprocessing.redenhance

# Segmentation training
python -m src.segmentation.train_segmentation \
  --dataset_name seg_dataset_v001 --run_name seg_v001 --epochs 30

# Report generation
python -m src.pipeline.make_report --run_name pipeline_run_v001 --patient_ids 101_LEFT 101_RIGHT

# Classical ellipse tester (CLI, multiple methods)
python -m src.ellipse.ellipse_otsu_tester --input path/to/roi.png --method sweep

# Classical ellipse tester (standalone, no src.* imports — edit INPUT_IMAGE inside the file)
python experiments/otsu_ellipse_single.py

# Streamlit UI
streamlit run src/ui/app.py
```

---

## Refraction estimation model (src/analysis/) — v150526

### Physical model

```
Ellipse (major_px, minor_px, angle_deg)
    ↓
ratio      = minor / major                    [scale-invariant]
area_scaled = major × minor × SCALE_FACTOR²  [SCALE_FACTOR=1.3, 暫定]
p_est = solve quadratic in p:
    [S2·ratio + I2]·p² + [S1·ratio + I1]·p + [S0·ratio + I0 - area_scaled] = 0
    S2=928.28, S1=1780.95, S0=-872.10
    I2=-462.23, I1=3344.24, I0=-4477.24
    → keep root in [P_MIN=2, P_MAX=9] mm
D1, D2 = estimate_D_from_ratio_and_p(ratio, p_est)  [2 refraction solutions]
adopted_D = D2                                        [myopic side adopted]
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

Fitted relationships (all quadratic in p):
- `ratio = a(p)·D^2 + b(p)·D + c(p)` with a, b, c interpolated as quadratics in p
- Pupil model: `area = slope(p)·ratio + intercept(p)`, inverted as quadratic in p (see `pupil_estimator.py`)

### Pipeline output files (per patient, v150526)

| File | Contents |
|---|---|
| `per_image.csv` | ratio, area_scaled, p_est, adopted_D, angle_bin per image |
| `sca.csv` | S, C, A, SE, R2, n for the patient |
| `cos_curve.png` | D vs angle cosine fit |
| `angle_dist.png` | angle distribution histogram |
| `ellipse_grid.png` | representative ellipse overlay grid |

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

### AdaptDoG thresholding variants (deferred — current variant adopted)
Investigated 3 alternative thresholding strategies for AdaptDoG on 104_LEFT (38 images):
- **A**: post-Otsu erode (`kernel = minor_est * 0.12`) to remove gradient shoulders
- **B**: raise threshold to Otsu + 30% of remaining headroom
- **C**: CLAHE (clipLimit=2.0, tileGridSize=8×8) on DoG before Otsu

Results (see `experiments/method_compare_output/104_LEFT_variants/`):
- A gave consistently tighter masks (ratio ~0.03–0.05 lower, major ~2–6 px smaller) with
  one edge case (`134221_891`) where erode fragmented the core and pick_central_blob picked
  the wrong piece — a rare but real failure mode.
- B and C did not show clear improvement over current.
- **Decision: keep "current" (plain Otsu).** A is not bad but the fragmentation risk
  outweighs the marginal tightening benefit at this stage. Revisit if ratio accuracy
  becomes a bottleneck.

---

## Android implementation (SmartphoneApplication/SmaKIArt_Camera_RemoCon)

End-to-end clinical app in Kotlin + Jetpack Compose. Geometry-only — no ML.
Mirrors `pipeline_v150526` directly on-device.

### Module layout

| Path | Role |
|---|---|
| `analysis/EllipseAnalyzer.kt`    | AdaptDoG → ratio → p → D₂ per image (OpenCV for Android) |
| `analysis/SCAEstimator.kt`       | Cosine fit `D=P₀+P₁cos2α+P₂sin2α` → S/C/A (port of `pipeline_v150526`) |
| `analysis/EllipseConstants.kt`   | Calibration constants (S0/S1/S2, I0/I1/I2, A/B/C, P_MIN/P_MAX, SCALE_FACTOR, CROP_RATIO) |
| `camera/CameraController.kt`     | Camera2 + manual ISO/exposure/focus + display transform |
| `data/PhotoFileManager.kt`       | Save to MediaStore + enumerate gallery + EXIF-aware load |
| `data/CapturedPhoto.kt`          | `CapturedPhoto`, `PatientSummary` |
| `ui/CameraViewModel.kt`          | Live-preview loop, capture, gallery state machine, `runAllAnalyze` |
| `ui/CameraScreen.kt`             | Preview + ellipse overlay canvas + UI panels |
| `ui/GalleryScreen.kt`            | 4-level gallery (Patient → Eye → Image list → SCA result) + cos plot |
| `ui/PhotoAnalysisScreen.kt`      | Single-image overlay analysis (reused from gallery) |

### Storage path

`Pictures/SmaKIArtClinical/{patientId}/{eye}/IMG_{timestamp}.jpg`
(MediaStore on Android Q+, direct File on older).  `eye ∈ {RIGHT, LEFT}`.

### Live-preview overlay — the `getBitmap` trap

`TextureView.getBitmap()` does **not** include the `setTransform` matrix.  It
returns the raw SurfaceTexture content, which is sensor-rotation-corrected to
**portrait** for `SENSOR_ORIENTATION = 90°` even though `setDefaultBufferSize`
is landscape (e.g. `1920×864`).

Pitfall: requesting `getBitmap(previewSize.width, previewSize.height)` forces
portrait content into a landscape destination → horizontal stretch.

Correct call:

```kotlin
val ps = cameraController.getPreviewSize()
tv.getBitmap(ps.height, ps.width)   // e.g. 864×1920, portrait, no distortion
```

Then `EllipseCanvas` must mirror `applyPreviewTransform`'s `-90° CCW` rotation
and uniform scale to map portrait bitmap coords → landscape display coords:

```kotlin
val scale = maxOf(canvasW / bmpH, canvasH / bmpW)
cx = centerX + (cyPx - bmpH/2) * scale
cy = centerY - (cxPx - bmpW/2) * scale
screenAngle = angleDeg - 90f   // (+90° for REVERSE_LANDSCAPE)
```

### Capture & gallery flow

- Shutter → **save only** (no auto-navigation).  Live preview analysis keeps running.
- SessionPanel exposes **Gallery** (next to **End**).
- Gallery: Patient list (newest first) → Eye selector → Image list → per-image
  Analyze (reuses `PhotoAnalysisScreen`) or **All Analyze**.
- All Analyze: loads every JPEG, runs `EllipseAnalyzer`, accumulates `(αᵢ, Dᵢ)`,
  fits via `SCAEstimator` (min 3 valid samples), shows S/C/A/SE/R²/n + cos-curve.

### Calibration constants

Centralised in `analysis/EllipseConstants.kt` — same values as
`src/analysis/pupil_estimator.py` and `src/analysis/build_patient_model.py`.
Update both Python and Kotlin together when recalibrating.

---

## Next session roadmap (as of 2026-05-15)

1. **Reference data retake** — recollect model eye images with consistent optics and re-derive SCALE_FACTOR and the pupil estimation coefficients.  
   Known issues: axis error (~40° offset for 104_LEFT) and C overestimation (~2×) may be partially explained by stale reference data.  When new constants are ready, update `src/analysis/*.py` **and** `analysis/EllipseConstants.kt` in the Android app.

2. **More patient data** — run `pipeline_v150526` on additional patients; compare S/C/A outputs against ground-truth refraction records.

3. **Remove live-preview debug overlay** — the small `{bmpW}×{bmpH} angle=N°` text top-centre in `CameraScreen.kt:237` is left in place for verification.  Delete the `ellipseResult?.let { r -> Box(...) }` block when ready.

---

## What NOT to do

- Do not hardcode paths outside of `src/common/paths.py`
- Do not copy processing logic — import from the source module
- Do not use `argparse` at module level (only inside `parse_args()` / `main()`)
- Do not add `data/`, `models/`, `reports/` content to git (large files / patient data)
