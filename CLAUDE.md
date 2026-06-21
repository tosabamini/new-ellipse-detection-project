# CLAUDE.md — Project Context for Claude Code

## What this project is

End-to-end red reflex analysis pipeline for ophthalmic images.  
Three parallel pipelines:
- **ML-based** (legacy): RedEnhance → Classification → Segmentation → Ellipse fitting → Refraction estimation (S, C, A).
- **Geometry-only v150526** (current): RedEnhance → AdaptDoG → IQR filter → Pupil estimation → D estimation → D-IQR → SCA fit. No neural networks; suitable for Android/iOS deployment.
- **Joint solver (poly10, 2026-06-09, 採用)**: AdaptDoG → IQR → ratio+area 連立ソルバー（10次多項式モデル）→ D-IQR → SCA fit。`src/analysis/ratio_model.py` + `src/analysis/area_model.py` が多項式版に更新済み。

A secondary workflow generates reference ellipse data from a model eye at known refraction powers, used as calibration data for refraction estimation.

A third workflow uses **optical simulation data** (`data/Simulation/`) to derive ratio–D fitting models directly from ray-traced images at known refraction powers.

---

## Architecture overview

All processing logic lives in `src/`. Paths are centralized in `src/common/paths.py` — always import from there, never hardcode paths.

Key modules:
- `src/preprocessing/preprocess_utils.py` — RedEnhance core (`center_crop(img, crop_ratio, left_shift=0.0)`, `process_red_by_mode`, etc.)
- `src/classify/classifier_model.py` — SmallClassifier (1-ch CNN, input 160×72, threshold 0.9)
- `src/segmentation/segmentation_model.py` — UNetSmall (1-ch input/output, threshold 0.5)
- `src/ellipse/ellipse_utils.py` — `fit_ellipse_from_mask`, `make_pred_overlay`, `add_text_block`
- `src/ellipse/ellipse_otsu_tester.py` — classical (DoG+Otsu) ellipse CLI tester; methods: auto/otsu/percentile/center_hull/sweep/top10_hull/thin_hull; imports from `src.*`
- `src/ellipse/adaptdog.py` — **[v150526]** AdaptDoG ellipse fitting (adaptive sigma, angle-aware dilation) + `iqr_filter` + `d_iqr_filter`
- `src/pipeline/main.py` — end-to-end runner for patient data (ML-based); also exports reusable functions; supports `101_LEFT`/`101_RIGHT` patient ID format (→ `data/101/LEFT/`)
- `src/pipeline/run_model_eye.py` — model eye batch runner; imports directly from `main.py`
- `src/pipeline/make_report.py` — report image generator: cos-curve, ellipse grid, classify grid per patient; angle-bin summary CSV
- `src/pipeline/pipeline_v150526.py` — **[v150526]** geometry-only end-to-end pipeline (Raw → SCA, no ML); CLI: `--patient_ids`, `--run_name`, `--exclude_prefixes`
- `src/pipeline/pipeline_simulation.py` — **[Simulation]** Simulation data pipeline (Raw PNG → ellipse fitting); crop 60% keep + 10% left shift; CLI: `--run_name`, `--pupil_groups`
- `src/pipeline/simulation_ellipse_from_json.py` — **[Simulation]** Labelme JSON → mask → `fit_ellipse_from_mask` → `per_image_label.csv`; CLI: `--run_name`, `--pupil_group`
- `src/analysis/build_patient_model.py` — model eye reference calibration; `estimate_D_from_ratio_and_p(ratio, p)` → (D1, D2)
- `src/analysis/refraction_estimator.py` — refraction pipeline module; per-image D estimation + SCA trigonometric fit
- `src/analysis/pupil_estimator.py` — **[v150526]** pupil diameter estimation from (ratio, area_scaled) via quadratic formula; `SCALE_FACTOR=1.3` (暫定)
- `src/analysis/sim_ratio_model.py` — **[sim_ratio]** Simulation C⁰ Logistic による ratio→D 直接逆算（暫定、瞳孔径依存性無視）; `estimate_D_from_ratio_sim(ratio)` → (D_myopia, D_hyperopia)
- `src/analysis/ratio_model.py` — **[poly10, 採用]** 10次多項式 `ratio_real(D, p_mm)`; poly_model.npz をロード
- `src/analysis/area_model.py` — **[poly10, 採用]** 10次多項式 `area_real(D, p_mm)`; α(p)・k(p) 込み; poly_model.npz をロード
- `src/pipeline/pipeline_sim_ratio.py` — **[sim_ratio]** v150526 の Step5+6 のみ `sim_ratio_model` に差し替えたパイプライン; `--data_dir` で任意データルート指定可能
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
│   ├── model_eye_runs/       # output of src/pipeline/run_model_eye.py
│   └── simulation_runs/      # output of src/pipeline/pipeline_simulation.py
├── annotations/
└── Simulation/               # optical simulation images (ray-traced)
    ├── p10/                  # pupil radius 10 units — camera_p10_<D>.png / .ras
    ├── p15/
    ├── ...
    └── p45/
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

## Data layout (追記)

```
data/
├── Repeatability/            # 繰り返し測定データ (患者個人情報含む — git管理外)
│   └── 0603/                 # 2026-06-03 撮影, 18名 × LEFT/RIGHT
│       └── <name>/LEFT|RIGHT/*.jpg
```

**`data/Repeatability/` は `.gitignore` で除外済み。絶対にコミットしない。**

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

# [sim_ratio] Simulation ratio モデルを使ったパイプライン (任意データルート指定可)
python -m src.pipeline.pipeline_sim_ratio \
    --patient_ids 101_LEFT 101_RIGHT \
    --run_name sim_ratio_run01

# [sim_ratio] Repeatability データへの適用例 (患者名フォルダ構造)
python -m src.pipeline.pipeline_sim_ratio \
    --patient_ids Name_LEFT Name_RIGHT \
    --run_name repeatability_0603_sim_ratio \
    --data_dir data/Repeatability/0603

# [sim_ratio] 統一フィット曲線の再生成 (p20/p30/p40 平均)
python experiments/simulation_unified_fit.py
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

### 楕円フィッティングの限界 — 細線画像での major 過大算出 (2026-06-03 確認)

Repeatability データ (data/Repeatability/0603/, 18名×2眼) に `pipeline_sim_ratio` を適用した結果、
一部患者で S や C が過大に算出されることが判明。

**原因:** 赤反射が極端に細い線状（minor が数 px 程度）の場合、AdaptDoG + `cv2.fitEllipse` が
楕円の長軸・短軸を正しく認識できず、major を実際より大きく（または形状を誤って）算出する。
→ ratio が不正確になり → D 推定・SCA フィットが崩れる。

**再現条件:** 近正視付近など red reflex が非常に細い場合に顕在化しやすい。
**対策 (未実施):** major/minor 絶対値の下限フィルタ、または細線検出後の別ルート処理を検討。
現時点では「R² < 0.3 の結果は信頼性低」として扱うことで暫定的に判別可能。

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
| `analysis/EllipseAnalyzer.kt`    | **[poly10]** crop → red → core fit (no dilation) → ratio + area_norm → D per image (OpenCV for Android) |
| `analysis/RefractionModel.kt`    | **[poly10]** joint solver: poly eval `ratio_real`/`area_real` + coarse grid + bounded Nelder–Mead (port of `refraction_from_ratio_area.py`; matches scipy L-BFGS-B to <1e-5 D) |
| `analysis/RefractionFilters.kt`  | **[poly10]** pre-SCA outlier filters: major-axis IQR (k=0.5) + per-angle-bin D-IQR (k=1.5) (port of `iqr_filter`/`d_iqr_filter`) |
| `analysis/SCAEstimator.kt`       | Cosine fit `D=P₀+P₁cos2α+P₂sin2α` → S/C/A (port of `pipeline_v150526`) |
| `analysis/EllipseConstants.kt`   | **[poly10]** Calibration constants: COEF_RATIO/COEF_AREA (66-term deg-10 polynomials), DEG, D/P bounds, RATIO_THRESH, REF_LONG_SIDE, IQR_K, D_IQR_K, CROP_RATIO |
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

### Calibration constants (poly10, 2026-06-10 採用)

Centralised in `analysis/EllipseConstants.kt` — `COEF_RATIO` / `COEF_AREA` are the
66-term degree-10 polynomials from `poly_model.npz` (same source as `src/analysis/ratio_model.py`
/ `area_model.py`). Regenerate **both Python and Kotlin together** when recalibrating
(see `docs/model_formulas.md` 付録 for the canonical coefficient list).

**Area normalization (important):** `area_real` is calibrated on 4000px-long-side patient
images. `EllipseAnalyzer` normalises observed area by `(REF_LONG_SIDE / max(bmpW,bmpH))²`
so any capture/preview resolution maps onto the calibration scale. The authoritative
measurement is gallery "All Analyze" on full-res JPEGs; live-preview D is approximate
(preview aspect ratio differs from the 4:3 sensor).

**Unmeasurable handling:** `ratio < RATIO_THRESH (0.13)` → D=0 (near emmetropia), and these
samples are **kept** in the SCA fit (dEst=0, not null). dEst is null only on solver failure.

---

## Next session roadmap (as of 2026-06-10)

1. **Simulation pipeline — extend to all pupil groups** — p20/p30/p40 annotated and fitted. Next: annotate p10, p15, p25, p35, p45 with Labelme → `simulation_ellipse_from_json` → `build_poly_models.py` で多項式を再フィット（全グループ統合）。

2. **楕円フィッティング精度改善** — 細線（minor 数px）ケースで major が過大算出される問題（2026-06-03 確認）。
   対策候補: minor 絶対値下限フィルタ。まず minor < X px のケースを除外するフィルタを `run_repeatability_pipeline.py` / `pipeline_v150526` に追加する。

3. **Reference data retake** — 模型眼を再撮影し k(p) および α(p) を再校正。  
   α(p) は現在 Abhishek（p≈3mm）・Dilsha（p≈7mm）の 2点のみの暫定値。  
   更新後は `src/analysis/*.py` と Android `analysis/EllipseConstants.kt` を同時に更新すること。

4. **More patient data** — さらなる患者データで `run_repeatability_pipeline.py` を実行し、屈折検査値と比較。

5. **Remove live-preview debug overlay** — `CameraScreen.kt:237` の `{bmpW}×{bmpH} angle=N°` テキストを検証完了後に削除。`ellipseResult?.let { r -> Box(...) }` ブロックを削除する。

### 2026-06-09〜10 実施内容（完了済み）

- 光学シミュレーショングリッド（p10〜p45 × D −8〜+8）に対し 10次 2変数多項式をフィット（RMSE: ratio=0.0125, area=311px²）
- `src/analysis/ratio_model.py` / `area_model.py` を線形補間から多項式に差し替え（α補正を area モデル内に統合）
- `poly_model.npz` に係数保存（66項×ratio/area）
- 繰り返し測定データ（0603/0604, 12名×両眼）に新モデルを適用し SCA 推定
- 線形補間との差は軽微: |ΔSE|MAE=0.056D, |ΔC|MAE=0.134D（採用理由: 論文で数式として記述可能）

### 2026-06-10 バグ修正（重要）

`run_repeatability_pipeline.py` / `plot_cos_curves_all.py` に2件のバグを発見・修正:
1. **center_crop が幅のみ切り出していた**（高さ全体を保持）→ 細い赤反射で虹彩縁等を誤検出。
   正規の `preprocess_utils.center_crop`（幅20%×高さ20%）を import して解消。
2. **楕円フィットに `run_adaptive_dog`（Final/dilated版）を使用**していた → 現行手法の
   **core fit（mask_core から直接 fitEllipse, dilation なし）** に統一。
   `pickup_mask_core_fit.py` / `sim_mask_core_fit.py` と同一ロジック。

検証: 生jpg→自前処理 == PickUP保存PNG == ellipse_results.csv がバイト単位で一致。
修正後の日差再現性: **|ΔSE|=0.49D, |ΔS|=0.40D, |ΔC|=0.64D**（バグ版 0.87/0.93/0.83 から改善）。
Dilsha は正視（plano）に正常化。

**教訓: 生画像処理スクリプトでは center_crop / red_channel / stretch / 楕円フィットを
再実装せず必ず src からimportすること（CLAUDE.md「Shared-logic convention」遵守）。**

---

## Simulation data pipeline (2026-06-01)

### Data layout

```
data/Simulation/
├── p10/   camera_p10_D000.png  camera_p10_Dm25.png  ...  camera_p10_Dp800.png
├── p15/   (same structure)
├── ...
└── p45/
```

- Folder name `p<N>` = pupil radius (simulation units).  Currently p10, p15, p20, p25, p30, p35, p40, p45.
- Filename `D000` = 0.00 D, `Dm<N>` = −N/100 D (myopia), `Dp<N>` = +N/100 D (hyperopia).
- `.ras` files coexist — **ignore them**; process `.png` only.
- Image size: 1127 × 1152 px.

### Crop settings (Simulation-specific)

Standard patient crop (CROP_RATIO=0.2) does not work for Simulation images.  
Simulation uses: **CROP_RATIO = 0.60, LEFT_SHIFT = 0.10** (keeps 60%, centre shifted 10% left).  
`center_crop` in `preprocess_utils.py` accepts `left_shift` as an optional argument (default 0.0, backwards-compatible).

### Ellipse fitting approach

AdaptDoG auto-fitting failed on Simulation images (edges not visible enough).  
Adopted workflow: **manual Labelme annotation → JSON → mask → `fit_ellipse_from_mask`**.

```
STEP 1  pipeline_simulation.py     RedEnhance + crop → roi/  (also tries AdaptDoG, keep overlays)
STEP 2  Labelme (manual)           annotate roi/ images, label = "red_reflex"
STEP 3  simulation_ellipse_from_json.py  JSON→mask→ellipse → per_image_label.csv + ellipse_label/
STEP 4  simulation_p30_fit_full.py       fitting analysis (ratio / major / minor / area vs D)
```

### Run commands

```bash
# STEP 1: crop + AdaptDoG (preview)
python -m src.pipeline.pipeline_simulation --run_name sim_run01

# STEP 3: JSON → ellipse (per group, for incremental checking)
python -m src.pipeline.simulation_ellipse_from_json --run_name sim_run01 --pupil_group p30
python -m src.pipeline.simulation_ellipse_from_json --run_name sim_run01  # all groups

# STEP 4: fitting analysis
python experiments/simulation_p30_fit.py         # C⁰ logistic (ratio, both sides)
python experiments/simulation_p30_fit_full.py    # ratio + major + minor + area
python experiments/simulation_p30_fit_C2.py      # C² Hill (reference only, not adopted)
```

### Fitting results (p30, 2026-06-01)

Annotated: 65 images (myopia 0D～−8D + hyperopia 0D～+8D).  
Adopted model: **C⁰ Logistic anchored at D=0** (`f(0) = ratio_0 = 0.0204`).

| Variable | Myopia R² | Hyperopia R² | Notes |
|---|---|---|---|
| ratio | 0.9962 | 0.9916 | Main variable; adopted |
| minor | 0.9957 | 0.9953 | Good fit |
| area (major×minor) | 0.9868 | 0.9895 | Good fit |
| major | 0.3224 | 0.7663 | Poor — noisy, not reliable |

Logistic formula (anchored):
```
f(|D|) = a / (1 + exp(-k·(|D| - x0))) + offset
offset = ratio_0 - a / (1 + exp(k·x0))   ← derived, not free
```

Myopia:    a=1.0062, k=0.7814, x0=3.4325  
Hyperopia: a=0.7336, k=0.5323, x0=3.5265  
Both sides share `ratio_0 = 0.0204` (measured 0D value) → C⁰ continuous at D=0 (cusp).

C² attempt (Hill equation, n>2) stored in `experiments/simulation_p30_fit_C2.py` for reference;  
myopia side achieves C² (n=2.39) but hyperopia hits lower bound (n≈2, C¹ only) — not adopted.

### Output files (per pupil group, sim_run01)

```
data/processed/simulation_runs/sim_run01/<pupil_group>/
├── red/                   RedEnhance (full image)
├── roi/                   60%-crop colour images + Labelme JSONs
├── ellipse/               AdaptDoG overlay (auto, for reference)
├── ellipse_label/         Labelme-based ellipse overlay
├── per_image.csv          AdaptDoG results
├── per_image_label.csv    Labelme-based: stem, major, minor, ratio, angle, mask_area
├── ellipse_grid.png       AdaptDoG grid
└── fitting/               (p30 only so far)
    ├── ratio_both_sides.png
    ├── ratio_myopia.png / ratio_hyperopia.png
    ├── major/minor/area_both_sides.png
    ├── fit_summary.csv       C⁰ logistic parameters
    ├── fit_summary_full.csv  all variables
    └── fit_summary_C2.csv    Hill C² (reference)
```

---

## What NOT to do

- Do not hardcode paths outside of `src/common/paths.py`
- Do not copy processing logic — import from the source module
- Do not use `argparse` at module level (only inside `parse_args()` / `main()`)
- Do not add `data/`, `models/`, `reports/` content to git (large files / patient data)
