# CLAUDE.md — Project Context for Claude Code

## What this project is

End-to-end red reflex analysis pipeline for ophthalmic images.  
Processes patient images through: RedEnhance → Classification → Segmentation → Ellipse fitting.  
A secondary workflow generates reference ellipse data from a model eye at known refraction powers.

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

---

## Shared-logic convention

`run_model_eye.py` intentionally imports `run_classifier_on_red`, `make_classify_overlay`, `load_classifier`, `EXTENSIONS`, `CLASSIFIER_THRESHOLD`, `DEVICE`, and `ensure_dir` **from `src.pipeline.main`** — not copied.  
When modifying classification or preprocessing logic, change it in `main.py` / `preprocess_utils.py` only; downstream scripts update automatically.

---

## Data layout

```
data/
├── raw/patient_data/         # source patient images
├── model_eye/                # model eye images (49 refraction folders, -6.00D to +6.00D, 0.25D steps)
├── processed/
│   ├── pipeline_runs/        # output of src/pipeline/main.py
│   └── model_eye_runs/       # output of src/pipeline/run_model_eye.py
└── annotations/
```

### Model eye folder naming

Format: `CODE_M/Z/P_DD_DDd`  
Code = 1600 + refraction × 100. M = minus, Z = zero, P = plus.  
Examples: `1000_M_06_00D`, `1600_Z_00_00D`, `2200_P_06_00D`

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

---

## Run commands

```bash
# Patient data end-to-end
python -m src.pipeline.main --patient_ids 01 02 --run_name pipeline_run_v001

# Model eye: RedEnhance + classification
python -m src.pipeline.run_model_eye --run_name model_eye_v001

# Model eye: specific folders only
python -m src.pipeline.run_model_eye --run_name test --folders 1600_Z_00_00D 1625_P_00_25D

# Standalone RedEnhance (patient data)
python -m src.preprocessing.redenhance

# Segmentation training
python -m src.segmentation.train_segmentation \
  --dataset_name seg_dataset_v001 --run_name seg_v001 --epochs 30

# Streamlit UI
streamlit run src/ui/app.py
```

---

## What NOT to do

- Do not hardcode paths outside of `src/common/paths.py`
- Do not copy processing logic — import from the source module
- Do not use `argparse` at module level (only inside `parse_args()` / `main()`)
- Do not add `data/`, `models/`, `reports/` content to git (large files / patient data)
