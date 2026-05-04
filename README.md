# Ellipse Detection Project (Red Reflex Pipeline)

End-to-end pipeline for **red reflex analysis** from ophthalmic images.  
Covers preprocessing, image quality classification, segmentation, and ellipse fitting.

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
│   │   ├── ellipse_utils.py      # ellipse fitting core functions
│   │   ├── fit_ellipse.py        # standalone ellipse runner
│   │   └── compare_ellipse_gt_pred.py
│   ├── pipeline/
│   │   ├── main.py               # end-to-end pipeline (patient data)
│   │   └── run_model_eye.py      # RedEnhance + classification (model eye)
│   └── ui/
│       └── app.py                # Streamlit UI
│
├── experiments/
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
- Outputs per-refraction summary CSV

```
model_eye_runs/<run_name>/ellipse_results/
├── 1600_Z_00_00D/
│   ├── masks/
│   └── overlay/
├── ...
└── ellipse_summary.csv
```

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

## Notes

Designed for ophthalmic imaging / red reflex analysis in low-cost smartphone-based diagnostics.
