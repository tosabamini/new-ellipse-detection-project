# Ellipse Detection Project (Red Reflex Pipeline)

This project performs **end-to-end red reflex analysis** from patient images:

- Red enhancement (preprocessing)
- Image quality classification (ML)
- Segmentation (ML)
- Ellipse fitting
- Quantitative evaluation

---

# 📁 Project Structure


project/
├─ data/
│ ├─ raw/ # original patient data
│ ├─ processed/ # all generated outputs
│ └─ annotations/ # labeled datasets
│
├─ models/
│ ├─ classify/
│ └─ segmentation/
│
├─ src/
│ ├─ preprocessing/
│ ├─ classify/
│ ├─ segmentation/
│ ├─ ellipse/
│ ├─ pipeline/
│ └─ common/
│
├─ experiments/
├─ reports/
└─ README.md


---

# 🚀 Full Pipeline Overview


RAW IMAGE
↓
RedEnhance
↓
Classifier (good / bad image)
↓
Segmentation (mask)
↓
Single component extraction
↓
Ellipse fitting
↓
Quantitative metrics


---

# 🧪 1. Red Enhancement

Convert raw patient images into enhanced red-reflex images.

## Command

```bash
python -m src.preprocessing.redenhance
Output
data/processed/redenhance/<version>/
├─ <patient_id>/
│  ├─ roi/
│  ├─ red/
│  └─ debug/
└─ red_enhance_log.csv

🧠 2. Classification (Inference)

Filter usable images using trained classifier.

Command
python -m src.classify.infer_classifier \
  --redenhance_version default \
  --run_name clf_v001_on_default
Output
data/processed/classify_outputs/<run_name>/
├─ positive_for_mask/
├─ negative/
├─ all_results/
└─ predictions.csv
🧩 3. Segmentation Dataset Preparation

Create dataset for annotation (patient-balanced sampling).

Command
python -m src.segmentation.prepare_segmentation_images \
  --classify_run_name clf_v001_on_default \
  --dataset_name seg_dataset_v001 \
  --max_train_images 120 \
  --max_val_images 40 \
  --max_test_images 40
Then annotate using Labelme
Open folders:
images/train
images/val
images/test
Label name must be:
red_reflex
🧾 4. JSON → Mask Conversion

Convert Labelme annotations into binary masks.

python -m src.segmentation.json_to_mask \
  --dataset_name seg_dataset_v001
🏋️ 5. Train Segmentation Model
python -m src.segmentation.train_segmentation \
  --dataset_name seg_dataset_v001 \
  --run_name seg_v001_on_seg_dataset_v001 \
  --batch_size 4 \
  --epochs 30 \
  --lr 1e-3
Outputs
models/segmentation/
├─ best_segmentation_model.pth
├─ <run_name>/
│  ├─ train_log.csv
│  ├─ config.json
│  ├─ val_preds/
│  └─ test_preds/
🔍 6. Segmentation Inference

Run segmentation on classified images.

python -m src.segmentation.infer_segmentation \
  --classify_run_name clf_v001_on_default \
  --run_name seginf_v001_on_clf_v001_on_default
🔵 7. Ellipse Fitting (Prediction Only)
python -m src.ellipse.fit_ellipse \
  --segmentation_run_name seginf_v001_on_clf_v001_on_default \
  --redenhance_version default \
  --run_name ellipse_v001
📊 8. Ellipse Evaluation (GT vs Pred)
python -m src.ellipse.compare_ellipse_gt_pred \
  --segmentation_train_run_name seg_v001_on_seg_dataset_v001 \
  --redenhance_version default \
  --run_name ellipse_compare_v001
Outputs
Area error
Major/minor axis error
Angle error
Visualization overlays
⚙️ 9. End-to-End Pipeline (Production)

Run everything on new patient data.

python -m src.pipeline.main \
  --patient_ids 01 02 \
  --run_name pipeline_run_v001
Output
data/processed/pipeline_runs/<run_name>/
├─ roi/
├─ red/
├─ classify_overlay/
├─ seg_pred/
├─ ellipse_overlay/
└─ results.csv
📌 Key Design Principles
1. Never overwrite results

Use versioned folders:

red_v001
clf_v002
seg_v003
2. Keep stages independent
Stage	Input	Output
RedEnhance	raw	processed
Classify	red	filtered
Segmentation	filtered	mask
Ellipse	mask	geometry
3. Patient-level split

Avoid data leakage:

train / val / test must be patient-separated
⚠️ Common Pitfalls
❌ Model not found

Check:

models/classify/best_classifier.pth
models/segmentation/best_segmentation_model.pth
❌ Empty segmentation output
classifier threshold too strict
preprocessing mismatch
❌ Ellipse fails
multiple mask regions
insufficient contour points (<5)
🔧 Future Improvements
Improve RedEnhance (lighting normalization)
Retrain classifier with more data
Increase segmentation dataset size
Add temporal consistency (video)
Improve ellipse robustness
👤 Author Notes

This pipeline is designed for:

ophthalmic imaging
red reflex analysis
low-cost smartphone-based diagnostics
🏁 Summary

This project provides a full pipeline:

Patient Image → ML → Geometry → Quantitative Analysis

and is structured for:

reproducibility
modular experimentation
scalable improvement