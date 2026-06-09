from pathlib import Path

# =========================
# Project root
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =========================
# Data directories
# =========================
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

PATIENT_DATA_DIR = RAW_DIR / "patient_data"
SIMULATION_DIR   = DATA_DIR / "Simulation"

# processed
SIMULATION_RUNS_DIR = PROCESSED_DIR / "simulation_runs"
REDENHANCE_DIR = PROCESSED_DIR / "redenhance"
CLASSIFY_INPUTS_DIR = PROCESSED_DIR / "classify_inputs"
CLASSIFY_OUTPUTS_DIR = PROCESSED_DIR / "classify_outputs"
SEGMENTATION_DATASET_DIR = PROCESSED_DIR / "segmentation_dataset"
SEGMENTATION_INFERENCE_DIR = PROCESSED_DIR / "segmentation_inference"
ELLIPSE_OUTPUTS_DIR = PROCESSED_DIR / "ellipse_outputs"

# annotations
CLASSIFY_ANNOTATIONS_DIR = ANNOTATIONS_DIR / "classify"
SEGMENTATION_ANNOTATIONS_DIR = ANNOTATIONS_DIR / "segmentation"

# =========================
# Models
# =========================
MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFY_MODELS_DIR = MODELS_DIR / "classify"
SEGMENTATION_MODELS_DIR = MODELS_DIR / "segmentation"
ARCHIVED_MODELS_DIR = MODELS_DIR / "archived"

CLASSIFY_CHECKPOINTS_DIR = CLASSIFY_MODELS_DIR / "checkpoints"
SEGMENTATION_CHECKPOINTS_DIR = SEGMENTATION_MODELS_DIR / "checkpoints"

BEST_CLASSIFIER_MODEL_PATH = CLASSIFY_MODELS_DIR / "best_classifier.pth"
BEST_SEGMENTATION_MODEL_PATH = SEGMENTATION_MODELS_DIR / "best_segmentation_model.pth"

# =========================
# Reports / experiments
# =========================
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
NOTES_DIR = REPORTS_DIR / "notes"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# =========================
# Calibrated reference models (Simulation -> real px)
# =========================
SIMU_MASKED_DIR = DATA_DIR / "simu_masked" / "ellipse_flat75"
AREA_MODEL_NPZ  = SIMU_MASKED_DIR / "fitting_calibrated_spline" / "area_model.npz"
RATIO_MODEL_NPZ = SIMU_MASKED_DIR / "fitting_calibrated_spline" / "ratio_model.npz"
POLY_MODEL_NPZ  = SIMU_MASKED_DIR / "fitting_calibrated_spline" / "poly_model.npz"