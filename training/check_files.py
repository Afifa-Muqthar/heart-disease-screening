from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "training"
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"

print("Training folder exists:", TRAINING_DIR.is_dir())
print("Dataset folder exists:", DATASET_DIR.is_dir())
print("Files in training folder:", sorted(path.name for path in TRAINING_DIR.iterdir()))
print(
    "Heart Disease Model candidates exist:",
    any(MODELS_DIR.glob("heart_disease_*.joblib")),
)
print(
    "Lung Cancer Model exists:",
    any(MODELS_DIR.glob("lung_cancer_*.joblib")),
)
