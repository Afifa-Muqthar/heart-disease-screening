
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_heart_data():
    """Load the heart CSV, preserve its raw target, and mark sentinel zeros missing."""
    data_path = PROJECT_ROOT / "dataset" / "heart_disease_uci.csv"
    heart_data = pd.read_csv(data_path)
    for column in ("chol", "trestbps"):
        heart_data.loc[heart_data[column] == 0, column] = pd.NA
    return heart_data

# Function to split Heart Disease dataset
def split_heart_data():
    heart_data = load_heart_data()
    # id and dataset are excluded; num is retained unchanged in the raw frame,
    # with the modeling target derived separately as disease present (num > 0).
    X_heart = heart_data.drop(columns=["id", "dataset", "num"])
    y_heart = (heart_data["num"] > 0).astype("int8")
    groups = pd.util.hash_pandas_object(
        X_heart.astype("string").fillna("<MISSING>"), index=False
    )
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, test_indices = next(splitter.split(X_heart, y_heart, groups))
    return (
        X_heart.iloc[train_indices],
        X_heart.iloc[test_indices],
        y_heart.iloc[train_indices],
        y_heart.iloc[test_indices],
    )

# Function to split Lung Cancer dataset
def split_lung_data():
    lung_data = pd.read_csv("disease_detection\\dataset\\cancer patient data sets.csv",encoding='ISO-8859-1')
    X_lung = lung_data[['Age', 'Smoking', 'Chronic Lung Disease', 'Genetic Risk']]
    y_lung = lung_data['Level']
    return train_test_split(X_lung, y_lung, test_size=0.2, random_state=42)

# Function to split COPD dataset
def split_copd_data():
    copd_data = pd.read_csv("disease_detection\dataset\dataset.csv")
    X_copd = copd_data[['AGE', 'PackHistory', 'FEV1', 'FVCPRED']]
    y_copd = copd_data['diagnosis']
    return train_test_split(X_copd, y_copd, test_size=0.2, random_state=42)

# Function to split Breast Cancer dataset
def split_cancer_data():
    cancer_data = pd.read_csv("disease_detection\dataset\data.csv")
    X_cancer = cancer_data[['tumor_size', 'compactness_mean', 'concavity_mean', 'fractal_dimension_worst']]
    y_cancer = cancer_data['diagnosis']
    return train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=42)
