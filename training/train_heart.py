"""Leakage-aware heart disease model comparison and candidate artifact export."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from training.split_data import load_heart_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
TARGET_COLUMN = "num"
SITE_COLUMN = "dataset"
EXCLUDED_COLUMNS = ("id", SITE_COLUMN, TARGET_COLUMN)
USER_FRIENDLY_FEATURES = ("age", "sex", "cp", "trestbps", "restecg")


def _make_pipeline(features: pd.DataFrame) -> Pipeline:
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _metrics(y_true, probabilities) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def _train_defaults(features: pd.DataFrame) -> dict[str, object]:
    defaults = {}
    for column in features:
        series = features[column].dropna()
        if pd.api.types.is_numeric_dtype(features[column]):
            defaults[column] = float(series.median())
        else:
            modes = series.mode()
            defaults[column] = modes.iloc[0] if not modes.empty else "Unknown"
    return defaults


def _apply_input_defaults(
    features: pd.DataFrame,
    chosen_inputs: tuple[str, ...],
    defaults: dict[str, object],
) -> pd.DataFrame:
    filled = features.copy()
    for column, value in defaults.items():
        if column not in chosen_inputs:
            filled[column] = value
    return filled


def _group_ids(features: pd.DataFrame) -> np.ndarray:
    # Group records with identical clinical predictors, even when their source
    # site or row ID differs, so a repeated record cannot cross a split.
    normalized = features.astype("string").fillna("<MISSING>")
    return pd.util.hash_pandas_object(normalized, index=False).to_numpy()


def _print_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(frame.to_string(index=False))


def main() -> None:
    source_path = PROJECT_ROOT / "dataset" / "heart_disease_uci.csv"
    source_raw = pd.read_csv(source_path)
    zero_sentinels = {
        column: int((source_raw[column] == 0).sum())
        for column in ("chol", "trestbps")
    }
    raw = load_heart_data()
    target = (raw[TARGET_COLUMN] > 0).astype("int8")
    features = raw.drop(columns=list(EXCLUDED_COLUMNS))
    groups = _group_ids(features)

    print("Dataset:", source_path)
    print("Rows:", len(raw), "| features used:", len(features.columns))
    print("Raw num distribution:", raw[TARGET_COLUMN].value_counts().sort_index().to_dict())
    binary_counts = target.value_counts().sort_index().to_dict()
    print(
        "Binary target (num > 0):",
        binary_counts,
        "| prevalence:",
        f"{target.mean():.3%}",
    )

    site_stats = pd.DataFrame(
        {
            "records": raw.groupby(SITE_COLUMN).size(),
            "positive": target.groupby(raw[SITE_COLUMN]).sum(),
            "prevalence": target.groupby(raw[SITE_COLUMN]).mean(),
        }
    ).reset_index()
    site_stats["prevalence"] = site_stats["prevalence"].map(lambda value: f"{value:.1%}")
    _print_table("Disease prevalence by source site:", site_stats)

    missing = raw.drop(columns=[TARGET_COLUMN]).isna().sum()
    # Include the target explicitly in the report without altering its raw values.
    missing[TARGET_COLUMN] = raw[TARGET_COLUMN].isna().sum()
    _print_table(
        "Missing values per column (chol/trestbps zero sentinels already marked missing):",
        missing.rename_axis("column").reset_index(name="missing_rows"),
    )
    print("Zero sentinels treated missing:", zero_sentinels)

    exact_duplicates = int(raw.duplicated().sum())
    grouped_duplicates = int(pd.Series(groups).duplicated().sum())
    print("Exact duplicate rows:", exact_duplicates)
    print("Repeated clinical predictor rows (ignoring id/site):", grouped_duplicates)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(features, target, groups))
    X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
    train_groups, test_groups = set(groups[train_idx]), set(groups[test_idx])
    overlap = train_groups.intersection(test_groups)
    print(
        f"Split: StratifiedGroupKFold fold 1/5 (random_state={RANDOM_STATE});",
        f"train={len(train_idx)}, test={len(test_idx)};",
        f"train prevalence={y_train.mean():.3%}, test prevalence={y_test.mean():.3%}",
    )
    print("Duplicate clinical groups crossing train/test:", len(overlap))

    full_model = _make_pipeline(X_train)
    full_model.fit(X_train, y_train)
    full_probability = full_model.predict_proba(X_test)[:, 1]
    results = {"All features": _metrics(y_test, full_probability)}

    from sklearn.inspection import permutation_importance

    importance = permutation_importance(
        full_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_table = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance": importance.importances_mean,
            "Std": importance.importances_std,
        }
    ).sort_values("Importance", ascending=False)
    best_five = tuple(importance_table["Feature"].head(5))
    defaults = _train_defaults(X_train)

    for label, selected in (
        ("Best-performing 5 (permutation-ranked)", best_five),
        ("Most user-friendly 5", USER_FRIENDLY_FEATURES),
    ):
        masked_test = _apply_input_defaults(X_test, selected, defaults)
        results[label] = _metrics(y_test, full_model.predict_proba(masked_test)[:, 1])

    _print_table(
        "Test-set permutation importance (ROC-AUC decrease; mean +/- std):",
        importance_table.assign(
            Importance=importance_table["Importance"].map(lambda value: f"{value:.4f}"),
            Std=importance_table["Std"].map(lambda value: f"{value:.4f}"),
        ),
    )
    print("Best-performing five:", list(best_five))
    print("Most user-friendly five:", list(USER_FRIENDLY_FEATURES))
    _print_table(
        "Feature selection and non-input handling:",
        importance_table.assign(
            **{
                "Chosen?": importance_table["Feature"].map(
                    lambda name: "Best 5" if name in best_five else "Friendly 5" if name in USER_FRIENDLY_FEATURES else "No"
                ),
                "If not chosen": importance_table["Feature"].map(
                    lambda name: "" if name in USER_FRIENDLY_FEATURES or name in best_five else f"Training {('median' if pd.api.types.is_numeric_dtype(X_train[name]) else 'mode')}: {defaults[name]}"
                ),
            }
        )[["Feature", "Importance", "Chosen?", "If not chosen"]],
    )

    result_table = pd.DataFrame(results).T.reset_index(names="Model / input set")
    _print_table(
        "Held-out test metrics (five-input cases fill omitted features with train-set median/mode):",
        result_table.assign(**{column: result_table[column].map(lambda value: f"{value:.4f}") for column in result_table.columns[1:]}),
    )
    for label, values in (
        ("Best five", _metrics(y_test, full_model.predict_proba(_apply_input_defaults(X_test, best_five, defaults))[:, 1])),
        ("User-friendly five", results["Most user-friendly 5"]),
    ):
        recall_loss = results["All features"]["recall"] - values["recall"]
        auc_loss = results["All features"]["roc_auc"] - values["roc_auc"]
        print(f"{label} trade-off vs full: recall change={-recall_loss:+.4f}; ROC-AUC change={-auc_loss:+.4f}")

    # Grouped, stratified five-fold CV. Defaults are re-estimated only from each
    # fold's training partition, and repeated clinical rows stay within a fold.
    cv_splits = list(splitter.split(features, target, groups))
    cv_names = ("All features", "Best-performing 5", "Most user-friendly 5")
    cv_scores = {name: [] for name in cv_names}
    for fold_train, fold_test in cv_splits:
        fold_X_train, fold_X_test = features.iloc[fold_train], features.iloc[fold_test]
        fold_y_train, fold_y_test = target.iloc[fold_train], target.iloc[fold_test]
        fold_defaults = _train_defaults(fold_X_train)
        fold_model = _make_pipeline(fold_X_train)
        fold_model.fit(fold_X_train, fold_y_train)
        cv_scores["All features"].append(
            _metrics(fold_y_test, fold_model.predict_proba(fold_X_test)[:, 1])
        )
        for name, selected in (
            ("Best-performing 5", best_five),
            ("Most user-friendly 5", USER_FRIENDLY_FEATURES),
        ):
            masked = _apply_input_defaults(fold_X_test, selected, fold_defaults)
            cv_scores[name].append(_metrics(fold_y_test, fold_model.predict_proba(masked)[:, 1]))
    cv_rows = []
    for name, folds in cv_scores.items():
        for metric in ("accuracy", "recall", "precision", "f1", "roc_auc"):
            values = np.array([fold[metric] for fold in folds])
            cv_rows.append({"Model / input set": name, "Metric": metric, "Mean +/- SD": f"{values.mean():.4f} +/- {values.std(ddof=1):.4f}"})
    _print_table("Grouped stratified 5-fold CV (mean +/- SD):", pd.DataFrame(cv_rows))

    # Leave one source site out at a time. dataset is a grouping variable only,
    # never a model feature.
    logo = LeaveOneGroupOut()
    site_rows = []
    site_groups = raw[SITE_COLUMN].to_numpy()
    for fold_train, fold_test in logo.split(features, target, site_groups):
        site_X_train, site_X_test = features.iloc[fold_train], features.iloc[fold_test]
        site_y_train, site_y_test = target.iloc[fold_train], target.iloc[fold_test]
        site_defaults = _train_defaults(site_X_train)
        site_model = _make_pipeline(site_X_train)
        site_model.fit(site_X_train, site_y_train)
        held_out_site = raw.iloc[fold_test][SITE_COLUMN].iloc[0]
        for label, selected in (
            ("All features", tuple(features.columns)),
            ("Best-performing 5", best_five),
            ("Most user-friendly 5", USER_FRIENDLY_FEATURES),
        ):
            evaluated = (
                site_X_test
                if label == "All features"
                else _apply_input_defaults(site_X_test, selected, site_defaults)
            )
            site_probability = site_model.predict_proba(evaluated)[:, 1]
            site_rows.append(
                {
                    "Held-out site": held_out_site,
                    "Input set": label,
                    **_metrics(site_y_test, site_probability),
                    "n": len(fold_test),
                }
            )
    _print_table("Site-grouped evaluation (leave one source site out):", pd.DataFrame(site_rows))

    # Report associations across numeric, categorical, and mixed-type inputs.
    numeric = X_train.select_dtypes(include="number")
    corr = numeric.corr(method="spearman").abs()
    corr_pairs = [
        {"Feature A": a, "Feature B": b, "Association": corr.loc[a, b], "Method": "|Spearman rho|"}
        for index, a in enumerate(corr.columns)
        for b in corr.columns[index + 1 :]
        if corr.loc[a, b] >= 0.70
    ]
    categorical = [column for column in X_train if column not in numeric.columns]
    from scipy.stats import chi2_contingency

    for index, first in enumerate(categorical):
        for second in categorical[index + 1 :]:
            pair = X_train[[first, second]].dropna()
            table = pd.crosstab(pair[first], pair[second])
            if min(table.shape) > 1:
                chi2 = chi2_contingency(table, correction=False)[0]
                value = np.sqrt(chi2 / (table.to_numpy().sum() * min(table.shape[0] - 1, table.shape[1] - 1)))
                if value >= 0.70:
                    corr_pairs.append({"Feature A": first, "Feature B": second, "Association": value, "Method": "Cramer's V"})
    for cat in categorical:
        for num in numeric.columns:
            pair = X_train[[cat, num]].dropna()
            overall = pair[num].mean()
            total = ((pair[num] - overall) ** 2).sum()
            if total == 0 or pair[cat].nunique() < 2:
                continue
            between = sum(
                len(group) * (group[num].mean() - overall) ** 2
                for _, group in pair.groupby(cat, observed=True)
            )
            value = np.sqrt(between / total)
            if value >= 0.70:
                corr_pairs.append({"Feature A": cat, "Feature B": num, "Association": value, "Method": "Correlation ratio (eta)"})
    if corr_pairs:
        _print_table(
            "Feature pairs with association >= 0.70 (training partition):",
            pd.DataFrame(corr_pairs).sort_values("Association", ascending=False),
        )
    else:
        print("No feature pairs had association >= 0.70 in the training partition.")

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    artifacts = (
        ("heart_disease_best_performance_5.joblib", best_five),
        ("heart_disease_user_friendly_5.joblib", USER_FRIENDLY_FEATURES),
    )
    for filename, selected in artifacts:
        artifact_path = models_dir / filename
        if artifact_path.exists():
            print("Kept existing model artifact (not overwritten):", artifact_path)
            continue
        joblib.dump(
            {
                "estimator": full_model,
                "input_features": list(selected),
                "default_values": defaults,
                "training_target": "num > 0",
                "excluded_columns": list(EXCLUDED_COLUMNS),
                "random_state": RANDOM_STATE,
            },
            artifact_path,
        )
        print("Saved model candidate:", artifact_path)

    print(
        "Note: permutation ranking and input-set selection use this test set as requested;"
        " treat test metrics as exploratory. A final unbiased estimate needs nested CV or a new holdout."
    )


if __name__ == "__main__":
    main()
