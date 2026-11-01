"""Rubric 3: cross-validated tuning, MLflow experiments, evaluation, model saving."""
from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
try:
    from .features import FEATURE_COLUMNS, CATEGORICAL, NUMERIC, TARGET
except ImportError:
    from features import FEATURE_COLUMNS, CATEGORICAL, NUMERIC, TARGET

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
DEPLOY_DIR = ROOT / "deployment"


def build_pipeline():
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(random_state=42, n_jobs=1)),
    ])


def run_training():
    train_path, test_path = ARTIFACT_DIR / "train.csv", ARTIFACT_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Prepared train.csv/test.csv missing. Run prep.py or download workflow artifact.")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    x_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    x_test, y_test = test[FEATURE_COLUMNS], test[TARGET]
    params = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [8, None],
        "model__min_samples_leaf": [1, 3],
        "model__class_weight": ["balanced_subsample"],
    }
    search = GridSearchCV(
        build_pipeline(), param_grid=params,
        scoring="f1", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=2, refit=True, return_train_score=False,
    )
    search.fit(x_train, y_train)
    best = search.best_estimator_
    pred = best.predict(x_test)
    prob = best.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 5),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 5),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 5),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 5),
        "roc_auc": round(float(roc_auc_score(y_test, prob)), 5),
        "cv_f1": round(float(search.best_score_), 5),
    }
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, DEPLOY_DIR / "model.joblib")
    values = {}
    for name in FEATURE_COLUMNS:
        col = x_train[name]
        if name in CATEGORICAL:
            values[name] = {"kind": "category", "choices": sorted(col.dropna().astype(str).unique().tolist())}
        else:
            sample = col.dropna().astype(float)
            values[name] = {
                "kind": "number",
                "minimum": float(sample.min()),
                "maximum": float(sample.max()),
                "default": float(sample.median()),
                "integer": bool(np.allclose(sample, sample.round())),
            }
    (DEPLOY_DIR / "feature_schema.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
    (DEPLOY_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACT_DIR / "classification_report.txt"
    report_path.write_text(classification_report(y_test, pred, digits=4, zero_division=0), encoding="utf-8")
    (ARTIFACT_DIR / "confusion_matrix.json").write_text(
        json.dumps(confusion_matrix(y_test, pred).tolist(), indent=2), encoding="utf-8"
    )
    cv_df = pd.DataFrame(search.cv_results_)
    cv_df[["rank_test_score", "mean_test_score", "std_test_score", "params"]].sort_values("rank_test_score").to_csv(
        ARTIFACT_DIR / "grid_search_results.csv", index=False
    )
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri((ROOT / "mlruns").resolve().as_uri())
        mlflow.set_experiment("VisitWithUs_Tourism_Purchase_Prediction")
        with mlflow.start_run(run_name="RandomForest_GridSearch"):
            for index, candidate_params in enumerate(search.cv_results_["params"]):
                with mlflow.start_run(run_name=f"candidate_{index + 1}", nested=True):
                    mlflow.log_params({k: str(v) for k, v in candidate_params.items()})
                    mlflow.log_metric("mean_cv_f1", float(search.cv_results_["mean_test_score"][index]))
                    mlflow.log_metric("std_cv_f1", float(search.cv_results_["std_test_score"][index]))
            mlflow.log_params({f"best_{k}": str(v) for k, v in search.best_params_.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
            mlflow.log_artifact(str(DEPLOY_DIR / "model.joblib"), artifact_path="model")
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")
            mlflow.log_artifact(str(ARTIFACT_DIR / "grid_search_results.csv"), artifact_path="evaluation")
        print("MLFLOW: Logged every tuned parameter candidate, CV scores and test metrics")
    else:
        print("MLFLOW NOT INSTALLED: Local metrics and CV results saved; install full requirements for MLflow tracking")
    print("MODEL TRAINING: PASSED")
    print("Best hyperparameters:", search.best_params_)
    print("Test metrics:", metrics)
    print("Confusion matrix:", confusion_matrix(y_test, pred).tolist())
    print(f"Saved model to {DEPLOY_DIR / 'model.joblib'}")
    return metrics


if __name__ == "__main__":
    run_training()
