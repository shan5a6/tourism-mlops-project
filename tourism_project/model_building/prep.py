"""Rubric 2: clean, split and save CSVs for the next GitHub Actions job."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
try:
    from .data_register import validate_dataset
    from .features import FEATURE_COLUMNS, TARGET, clean_categories
except ImportError:
    from data_register import validate_dataset
    from features import FEATURE_COLUMNS, TARGET, clean_categories

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"


def prepare_data(test_size=0.2):
    source = validate_dataset()
    source = clean_categories(source)
    cols = FEATURE_COLUMNS + [TARGET]
    df = source[cols].copy()  # Explicitly remove identifiers and post-contact leakage
    for name in FEATURE_COLUMNS:
        if name not in ("Occupation", "Gender", "MaritalStatus", "Designation"):
            df[name] = pd.to_numeric(df[name], errors="coerce")
    train, test = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df[TARGET]
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(ARTIFACT_DIR / "train.csv", index=False)
    test.to_csv(ARTIFACT_DIR / "test.csv", index=False)
    print("DATA PREPARATION: PASSED")
    print(f"Selected pre-contact features: {len(FEATURE_COLUMNS)}")
    print(f"Train: {train.shape[0]:,} | Test: {test.shape[0]:,}")
    print("Train purchase distribution:", train[TARGET].value_counts().sort_index().to_dict())
    print("Test purchase distribution:", test[TARGET].value_counts().sort_index().to_dict())
    print(f"Saved train/test splits in {ARTIFACT_DIR}")
    return train, test


if __name__ == "__main__":
    prepare_data()
