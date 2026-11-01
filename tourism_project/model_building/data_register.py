"""Rubric 1: register dataset by validating schema and printing a summary."""
from pathlib import Path
import pandas as pd
try:
    from .features import EXPECTED_COLUMNS, TARGET
except ImportError:
    from features import EXPECTED_COLUMNS, TARGET

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "tourism.csv"


def validate_dataset(path=INPUT_PATH):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if df.empty:
        raise ValueError("Dataset must contain at least one record")
    if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("ProdTaken must contain only 0 and 1, without nulls")
    if df["CustomerID"].duplicated().any():
        raise ValueError("CustomerID is not unique")
    print("DATA REGISTRATION: PASSED")
    print(f"Source: {path}")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    print(f"Missing cells: {int(df.isna().sum().sum())}")
    print(f"Unique customers: {df['CustomerID'].nunique():,}")
    print("Target counts:", df[TARGET].value_counts().sort_index().to_dict())
    return df


if __name__ == "__main__":
    validate_dataset()
