"""Features available before the sales pitch, plus cleaning rules."""
TARGET = "ProdTaken"
IDENTIFIERS = ["Unnamed: 0", "CustomerID"]
POST_CONTACT = ["DurationOfPitch", "NumberOfFollowups", "PitchSatisfactionScore", "ProductPitched"]
# TypeofContact is excluded to make predictions independent of subsequent outreach method.
EXCLUDED = IDENTIFIERS + POST_CONTACT + ["TypeofContact"]
FEATURE_COLUMNS = [
    "Age", "CityTier", "Occupation", "Gender", "NumberOfPersonVisiting",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
CATEGORICAL = ["Occupation", "Gender", "MaritalStatus", "Designation"]
NUMERIC = [name for name in FEATURE_COLUMNS if name not in CATEGORICAL]
EXPECTED_COLUMNS = [TARGET] + FEATURE_COLUMNS + EXCLUDED


def clean_categories(df):
    """Apply consistent corrections before splitting; does not fit transformations."""
    df = df.copy()
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
    return df
