"""Run after training to validate reproducible artifacts and predictions."""
import json
import unittest
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class TourismPipelineTests(unittest.TestCase):
    def test_registered_data(self):
        df = pd.read_csv(ROOT / "tourism_project/data/tourism.csv")
        self.assertIn("ProdTaken", df.columns)
        self.assertTrue(set(df["ProdTaken"].unique()) <= {0, 1})
        self.assertFalse(df["CustomerID"].duplicated().any())

    def test_split_has_no_identifier_or_post_contact_leakage(self):
        from tourism_project.model_building.features import FEATURE_COLUMNS, TARGET, EXCLUDED
        train = pd.read_csv(ROOT / "tourism_project/artifacts/train.csv")
        test = pd.read_csv(ROOT / "tourism_project/artifacts/test.csv")
        self.assertEqual(list(train.columns), FEATURE_COLUMNS + [TARGET])
        self.assertFalse(any(name in train for name in EXCLUDED))
        self.assertEqual(len(train) + len(test), 4128)

    def test_model_and_streamlit_schema(self):
        path = ROOT / "tourism_project/deployment"
        model = joblib.load(path / "model.joblib")
        schema = json.loads((path / "feature_schema.json").read_text())
        from tourism_project.model_building.features import FEATURE_COLUMNS
        self.assertEqual(list(schema), FEATURE_COLUMNS)
        example = {}
        for k, v in schema.items():
            example[k] = v["choices"][0] if v["kind"] == "category" else v["default"]
        test_input = pd.DataFrame([example], columns=FEATURE_COLUMNS)
        self.assertEqual(len(model.predict(test_input)), 1)
        self.assertGreaterEqual(float(model.predict_proba(test_input)[0, 1]), 0)


if __name__ == "__main__":
    unittest.main()
