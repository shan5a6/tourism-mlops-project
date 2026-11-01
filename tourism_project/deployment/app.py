"""Rubric 4: Streamlit front end for pre-contact lead purchase predictions."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.joblib"
SCHEMA_PATH = HERE / "feature_schema.json"

st.set_page_config(page_title="Visit with Us | Tourism Purchase Prediction", page_icon="✈️", layout="centered")
st.title("Tourism Package Purchase Prediction")
st.write("Predict a customer's likelihood of purchasing a travel package before an outreach call.")
st.info("Educational model trained on historical package purchases; not a validated Wellness-package-specific forecast.")
if not MODEL_PATH.is_file() or not SCHEMA_PATH.is_file():
    st.error("The trained model is not in the repository yet. Run GitHub Actions → Tourism Project Pipeline and let it commit the model artifacts.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

pipeline = load_model()
schema = load_schema()
with st.form("prediction_form"):
    st.subheader("Customer information")
    cols = st.columns(2)
    user_values = {}
    for idx, (name, config) in enumerate(schema.items()):
        with cols[idx % 2]:
            if config["kind"] == "category":
                user_values[name] = st.selectbox(name, config["choices"])
            elif name in ("Passport", "OwnCar"):
                user_values[name] = int(st.selectbox(name, [0, 1], index=int(config["default"] == 1)))
            elif name in ("CityTier", "PreferredPropertyStar"):
                opts = list(range(int(config["minimum"]), int(config["maximum"]) + 1))
                user_values[name] = st.selectbox(name, opts, index=opts.index(int(config["default"])))
            elif config["integer"]:
                user_values[name] = st.number_input(
                    name, min_value=int(config["minimum"]), max_value=int(config["maximum"]),
                    value=int(config["default"]), step=1
                )
            else:
                user_values[name] = st.number_input(
                    name, min_value=float(config["minimum"]), max_value=float(config["maximum"]),
                    value=float(config["default"])
                )
    submitted = st.form_submit_button("Predict purchase likelihood", use_container_width=True)

if submitted:
    sample = pd.DataFrame([user_values], columns=list(schema.keys()))
    likelihood = float(pipeline.predict_proba(sample)[0, 1])
    prediction = int(pipeline.predict(sample)[0])
    st.metric("Estimated purchase likelihood", f"{likelihood * 100:.1f}%")
    st.success("Likely to purchase" if prediction == 1 else "Unlikely to purchase")
    st.caption("The probability is a model estimate, not a guarantee. Use alongside human review and marketing consent.")
