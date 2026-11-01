# Advanced Machine Learning and MLOps: Tourism Package Prediction

**Visit with Us — Wellness Tourism Package project (course assignment)**

Historical `ProdTaken` means purchase of a *historical travel package*, not proven demand for Wellness. All 13 model features are potentially available prior to an outreach pitch; sales-pitch follow-ups and satisfaction variables are excluded to prevent future-information leakage.

## Implemented sections

| Criterion | Implementation |
|---|---|
| Data Registration | `tourism_project/model_building/data_register.py` |
| Data Preparation | `tourism_project/model_building/prep.py` + `prepared-data` GitHub Actions artifact |
| Model Building | `tourism_project/model_building/train.py` + MLflow logs + saved model |
| Deployment | `tourism_project/deployment/app.py` + deployment requirements |
| GitHub Actions | `.github/workflows/pipeline.yml` |
| Output Evaluation | screenshots and live links to place in notebook |
| Notebook Quality | `notebooks/Tourism_MLOps_Project_Submission.ipynb`, exported HTML |

## Prepare your repository

1. Open and execute the prepared Google Colab notebook. It creates the repository, generates source files and pushes them to `main`.
4. Open **Actions → Tourism Project Pipeline**, inspect the green jobs `Data Registration`, `Data Preparation`, `Model Training, Tracking and Commit`.
5. The last job automatically commits `tourism_project/deployment/model.joblib`, `feature_schema.json`, and `metrics.json` to the repository. A push using `GITHUB_TOKEN` ordinarily does not retrigger this pipeline; the workflow also limits paths.
6. Download the `prepared-data`, `evaluation-and-model`, and `mlflow-experiments` Actions artifacts as evidence, if desired.
7. Open https://share.streamlit.io → Create app → Your GitHub repo → branch `main` → main file `tourism_project/deployment/app.py` → deploy. Choose **Python 3.11** in advanced settings when available.
8. Add your actual public repository URL and Streamlit URL and screenshots to the notebook, execute code top to bottom, export it to a *single HTML* for submission.

## Local validation

Use Python 3.11:

```bash
python -m pip install -r tourism_project/requirements.txt
python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py
python -m unittest discover -s tests -v
python -m pip install -r tourism_project/deployment/requirements.txt
streamlit run tourism_project/deployment/app.py
```

If not using a local MLflow installation, train.py still saves local model/metrics/results but does not produce MLflow experiment history. GitHub Actions installs the pinned MLflow dependency to record the experiments.

## Security and reproducibility

- No personal access token is required for the standard GitHub Actions `GITHUB_TOKEN` write-back route. Do not include credentials in notebook outputs or repository files.
- Stratified 80/20 split, fixed random seed, 3-fold CV hyperparameter grid, evaluation on held-out test split.
- Numerics imputed by training medians and categoricals by training modes; encoding is learned only on the training split within the fitted pipeline.
- Initial Streamlit app may display **model not available** before first successful training workflow. The workflow commits the model after successful training.
- Metric choice: F1 as cross-validation tuning score, report accuracy, precision, recall, F1 and ROC-AUC due to minority positive class.

## Submission

Only the exported `.html` is submitted to the course portal. Submit after inserting actual live URLs and screenshots, and rerunning the notebook. Do not claim GitHub or Streamlit execution evidence until these services have actually run.
