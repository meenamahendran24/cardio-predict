  # Cardiovascular Disease Prediction

A machine-learning project that predicts the likelihood of cardiovascular disease from patient health and lifestyle information.

## Project goal

The goal is to classify whether a patient is likely to have cardiovascular disease.

- `0` = no cardiovascular disease
- `1` = cardiovascular disease

## Dataset

The dataset contains 70,000 patient records with these features:

- Age
- Gender
- Height and weight
- Systolic and diastolic blood pressure
- Cholesterol and glucose levels
- Smoking, alcohol consumption, and physical activity
- Cardiovascular disease target (`cardio`)

During cleaning, invalid medical measurements were removed. The final cleaned dataset contains 68,598 records.

## Model results

Two models were evaluated:

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.73 | 0.7916 |
| Random Forest | 0.71 | 0.7751 |

Logistic Regression was selected as the final model because it achieved the best ROC-AUC score and overall accuracy.

## Installation

Create and activate a virtual environment:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`.

## Deployment

This project cannot run on GitHub Pages: Pages serves static HTML, CSS, and JavaScript, while CardioPredict needs a Python/Streamlit server to load and run `cardio_logistic_model.joblib`.

Use **Streamlit Community Cloud** for a free live deployment:

1. Push the repository to GitHub, including `app.py`, `cardio_logistic_model.joblib`, `requirements.txt`, and `.streamlit/config.toml`.
2. In Streamlit Community Cloud, choose **Create app**, then select the repository, `main` branch, and `app.py` entry point.
3. Select **Deploy**. The URL will look like `https://your-github-username-cardiopredict.streamlit.app`.

No environment variables or secrets are required.

## Repository notes

- `data/` is excluded from Git because it is not needed to run the live app; it is only used for analysis and retraining.
- `cardio_random_forest_model.joblib` is excluded because it is a 440 MB experimental model and exceeds GitHub's normal file-size limit. The selected Logistic Regression model remains included.

## Disclaimer

This project is for education only. A model output is not a diagnosis and must not replace professional medical advice.
