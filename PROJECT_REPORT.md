# Cardiovascular Disease Prediction Project Report

## 1. Introduction

Cardiovascular disease is a major health concern. This project uses machine learning to predict the likelihood of cardiovascular disease from patient health measurements and lifestyle information.

## 2. Dataset

The dataset originally contained 70,000 patient records and 13 columns.

The target column is `cardio`:

- `0`: No cardiovascular disease
- `1`: Cardiovascular disease

The dataset includes age, gender, height, weight, blood pressure, cholesterol, glucose level, smoking, alcohol consumption, and physical activity.

## 3. Data Cleaning

The dataset had no missing values and no duplicate rows.

Some records contained implausible blood pressure values. Rows with unreliable height, weight, or blood-pressure measurements were removed.

After cleaning:

- Original rows: 70,000
- Removed rows: 1,402
- Final rows: 68,598

Two new features were created:

- `age_years`: Age converted from days to years
- `bmi`: Body Mass Index calculated from height and weight

## 4. Model Development

The data was divided into:

- Training data: 80%
- Testing data: 20%

Two classification models were trained:

1. Logistic Regression
2. Random Forest Classifier

## 5. Model Results

| Model | Accuracy | ROC-AUC | Recall for Disease Class |
|---|---:|---:|---:|
| Logistic Regression | 0.73 | 0.7916 | 0.67 |
| Random Forest | 0.71 | 0.7751 | 0.70 |

Logistic Regression was selected as the final model because it had the best overall accuracy and ROC-AUC score.

## 6. Final Application

A Streamlit web application was created. Users can enter patient information such as age, height, weight, blood pressure, cholesterol, glucose level, and lifestyle details.

The application calculates BMI and provides an estimated cardiovascular disease probability.

## 7. Conclusion

The final Logistic Regression model achieved a ROC-AUC score of 0.7916 and an accuracy of 73%. The project demonstrates how machine learning can support cardiovascular risk screening.

The application is for educational purposes only and is not a medical diagnosis tool.

## 8. Limitations

- The model is trained on one dataset only.
- A prediction does not replace clinical diagnosis.
- More medical information could improve future versions of the model.