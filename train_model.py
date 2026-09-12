import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Load the cleaned data
df = pd.read_csv("data/cardio_cleaned.csv")

# Separate input features from the target
X = df.drop(columns="cardio")
y = df["cardio"]

# Keep 20% of data for a final unbiased test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Logistic Regression baseline model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nClassification report:")
print(classification_report(y_test, predictions))

print("ROC-AUC score:", round(roc_auc_score(y_test, probabilities), 4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

# Save model for the future Streamlit app
joblib.dump(model, "cardio_logistic_model.joblib")
print("\nModel saved as cardio_logistic_model.joblib")


from sklearn.ensemble import RandomForestClassifier

# Random Forest comparison model
forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

forest_model.fit(X_train, y_train)

forest_predictions = forest_model.predict(X_test)
forest_probabilities = forest_model.predict_proba(X_test)[:, 1]

print("\n--- Random Forest Results ---")
print(classification_report(y_test, forest_predictions))

print(
    "Random Forest ROC-AUC:",
    round(roc_auc_score(y_test, forest_probabilities), 4)
)

print("\nRandom Forest confusion matrix:")
print(confusion_matrix(y_test, forest_predictions))

joblib.dump(forest_model, "cardio_random_forest_model.joblib")
print("\nRandom Forest model saved as cardio_random_forest_model.joblib")
