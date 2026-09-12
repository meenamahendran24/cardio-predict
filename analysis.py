import pandas as pd

# The supplied file is an Excel workbook with semicolon-separated values in one column.
raw = pd.read_excel("data/cardio_train.csv", engine="openpyxl")

# Split that one column into the actual 13 dataset columns.
column_names = raw.columns[0].split(";")
df = raw.iloc[:, 0].astype(str).str.split(";", expand=True)
df.columns = column_names

# Convert values from text to numeric data.
df = df.apply(pd.to_numeric, errors="coerce")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())
print("\nCardiovascular disease counts:")
print(df["cardio"].value_counts())

print("\nCardiovascular disease percentages:")
print((df["cardio"].value_counts(normalize=True) * 100).round(2))

print("\nDuplicate rows:", df.duplicated().sum())

print("\nSummary statistics:")
print(df.describe())

invalid_bp = df[
    (df["ap_hi"] <= df["ap_lo"]) |
    (df["ap_hi"] < 70) |
    (df["ap_hi"] > 250) |
    (df["ap_lo"] < 40) |
    (df["ap_lo"] > 150)
]

print("\nRows with implausible blood pressure:", len(invalid_bp))
# Create a cleaned dataset
clean_df = df.copy()

# Keep medically plausible measurements
clean_df = clean_df[
    clean_df["height"].between(120, 220) &
    clean_df["weight"].between(35, 200) &
    clean_df["ap_hi"].between(70, 250) &
    clean_df["ap_lo"].between(40, 150) &
    (clean_df["ap_hi"] > clean_df["ap_lo"])
].copy()

# Feature engineering
clean_df["age_years"] = (clean_df["age"] / 365.25).round(1)
clean_df["bmi"] = (
    clean_df["weight"] / (clean_df["height"] / 100) ** 2
).round(2)

# Remove the ID and the age-in-days column
clean_df = clean_df.drop(columns=["id", "age"])

print("\nCleaned dataset shape:", clean_df.shape)
print("Rows removed during cleaning:", len(df) - len(clean_df))

print("\nNew features:")
print(clean_df[["age_years", "bmi"]].head())

# Save it for the modelling stage
clean_df.to_csv("data/cardio_cleaned.csv", index=False)
