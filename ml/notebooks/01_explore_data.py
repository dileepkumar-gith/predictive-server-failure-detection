# ml/notebooks/01_explore_data.py

import pandas as pd

# Relative path - works because we run this script from the project root folder
df = pd.read_csv("ml/data/Failure_Prediction.csv")

print("Columns found in the dataset:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape (rows, columns):", df.shape)

print("\nColumn info:")
print(df.info())

# Check for missing values in each column
print("\nMissing values per column:")
print(df.isnull().sum())

# Check the unique failure_risk categories and how many rows fall into each
print("\nFailure risk value counts:")
print(df["failure_risk"].value_counts())

# Check basic statistics for the numeric columns
print("\nStatistical summary:")
print(df.describe())

# Check correlation between numeric features (helps us see which metrics move together)
print("\nCorrelation matrix (numeric columns only):")
print(df.corr(numeric_only=True))


# Sanity check: do higher-risk servers actually show higher CPU/temp/power on average?
print("\nAverage metrics grouped by failure risk:")
print(df.groupby("failure_risk")[["cpu_usage_percent", "temperature_celsius", "power_consumption_watts"]].mean())
