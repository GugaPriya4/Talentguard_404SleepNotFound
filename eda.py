import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

BASE_DIR = Path(__file__).resolve().parent
XLSX_PATH = BASE_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.xlsx"

# 1. Load the dataset
xls = pd.ExcelFile(XLSX_PATH)
print("Sheets available:", xls.sheet_names)
df = pd.read_excel(XLSX_PATH, sheet_name=xls.sheet_names[0])

# 2. Inspect its structure
print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types and non-null counts:")
df.info()

# 3. Check missing values
missing = df.isnull().sum().sort_values(ascending=False)
print("\nMissing values by column:")
print(missing[missing > 0] if missing.sum() else "No missing values.")

# 4. Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# 5. Check constant columns
constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
print("\nConstant columns:", constant_cols)

# 6. Remove known non-predictive columns
drop_cols = ["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"]
df = df.drop(columns=drop_cols, errors="ignore")
print("\nShape after dropping columns:", df.shape)

# 7. Target distribution
print("\nAttrition counts:")
print(df["Attrition"].value_counts())

print("\nAttrition percentages:")
print((df["Attrition"].value_counts(normalize=True) * 100).round(2))

# 8. Attrition rate by overtime
print("\nAttrition rate by overtime (%):")
print(
    pd.crosstab(df["OverTime"], df["Attrition"], normalize="index")
    .mul(100)
    .round(2)
)

# 9. Attrition rate by department
print("\nAttrition rate by department (%):")
print(
    pd.crosstab(df["Department"], df["Attrition"], normalize="index")
    .mul(100)
    .round(2)
    .sort_values("Yes", ascending=False)
)

# 10. Attrition rate by job role
print("\nAttrition rate by job role (%):")
print(
    pd.crosstab(df["JobRole"], df["Attrition"], normalize="index")
    .mul(100)
    .round(2)
    .sort_values("Yes", ascending=False)
)