import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

# ------------------------------------------------------------------
# 0. Load and prepare
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
XLSX_PATH = BASE_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.xlsx"

xls = pd.ExcelFile(XLSX_PATH)
df = pd.read_excel(XLSX_PATH, sheet_name=xls.sheet_names[0])

df = df.drop(columns=["EmployeeCount", "StandardHours", "Over18"],
             errors="ignore")

# CRITICAL: create the binary target used throughout this script
df["AttritionBinary"] = (df["Attrition"] == "Yes").astype(int)

print("Dataset loaded:", df.shape)
print("Baseline attrition rate:",
      round(df["AttritionBinary"].mean() * 100, 2), "%")


# ------------------------------------------------------------------
# A1. Attrition rate by satisfaction and work-life balance
# ------------------------------------------------------------------
satisfaction_cols = [
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "WorkLifeBalance",
    "JobInvolvement",
    "RelationshipSatisfaction",
]

satisfaction_tables = {}

for col in satisfaction_cols:
    print(f"\n{'=' * 60}")
    print(f"ATTRITION RATE BY {col.upper()}")
    print("=" * 60)

    result = (
        df.groupby(col, observed=True)
          .agg(
              Employees=("AttritionBinary", "size"),
              Employees_Left=("AttritionBinary", "sum"),
              Attrition_Rate=("AttritionBinary", "mean"),
          )
    )
    result["Attrition_Rate"] = (result["Attrition_Rate"] * 100).round(2)
    satisfaction_tables[col] = result
    print(result)


# ------------------------------------------------------------------
# A2. Compare satisfaction by attrition status
# ------------------------------------------------------------------
print("\n=== SATISFACTION: EMPLOYEES WHO LEFT VS STAYED ===")

for col in satisfaction_cols:
    summary = df.groupby("Attrition")[col].agg(["count", "mean", "median"])
    print(f"\n{col}")
    print(summary.round(2))


# ------------------------------------------------------------------
# A3. Overtime x satisfaction x attrition
# ------------------------------------------------------------------
print("\n=== OVERTIME + SATISFACTION + ATTRITION ===")

for col in ["JobSatisfaction", "WorkLifeBalance"]:
    rates = (
        df.groupby([col, "OverTime"], observed=True)["AttritionBinary"]
          .agg(["count", "sum", "mean"])
          .rename(columns={
              "count": "Employees",
              "sum": "Employees_Left",
              "mean": "Attrition_Rate",
          })
    )
    rates["Attrition_Rate"] = (rates["Attrition_Rate"] * 100).round(2)
    print(f"\n{col}")
    print(rates)


# ------------------------------------------------------------------
# B1. Career factors: left vs stayed
# ------------------------------------------------------------------
career_cols = [
    "MonthlyIncome",
    "PercentSalaryHike",
    "YearsSinceLastPromotion",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsWithCurrManager",
    "TotalWorkingYears",
    "NumCompaniesWorked",
    "StockOptionLevel",
]

print("\n=== CAREER FACTORS: LEFT VS STAYED ===")

for col in career_cols:
    print(f"\n{'=' * 50}")
    print(col)
    print("=" * 50)
    print(df.groupby("Attrition")[col]
            .agg(["count", "mean", "median"]).round(2))


# ------------------------------------------------------------------
# B2. Attrition by income quartile
# ------------------------------------------------------------------
df["IncomeBand"] = pd.qcut(df["MonthlyIncome"], q=4, duplicates="drop")

print("\n=== ATTRITION BY INCOME QUARTILE ===")

income_result = (
    df.groupby("IncomeBand", observed=True)
      .agg(
          Employees=("AttritionBinary", "size"),
          Employees_Left=("AttritionBinary", "sum"),
          Attrition_Rate=("AttritionBinary", "mean"),
          Median_Income=("MonthlyIncome", "median"),
      )
)
income_result["Attrition_Rate"] = (income_result["Attrition_Rate"] * 100).round(2)
print(income_result)

# Excel-safe copy (Interval objects cannot be serialised)
income_export = income_result.copy()
income_export.index = income_export.index.astype(str)


# ------------------------------------------------------------------
# B3. Attrition by promotion delay
# ------------------------------------------------------------------
print("\n=== ATTRITION BY YEARS SINCE LAST PROMOTION ===")

promotion_result = (
    df.groupby("YearsSinceLastPromotion", observed=True)
      .agg(
          Employees=("AttritionBinary", "size"),
          Employees_Left=("AttritionBinary", "sum"),
          Attrition_Rate=("AttritionBinary", "mean"),
      )
)
promotion_result["Attrition_Rate"] = (
    promotion_result["Attrition_Rate"] * 100
).round(2)
print(promotion_result)


# ------------------------------------------------------------------
# B4. Career factors by overtime and attrition
# ------------------------------------------------------------------
print("\n=== CAREER FACTORS BY OVERTIME AND ATTRITION ===")

for col in ["MonthlyIncome", "YearsSinceLastPromotion",
            "YearsAtCompany", "YearsInCurrentRole"]:
    print(f"\n{col}")
    print(df.groupby(["OverTime", "Attrition"], observed=True)[col]
            .agg(["count", "mean", "median"]).round(2))


# ------------------------------------------------------------------
# C1. Chi-square: overtime vs attrition
# ------------------------------------------------------------------
print("\n=== CHI-SQUARE TEST: OVERTIME VS ATTRITION ===")

table = pd.crosstab(df["OverTime"], df["Attrition"])
print("\nContingency table:")
print(table)

chi2, p_value, dof, expected = chi2_contingency(table)
n = table.to_numpy().sum()
cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))

print(f"\nChi-square statistic: {chi2:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p_value:.6g}")
print(f"Cramer's V: {cramers_v:.4f}")
print("Result:",
      "Evidence of an association between overtime and attrition."
      if p_value < 0.05 else
      "Insufficient evidence at the 5% level.")


# ------------------------------------------------------------------
# C2. Chi-square across ALL categorical drivers (ranked)
# ------------------------------------------------------------------
print("\n=== CATEGORICAL DRIVER RANKING (Cramer's V) ===")

cat_cols = [
    "OverTime", "BusinessTravel", "MaritalStatus", "JobRole",
    "Department", "EducationField", "Gender", "JobLevel",
    "StockOptionLevel", "JobSatisfaction", "EnvironmentSatisfaction",
    "WorkLifeBalance", "JobInvolvement", "RelationshipSatisfaction",
]

rows = []
for col in cat_cols:
    ct = pd.crosstab(df[col], df["Attrition"])
    c2, p, _, _ = chi2_contingency(ct)
    v = np.sqrt(c2 / (ct.to_numpy().sum() * (min(ct.shape) - 1)))
    rows.append({
        "Variable": col,
        "ChiSquare": round(c2, 2),
        "P_Value": p,
        "Cramers_V": round(v, 4),
        "Significant_5pct": p < 0.05,
    })

driver_rank = (pd.DataFrame(rows)
                 .sort_values("Cramers_V", ascending=False)
                 .reset_index(drop=True))
print(driver_rank.to_string(index=False))


# ------------------------------------------------------------------
# C3. Fisher's exact: Sales Representatives
# ------------------------------------------------------------------
print("\n=== FISHER'S EXACT TEST: SALES REPRESENTATIVES ===")

sales_rep = df[df["JobRole"] == "Sales Representative"]
sr_table = (pd.crosstab(sales_rep["OverTime"], sales_rep["Attrition"])
              .reindex(index=["No", "Yes"], columns=["No", "Yes"],
                       fill_value=0))

print("\nContingency table:")
print(sr_table)

odds_ratio, p_value = fisher_exact(sr_table.to_numpy())
print(f"\nOdds ratio: {odds_ratio:.4f}")
print(f"P-value: {p_value:.6g}")
print("Result:",
      "Evidence of an association within Sales Representatives."
      if p_value < 0.05 else
      "Insufficient evidence at the 5% level.")


# ------------------------------------------------------------------
# C4. Mann-Whitney U with rank-biserial effect size
# ------------------------------------------------------------------
print("\n=== MANN-WHITNEY U TESTS ===")

numeric_cols = [
    "MonthlyIncome", "PercentSalaryHike", "YearsSinceLastPromotion",
    "YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager",
    "TotalWorkingYears", "Age", "DistanceFromHome", "NumCompaniesWorked",
]

mw_rows = []
for col in numeric_cols:
    stayed = df.loc[df["Attrition"] == "No", col].dropna()
    left = df.loc[df["Attrition"] == "Yes", col].dropna()

    u_stat, p_val = mannwhitneyu(stayed, left, alternative="two-sided")
    rbc = 1 - (2 * u_stat) / (len(stayed) * len(left))   # effect size

    mw_rows.append({
        "Variable": col,
        "Stayed_Median": round(stayed.median(), 2),
        "Left_Median": round(left.median(), 2),
        "U_Statistic": round(u_stat, 1),
        "P_Value": p_val,
        "RankBiserial": round(rbc, 4),
        "Significant_5pct": p_val < 0.05,
    })

mw_results = (pd.DataFrame(mw_rows)
                .reindex(columns=["Variable", "Stayed_Median", "Left_Median",
                                  "U_Statistic", "P_Value", "RankBiserial",
                                  "Significant_5pct"])
                .assign(AbsEffect=lambda d: d["RankBiserial"].abs())
                .sort_values("AbsEffect", ascending=False)
                .drop(columns="AbsEffect")
                .reset_index(drop=True))

print(mw_results.to_string(index=False))


# ------------------------------------------------------------------
# D. Export everything
# ------------------------------------------------------------------
OUT_PATH = BASE_DIR / "talentguard_deeper_eda.xlsx"

with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:

    for col, tbl in satisfaction_tables.items():
        tbl.to_excel(writer, sheet_name=col[:31])

    career_summary = df.groupby("Attrition")[career_cols].agg(["mean", "median"])
    career_summary.to_excel(writer, sheet_name="Career_Summary")

    income_export.to_excel(writer, sheet_name="Income_Quartiles")
    promotion_result.to_excel(writer, sheet_name="Promotion_Delay")

    overtime_summary = (df.groupby(["OverTime", "Attrition"], observed=True)
                          .size().unstack(fill_value=0))
    overtime_summary.to_excel(writer, sheet_name="Overtime_Attrition")

    driver_rank.to_excel(writer, sheet_name="Categorical_Ranking", index=False)
    mw_results.to_excel(writer, sheet_name="Numeric_Tests", index=False)

print(f"\nSaved results to {OUT_PATH.name}")