import pandas as pd
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

# ------------------------------------------------------------------
# 0. Load the dataset (Excel, sheet auto-detected)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
XLSX_PATH = BASE_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.xlsx"

xls = pd.ExcelFile(XLSX_PATH)
df = pd.read_excel(XLSX_PATH, sheet_name=xls.sheet_names[0])

# Drop non-predictive / constant columns
df = df.drop(
    columns=["EmployeeNumber", "EmployeeCount", "StandardHours", "Over18"],
    errors="ignore",
)

print("Dataset loaded:", df.shape)


# ------------------------------------------------------------------
# Helper: build a Yes/No crosstab with Total and percentage columns
# ------------------------------------------------------------------
def rate_table(frame: pd.DataFrame, index_col: str, flag_col: str,
               label: str) -> pd.DataFrame:
    table = pd.crosstab(frame[index_col], frame[flag_col])
    for col in ("Yes", "No"):
        if col not in table.columns:
            table[col] = 0
    table = table[["No", "Yes"]]
    table["Total"] = table.sum(axis=1)
    table[label] = (table["Yes"] / table["Total"] * 100).round(2)
    return table.sort_values(label, ascending=False)


# ------------------------------------------------------------------
# 1. Department: who works overtime?
# ------------------------------------------------------------------
print("\n=== OVERTIME BY DEPARTMENT ===")
dept_overtime = rate_table(df, "Department", "OverTime", "Overtime %")
print(dept_overtime)


# ------------------------------------------------------------------
# 2. Job role: who works overtime?
# ------------------------------------------------------------------
print("\n=== OVERTIME BY JOB ROLE ===")
role_overtime = rate_table(df, "JobRole", "OverTime", "Overtime %")
print(role_overtime)


# ------------------------------------------------------------------
# 3. Department + job role + overtime vs attrition
# ------------------------------------------------------------------
print("\n=== DEPARTMENT, JOB ROLE, OVERTIME & ATTRITION ===")

group_analysis = (
    df.groupby(["Department", "JobRole", "OverTime"], observed=True)
      .agg(
          Employees=("Attrition", "size"),
          Attrition_Count=("Attrition", lambda x: (x == "Yes").sum()),
          Attrition_Rate=("Attrition", lambda x: (x == "Yes").mean() * 100),
      )
      .reset_index()
)
group_analysis["Attrition_Rate"] = group_analysis["Attrition_Rate"].round(2)

print(
    group_analysis
    .sort_values(["Department", "JobRole", "OverTime"])
    .to_string(index=False)
)


# ------------------------------------------------------------------
# 3b. Overtime impact per role (pivoted, with delta)
# ------------------------------------------------------------------
print("\n=== OVERTIME IMPACT ON ATTRITION BY ROLE ===")

pivot = group_analysis.pivot_table(
    index=["Department", "JobRole"],
    columns="OverTime",
    values="Attrition_Rate",
).rename(columns={"No": "Attrition% (No OT)", "Yes": "Attrition% (OT)"})

pivot["Delta (pp)"] = (
    pivot["Attrition% (OT)"] - pivot["Attrition% (No OT)"]
).round(2)

print(pivot.sort_values("Delta (pp)", ascending=False).to_string())


# ------------------------------------------------------------------
# 4. Sales Representatives specifically
# ------------------------------------------------------------------
print("\n=== SALES REPRESENTATIVES: OVERTIME VS ATTRITION ===")

sales_rep = df[df["JobRole"] == "Sales Representative"]

sales_rep_analysis = (
    sales_rep.groupby("OverTime", observed=True)
             .agg(
                 Employees=("Attrition", "size"),
                 Attrition_Count=("Attrition", lambda x: (x == "Yes").sum()),
                 Attrition_Rate=("Attrition", lambda x: (x == "Yes").mean() * 100),
             )
)
sales_rep_analysis["Attrition_Rate"] = sales_rep_analysis["Attrition_Rate"].round(2)
print(sales_rep_analysis)


# ------------------------------------------------------------------
# 5. Sales department: overtime vs attrition
# ------------------------------------------------------------------
print("\n=== SALES DEPARTMENT: OVERTIME VS ATTRITION (%) ===")

sales_dept = df[df["Department"] == "Sales"]
print(
    pd.crosstab(sales_dept["OverTime"], sales_dept["Attrition"],
                normalize="index").mul(100).round(2)
)


# ------------------------------------------------------------------
# 6. Export all tables to one Excel workbook
# ------------------------------------------------------------------
OUT_PATH = BASE_DIR / "overtime_attrition_analysis.xlsx"

with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    dept_overtime.to_excel(writer, sheet_name="OT_by_Department")
    role_overtime.to_excel(writer, sheet_name="OT_by_JobRole")
    group_analysis.to_excel(writer, sheet_name="Dept_Role_OT_Attrition", index=False)
    pivot.to_excel(writer, sheet_name="OT_Impact_by_Role")
    sales_rep_analysis.to_excel(writer, sheet_name="Sales_Rep_OT")

print(f"\nAll tables exported to: {OUT_PATH.name}")