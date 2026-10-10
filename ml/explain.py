"""SHAP explanations -> top-3 human-readable drivers per employee."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import Pool

FEATURE_LABELS = {
    "OverTime": "Works overtime",
    "UncompensatedOvertime": "Overtime with bottom-quartile pay",
    "OT_x_JobLevel": "Overtime at junior job level",
    "OT_x_LowInvolvement": "Overtime with low job involvement",
    "OT_x_WLBCritical": "Overtime with critical work-life balance",
    "MonthlyIncome": "Monthly income",
    "LowIncomeFlag": "Income in bottom quartile",
    "IncomeVsRoleMedian": "Pay relative to role peers",
    "IncomeVsLevelMedian": "Pay relative to job-level peers",
    "StockOptionLevel": "Stock option level",
    "JobInvolvement": "Job involvement",
    "LowInvolvement": "Low job involvement",
    "EnvSat_Critical": "Lowest environment satisfaction",
    "JobSat_Critical": "Lowest job satisfaction",
    "WLB_Critical": "Lowest work-life balance",
    "EnvironmentSatisfaction": "Environment satisfaction",
    "JobSatisfaction": "Job satisfaction",
    "WorkLifeBalance": "Work-life balance",
    "JobLevel": "Job level",
    "JobRole": "Job role",
    "Department": "Department",
    "YearsAtCompany": "Tenure at company",
    "IsNewJoiner": "New joiner (under 2 years)",
    "TenureRatio": "Share of career spent here",
    "RoleStability": "Stability in current role",
    "ManagerStability": "Stability with current manager",
    "YearsWithCurrManager": "Years with current manager",
    "YearsInCurrentRole": "Years in current role",
    "YearsSinceLastPromotion": "Years since last promotion",
    "PromotionStagnationRate": "Promotion stagnation",
    "BusinessTravel": "Business travel frequency",
    "DistanceFromHome": "Commute distance",
    "LongCommute": "Long commute",
    "Age": "Age",
    "TotalWorkingYears": "Total working experience",
    "EducationField": "Education field",
    "Education": "Education level",
    "TrainingTimesLastYear": "Training sessions last year",
}


def pretty(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def _catboost_component(artefact: dict):
    """SHAP is computed from the CatBoost component (60% of the blend)."""
    model = artefact["model"]
    return model["catboost"] if isinstance(model, dict) else model


def compute_shap(artefact: dict, X: pd.DataFrame) -> np.ndarray:
    cb = _catboost_component(artefact)
    pool = Pool(X, cat_features=artefact["cat_features"])
    values = cb.get_feature_importance(pool, type="ShapValues")
    return values[:, :-1]          # drop the expected-value column


def top_drivers(shap_values: np.ndarray, X: pd.DataFrame,
                n: int = 3) -> pd.DataFrame:
    cols = list(X.columns)
    records = []

    for i in range(shap_values.shape[0]):
        row = shap_values[i]
        order = np.argsort(-np.abs(row))[:n]
        labels, extras = [], {}
        for rank, j in enumerate(order, start=1):
            direction = "increases" if row[j] > 0 else "reduces"
            labels.append(f"{pretty(cols[j])} ({direction} risk)")
            extras[f"driver_{rank}"] = cols[j]
            extras[f"driver_{rank}_label"] = pretty(cols[j])
            extras[f"driver_{rank}_value"] = X.iloc[i, j]
            extras[f"driver_{rank}_shap"] = round(float(row[j]), 4)
        records.append({"top_factors": " | ".join(labels), **extras})

    return pd.DataFrame(records, index=X.index)


def global_importance(shap_values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    return (pd.DataFrame({
        "Feature": list(X.columns),
        "Label": [pretty(c) for c in X.columns],
        "MeanAbsSHAP": np.abs(shap_values).mean(axis=0).round(5),
    })
    .sort_values("MeanAbsSHAP", ascending=False)
    .reset_index(drop=True))