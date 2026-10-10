"""Export model outputs as JSON for the Next.js dashboard."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as cfg

WEB = cfg.ROOT_DIR / "web" / "public" / "data"

INSIGHTS = [
    {"id": "overtime", "severity": "critical", "metric": "2.9x",
     "title": "Overtime is the strongest single driver",
     "detail": "Employees working overtime leave at 30.53% versus 10.44%. "
               "Around 35% of all departures are attributable to overtime."},
    {"id": "equity", "severity": "high", "metric": "#1 driver",
     "title": "Stock options rank first in SHAP importance",
     "detail": "Stock option level is the model's strongest single feature, "
               "ahead of every satisfaction metric. Stayers average 0.85, "
               "leavers 0.53."},
    {"id": "pay-floor", "severity": "high", "metric": "45.6%",
     "title": "Attrition is a pay-floor problem",
     "detail": "The bottom income quartile produces 45.6% of all attrition "
               "from 25% of headcount. Above roughly 5,000 per month the "
               "retention return on pay is negligible."},
    {"id": "sentiment", "severity": "medium", "metric": "21.1% vs 17.6%",
     "title": "Behaviour beats sentiment",
     "detail": "Highly satisfied employees who work overtime leave more often "
               "than dissatisfied employees who do not."},
    {"id": "targeting", "severity": "info", "metric": "5.7x",
     "title": "Top 50 watchlist is 5.7x denser in risk",
     "detail": "46 of the top 50 ranked employees genuinely left. "
               "Random selection would find 8."},
    {"id": "fairness", "severity": "info", "metric": "1.14 pp",
     "title": "No gender disparity detected",
     "detail": "Gender flag-rate parity gap is 1.14 percentage points. "
               "Gender is excluded from model features."},
]

REGISTER_COLS = [
    "Age", "Department", "JobRole", "JobLevel", "OverTime", "MonthlyIncome",
    "StockOptionLevel", "YearsAtCompany", "YearsInCurrentRole",
    "YearsWithCurrManager", "JobSatisfaction", "EnvironmentSatisfaction",
    "WorkLifeBalance", "JobInvolvement", "BusinessTravel", "DistanceFromHome",
    "RiskScore", "RiskTier", "top_factors",
    "driver_1_label", "driver_1_shap", "driver_2_label", "driver_2_shap",
    "driver_3_label", "driver_3_shap",
]


def _clean(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def export_all(scored: pd.DataFrame, importance: pd.DataFrame,
               recs: pd.DataFrame, model_card: dict) -> None:
    print("\nExporting dashboard artefacts...")
    WEB.mkdir(parents=True, exist_ok=True)

    T = cfg.TARGET_BINARY
    total, left = len(scored), int(scored[T].sum())
    m = model_card.get("metrics", {})

    def breakdown(col, key):
        return [{key: k, "employees": int(len(g)),
                 "attritionRate": round(g[T].mean() * 100, 2),
                 "avgRiskScore": round(g["RiskScore"].mean(), 2),
                 "highRisk": int(g["RiskTier"].isin(["Critical", "High"]).sum())}
                for k, g in scored.groupby(col)]

    kpis = {
        "generatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
        "modelName": model_card.get("algorithm", "ensemble"),
        "modelVersion": model_card.get("version", "1.1.0"),
        "totalEmployees": total,
        "historicalAttritionCount": left,
        "historicalAttritionRate": round(left / total * 100, 2),
        "overtimeAttritionRate": round(
            scored.loc[scored["OverTime"] == "Yes", T].mean() * 100, 2),
        "nonOvertimeAttritionRate": round(
            scored.loc[scored["OverTime"] == "No", T].mean() * 100, 2),
        "highRiskCount": int(scored["RiskTier"].isin(["Critical", "High"]).sum()),
        "riskTierStats": [
            {"tier": t, "employees": int(len(g)), "leavers": int(g[T].sum()),
             "attritionRate": round(g[T].mean() * 100, 2),
             "avgRiskScore": round(g["RiskScore"].mean(), 2)}
            for t, g in scored.groupby("RiskTier")],
        "modelMetrics": {
            "prAuc": m.get("PR_AUC"), "rocAuc": m.get("ROC_AUC"),
            "precisionAt50": m.get("Precision@50"),
            "recallAt50": m.get("Recall@50"), "liftAt50": m.get("Lift@50"),
            "brier": model_card.get("calibration", {}).get("brierAfter")},
        "operatingPoint": model_card.get("operatingPoint", {}),
        "departmentBreakdown": breakdown("Department", "department"),
        "jobRoleBreakdown": breakdown("JobRole", "jobRole"),
    }

    cols = [cfg.ID_COL] + [c for c in REGISTER_COLS if c in scored.columns]
    register = scored[cols].rename(columns={cfg.ID_COL: "employeeId"})

    payloads = {
        "kpis.json": kpis,
        "risk-register.json": register.to_dict("records"),
        "watchlist.json": register.head(cfg.WATCHLIST_K).to_dict("records"),
        "drivers.json": importance.head(20).to_dict("records"),
        "recommendations.json": recs.to_dict("records"),
        "model-card.json": model_card,
        "insights.json": INSIGHTS,
    }

    for name, payload in payloads.items():
        with open(WEB / name, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_clean)
        print(f"  wrote web/public/data/{name}")

    print("Export complete.\n")