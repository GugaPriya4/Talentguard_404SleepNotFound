"""Deterministic agent routing. Auditable rules, no LLM in the core."""
from __future__ import annotations

import pandas as pd

AGENTS = {
    "WORKLOAD": "Workload & Well-being Agent",
    "CAREER": "Career & Compensation Agent",
    "MANAGER": "Manager Support Agent",
    "REVIEW": "Risk Review & Routing Agent",
}


def _rules(row: pd.Series) -> list:
    out = []
    ot = row.get("OverTime") == "Yes"

    # --- Workload & Well-being ---
    if ot and row.get("WorkLifeBalance", 3) <= 2:
        out.append((AGENTS["WORKLOAD"],
                    "Overtime with poor work-life balance",
                    "Workload review and recovery-time planning with line manager"))
    if ot and row.get("JobInvolvement", 3) <= 2:
        out.append((AGENTS["WORKLOAD"],
                    "Overtime with low job involvement",
                    "Role-fit conversation; reassess task allocation"))
    if ot and row.get("JobLevel", 3) <= 2:
        out.append((AGENTS["WORKLOAD"],
                    "Overtime at junior job level",
                    "Apply overtime cap; require manager approval above threshold"))

    # --- Career & Compensation ---
    if row.get("LowIncomeFlag", 0) == 1:
        out.append((AGENTS["CAREER"],
                    "Pay in bottom quartile",
                    "Compensation benchmarking against role and level peers"))
    if row.get("IncomeVsRoleMedian", 1.0) < 0.85:
        out.append((AGENTS["CAREER"],
                    "Paid below role-peer median",
                    "Pay-equity review within job role"))
    if row.get("StockOptionLevel", 1) == 0:
        out.append((AGENTS["CAREER"],
                    "No stock options granted",
                    "Assess eligibility for equity grant (high-ROI retention lever)"))
    if row.get("YearsInCurrentRole", 5) <= 2 and row.get("YearsAtCompany", 5) >= 4:
        out.append((AGENTS["CAREER"],
                    "Long tenure, short time in current role",
                    "Career-path discussion and internal mobility options"))

    # --- Manager Support ---
    if row.get("YearsWithCurrManager", 5) <= 1:
        out.append((AGENTS["MANAGER"],
                    "Recently changed manager",
                    "Schedule structured 1:1 and skip-level check-in"))
    if row.get("EnvironmentSatisfaction", 3) == 1:
        out.append((AGENTS["MANAGER"],
                    "Lowest environment satisfaction",
                    "Confidential stay interview on workplace blockers"))
    if row.get("JobSatisfaction", 3) == 1:
        out.append((AGENTS["MANAGER"],
                    "Lowest job satisfaction",
                    "Stay interview on role expectations and support needs"))
    if row.get("IsNewJoiner", 0) == 1:
        out.append((AGENTS["MANAGER"],
                    "Within first two years",
                    "Onboarding health check at 30/60/90-day cadence"))

    if not out:
        out.append((AGENTS["REVIEW"],
                    "No specific rule triggered",
                    "Manual HR review of model explanation before any action"))
    return out


def build_recommendations(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in scored.iterrows():
        for agent, trigger, action in _rules(row):
            rows.append({
                "employeeId": row.get("EmployeeNumber"),
                "RiskScore": row.get("RiskScore"),
                "RiskTier": row.get("RiskTier"),
                "Agent": agent,
                "Trigger": trigger,
                "RecommendedAction": action,
            })
    return pd.DataFrame(rows)


def summarise_per_employee(recs: pd.DataFrame) -> pd.DataFrame:
    return (recs.groupby("employeeId")
                .agg(Agents=("Agent", lambda s: " | ".join(sorted(set(s)))),
                     Actions=("RecommendedAction", lambda s: " | ".join(s)),
                     TriggerCount=("Trigger", "size"))
                .reset_index())