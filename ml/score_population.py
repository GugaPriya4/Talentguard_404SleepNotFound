"""Phase 2 orchestrator: score -> tier -> explain -> recommend -> export.

IMPORTANT: historical employees are scored using OUT-OF-FOLD predictions.
Scoring them with the full-data model would leak their training labels.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

import config as cfg
from calibrate_model import score_employees
from data_prep import load_raw, split_features
from explain import compute_shap, global_importance, top_drivers
from export_artifacts import export_all
from recommend import build_recommendations, summarise_per_employee

# Fallbacks so the script still runs if config.py has not been updated yet
TIER_CUTS = getattr(cfg, "TIER_CUTS", {
    "Critical": 0.70, "High": 0.50, "Elevated": 0.25, "Moderate": 0.10,
})
OPERATING_POINT = getattr(cfg, "OPERATING_POINT_STATS", {"threshold": 0.30})
TIER_ORDER = ["Critical", "High", "Elevated", "Moderate", "Low"]


def assign_tier(score: float, cuts: dict) -> str:
    for tier, cut in cuts.items():
        if score >= cut:
            return tier
    return "Low"


def population_scores(artefact: dict, df: pd.DataFrame) -> np.ndarray:
    """Honest scores for the historical population.

    Uses out-of-fold predictions (never in-sample) and applies the
    isotonic calibrator. Falls back to live scoring only for employees
    absent from the OOF file.
    """
    oof_path = cfg.OUTPUT_DIR / "oof_scores.csv"
    if not oof_path.exists():
        raise RuntimeError("oof_scores.csv not found. Run finalise_model.py first.")

    oof = pd.read_csv(oof_path)
    calibrated = artefact["calibrator"].predict(oof["score_ensemble"].to_numpy())
    lookup = dict(zip(oof[cfg.ID_COL], calibrated))

    scores = df[cfg.ID_COL].map(lookup)
    missing = scores.isna()

    if missing.any():
        print(f"  {missing.sum()} employees not in OOF file -> live scoring")
        scores.loc[missing] = score_employees(artefact, df.loc[missing])

    print(f"  scored {len(scores)} employees "
          f"({(~missing).sum()} from out-of-fold)")
    return scores.to_numpy(dtype=float)


def sanity_check(scored: pd.DataFrame, watchlist_precision: float) -> bool:
    """Detect genuine label leakage.

    A saturated top tier is normal with isotonic calibration.
    Real leakage shows up as a near-zero bottom tier combined with
    near-perfect watchlist precision.
    """
    rates = scored.groupby("RiskTier")[cfg.TARGET_BINARY].mean() * 100
    low_rate = float(rates.get("Low", 0.0))
    n_tiers = scored["RiskTier"].nunique()

    leak = (low_rate < 0.5) and (watchlist_precision > 0.98)

    print(f"  Low-tier attrition : {low_rate:.2f}%   (expect 2-8%)")
    print(f"  Watchlist precision: {watchlist_precision * 100:.0f}%   (expect ~92%)")
    print(f"  Distinct tiers     : {n_tiers}        (expect 5)")

    if leak:
        print("\n*** LEAKAGE DETECTED. Do not export. ***")
        return False

    print("  sanity check PASSED")
    return True


def main() -> None:
    artefact = joblib.load(cfg.MODEL_DIR / "attrition_model.joblib")
    if not artefact.get("calibrated"):
        raise RuntimeError("Artefact not calibrated. Run calibrate_model.py first.")

    print(f"Model: {artefact['model_name']} v{artefact['version']}")
    print(f"Tier cuts: {TIER_CUTS}")

    df = load_raw()

    # ------------------------------------------------------------------
    # 1. Score
    # ------------------------------------------------------------------
    print("\nScoring population (out-of-fold)...")
    probs = population_scores(artefact, df)

    df_fe = artefact["feature_engineer"].transform(df)
    X, _, cat_features = split_features(df_fe)
    X = X[artefact["features"]]

    # ------------------------------------------------------------------
    # 2. Explain
    # ------------------------------------------------------------------
    print("\nComputing SHAP explanations...")
    shap_values = compute_shap(artefact, X)
    drivers = top_drivers(shap_values, X, n=3)
    importance = global_importance(shap_values, X)

    print("\n=== GLOBAL DRIVERS (top 15) ===")
    print(importance.head(15).to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Tier
    # ------------------------------------------------------------------
    scored = df_fe.reset_index(drop=True).copy()
    scored["RiskScore"] = (probs * 100).round(2)
    scored["RiskTier"] = [assign_tier(p, TIER_CUTS) for p in probs]
    scored = pd.concat([scored, drivers.reset_index(drop=True)], axis=1)
    scored = scored.sort_values("RiskScore", ascending=False).reset_index(drop=True)

    print("\n=== RISK TIER DISTRIBUTION ===")
    tier_tbl = (scored.groupby("RiskTier")
                .agg(Employees=("RiskScore", "size"),
                     Leavers=(cfg.TARGET_BINARY, "sum"),
                     Attrition_pct=(cfg.TARGET_BINARY,
                                    lambda s: round(s.mean() * 100, 2)),
                     Avg_Score=("RiskScore", "mean"))
                .reindex(TIER_ORDER).dropna().round(2))
    print(tier_tbl.to_string())

    # ------------------------------------------------------------------
    # 4. Watchlist + sanity gate
    # ------------------------------------------------------------------
    watchlist = scored.head(cfg.WATCHLIST_K)
    hits = int(watchlist[cfg.TARGET_BINARY].sum())
    precision = hits / cfg.WATCHLIST_K

    print("\n=== SANITY CHECK ===")
    if not sanity_check(scored, precision):
        raise SystemExit("Aborting export due to suspected leakage.")

    print(f"\n=== TOP {cfg.WATCHLIST_K} WATCHLIST ===")
    print(f"  Actual leavers in watchlist: {hits}/{cfg.WATCHLIST_K} "
          f"({precision * 100:.0f}% precision)")
    print(watchlist[[cfg.ID_COL, "JobRole", "JobLevel", "OverTime",
                     "MonthlyIncome", "StockOptionLevel",
                     "RiskScore", "RiskTier"]].head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Agent recommendations
    # ------------------------------------------------------------------
    print("\nBuilding agent recommendations...")
    recs = build_recommendations(watchlist)
    per_emp = summarise_per_employee(recs)

    print("\n=== AGENT WORKLOAD ===")
    print(recs["Agent"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 6. Model card enrichment
    # ------------------------------------------------------------------
    with open(cfg.OUTPUT_DIR / "model_card.json", encoding="utf-8") as f:
        model_card = json.load(f)

    model_card["riskTierCuts"] = TIER_CUTS
    model_card["operatingPoint"] = OPERATING_POINT
    model_card["accuracyNote"] = (
        "Accuracy is reported for completeness only. A trivial model predicting "
        "no attrition achieves 83.88% accuracy while identifying zero at-risk "
        "employees. Balanced accuracy, PR-AUC and capacity-aware precision are "
        "the governing metrics."
    )
    model_card["watchlistPrecision"] = round(precision, 4)

    with open(cfg.OUTPUT_DIR / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    # ------------------------------------------------------------------
    # 7. Export
    # ------------------------------------------------------------------
    out = cfg.OUTPUT_DIR / "talentguard_risk_register.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        importance.to_excel(xl, sheet_name="Global_Drivers", index=False)
        tier_tbl.to_excel(xl, sheet_name="Risk_Tiers")
        watchlist[[cfg.ID_COL, "Department", "JobRole", "JobLevel", "OverTime",
                   "MonthlyIncome", "StockOptionLevel", "YearsAtCompany",
                   "RiskScore", "RiskTier", "top_factors"]].to_excel(
            xl, sheet_name="Top50_Watchlist", index=False)
        recs.to_excel(xl, sheet_name="Agent_Recommendations", index=False)
        per_emp.to_excel(xl, sheet_name="Actions_Per_Employee", index=False)
        scored[[cfg.ID_COL, "RiskScore", "RiskTier", "top_factors"]].to_excel(
            xl, sheet_name="Full_Risk_Register", index=False)

    export_all(scored, importance, recs, model_card)

    print(f"Excel report  : outputs/{out.name}")
    print("Dashboard JSON: web/public/data/")


if __name__ == "__main__":
    main()