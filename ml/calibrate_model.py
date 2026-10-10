"""Calibrate the ensemble so scores become true probabilities.

Run AFTER finalise_model.py. Does not retrain anything.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from catboost import Pool
from scipy.stats import percentileofscore
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

import config as cfg
from data_prep import load_raw, split_features

ARTEFACT_PATH = cfg.MODEL_DIR / "attrition_model.joblib"


def main() -> None:
    artefact = joblib.load(ARTEFACT_PATH)
    oof = pd.read_csv(cfg.OUTPUT_DIR / "oof_scores.csv")

    y = oof["y_true"].to_numpy()
    blend_oof = oof["score_ensemble"].to_numpy()
    cat_oof = oof["score_catboost"].to_numpy()
    xgb_oof = oof["score_xgboost"].to_numpy()

    print("Before calibration:")
    print(f"  Brier (ensemble ranks) = "
          f"{brier_score_loss(y, np.clip(blend_oof, 0, 1)):.4f}")

    # --- Fit isotonic on honest out-of-fold scores ---
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(blend_oof, y)
    calibrated = iso.predict(blend_oof)

    print("After calibration:")
    print(f"  Brier (calibrated)     = "
          f"{brier_score_loss(y, calibrated):.4f}")

    # --- Reliability check ---
    bins = pd.cut(calibrated, bins=[0, .1, .2, .4, .6, .8, 1.0],
                  include_lowest=True)
    reliability = (pd.DataFrame({"bin": bins, "pred": calibrated, "actual": y})
                   .groupby("bin", observed=True)
                   .agg(N=("actual", "size"),
                        Mean_Predicted=("pred", "mean"),
                        Actual_Rate=("actual", "mean"))
                   .round(4))
    print("\nReliability (predicted vs actual):")
    print(reliability.to_string())

    # --- Recompute tier cut-offs on the calibrated scale ---
    tier_cuts = {
        "Critical": float(np.quantile(calibrated, 0.97)),
        "High": float(np.quantile(calibrated, 0.90)),
        "Elevated": float(np.quantile(calibrated, 0.75)),
        "Moderate": float(np.quantile(calibrated, 0.50)),
    }

    def tier_of(s):
        for name, cut in tier_cuts.items():
            if s >= cut:
                return name
        return "Low"

    tier_summary = (pd.DataFrame({"Tier": [tier_of(s) for s in calibrated],
                                  "y": y})
                    .groupby("Tier")
                    .agg(Employees=("y", "size"), Leavers=("y", "sum"),
                         Attrition_pct=("y", lambda s: round(s.mean() * 100, 2)))
                    .reindex(["Critical", "High", "Elevated", "Moderate", "Low"])
                    .dropna())
    print("\nRecalibrated risk tiers:")
    print(tier_summary.to_string())

    # --- Store everything needed for single-record scoring ---
    artefact["calibrator"] = iso
    artefact["reference_catboost"] = np.sort(cat_oof)
    artefact["reference_xgboost"] = np.sort(xgb_oof)
    artefact["tier_cuts"] = {k: round(v, 4) for k, v in tier_cuts.items()}
    artefact["calibrated"] = True
    artefact["version"] = "1.1.0"

    joblib.dump(artefact, ARTEFACT_PATH)

    card_path = cfg.OUTPUT_DIR / "model_card.json"
    with open(card_path, encoding="utf-8") as f:
        card = json.load(f)
    card["version"] = "1.1.0"
    card["calibration"] = {
        "method": "Isotonic regression on out-of-fold ensemble ranks",
        "brierBefore": round(float(brier_score_loss(y, np.clip(blend_oof, 0, 1))), 4),
        "brierAfter": round(float(brier_score_loss(y, calibrated)), 4),
    }
    card["riskTierCuts"] = artefact["tier_cuts"]
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    print(f"\nCalibrated artefact saved: {ARTEFACT_PATH.name} (v1.1.0)")


def score_employees(artefact: dict, df_raw: pd.DataFrame) -> np.ndarray:
    """Reusable scorer. Works for 1 employee or the whole population."""
    fe = artefact["feature_engineer"]
    X, _, cat_features = split_features(fe.transform(df_raw))
    X = X[artefact["features"]]

    models = artefact["model"]
    p_cat = models["catboost"].predict_proba(
        Pool(X, cat_features=cat_features))[:, 1]

    xf = X.copy()
    for c in cat_features:
        xf[c] = xf[c].astype("category")
    p_xgb = models["xgboost"].predict_proba(xf)[:, 1]

    # Percentile against the stored OOF reference distributions
    r_cat = np.array([percentileofscore(artefact["reference_catboost"], p) / 100
                      for p in p_cat])
    r_xgb = np.array([percentileofscore(artefact["reference_xgboost"], p) / 100
                      for p in p_xgb])

    w = models["weight"]
    blend = w * r_cat + (1 - w) * r_xgb
    return artefact["calibrator"].predict(blend)


if __name__ == "__main__":
    main()