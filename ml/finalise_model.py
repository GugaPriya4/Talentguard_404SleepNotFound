"""Model finalisation: benchmark, validate, calibrate, freeze."""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import config as cfg
from data_prep import FeatureEngineer, load_raw, split_features
from metrics import (
    evaluate,
    fairness_audit,
    find_best_threshold,
    lift_at_k,
    precision_at_k,
    recall_at_k,
)

warnings.filterwarnings("ignore")

N_SPLITS = 5
N_REPEATS = 3
K = cfg.WATCHLIST_K

ACCEPTANCE = {
    "PR_AUC": 0.55,
    f"Recall@{K}": 0.12,
    f"Precision@{K}": 0.50,
    f"Lift@{K}": 3.0,
    "max_prauc_std": 0.08,
    "min_uplift_over_baseline": 0.03,
    "max_gender_parity_pp": 10.0,
}


def make_catboost(spw):
    return CatBoostClassifier(
        iterations=1000, learning_rate=0.03, depth=5, l2_leaf_reg=6.0,
        loss_function="Logloss", eval_metric="PRAUC", scale_pos_weight=spw,
        random_seed=cfg.RANDOM_STATE, early_stopping_rounds=120,
        verbose=False, allow_writing_files=False,
    )


def make_xgboost(spw):
    return XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=4,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=2.0, scale_pos_weight=spw,
        objective="binary:logistic", eval_metric="aucpr",
        tree_method="hist", enable_categorical=True,
        early_stopping_rounds=120, random_state=cfg.RANDOM_STATE, n_jobs=-1,
    )


def make_baseline(cat_features, num_features):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_features),
        ("num", StandardScaler(), num_features),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   C=0.5, random_state=cfg.RANDOM_STATE)),
    ])


def to_xgb_frame(X, cat_features):
    out = X.copy()
    for c in cat_features:
        out[c] = out[c].astype("category")
    return out


def blend(cat_scores, xgb_scores, w=0.6):
    n = len(cat_scores)
    return w * (rankdata(cat_scores) / n) + (1 - w) * (rankdata(xgb_scores) / n)


def repeated_cv():
    df = load_raw()
    y = df[cfg.TARGET_BINARY].to_numpy()
    n = len(df)

    models = ["baseline_logreg", "catboost", "xgboost", "ensemble"]
    oof_repeats = {m: np.zeros((N_REPEATS, n)) for m in models}
    fold_metrics = []

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=cfg.RANDOM_STATE)

    print(f"Running {N_SPLITS}-fold CV x {N_REPEATS} repeats "
          f"({N_SPLITS * N_REPEATS} fits per model)\n")

    for i, (tr_idx, va_idx) in enumerate(rskf.split(df, y)):
        rep, fold = i // N_SPLITS, i % N_SPLITS + 1
        train_df, valid_df = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()

        fe = FeatureEngineer().fit(train_df)
        X_tr, y_tr, cat_features = split_features(fe.transform(train_df))
        X_va, y_va, _ = split_features(fe.transform(valid_df))

        num_features = [c for c in X_tr.columns if c not in cat_features]
        spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

        preds = {}

        base = make_baseline(cat_features, num_features)
        base.fit(X_tr, y_tr)
        preds["baseline_logreg"] = base.predict_proba(X_va)[:, 1]

        cb = make_catboost(spw)
        cb.fit(Pool(X_tr, y_tr, cat_features=cat_features),
               eval_set=Pool(X_va, y_va, cat_features=cat_features))
        preds["catboost"] = cb.predict_proba(
            Pool(X_va, cat_features=cat_features))[:, 1]

        xgb = make_xgboost(spw)
        xgb.fit(to_xgb_frame(X_tr, cat_features), y_tr,
                eval_set=[(to_xgb_frame(X_va, cat_features), y_va)], verbose=False)
        preds["xgboost"] = xgb.predict_proba(
            to_xgb_frame(X_va, cat_features))[:, 1]

        preds["ensemble"] = blend(preds["catboost"], preds["xgboost"], 0.6)

        k_fold = max(int(round(K * len(va_idx) / n)), 5)
        for m in models:
            oof_repeats[m][rep, va_idx] = preds[m]
            fold_metrics.append({
                "model": m, "repeat": rep + 1, "fold": fold,
                "PR_AUC": round(average_precision_score(y_va, preds[m]), 4),
                "Precision_at_k": round(precision_at_k(y_va, preds[m], k_fold), 4),
                "Recall_at_k": round(recall_at_k(y_va, preds[m], k_fold), 4),
                "Lift_at_k": round(lift_at_k(y_va, preds[m], k_fold), 2),
            })

        print(f"  repeat {rep + 1} fold {fold} done")

    oof = {m: oof_repeats[m].mean(axis=0) for m in models}
    return {"df": df, "y": y, "oof": oof,
            "fold_metrics": pd.DataFrame(fold_metrics)}


def build_leaderboard(y, oof):
    board = pd.DataFrame([evaluate(y, oof[m], k=K, label=m) for m in oof])
    board["Brier"] = [round(brier_score_loss(y, np.clip(oof[m], 0, 1)), 4)
                      for m in oof]
    return board.sort_values([f"Recall@{K}", "PR_AUC"], ascending=False)


def stability_report(fold_metrics):
    return (fold_metrics.groupby("model")["PR_AUC"]
            .agg(Mean="mean", Std="std", Min="min", Max="max")
            .round(4).sort_values("Mean", ascending=False))


def decile_table(y, scores):
    order = np.argsort(-scores)
    y_sorted, s_sorted = y[order], scores[order]
    n = len(y)
    rows = []
    for d in range(10):
        lo, hi = int(n * d / 10), int(n * (d + 1) / 10)
        seg = y_sorted[lo:hi]
        rows.append({
            "Decile": d + 1,
            "Employees": len(seg),
            "Leavers": int(seg.sum()),
            "Attrition_%": round(seg.mean() * 100, 2),
            "Lift": round(seg.mean() / y.mean(), 2),
            "Score_Min": round(float(s_sorted[hi - 1]), 4),
            "Score_Max": round(float(s_sorted[lo]), 4),
            "Cum_Recall_%": round(y_sorted[:hi].sum() / y.sum() * 100, 2),
        })
    return pd.DataFrame(rows)


def calibrate_tiers(y, scores):
    cuts = {
        "Critical": float(np.quantile(scores, 0.97)),
        "High": float(np.quantile(scores, 0.90)),
        "Elevated": float(np.quantile(scores, 0.75)),
        "Moderate": float(np.quantile(scores, 0.50)),
    }

    def tier_of(s):
        for name, cut in cuts.items():
            if s >= cut:
                return name
        return "Low"

    summary = (pd.DataFrame({"Tier": [tier_of(s) for s in scores], "y": y})
               .groupby("Tier")
               .agg(Employees=("y", "size"), Leavers=("y", "sum"),
                    Attrition_pct=("y", lambda s: round(s.mean() * 100, 2)))
               .reindex(["Critical", "High", "Elevated", "Moderate", "Low"])
               .dropna())
    return {"cuts": {k: round(v, 4) for k, v in cuts.items()},
            "summary": summary}


def check_acceptance(board, stability, winner, audit):
    w = board[board["model"] == winner].iloc[0]
    base = board[board["model"] == "baseline_logreg"].iloc[0]
    prauc_std = float(stability.loc[winner, "Std"])

    gender = audit[audit["Attribute"] == "Gender"] if len(audit) else audit
    parity = (float(gender["Flag_Rate_%"].max() - gender["Flag_Rate_%"].min())
              if len(gender) else 0.0)

    checks = [
        ("PR-AUC", float(w["PR_AUC"]), ACCEPTANCE["PR_AUC"], ">="),
        (f"Recall@{K}", float(w[f"Recall@{K}"]), ACCEPTANCE[f"Recall@{K}"], ">="),
        (f"Precision@{K}", float(w[f"Precision@{K}"]),
         ACCEPTANCE[f"Precision@{K}"], ">="),
        (f"Lift@{K}", float(w[f"Lift@{K}"]), ACCEPTANCE[f"Lift@{K}"], ">="),
        ("PR-AUC fold std", prauc_std, ACCEPTANCE["max_prauc_std"], "<="),
        ("Uplift vs baseline", round(float(w["PR_AUC"] - base["PR_AUC"]), 4),
         ACCEPTANCE["min_uplift_over_baseline"], ">="),
        ("Gender parity gap (pp)", round(parity, 2),
         ACCEPTANCE["max_gender_parity_pp"], "<="),
    ]

    rows = []
    for name, actual, target, op in checks:
        passed = actual >= target if op == ">=" else actual <= target
        rows.append({"Criterion": name, "Actual": actual,
                     "Target": f"{op} {target}",
                     "Status": "PASS" if passed else "FAIL"})
    return pd.DataFrame(rows)


def freeze(winner, threshold, tier_cuts):
    df = load_raw()
    fe = FeatureEngineer().fit(df)
    X, y, cat_features = split_features(fe.transform(df))
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))

    if winner == "catboost":
        model = make_catboost(spw)
        model.set_params(early_stopping_rounds=None, iterations=800)
        model.fit(Pool(X, y, cat_features=cat_features))
    elif winner == "xgboost":
        model = make_xgboost(spw)
        model.set_params(early_stopping_rounds=None, n_estimators=800)
        model.fit(to_xgb_frame(X, cat_features), y, verbose=False)
    else:
        cb = make_catboost(spw)
        cb.set_params(early_stopping_rounds=None, iterations=800)
        cb.fit(Pool(X, y, cat_features=cat_features))
        xg = make_xgboost(spw)
        xg.set_params(early_stopping_rounds=None, n_estimators=800)
        xg.fit(to_xgb_frame(X, cat_features), y, verbose=False)
        model = {"catboost": cb, "xgboost": xg, "weight": 0.6}

    artefact = {
        "model": model, "model_name": winner,
        "feature_engineer": fe, "features": list(X.columns),
        "cat_features": cat_features, "threshold": threshold,
        "tier_cuts": tier_cuts,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }
    joblib.dump(artefact, cfg.MODEL_DIR / "attrition_model.joblib")
    return artefact


def write_model_card(artefact, board, stability, acceptance,
                     tiers, deciles, audit):
    w = board[board["model"] == artefact["model_name"]].iloc[0].to_dict()

    card = {
        "modelName": "TalentGuard Attrition Risk Model",
        "version": artefact["version"],
        "frozenAt": artefact["frozen_at"],
        "algorithm": artefact["model_name"],
        "validation": f"{N_SPLITS}-fold stratified CV x {N_REPEATS} repeats, "
                      "feature engineering fitted per fold",
        "featureCount": len(artefact["features"]),
        "decisionThreshold": artefact["threshold"],
        "riskTierCuts": artefact["tier_cuts"],
        "metrics": {k: v for k, v in w.items() if k != "model"},
        "acceptance": acceptance.to_dict(orient="records"),
        "excludedFeatures": {
            "notSignificant": cfg.NOISE_COLS,
            "zeroVariance": cfg.ZERO_VARIANCE,
            "protectedAttributes": cfg.PROTECTED_COLS,
        },
        "intendedUse": "Decision support for HR retention planning. "
                       "Outputs are advisory and require human review.",
        "prohibitedUse": "Must not be used for termination, promotion denial, "
                         "compensation reduction, or any adverse action.",
        "limitations": [
            "Trained on a single-snapshot public dataset (IBM HR Analytics).",
            "Associations are correlational, not causal.",
            "WorkLifeBalance level 4 shows a non-monotonic anomaly (n=153).",
            "Subgroup rates with fewer than 30 employees are low-confidence.",
            "Requires revalidation on live HRIS data before production use.",
        ],
    }

    with open(cfg.OUTPUT_DIR / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    with pd.ExcelWriter(cfg.OUTPUT_DIR / "model_finalisation.xlsx",
                        engine="openpyxl") as xl:
        board.to_excel(xl, sheet_name="Leaderboard", index=False)
        stability.to_excel(xl, sheet_name="Fold_Stability")
        acceptance.to_excel(xl, sheet_name="Acceptance_Gate", index=False)
        deciles.to_excel(xl, sheet_name="Decile_Lift", index=False)
        tiers["summary"].to_excel(xl, sheet_name="Risk_Tiers")
        if len(audit):
            audit.to_excel(xl, sheet_name="Fairness_Audit", index=False)


def main():
    cv = repeated_cv()
    y, oof, df = cv["y"], cv["oof"], cv["df"]

    board = build_leaderboard(y, oof)
    print("\n" + "=" * 70)
    print("LEADERBOARD (out-of-fold, averaged over repeats)")
    print("=" * 70)
    print(board[["model", "PR_AUC", "ROC_AUC", f"Precision@{K}",
                 f"Recall@{K}", f"Lift@{K}", "F2", "Brier",
                 "Threshold"]].to_string(index=False))

    stability = stability_report(cv["fold_metrics"])
    print("\n" + "=" * 70)
    print("FOLD STABILITY (PR-AUC across 15 folds)")
    print("=" * 70)
    print(stability.to_string())

    winner = board.iloc[0]["model"]
    if winner == "baseline_logreg":
        print("\nWARNING: baseline beat both GBMs. Investigate overfitting.")
    print(f"\nSELECTED MODEL: {winner}")

    scores = oof[winner]
    threshold, _ = find_best_threshold(y, scores, beta=2.0)

    deciles = decile_table(y, scores)
    print("\n" + "=" * 70)
    print("DECILE LIFT TABLE")
    print("=" * 70)
    print(deciles.to_string(index=False))

    tiers = calibrate_tiers(y, scores)
    print("\n" + "=" * 70)
    print("CALIBRATED RISK TIERS")
    print("=" * 70)
    print("Cut-offs:", tiers["cuts"])
    print(tiers["summary"].to_string())

    audit = fairness_audit(df, scores, threshold, cfg.PROTECTED_COLS)
    print("\n" + "=" * 70)
    print("FAIRNESS AUDIT")
    print("=" * 70)
    print(audit.to_string(index=False))

    acceptance = check_acceptance(board, stability, winner, audit)
    print("\n" + "=" * 70)
    print("ACCEPTANCE GATE")
    print("=" * 70)
    print(acceptance.to_string(index=False))

    if (acceptance["Status"] == "FAIL").any():
        print("\n>>> MODEL NOT READY. Failed criteria above. Tune before freezing.")
    else:
        print("\n>>> ALL CRITERIA PASSED. Freezing model...")

    artefact = freeze(winner, threshold, tiers["cuts"])
    write_model_card(artefact, board, stability, acceptance,
                     tiers, deciles, audit)

    pd.DataFrame({
        cfg.ID_COL: df[cfg.ID_COL], "y_true": y,
        **{f"score_{m}": oof[m] for m in oof},
    }).to_csv(cfg.OUTPUT_DIR / "oof_scores.csv", index=False)

    print("\nFrozen artefact : models/attrition_model.joblib")
    print("Model card      : outputs/model_card.json")
    print("Full report     : outputs/model_finalisation.xlsx")


if __name__ == "__main__":
    main()