"""Capacity-aware evaluation metrics for retention triage."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    idx = np.argsort(-y_score)[:k]
    return float(np.asarray(y_true)[idx].mean())


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true)
    idx = np.argsort(-y_score)[:k]
    total = y_true.sum()
    return float(y_true[idx].sum() / total) if total else 0.0


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    base = np.asarray(y_true).mean()
    return precision_at_k(y_true, y_score, k) / base if base else 0.0


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray,
                        beta: float = 2.0):
    """Cut-off maximising F-beta. beta=2 favours recall."""
    best_t, best_s = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        s = fbeta_score(y_true, (y_score >= t).astype(int),
                        beta=beta, zero_division=0)
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, best_s


def evaluate(y_true: np.ndarray, y_score: np.ndarray, k: int = 50,
             threshold: float | None = None, label: str = "model") -> dict:
    if threshold is None:
        threshold, _ = find_best_threshold(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "model": label,
        "PR_AUC": round(average_precision_score(y_true, y_score), 4),
        "ROC_AUC": round(roc_auc_score(y_true, y_score), 4),
        f"Precision@{k}": round(precision_at_k(y_true, y_score, k), 4),
        f"Recall@{k}": round(recall_at_k(y_true, y_score, k), 4),
        f"Lift@{k}": round(lift_at_k(y_true, y_score, k), 2),
        "Threshold": round(threshold, 3),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "F2": round(fbeta_score(y_true, y_pred, beta=2, zero_division=0), 4),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def fairness_audit(df: pd.DataFrame, y_score: np.ndarray,
                   threshold: float, attrs: list) -> pd.DataFrame:
    target = "AttritionBinary" if "AttritionBinary" in df.columns else "Attrition"
    work = df.assign(_flag=(y_score >= threshold).astype(int))

    rows = []
    for attr in attrs:
        if attr not in work.columns:
            continue
        for value, grp in work.groupby(attr):
            rows.append({
                "Attribute": attr,
                "Group": str(value),
                "N": len(grp),
                "Flag_Rate_%": round(grp["_flag"].mean() * 100, 2),
                "Actual_Attrition_%": round(grp[target].mean() * 100, 2),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Parity_Diff_pp"] = out.groupby("Attribute")["Flag_Rate_%"].transform(
        lambda s: (s - s.mean()).round(2)
    )
    return out