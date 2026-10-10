"""Compute accuracy and the full confusion matrix for the frozen model."""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

import config as cfg

artefact = joblib.load(cfg.MODEL_DIR / "attrition_model.joblib")
oof = pd.read_csv(cfg.OUTPUT_DIR / "oof_scores.csv")

y = oof["y_true"].to_numpy()
prob = artefact["calibrator"].predict(oof["score_ensemble"].to_numpy())

print(f"Baseline (predict all 'No') accuracy : "
      f"{(1 - y.mean()) * 100:.2f}%\n")

for t in [0.15, 0.20, 0.30, 0.40, 0.50]:
    pred = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f"Threshold {t:.2f} | "
          f"Accuracy {accuracy_score(y, pred)*100:5.2f}% | "
          f"Balanced {balanced_accuracy_score(y, pred)*100:5.2f}% | "
          f"Precision {tp/(tp+fp)*100:5.2f}% | "
          f"Recall {tp/(tp+fn)*100:5.2f}% | "
          f"Flagged {pred.sum():4d} | "
          f"TP {tp:3d} FP {fp:4d} FN {fn:3d} TN {tn:4d}")