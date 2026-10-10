# TalentGuard — Employee Attrition Intelligence

> 1,470 employees. HR has capacity for 50 conversations.
> **Which 50 — and what do we say?**

A calibrated ML pipeline that ranks employees by attrition risk, explains each
prediction with SHAP, and routes specific retention actions to four specialised
agents. Deployed as a Next.js dashboard on Vercel.

---

## Headline Results

| Metric | Value |
|---|---|
| PR-AUC (out-of-fold) | **0.6223** |
| ROC-AUC | 0.8362 |
| **Precision@50** | **0.92** — 46 of the top 50 genuinely left |
| Recall@50 | 0.1941 |
| **Lift@50** | **5.71×** over random selection |
| Brier score (calibrated) | 0.0902 |
| Fold stability (15 folds) | ±0.0635 |

### Risk tier validation

| Tier | Employees | Leavers | Actual attrition |
|---|---|---|---|
| Critical | 53 | 49 | **92.45%** |
| High | 98 | 56 | 57.14% |
| Elevated | 123 | 45 | 36.59% |
| Moderate | 208 | 36 | 17.31% |
| Low | 988 | 51 | **5.16%** |

**18× separation, perfectly monotonic.**

---

## Why Not Accuracy?

| Model | Accuracy | Employees helped |
|---|---|---|
| "Nobody ever leaves" | **83.88%** | **0 of 237** |
| TalentGuard @ threshold 0.30 | 85.65% | **150 of 237** |

On 16.12% attrition, accuracy is a trap. We optimise for **PR-AUC and
capacity-tied precision**, with the operating threshold selected from an
HR-workload trade-off table.

---

## Leakage Prevention

| # | Safeguard |
|---|---|
| 1 | `RepeatedStratifiedKFold(5 × 3)` — **split before fit**, always |
| 2 | `FeatureEngineer` fitted **inside each training fold only** |
| 3 | `EmployeeNumber` (ID column) excluded from the feature set |
| 4 | Dashboard scores come from **out-of-fold predictions**, never in-sample |
| 5 | Automated `sanity_check()` aborts export if leakage is detected |

---

## Model

**Calibrated ensemble** — CatBoost (0.6) + XGBoost (0.4), rank-averaged,
isotonic-calibrated.

Selected because neither boosted model individually beat a logistic regression
baseline by the pre-registered +0.03 PR-AUC margin. Only the blend did
(+0.0355).

| Model | PR-AUC | Uplift vs baseline |
|---|---|---|
| **Ensemble** | **0.6223** | **+0.0355** ✓ |
| CatBoost | 0.6101 | +0.0233 ✗ |
| XGBoost | 0.5874 | +0.0006 ✗ |
| Logistic baseline | 0.5868 | — |

---

## Project Structure

```text
.
├── data/                      Dataset (xlsx)
├── ml/                        Offline ML pipeline
│   ├── config.py              Feature lists, tier cuts, exclusions
│   ├── data_prep.py           Leakage-safe feature engineering
│   ├── metrics.py             Precision@K, Recall@K, Lift@K, fairness audit
│   ├── finalise_model.py      Benchmark 4 models, acceptance gate, freeze
│   ├── calibrate_model.py     Isotonic calibration
│   ├── explain.py             SHAP top-3 drivers per employee
│   ├── recommend.py           4-agent deterministic rule engine
│   ├── export_artifacts.py    JSON export for the dashboard
│   └── score_population.py    Orchestrator
└── web/                       Next.js dashboard (deployed to Vercel)
    ├── public/data/           7 generated JSON artefacts
    └── src/
```

---

## Setup

### ML pipeline (Python 3.12)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

cd ml
python finalise_model.py      # benchmark, validate, freeze  (~5 min)
python calibrate_model.py     # isotonic calibration          (~5 sec)
python score_population.py    # score, explain, export JSON   (~30 sec)
```

### Dashboard

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## Deployment

Hosted on Vercel with **Root Directory = `web`**.

Heavy ML dependencies (CatBoost + XGBoost + SHAP + SciPy ≈ 590 MB) exceed
Vercel's 250 MB serverless limit, so the pipeline runs offline and the
dashboard serves precomputed JSON — instant load, zero cold start.

---

## Responsible AI

- `Gender` and `MaritalStatus` excluded from model features (fairness audit only)
- Gender flag-rate parity gap: **1.14 pp** (Female 30.95% / Male 32.09%)
- Gender shows no statistically significant association with attrition (p = 0.291)
- Outputs are **advisory only** — human review required
- **Prohibited uses:** termination, promotion denial, compensation reduction
- Model card published at `outputs/model_card.json`

---

## Limitations

- Single-snapshot public dataset (IBM HR Analytics)
- Associations are correlational, not causal
- WorkLifeBalance level 4 shows a non-monotonic anomaly (n = 153)
- Subgroups with n < 30 flagged as low-confidence
- Calibration Brier is an in-sample estimate; requires revalidation on held-out data

---

## Licence

Hackathon submission. Dataset: IBM HR Analytics Employee Attrition (public sample).