"""Central configuration. All EDA-driven decisions live here."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.xlsx"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Attrition"
TARGET_BINARY = "AttritionBinary"
ID_COL = "EmployeeNumber"

RANDOM_STATE = 42
WATCHLIST_K = 50

# ------------------------------------------------------------------
# EDA-driven exclusions
# ------------------------------------------------------------------
ZERO_VARIANCE = ["EmployeeCount", "Over18", "StandardHours"]

NOISE_COLS = [
    "DailyRate",
    "HourlyRate",
    "MonthlyRate",
    "PerformanceRating",          # only values 3 and 4
    "PercentSalaryHike",          # p = 0.366, not significant
    "RelationshipSatisfaction",   # p = 0.155, not significant
    "NumCompaniesWorked",         # p = 0.242, not significant
]

# Protected attributes: excluded from features, kept for fairness auditing
PROTECTED_COLS = ["Gender", "MaritalStatus"]

DROP_FROM_FEATURES = (
    ZERO_VARIANCE + NOISE_COLS + PROTECTED_COLS + [ID_COL, TARGET, TARGET_BINARY]
)

CATEGORICAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "JobRole",
    "OverTime",
]

LOW_INCOME_THRESHOLD_FALLBACK = 2911
# Risk tiers on the CALIBRATED probability scale.
# Order matters: evaluated top to bottom, first match wins.
# Risk tiers on the CALIBRATED probability scale.
# Order matters: evaluated top to bottom, first match wins.
TIER_CUTS = {
    "Critical": 0.70,
    "High": 0.50,
    "Elevated": 0.25,
    "Moderate": 0.10,
}

# Chosen operating threshold. Best F1 (0.587) while exceeding the
# do-nothing accuracy baseline of 83.88%.
OPERATING_THRESHOLD = 0.30

OPERATING_POINT_STATS = {
    "threshold": 0.30,
    "accuracy": 85.65,
    "balancedAccuracy": 76.62,
    "precision": 54.74,
    "recall": 63.29,
    "f1": 0.587,
    "flagged": 274,
    "truePositives": 150,
    "falsePositives": 124,
    "falseNegatives": 87,
    "baselineAccuracy": 83.88,
}