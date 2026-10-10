"""Loading and leakage-safe feature engineering."""
from __future__ import annotations

import pandas as pd

import config as cfg


def load_raw() -> pd.DataFrame:
    xls = pd.ExcelFile(cfg.DATA_PATH)
    df = pd.read_excel(cfg.DATA_PATH, sheet_name=xls.sheet_names[0])
    df[cfg.TARGET_BINARY] = (df[cfg.TARGET] == "Yes").astype(int)
    df = df.drop(columns=cfg.ZERO_VARIANCE, errors="ignore")
    return df


class FeatureEngineer:
    """Fit statistics on TRAIN only, then transform any split."""

    def __init__(self) -> None:
        self.role_median_income_ = {}
        self.level_median_income_ = {}
        self.global_median_income_ = 0.0
        self.low_income_threshold_ = cfg.LOW_INCOME_THRESHOLD_FALLBACK
        self.fitted_ = False

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        self.role_median_income_ = (
            df.groupby("JobRole")["MonthlyIncome"].median().to_dict()
        )
        self.level_median_income_ = (
            df.groupby("JobLevel")["MonthlyIncome"].median().to_dict()
        )
        self.global_median_income_ = float(df["MonthlyIncome"].median())
        self.low_income_threshold_ = float(df["MonthlyIncome"].quantile(0.25))
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")

        out = df.copy()
        ot = (out["OverTime"] == "Yes").astype(int)

        # Threshold flags: EDA showed signal lives at level 1 only
        out["EnvSat_Critical"] = (out["EnvironmentSatisfaction"] == 1).astype(int)
        out["JobSat_Critical"] = (out["JobSatisfaction"] == 1).astype(int)
        out["WLB_Critical"] = (out["WorkLifeBalance"] == 1).astype(int)
        out["LowInvolvement"] = (out["JobInvolvement"] <= 2).astype(int)

        # Pay floor: effect concentrated in bottom quartile
        out["LowIncomeFlag"] = (
            out["MonthlyIncome"] <= self.low_income_threshold_
        ).astype(int)

        role_med = (out["JobRole"].map(self.role_median_income_)
                    .fillna(self.global_median_income_))
        level_med = (out["JobLevel"].map(self.level_median_income_)
                     .fillna(self.global_median_income_))
        out["IncomeVsRoleMedian"] = out["MonthlyIncome"] / role_med
        out["IncomeVsLevelMedian"] = out["MonthlyIncome"] / level_med

        # The lethal combination identified in EDA
        out["UncompensatedOvertime"] = (ot & out["LowIncomeFlag"]).astype(int)

        # Interactions
        out["OT_x_JobLevel"] = ot * out["JobLevel"]
        out["OT_x_LowInvolvement"] = ot * out["LowInvolvement"]
        out["OT_x_WLBCritical"] = ot * out["WLB_Critical"]

        # Tenure structure
        out["TenureRatio"] = out["YearsAtCompany"] / (out["TotalWorkingYears"] + 1)
        out["RoleStability"] = out["YearsInCurrentRole"] / (out["YearsAtCompany"] + 1)
        out["ManagerStability"] = out["YearsWithCurrManager"] / (out["YearsAtCompany"] + 1)
        out["PromotionStagnationRate"] = (
            out["YearsSinceLastPromotion"] / (out["YearsAtCompany"] + 1)
        )
        out["IsNewJoiner"] = (out["YearsAtCompany"] <= 2).astype(int)

        # Commute burden
        out["LongCommute"] = (out["DistanceFromHome"] >= 15).astype(int)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in cfg.DROP_FROM_FEATURES]


def split_features(df: pd.DataFrame):
    features = get_feature_columns(df)
    cat_features = [c for c in cfg.CATEGORICAL_FEATURES if c in features]
    X = df[features].copy()
    for col in cat_features:
        X[col] = X[col].astype(str)
    y = df[cfg.TARGET_BINARY].copy()
    return X, y, cat_features

