"""Feature engineering shared by training and the Streamlit prediction app."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Fields entered by the user and stored in the original synthetic dataset.
INPUT_FEATURES = [
    "age",
    "annual_income",
    "total_debt",
    "payment_history",
    "employment_years",
    "credit_utilization",
    "existing_loans",
    "late_payments",
]

BASE_NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "total_debt",
    "employment_years",
    "credit_utilization",
    "existing_loans",
    "late_payments",
]
ENGINEERED_NUMERIC_FEATURES = ["debt_to_annual_income_ratio"]
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
CATEGORICAL_FEATURES = ["payment_history"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with debt divided by annual income added.

    This is called a debt-to-annual-income ratio to distinguish it from the
    conventional debt-to-income ratio, which is commonly based on monthly
    debt payments divided by gross monthly income. Zero/missing income yields
    NaN; the model's numeric imputer handles that safely.
    """
    missing = [column for column in INPUT_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required input feature(s): {', '.join(missing)}")

    result = frame.copy()
    annual_income = pd.to_numeric(result["annual_income"], errors="coerce")
    total_debt = pd.to_numeric(result["total_debt"], errors="coerce")
    denominator = annual_income.where(annual_income > 0)
    result["debt_to_annual_income_ratio"] = total_debt / denominator
    return result
