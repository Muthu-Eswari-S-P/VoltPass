"""
ml_model.py — Battery Passport ML component

Pipeline:
  1. Load battery sensor/cycle data.
  2. Train Random Forest models to predict SOH (%) and RUL (cycles).
  3. For every new prediction, compute a CONFIDENCE score from the
     agreement between the individual trees in the forest (this is the
     "estimator-agreement" check referenced in the project README).
  4. Derive a rule-based COMPATIBILITY label from the predicted SOH
     (what the battery is still fit for: EV reuse / second-life storage /
     recycle-only).
  5. Only if confidence clears the threshold does the record get packaged
     as an event ready for oracle.js to push on-chain; otherwise it is
     flagged for manual review instead of being written to the ledger.

Run:
    python3 generate_battery_dataset.py   # creates battery_dataset.csv
    python3 ml_model.py
"""

from dataclasses import dataclass, asdict
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

FEATURES = [
    "cycle_number",
    "ambient_temp_c",
    "c_rate",
    "voltage_mean",
    "voltage_std",
    "current_mean",
    "temperature_mean",
    "internal_resistance_mohm",
    "charge_time_min",
    "discharge_capacity_ah",
]
TARGETS = ["soh", "rul"]

# Confidence threshold: minimum estimator agreement required before a
# prediction is trusted enough to be written on-chain. Tune this against
# your real data distribution rather than trusting this placeholder.
CONFIDENCE_THRESHOLD = 0.90


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def load_data(path: str = "battery_dataset.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def train_models(df: pd.DataFrame):
    X = df[FEATURES]
    y_soh = df["soh"]
    y_rul = df["rul"]

    X_train, X_test, y_soh_train, y_soh_test, y_rul_train, y_rul_test = train_test_split(
        X, y_soh, y_rul, test_size=0.2, random_state=42
    )

    soh_model = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=42
    )
    rul_model = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=42
    )

    soh_model.fit(X_train, y_soh_train)
    rul_model.fit(X_train, y_rul_train)

    for name, model, y_test in (("SOH", soh_model, y_soh_test), ("RUL", rul_model, y_rul_test)):
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)
        print(f"[{name}] MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")

    return soh_model, rul_model


# --------------------------------------------------------------------------
# Confidence check (estimator agreement across the forest's trees)
# --------------------------------------------------------------------------

def predict_with_confidence(model: RandomForestRegressor, x_row: pd.DataFrame):
    """
    Returns (mean_prediction, confidence) for a single sample.

    Confidence is derived from the coefficient of variation across the
    individual trees' predictions: low spread relative to the mean means
    the trees agree, which we treat as high confidence. This is deliberately
    simple and interpretable — swap in a calibrated method (e.g. quantile
    regression forests) once you have enough real data to validate it.
    """
    x_values = x_row.values
    tree_preds = np.array([tree.predict(x_values)[0] for tree in model.estimators_])
    mean_pred = tree_preds.mean()
    std_pred = tree_preds.std()

    if abs(mean_pred) < 1e-6:
        coeff_variation = 0.0
    else:
        coeff_variation = std_pred / abs(mean_pred)

    confidence = float(np.clip(1.0 - coeff_variation, 0.0, 1.0))
    return float(mean_pred), confidence


# --------------------------------------------------------------------------
# Compatibility rule (derived from predicted SOH)
# --------------------------------------------------------------------------

def compatibility_from_soh(soh: float) -> str:
    if soh >= 80:
        return "EV_REUSE"
    elif soh >= 60:
        return "SECOND_LIFE_STORAGE"
    else:
        return "RECYCLE_ONLY"


# --------------------------------------------------------------------------
# Oracle-facing decision payload
# --------------------------------------------------------------------------

@dataclass
class BatteryHealthEvent:
    battery_id: str
    cycle_number: int
    predicted_soh: float
    soh_confidence: float
    predicted_rul: int
    rul_confidence: float
    compatibility: str
    overall_confidence: float
    decision: str  # "PUSH_ON_CHAIN" or "FLAG_FOR_REVIEW"


def evaluate_battery(battery_id: str, x_row: pd.DataFrame, soh_model, rul_model) -> BatteryHealthEvent:
    soh_pred, soh_conf = predict_with_confidence(soh_model, x_row)
    rul_pred, rul_conf = predict_with_confidence(rul_model, x_row)

    overall_confidence = min(soh_conf, rul_conf)  # gate on the weaker of the two
    decision = "PUSH_ON_CHAIN" if overall_confidence >= CONFIDENCE_THRESHOLD else "FLAG_FOR_REVIEW"

    return BatteryHealthEvent(
        battery_id=battery_id,
        cycle_number=int(x_row["cycle_number"].iloc[0]),
        predicted_soh=round(soh_pred, 2),
        soh_confidence=round(soh_conf, 3),
        predicted_rul=int(round(rul_pred)),
        rul_confidence=round(rul_conf, 3),
        compatibility=compatibility_from_soh(soh_pred),
        overall_confidence=round(overall_confidence, 3),
        decision=decision,
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data()
    soh_model, rul_model = train_models(df)

    joblib.dump(soh_model, "soh_model.joblib")
    joblib.dump(rul_model, "rul_model.joblib")
    print("\nSaved soh_model.joblib and rul_model.joblib")

    # Demo: run the confidence-gated pipeline on a few sample readings,
    # exactly like oracle.js would when new BMS data comes in.
    print("\n--- Sample oracle decisions ---")
    sample = df.sample(5, random_state=1)
    for _, row in sample.iterrows():
        x_row = pd.DataFrame([row[FEATURES]])
        event = evaluate_battery(f"cell-{int(row.cell_id)}", x_row, soh_model, rul_model)
        print(json.dumps(asdict(event), indent=2))
