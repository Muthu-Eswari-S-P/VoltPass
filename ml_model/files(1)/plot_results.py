"""
plot_results.py — Generates the visuals worth showing in a review slide:

  1. soh_degradation_curve.png  - actual vs predicted SOH over cycles for one
     cell, so the audience can SEE the model tracking real degradation.
  2. predicted_vs_actual.png    - scatter plots (SOH and RUL) showing how
     tightly predictions cluster around the perfect-prediction line.
  3. feature_importance.png     - which sensor readings the Random Forest
     actually relies on. Good for defending "why these features" questions.
  4. confidence_gate_summary.png - how many readings get pushed on-chain vs
     flagged for review, and how confidence tracks with prediction error.

Run after ml_model.py so soh_model.joblib / rul_model.joblib exist:
    python3 plot_results.py
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ml_model import (
    FEATURES, predict_with_confidence, compatibility_from_soh,
    CONFIDENCE_THRESHOLD,
)

df = pd.read_csv("battery_dataset.csv")
soh_model = joblib.load("soh_model.joblib")
rul_model = joblib.load("rul_model.joblib")

X = df[FEATURES]
X_train, X_test, idx_train, idx_test = train_test_split(
    X, df.index, test_size=0.2, random_state=42
)
test_df = df.loc[idx_test].copy()
test_df["soh_pred"] = soh_model.predict(X_test)
test_df["rul_pred"] = rul_model.predict(X_test)

# --------------------------------------------------------------------
# 1. Degradation curve for one held-out cell: actual vs predicted SOH
# --------------------------------------------------------------------
sample_cell = df.cell_id.unique()[0]
cell_df = df[df.cell_id == sample_cell].copy()
cell_df["soh_pred"] = soh_model.predict(cell_df[FEATURES])

plt.figure(figsize=(8, 5))
plt.plot(cell_df.cycle_number, cell_df.soh, label="Actual SOH", linewidth=2)
plt.plot(cell_df.cycle_number, cell_df.soh_pred, label="Predicted SOH",
          linestyle="--", linewidth=2)
plt.axhline(80, color="red", linestyle=":", label="80% EOL threshold")
plt.xlabel("Cycle number")
plt.ylabel("State of Health (%)")
plt.title(f"SOH degradation over life — Cell {sample_cell}")
plt.legend()
plt.tight_layout()
plt.savefig("soh_degradation_curve.png", dpi=150)
plt.close()

# --------------------------------------------------------------------
# 2. Predicted vs actual scatter (SOH and RUL) on the held-out test set
# --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

axes[0].scatter(test_df.soh, test_df.soh_pred, s=4, alpha=0.3)
lims = [test_df.soh.min(), test_df.soh.max()]
axes[0].plot(lims, lims, color="red", linestyle="--", label="Perfect prediction")
axes[0].set_xlabel("Actual SOH (%)")
axes[0].set_ylabel("Predicted SOH (%)")
axes[0].set_title("SOH: predicted vs actual")
axes[0].legend()

axes[1].scatter(test_df.rul, test_df.rul_pred, s=4, alpha=0.3, color="orange")
lims = [test_df.rul.min(), test_df.rul.max()]
axes[1].plot(lims, lims, color="red", linestyle="--", label="Perfect prediction")
axes[1].set_xlabel("Actual RUL (cycles)")
axes[1].set_ylabel("Predicted RUL (cycles)")
axes[1].set_title("RUL: predicted vs actual")
axes[1].legend()

plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=150)
plt.close()

# --------------------------------------------------------------------
# 3. Feature importance
# --------------------------------------------------------------------
importances = pd.DataFrame({
    "feature": FEATURES,
    "soh_importance": soh_model.feature_importances_,
    "rul_importance": rul_model.feature_importances_,
}).sort_values("soh_importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
y_pos = np.arange(len(importances))
ax.barh(y_pos - 0.2, importances.soh_importance, height=0.4, label="SOH model")
ax.barh(y_pos + 0.2, importances.rul_importance, height=0.4, label="RUL model")
ax.set_yticks(y_pos)
ax.set_yticklabels(importances.feature)
ax.set_xlabel("Feature importance")
ax.set_title("Which sensor readings drive each prediction")
ax.legend()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

# --------------------------------------------------------------------
# 4. Confidence-gate summary: decision counts + confidence vs error
# --------------------------------------------------------------------
sample = test_df.sample(min(400, len(test_df)), random_state=7)
records = []
for _, row in sample.iterrows():
    x_row = pd.DataFrame([row[FEATURES]])
    soh_pred, soh_conf = predict_with_confidence(soh_model, x_row)
    rul_pred, rul_conf = predict_with_confidence(rul_model, x_row)
    overall_conf = min(soh_conf, rul_conf)
    decision = "PUSH_ON_CHAIN" if overall_conf >= CONFIDENCE_THRESHOLD else "FLAG_FOR_REVIEW"
    soh_error = abs(soh_pred - row.soh)
    records.append((overall_conf, soh_error, decision))

conf_df = pd.DataFrame(records, columns=["confidence", "soh_error", "decision"])

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

counts = conf_df.decision.value_counts()
axes[0].bar(counts.index, counts.values, color=["seagreen", "firebrick"])
axes[0].set_title("Oracle decisions on 400 sampled readings")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 2, str(v), ha="center")

colors = conf_df.decision.map({"PUSH_ON_CHAIN": "seagreen", "FLAG_FOR_REVIEW": "firebrick"})
axes[1].scatter(conf_df.confidence, conf_df.soh_error, c=colors, s=15, alpha=0.6)
axes[1].axvline(CONFIDENCE_THRESHOLD, color="black", linestyle="--",
                 label=f"Threshold = {CONFIDENCE_THRESHOLD}")
axes[1].set_xlabel("Overall confidence")
axes[1].set_ylabel("Absolute SOH prediction error")
axes[1].set_title("Low confidence lines up with high error")
axes[1].legend()

plt.tight_layout()
plt.savefig("confidence_gate_summary.png", dpi=150)
plt.close()

print("Saved: soh_degradation_curve.png, predicted_vs_actual.png, "
      "feature_importance.png, confidence_gate_summary.png")
