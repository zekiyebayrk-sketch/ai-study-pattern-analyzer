"""
train_model.py
--------------
Loads the dataset, trains a regression model to predict productivity_score,
evaluates it, and saves plots for visual inspection.

Run after generate_data.py.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH  = "data/study_data.csv"
MODEL_PATH = "data/model.pkl"
SCALER_PATH = "data/scaler.pkl"

FEATURES = [
    "sleep_hours",
    "study_hours",
    "screen_time_hours",
    "energy_level",
    "stress_level",
    "focus_level",
]
TARGET = "productivity_score"

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}")

X = df[FEATURES]
y = df[TARGET]

# ── Train / test split (80 % / 20 %) ─────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Feature scaling (important for Linear Regression) ────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train two models and compare ──────────────────────────────────────────────

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
}

results = {}
print("\nTraining models...")

for name, model in models.items():
    # Random Forest doesn't need scaling, but it doesn't hurt to use raw features
    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    results[name] = {"mae": mae, "r2": r2, "model": model, "y_pred": y_pred}

    print(f"  {name}:")
    print(f"    MAE : {mae:.3f}  (average error in score units)")
    print(f"    R²  : {r2:.3f}  (1.0 = perfect, 0 = no better than mean)")

# ── Pick the better model (lower MAE wins) ────────────────────────────────────

best_name = min(results, key=lambda k: results[k]["mae"])
best_model  = results[best_name]["model"]
best_y_pred = results[best_name]["y_pred"]

print(f"\nBest model: {best_name}")

# ── Save the best model and scaler ────────────────────────────────────────────

joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"Model saved  → {MODEL_PATH}")
print(f"Scaler saved → {SCALER_PATH}")

# ── Plot 1: Actual vs Predicted ───────────────────────────────────────────────

os.makedirs("data", exist_ok=True)

plt.figure(figsize=(7, 5))
plt.scatter(y_test, best_y_pred, alpha=0.6, color="steelblue", edgecolors="white", linewidths=0.4)
plt.plot([1, 10], [1, 10], "r--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual Productivity Score")
plt.ylabel("Predicted Productivity Score")
plt.title(f"Actual vs Predicted — {best_name}")
plt.legend()
plt.tight_layout()
plt.savefig("data/actual_vs_predicted.png", dpi=120)
plt.close()

# ── Plot 2: Feature importance (Random Forest) or coefficients ────────────────

plt.figure(figsize=(7, 4))

if best_name == "Random Forest":
    importances = best_model.feature_importances_
    indices = np.argsort(importances)
    plt.barh([FEATURES[i] for i in indices], importances[indices], color="steelblue")
    plt.xlabel("Importance Score")
    plt.title("Feature Importances (Random Forest)")
else:
    coefs = best_model.coef_
    indices = np.argsort(np.abs(coefs))
    plt.barh([FEATURES[i] for i in indices], coefs[indices], color="steelblue")
    plt.xlabel("Coefficient Value")
    plt.title("Feature Coefficients (Linear Regression)")

plt.tight_layout()
plt.savefig("data/feature_importance.png", dpi=120)
plt.close()

print("\nPlots saved:")
print("  data/actual_vs_predicted.png")
print("  data/feature_importance.png")

# ── Quick sanity check: predict a sample student ──────────────────────────────

sample = pd.DataFrame([{
    "sleep_hours":       7.5,
    "study_hours":       5.0,
    "screen_time_hours": 3.0,
    "energy_level":      7.0,
    "stress_level":      4.0,
    "focus_level":       7.5,
}])

if best_name == "Linear Regression":
    sample_scaled = scaler.transform(sample)
    prediction = best_model.predict(sample_scaled)[0]
else:
    prediction = best_model.predict(sample)[0]

print(f"\nSample prediction for a well-rested student: {prediction:.2f} / 10")
