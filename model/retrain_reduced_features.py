# ============================================================
# Retrain with Reduced Core Features
# File   : model/retrain_reduced_features.py
# Run    : python model/retrain_reduced_features.py
# ============================================================

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.calibration import calibration_curve

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────
# CORE FEATURE SET
# Only clinically meaningful features
# + their lag/trend versions for time series
# ─────────────────────────────────────────
CORE_FEATURES = [
    # Clinical measurements
    "FastingBloodSugar", "HbA1c", "PostPrandialGlucose",
    "BMI", "Insulin", "SystolicBP", "DiastolicBP",
    "CholesterolHDL", "CholesterolTriglycerides",
    "WaistCircumference", "SerumCreatinine",

    # Demographics + risk factors
    "Age", "Gender", "Smoking", "Hypertension",
    "FamilyHistoryDiabetes", "BMI_Change_30d",

    # Lifestyle
    "PhysicalActivity", "DietQuality", "MedicationAdherence",
    "StressLevel", "AlcoholConsumption",

    # Symptoms
    "FrequentUrination", "ExcessiveThirst",
    "BlurredVision", "TinglingHandsFeet",

    # Engineered scores
    "InsulinResistanceScore", "GlucoseToInsulinRatio",
    "PostPrandialSpike", "FastingBloodSugar_Spike",
    "GlucoseVariability", "HbA1c_Deteriorating",

    # Time series lag features — HbA1c
    "HbA1c_lag1", "HbA1c_lag2", "HbA1c_lag3",
    "HbA1c_RollingAvg3", "HbA1c_Trend",

    # Time series lag features — FastingBloodSugar
    "FastingBloodSugar_lag1", "FastingBloodSugar_lag2", "FastingBloodSugar_lag3",
    "FastingBloodSugar_RollingAvg3", "FastingBloodSugar_Trend",

    # Time series lag features — PostPrandialGlucose
    "PostPrandialGlucose_lag1", "PostPrandialGlucose_lag2", "PostPrandialGlucose_lag3",
    "PostPrandialGlucose_RollingAvg3", "PostPrandialGlucose_Trend",

    # Time series lag features — BMI
    "BMI_Change_30d",

    # Time series lag features — Insulin
    "Insulin_lag1", "Insulin_lag2", "Insulin_lag3",
    "Insulin_RollingAvg3",

    # BP trends
    "SystolicBP_RollingAvg3", "SystolicBP_Trend",
]

# Remove duplicates while preserving order
seen = set()
CORE_FEATURES = [f for f in CORE_FEATURES if not (f in seen or seen.add(f))]
print(f"Total core features: {len(CORE_FEATURES)}")

# ─────────────────────────────────────────
# 1. LOAD SYNTHETIC DATASET
# ─────────────────────────────────────────
print("\nLoading synthetic dataset...")
df_syn  = pd.read_csv(os.path.join(BASE_DIR, "data", "diabetes_timeseries_v3.csv"))
df_syn  = df_syn.drop(columns=["PatientID","visit_date","DoctorInCharge"], errors="ignore")

# Keep only core features
available = [f for f in CORE_FEATURES if f in df_syn.columns]
print(f"  Available in synthetic: {len(available)}/{len(CORE_FEATURES)}")
df_syn = df_syn[available + ["Diagnosis"]]
print(f"  Synthetic rows: {len(df_syn)}")

# ─────────────────────────────────────────
# 2. LOAD REAL KAGGLE DATASET
# ─────────────────────────────────────────
print("\nLoading real Kaggle dataset...")
df_kaggle = pd.read_csv(os.path.join(BASE_DIR, "data", "diabetes_prediction_dataset.csv"))
df_kaggle = df_kaggle[df_kaggle['age'] >= 18].copy()

# Balance
diabetic     = df_kaggle[df_kaggle['diabetes'] == 1]
non_diabetic = df_kaggle[df_kaggle['diabetes'] == 0].sample(n=len(diabetic), random_state=42)
df_kaggle    = pd.concat([diabetic, non_diabetic]).reset_index(drop=True)
print(f"  Kaggle rows (balanced): {len(df_kaggle)}")

# Compute synthetic means for filling missing columns
syn_means = df_syn[available].mean()

real_rows = []
for _, row in df_kaggle.iterrows():
    r = {}
    # Default to synthetic mean
    for f in available:
        r[f] = float(syn_means[f])

    # Override with genuine Kaggle values
    r["Age"]               = float(row["age"])
    r["Gender"]            = 0.0 if row["gender"] == "Female" else 1.0
    r["BMI"]               = float(row["bmi"])
    r["Hypertension"]      = float(row["hypertension"])
    r["HbA1c"]             = float(row["HbA1c_level"])
    r["FastingBloodSugar"] = float(row["blood_glucose_level"])
    r["Smoking"]           = 1.0 if row["smoking_history"] in ["current","ever","former"] else 0.0
    r["PostPrandialGlucose"] = float(row["blood_glucose_level"]) * 1.35

    # Lag = current value (no history)
    for field in ["HbA1c","FastingBloodSugar","PostPrandialGlucose","Insulin"]:
        for lag in ["lag1","lag2","lag3"]:
            key = f"{field}_{lag}"
            if key in available:
                r[key] = r.get(field, float(syn_means.get(field, 0)))
        avg_key = f"{field}_RollingAvg3"
        if avg_key in available:
            r[avg_key] = r.get(field, float(syn_means.get(field, 0)))
        trend_key = f"{field}_Trend"
        if trend_key in available:
            r[trend_key] = 0.0

    # BP trends
    if "SystolicBP_RollingAvg3" in available:
        r["SystolicBP_RollingAvg3"] = r.get("SystolicBP", float(syn_means.get("SystolicBP", 130)))
    if "SystolicBP_Trend" in available:
        r["SystolicBP_Trend"] = 0.0

    # Engineered scores
    insulin = r.get("Insulin", float(syn_means.get("Insulin", 17)))
    fbs     = r["FastingBloodSugar"]
    ppg     = r["PostPrandialGlucose"]
    hba1c   = r["HbA1c"]

    r["InsulinResistanceScore"] = round(fbs * insulin / 405, 4)
    r["GlucoseToInsulinRatio"]  = round(fbs / insulin if insulin > 0 else 0, 4)
    r["PostPrandialSpike"]      = 1.0 if ppg > 140 else 0.0
    r["FastingBloodSugar_Spike"]= 1.0 if fbs > 126 else 0.0
    r["GlucoseVariability"]     = round(abs(ppg - fbs), 4)
    r["HbA1c_Deteriorating"]    = 0.0
    r["BMI_Change_30d"]         = 0.0

    real_rows.append(r)

df_real            = pd.DataFrame(real_rows)[available]
df_real["Diagnosis"] = df_kaggle["diabetes"].values
print(f"  Real rows formatted: {len(df_real)}")

# ─────────────────────────────────────────
# 3. COMBINE — Real data 2x weight
# ─────────────────────────────────────────
print("\nCombining...")
df_real_weighted = pd.concat([df_real] * 2, ignore_index=True)
df_combined      = pd.concat([df_syn, df_real_weighted], ignore_index=True).fillna(0)
print(f"  Total rows: {len(df_combined)}")
print(f"  Class dist — 0: {(df_combined['Diagnosis']==0).sum()}, 1: {(df_combined['Diagnosis']==1).sum()}")

# ─────────────────────────────────────────
# 4. TRAIN
# ─────────────────────────────────────────
X = df_combined[available]
y = df_combined["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

print("\nTraining reduced feature model...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=neg/pos,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)
print("  Done.")

# ─────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────
y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc          = roc_auc_score(y_test, y_pred_proba)
f1           = f1_score(y_test, y_pred)
acc          = accuracy_score(y_test, y_pred)

print(f"\n--- New Reduced Model ---")
print(f"  Features : {len(available)}")
print(f"  AUC-ROC  : {auc:.4f}")
print(f"  F1 Score : {f1:.4f}")
print(f"  Accuracy : {acc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["Non-Diabetic","Diabetic"]))

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"CV Mean AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# Probability distribution
print(f"\nProbability distribution on test set:")
print(f"  0-30%  : {(y_pred_proba < 0.30).sum()} patients")
print(f"  30-60% : {((y_pred_proba >= 0.30) & (y_pred_proba < 0.60)).sum()} patients")
print(f"  60-80% : {((y_pred_proba >= 0.60) & (y_pred_proba < 0.80)).sum()} patients")
print(f"  80-95% : {((y_pred_proba >= 0.80) & (y_pred_proba < 0.95)).sum()} patients")
print(f"  95%+   : {(y_pred_proba >= 0.95).sum()} patients")

# ─────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────
PLOTS_DIR = os.path.join(BASE_DIR, "model", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Diabetic","Diabetic"],
            yticklabels=["Non-Diabetic","Diabetic"])
plt.title("Confusion Matrix — Reduced Feature Model")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_reduced.png"), dpi=150)
plt.close()

# Probability distribution
plt.figure(figsize=(8,4))
plt.hist(y_pred_proba[y_test==0], bins=40, alpha=0.6, color="#1a237e", label="Non-Diabetic")
plt.hist(y_pred_proba[y_test==1], bins=40, alpha=0.6, color="#c62828", label="Diabetic")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Probability Distribution — Reduced Feature Model")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "probability_distribution_reduced.png"), dpi=150)
plt.close()
print("  Saved plots.")

# ─────────────────────────────────────────
# 7. SAVE MODEL + NEW FEATURE NAMES
# ─────────────────────────────────────────
joblib.dump(model,   os.path.join(BASE_DIR, "model", "xgb_model.pkl"))
joblib.dump(available, os.path.join(BASE_DIR, "model", "feature_names.pkl"))
print(f"\n  Model saved    → model/xgb_model.pkl")
print(f"  Features saved → model/feature_names.pkl")
print(f"\nRestart backend: python backend/app.py")