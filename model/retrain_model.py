# ============================================================
# Periodic Retraining Script — Doctor Feedback Loop
# File   : model/retrain_model.py
# Run    : python model/retrain_model.py
#
# This script runs whenever enough doctor confirmations
# have been collected (threshold: 10 confirmed cases).
# It combines the original training data with confirmed
# real cases from the database and retrains the model.
# ============================================================

import os
import sys
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from database.setup_database import get_connection

# ─────────────────────────────────────────
# 1. LOAD CURRENT FEATURE NAMES
# Always load from saved file so this script
# works with whatever feature set is active
# (50 features after reduction, or 110 original)
# ─────────────────────────────────────────
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_names.pkl")
feature_names = joblib.load(FEATURES_PATH)
print(f"Feature set loaded: {len(feature_names)} features")

# ─────────────────────────────────────────
# 2. LOAD ORIGINAL TRAINING DATASET
# ─────────────────────────────────────────
print("\nLoading original training dataset...")
DATA_PATH    = os.path.join(BASE_DIR, "data", "diabetes_timeseries_v3.csv")
df_original  = pd.read_csv(DATA_PATH)
DROP_COLS    = ["PatientID", "visit_date", "DoctorInCharge"]
df_original  = df_original.drop(columns=DROP_COLS, errors="ignore")

# Keep only active feature set
available    = [f for f in feature_names if f in df_original.columns]
df_original  = df_original[available + ["Diagnosis"]].copy()
print(f"  Original rows    : {len(df_original)}")
print(f"  Features matched : {len(available)}/{len(feature_names)}")

# ─────────────────────────────────────────
# 3. LOAD CONFIRMED CASES FROM DATABASE
# ─────────────────────────────────────────
print("\nLoading confirmed cases from database...")
conn   = get_connection()
cursor = conn.cursor()

# Check if confirmed_diagnosis column exists
try:
    cursor.execute("""
        SELECT * FROM visits
        WHERE confirmed_diagnosis IS NOT NULL
    """)
    rows = cursor.fetchall()
except Exception as e:
    print(f"  No confirmed cases table yet: {e}")
    rows = []

conn.close()
print(f"  Confirmed cases found: {len(rows)}")

# ─────────────────────────────────────────
# 4. CHECK MINIMUM THRESHOLD
# ─────────────────────────────────────────
MIN_CASES = 5
if len(rows) < MIN_CASES:
    print(f"\n  Not enough confirmed cases to retrain.")
    print(f"  Need at least {MIN_CASES}, currently have {len(rows)}.")
    print(f"  Keep collecting doctor confirmations and try again.")
    sys.exit(0)

# ─────────────────────────────────────────
# 5. CONVERT DB ROWS TO DATAFRAME
# Map confirmed visit records to feature vectors
# ─────────────────────────────────────────
print("\nConverting confirmed cases to feature vectors...")
syn_means = df_original[available].mean()

real_rows = []
for row in rows:
    row = dict(row)
    r   = {}

    # Default everything to synthetic mean
    for f in available:
        r[f] = float(syn_means.get(f, 0))

    # Override with actual recorded values
    direct_fields = [
        "FastingBloodSugar", "HbA1c", "PostPrandialGlucose",
        "Insulin", "BMI", "SystolicBP", "DiastolicBP",
        "CholesterolTotal", "CholesterolLDL", "CholesterolHDL",
        "CholesterolTriglycerides", "SerumCreatinine", "BUNLevels",
        "WaistCircumference", "CarbohydrateIntake", "CalorieIntake",
        "PhysicalActivity", "SleepQuality", "DietQuality",
        "AlcoholConsumption", "StressLevel", "MedicationAdherence",
        "FatigueLevels", "QualityOfLifeScore", "MedicalCheckupsFrequency",
        "FrequentUrination", "ExcessiveThirst", "UnexplainedWeightLoss",
        "BlurredVision", "SlowHealingSores", "TinglingHandsFeet",
        "AntidiabeticMedications", "AntihypertensiveMedications", "Statins",
        "Hypertension"
    ]
    for field in direct_fields:
        if field in available and row.get(field) is not None:
            r[field] = float(row[field])

    # Recompute engineered features from recorded values
    fbs     = r.get("FastingBloodSugar", float(syn_means.get("FastingBloodSugar", 135)))
    insulin = r.get("Insulin", float(syn_means.get("Insulin", 17)))
    ppg     = r.get("PostPrandialGlucose", float(syn_means.get("PostPrandialGlucose", 209)))
    hba1c   = r.get("HbA1c", float(syn_means.get("HbA1c", 6.98)))

    if "InsulinResistanceScore" in available:
        r["InsulinResistanceScore"] = round(fbs * insulin / 405, 4)
    if "GlucoseToInsulinRatio" in available:
        r["GlucoseToInsulinRatio"]  = round(fbs / insulin if insulin > 0 else 0, 4)
    if "PostPrandialSpike" in available:
        r["PostPrandialSpike"]      = 1.0 if ppg > 140 else 0.0
    if "FastingBloodSugar_Spike" in available:
        r["FastingBloodSugar_Spike"]= 1.0 if fbs > 126 else 0.0
    if "GlucoseVariability" in available:
        r["GlucoseVariability"]     = round(abs(ppg - fbs), 4)
    if "HbA1c_Deteriorating" in available:
        r["HbA1c_Deteriorating"]    = 0.0
    if "BMI_Change_30d" in available:
        r["BMI_Change_30d"]         = float(row.get("BMI_Change_30d", 0) or 0)

    # Lag features = current value (single visit, no history)
    for base in ["FastingBloodSugar","HbA1c","PostPrandialGlucose","Insulin"]:
        for lag in ["lag1","lag2","lag3"]:
            key = f"{base}_{lag}"
            if key in available:
                r[key] = r.get(base, float(syn_means.get(base, 0)))
        avg_key = f"{base}_RollingAvg3"
        if avg_key in available:
            r[avg_key] = r.get(base, float(syn_means.get(base, 0)))
        trend_key = f"{base}_Trend"
        if trend_key in available:
            r[trend_key] = 0.0

    # BP trends
    if "SystolicBP_RollingAvg3" in available:
        r["SystolicBP_RollingAvg3"] = r.get("SystolicBP", float(syn_means.get("SystolicBP", 134)))
    if "SystolicBP_Trend" in available:
        r["SystolicBP_Trend"] = 0.0

    # Use confirmed diagnosis as the label
    r["Diagnosis"] = int(row.get("confirmed_diagnosis", 0))
    real_rows.append(r)

df_confirmed = pd.DataFrame(real_rows)
print(f"  Confirmed cases converted: {len(df_confirmed)}")

# ─────────────────────────────────────────
# 6. COMBINE DATASETS
# Confirmed real cases get 3x weight so they
# have meaningful influence despite small count
# ─────────────────────────────────────────
print("\nCombining datasets...")
df_confirmed_weighted = pd.concat([df_confirmed] * 3, ignore_index=True)
df_combined = pd.concat([df_original, df_confirmed_weighted], ignore_index=True).fillna(0)

print(f"  Original rows         : {len(df_original)}")
print(f"  Confirmed rows (3x)   : {len(df_confirmed_weighted)}")
print(f"  Combined total        : {len(df_combined)}")
print(f"  Class dist — 0: {(df_combined['Diagnosis']==0).sum()}, 1: {(df_combined['Diagnosis']==1).sum()}")

# ─────────────────────────────────────────
# 7. LOAD OLD MODEL FOR COMPARISON
# ─────────────────────────────────────────
old_model  = joblib.load(os.path.join(BASE_DIR, "model", "xgb_model.pkl"))

# ─────────────────────────────────────────
# 8. TRAIN NEW MODEL
# ─────────────────────────────────────────
X = df_combined[available]
y = df_combined["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

print("\nRetraining model...")
new_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=neg/pos,
    eval_metric="logloss",
    random_state=42
)
new_model.fit(X_train, y_train)
print("  Done.")

# ─────────────────────────────────────────
# 9. EVALUATE BOTH MODELS
# ─────────────────────────────────────────
old_pred   = old_model.predict(X_test)
old_proba  = old_model.predict_proba(X_test)[:, 1]
old_auc    = roc_auc_score(y_test, old_proba)
old_f1     = f1_score(y_test, old_pred)

new_pred   = new_model.predict(X_test)
new_proba  = new_model.predict_proba(X_test)[:, 1]
new_auc    = roc_auc_score(y_test, new_proba)
new_f1     = f1_score(y_test, new_pred)
new_acc    = accuracy_score(y_test, new_pred)

print(f"\n--- Model Comparison ---")
print(f"{'Metric':<12} {'Old Model':>12} {'New Model':>12} {'Change':>10}")
print(f"{'-'*50}")
print(f"{'AUC-ROC':<12} {old_auc:>12.4f} {new_auc:>12.4f} {new_auc-old_auc:>+10.4f}")
print(f"{'F1 Score':<12} {old_f1:>12.4f} {new_f1:>12.4f} {new_f1-old_f1:>+10.4f}")
print(f"{'Accuracy':<12} {'—':>12} {new_acc:>12.4f} {'—':>10}")
print()
print(classification_report(y_test, new_pred, target_names=["Non-Diabetic","Diabetic"]))

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(new_model, X, y, cv=cv, scoring="roc_auc")
print(f"CV Mean AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# Probability distribution
print(f"\nProbability distribution:")
print(f"  0-30%  : {(new_proba < 0.30).sum()} patients")
print(f"  30-60% : {((new_proba >= 0.30) & (new_proba < 0.60)).sum()} patients")
print(f"  60-80% : {((new_proba >= 0.60) & (new_proba < 0.80)).sum()} patients")
print(f"  80-95% : {((new_proba >= 0.80) & (new_proba < 0.95)).sum()} patients")
print(f"  95%+   : {(new_proba >= 0.95).sum()} patients")

# ─────────────────────────────────────────
# 10. BACKUP OLD + SAVE NEW
# ─────────────────────────────────────────
MODEL_DIR  = os.path.join(BASE_DIR, "model")
joblib.dump(old_model,  os.path.join(MODEL_DIR, "xgb_model_backup.pkl"))
joblib.dump(new_model,  os.path.join(MODEL_DIR, "xgb_model.pkl"))
print(f"\n  Old model backed up → model/xgb_model_backup.pkl")
print(f"  New model saved    → model/xgb_model.pkl")
print(f"\nDone! Restart backend to load the new model:")
print(f"  python backend/app.py")