# ============================================================
# Retrain with Real Data - Fixed Version
# File   : model/retrain_with_real_data.py
# Run    : python model/retrain_with_real_data.py
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────
# 1. LOAD ORIGINAL SYNTHETIC DATASET
# ─────────────────────────────────────────
print("Loading original synthetic dataset...")
synthetic_path = os.path.join(BASE_DIR, "data", "diabetes_timeseries_v3.csv")
df_synthetic   = pd.read_csv(synthetic_path)
DROP_COLS      = ["PatientID", "visit_date", "DoctorInCharge"]
df_synthetic   = df_synthetic.drop(columns=DROP_COLS, errors="ignore")
print(f"  Synthetic rows : {len(df_synthetic)}")

# ─────────────────────────────────────────
# 2. LOAD + REFORMAT REAL DATA
# Only use columns genuinely from Kaggle
# Everything else = synthetic dataset mean
# This avoids data leakage from estimated
# lifestyle features
# ─────────────────────────────────────────
print("Loading real world dataset...")
real_kaggle_path = os.path.join(BASE_DIR, "data", "diabetes_prediction_dataset.csv")
df_kaggle        = pd.read_csv(real_kaggle_path)
df_kaggle        = df_kaggle[df_kaggle['age'] >= 18].copy()

# Balance classes
diabetic     = df_kaggle[df_kaggle['diabetes'] == 1]
non_diabetic = df_kaggle[df_kaggle['diabetes'] == 0].sample(n=len(diabetic), random_state=42)
df_kaggle    = pd.concat([diabetic, non_diabetic]).reset_index(drop=True)
print(f"  Real rows (balanced): {len(df_kaggle)}")

# Load feature names
feature_names = joblib.load(os.path.join(BASE_DIR, "model", "feature_names.pkl"))

# Compute synthetic means for filling missing features
syn_means = df_synthetic[feature_names].mean()

real_rows = []
for _, row in df_kaggle.iterrows():
    r = {}

    # Default everything to synthetic mean
    for f in feature_names:
        r[f] = float(syn_means[f])

    # Override with genuine Kaggle columns only
    r["Age"]               = float(row["age"])
    r["Gender"]            = 0.0 if row["gender"] == "Female" else 1.0
    r["BMI"]               = float(row["bmi"])
    r["Hypertension"]      = float(row["hypertension"])
    r["HbA1c"]             = float(row["HbA1c_level"])
    r["FastingBloodSugar"] = float(row["blood_glucose_level"])
    r["Smoking"]           = 1.0 if row["smoking_history"] in ["current","ever","former"] else 0.0

    # PostPrandial — clinical approximation only, not derived from label
    r["PostPrandialGlucose"] = float(row["blood_glucose_level"]) * 1.35

    # Lag features = same as current (no history for real data)
    for field in ["FastingBloodSugar","HbA1c","PostPrandialGlucose"]:
        r[f"{field}_lag1"]        = r[field]
        r[f"{field}_lag2"]        = r[field]
        r[f"{field}_lag3"]        = r[field]
        r[f"{field}_RollingAvg3"] = r[field]
        r[f"{field}_Trend"]       = 0.0

    # Engineered scores from genuine values
    insulin = float(syn_means["Insulin"])
    r["InsulinResistanceScore"] = round(r["FastingBloodSugar"] * insulin / 405, 4)
    r["GlucoseToInsulinRatio"]  = round(r["FastingBloodSugar"] / insulin if insulin > 0 else 0, 4)
    r["PostPrandialSpike"]      = 1.0 if r["PostPrandialGlucose"] > 140 else 0.0
    r["FastingBloodSugar_Spike"]= 1.0 if r["FastingBloodSugar"] > 126 else 0.0
    r["GlucoseVariability"]     = round(abs(r["PostPrandialGlucose"] - r["FastingBloodSugar"]), 4)
    r["HbA1c_Deteriorating"]    = 0.0
    r["BMI_Change_30d"]         = 0.0

    real_rows.append(r)

df_real            = pd.DataFrame(real_rows)[feature_names]
df_real["Diagnosis"] = df_kaggle["diabetes"].values
print(f"  Real rows formatted : {len(df_real)}")

# ─────────────────────────────────────────
# 3. COMBINE — Real data 2x weight
# ─────────────────────────────────────────
print("\nCombining datasets...")
df_real_weighted = pd.concat([df_real] * 2, ignore_index=True)
df_combined      = pd.concat([df_synthetic, df_real_weighted], ignore_index=True)
df_combined      = df_combined.fillna(0)

print(f"  Synthetic rows : {len(df_synthetic)}")
print(f"  Real rows (2x) : {len(df_real_weighted)}")
print(f"  Combined total : {len(df_combined)}")
print(f"  Class dist     — 0: {(df_combined['Diagnosis']==0).sum()}, 1: {(df_combined['Diagnosis']==1).sum()}")

# ─────────────────────────────────────────
# 4. PREPARE FEATURES
# ─────────────────────────────────────────
X = df_combined[feature_names]
y = df_combined["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────
# 5. LOAD OLD MODEL FOR COMPARISON
# ─────────────────────────────────────────
backup_path    = os.path.join(BASE_DIR, "model", "xgb_model_backup.pkl")
old_model_path = backup_path if os.path.exists(backup_path) else os.path.join(BASE_DIR, "model", "xgb_model.pkl")
old_model      = joblib.load(old_model_path)
old_pred_proba = old_model.predict_proba(X_test)[:, 1]
old_auc        = roc_auc_score(y_test, old_pred_proba)
old_f1         = f1_score(y_test, old_model.predict(X_test))
print(f"\nOld model — AUC: {old_auc:.4f}, F1: {old_f1:.4f}")

# ─────────────────────────────────────────
# 6. TRAIN NEW MODEL
# ─────────────────────────────────────────
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

print("\nTraining new model on combined data...")
new_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)
new_model.fit(X_train, y_train)
print("  Training complete.")

# ─────────────────────────────────────────
# 7. EVALUATE
# ─────────────────────────────────────────
new_pred       = new_model.predict(X_test)
new_pred_proba = new_model.predict_proba(X_test)[:, 1]
new_auc        = roc_auc_score(y_test, new_pred_proba)
new_f1         = f1_score(y_test, new_pred)
new_acc        = accuracy_score(y_test, new_pred)

print(f"\n--- Model Comparison ---")
print(f"{'Metric':<12} {'Old Model':>12} {'New Model':>12} {'Change':>10}")
print(f"{'-'*50}")
print(f"{'AUC-ROC':<12} {old_auc:>12.4f} {new_auc:>12.4f} {new_auc-old_auc:>+10.4f}")
print(f"{'F1 Score':<12} {old_f1:>12.4f} {new_f1:>12.4f} {new_f1-old_f1:>+10.4f}")
print(f"{'Accuracy':<12} {'—':>12} {new_acc:>12.4f} {'—':>10}")
print()
print("New model classification report:")
print(classification_report(y_test, new_pred, target_names=["Non-Diabetic","Diabetic"]))

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(new_model, X, y, cv=cv, scoring="roc_auc")
print(f"CV Mean AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ─────────────────────────────────────────
# 8. SAVE PLOTS
# ─────────────────────────────────────────
PLOTS_DIR = os.path.join(BASE_DIR, "model", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

cm = confusion_matrix(y_test, new_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Diabetic","Diabetic"],
            yticklabels=["Non-Diabetic","Diabetic"])
plt.title("Confusion Matrix — Retrained Model")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_retrained.png"), dpi=150)
plt.close()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(old_pred_proba, bins=50, color="#1a237e", alpha=0.7)
plt.title("Old Model — Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.subplot(1, 2, 2)
plt.hist(new_pred_proba, bins=50, color="#2e7d32", alpha=0.7)
plt.title("New Model — Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "probability_distribution_comparison.png"), dpi=150)
plt.close()
print("  Saved plots.")

# ─────────────────────────────────────────
# 9. BACKUP OLD + SAVE NEW
# ─────────────────────────────────────────
old_to_backup = joblib.load(os.path.join(BASE_DIR, "model", "xgb_model.pkl"))
joblib.dump(old_to_backup, os.path.join(BASE_DIR, "model", "xgb_model_backup.pkl"))
joblib.dump(new_model,     os.path.join(BASE_DIR, "model", "xgb_model.pkl"))
print(f"\n  Old model backed up → model/xgb_model_backup.pkl")
print(f"  New model saved    → model/xgb_model.pkl")
print("\nDone! Restart backend to load the new model:")
print("  python backend/app.py")
