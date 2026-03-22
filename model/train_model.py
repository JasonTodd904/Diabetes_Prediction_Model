# ============================================================
# Step 1: Diabetes Prediction Model - Training Script
# File   : model/train_model.py
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
import shap

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("Loading dataset...")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes_timeseries_v3.csv")

df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

# ─────────────────────────────────────────
# 2. PREPARE FEATURES
# ─────────────────────────────────────────
# Drop columns that are not useful for prediction
DROP_COLS = ["PatientID", "visit_date", "DoctorInCharge", "Diagnosis"]

X = df.drop(columns=DROP_COLS)
y = df["Diagnosis"]

print(f"  Features: {X.shape[1]}")
print(f"  Class distribution — 0: {(y==0).sum()}, 1: {(y==1).sum()}")

# ─────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ─────────────────────────────────────────
# 4. CLASS IMBALANCE WEIGHT
# ─────────────────────────────────────────
# Handles the 1127 vs 752 imbalance
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight: {scale_pos_weight:.3f}")

# ─────────────────────────────────────────
# 5. TRAIN XGBOOST CLASSIFIER
# ─────────────────────────────────────────
print("\nTraining XGBoost classifier...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)
print("  Training complete.")

# ─────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────
print("\n--- Evaluation on Test Set ---")
y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
f1      = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"  Accuracy : {acc:.4f}")
print(f"  F1 Score : {f1:.4f}")
print(f"  AUC-ROC  : {auc_roc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["Non-Diabetic", "Diabetic"]))

# ─────────────────────────────────────────
# 7. CROSS VALIDATION
# ─────────────────────────────────────────
print("--- 5-Fold Cross Validation ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"  AUC per fold : {[round(s,4) for s in cv_scores]}")
print(f"  Mean AUC     : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ─────────────────────────────────────────
# 8. CONFUSION MATRIX PLOT
# ─────────────────────────────────────────
print("\nSaving plots...")
PLOTS_DIR = os.path.join(BASE_DIR, "model", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Diabetic", "Diabetic"],
            yticklabels=["Non-Diabetic", "Diabetic"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("  Saved: confusion_matrix.png")

# ─────────────────────────────────────────
# 9. FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────
importance_df = pd.DataFrame({
    "feature"   : X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False).head(20)

plt.figure(figsize=(8, 7))
sns.barplot(data=importance_df, x="importance", y="feature", palette="Blues_r")
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("  Saved: feature_importance.png")

# ─────────────────────────────────────────
# 10. PRECISION-RECALL CURVE
# ─────────────────────────────────────────
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color="purple", lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "precision_recall.png"), dpi=150)
plt.close()
print("  Saved: precision_recall.png")

# ─────────────────────────────────────────
# 11. CALIBRATION CURVE
# ─────────────────────────────────────────
fraction_of_positives, mean_predicted = calibration_curve(
    y_test, y_pred_proba, n_bins=10
)
plt.figure(figsize=(6, 5))
plt.plot(mean_predicted, fraction_of_positives, "s-", label="XGBoost")
plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "calibration_curve.png"), dpi=150)
plt.close()
print("  Saved: calibration_curve.png")

# ─────────────────────────────────────────
# 12. SHAP VALUES
# ─────────────────────────────────────────
print("\nComputing SHAP values (this may take a moment)...")
explainer    = shap.TreeExplainer(model)
shap_values  = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_summary.png")

# ─────────────────────────────────────────
# 13. SAVE MODEL + FEATURE NAMES
# ─────────────────────────────────────────
print("\nSaving model...")
MODEL_DIR = os.path.join(BASE_DIR, "model")
joblib.dump(model,          os.path.join(MODEL_DIR, "xgb_model.pkl"))
joblib.dump(list(X.columns),os.path.join(MODEL_DIR, "feature_names.pkl"))
print("  Saved: xgb_model.pkl")
print("  Saved: feature_names.pkl")

print("\nStep 1 complete! All files saved.")
print(f"  Model    -> model/xgb_model.pkl")
print(f"  Features -> model/feature_names.pkl")
print(f"  Plots    -> model/plots/")