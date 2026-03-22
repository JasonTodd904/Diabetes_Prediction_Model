# ============================================================
# Step 3: Diabetes Prediction Model - Flask Backend
# File   : backend/app.py
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import joblib
import shap
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# ─────────────────────────────────────────
# PATH SETUP
# Add project root to path so we can import
# from the database module
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from database.setup_database import (
    patient_exists, get_patient, get_last_n_visits,
    get_all_visits, save_new_patient, save_visit
)

app = Flask(__name__)
CORS(app)  # allows frontend to call this API

# ─────────────────────────────────────────
# LOAD MODEL + FEATURE NAMES
# ─────────────────────────────────────────
MODEL_PATH    = os.path.join(BASE_DIR, "model", "xgb_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_names.pkl")

print("Loading model...")
model         = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
explainer     = shap.TreeExplainer(model)
print(f"  Model loaded. Features: {len(feature_names)}")


# ─────────────────────────────────────────
# HELPER: RISK CATEGORY
# ─────────────────────────────────────────
def get_risk_category(probability):
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Moderate Risk"
    elif probability < 0.80:
        return "High Risk"
    else:
        return "Critical Risk"


# ─────────────────────────────────────────
# HELPER: COMPUTE LAG FEATURES
# Takes current visit data + list of past
# visits and computes lag/rolling/trend
# features dynamically for revisit patients
# ─────────────────────────────────────────
def compute_lag_features(current, past_visits):
    """
    current     : dict of current visit readings
    past_visits : list of dicts, most recent first
    Returns     : dict with lag/rolling/trend features added
    """
    features = dict(current)

    # Fields we compute time features for
    time_fields = [
        "FastingBloodSugar", "HbA1c", "Insulin",
        "PostPrandialGlucose", "CarbohydrateIntake",
        "SystolicBP", "DiastolicBP",
        "CholesterolTotal", "CholesterolLDL",
        "CholesterolTriglycerides", "BMI",
        "MedicationAdherence", "PhysicalActivity",
        "SleepQuality", "DietQuality", "AlcoholConsumption",
        "FatigueLevels", "WaistCircumference",
        "CalorieIntake", "StressLevel"
    ]

    for field in time_fields:
        current_val = float(current.get(field, 0) or 0)
        past_vals   = [float(v.get(field, 0) or 0) for v in past_visits]

        # Lag features
        features[f"{field}_lag1"] = past_vals[0] if len(past_vals) > 0 else current_val
        features[f"{field}_lag2"] = past_vals[1] if len(past_vals) > 1 else current_val
        features[f"{field}_lag3"] = past_vals[2] if len(past_vals) > 2 else current_val

        # Rolling average (current + up to last 2)
        window = [current_val] + past_vals[:2]
        features[f"{field}_RollingAvg3"] = round(sum(window) / len(window), 4)

        # Trend (current minus oldest in window)
        oldest = past_vals[0] if past_vals else current_val
        features[f"{field}_Trend"] = round(current_val - oldest, 4)

    # Engineered scores
    fbs     = float(current.get("FastingBloodSugar", 100) or 100)
    insulin = float(current.get("Insulin", 1) or 1)
    ppg     = float(current.get("PostPrandialGlucose", 100) or 100)
    hba1c   = float(current.get("HbA1c", 5) or 5)

    features["InsulinResistanceScore"] = round(fbs * insulin / 405, 4)
    features["GlucoseToInsulinRatio"]  = round(fbs / insulin if insulin > 0 else 0, 4)
    features["PostPrandialSpike"]      = 1 if ppg > 140 else 0
    features["HbA1c_Deteriorating"]   = 0  # updated below for revisits
    features["FastingBloodSugar_Spike"]= 1 if fbs > 126 else 0
    features["GlucoseVariability"]     = round(abs(ppg - fbs), 4)

    # HbA1c deteriorating — only meaningful for revisits
    if past_visits:
        prev_hba1c = float(past_visits[0].get("HbA1c", hba1c) or hba1c)
        features["HbA1c_Deteriorating"] = 1 if hba1c > prev_hba1c else 0

    # BMI change
    prev_bmi = float(past_visits[0].get("BMI", current.get("BMI", 25)) or 25) if past_visits else float(current.get("BMI", 25) or 25)
    features["BMI_Change_30d"] = round(float(current.get("BMI", 25) or 25) - prev_bmi, 4)

    return features


# ─────────────────────────────────────────
# HELPER: BUILD MODEL INPUT
# Aligns computed features with the exact
# feature list the model was trained on
# ─────────────────────────────────────────
def build_model_input(features):
    row = []
    for f in feature_names:
        row.append(float(features.get(f, 0) or 0))
    return np.array(row).reshape(1, -1)


# ─────────────────────────────────────────
# HELPER: TREND ALERTS
# Checks if key metrics are worsening
# across the last 3 visits
# ─────────────────────────────────────────
def get_trend_alerts(all_visits):
    alerts = []
    if len(all_visits) < 2:
        return alerts

    watch_fields = {
        "HbA1c"           : "HbA1c",
        "FastingBloodSugar": "Fasting Blood Sugar",
        "BMI"             : "BMI",
        "SystolicBP"      : "Systolic BP"
    }

    recent = all_visits[-3:] if len(all_visits) >= 3 else all_visits

    for field, label in watch_fields.items():
        vals = [float(v.get(field, 0) or 0) for v in recent if v.get(field)]
        if len(vals) >= 2 and all(vals[i] < vals[i+1] for i in range(len(vals)-1)):
            alerts.append(f"{label} has been rising across your last {len(vals)} visits")

    return alerts


# ─────────────────────────────────────────
# HELPER: POPULATION COMPARISON
# Compares patient's key values against
# dataset statistics (approximate)
# ─────────────────────────────────────────
DATASET_MEANS = {
    "HbA1c"           : 6.98,
    "FastingBloodSugar": 135.2,
    "BMI"             : 27.7,
    "SystolicBP"      : 134.1,
}

def get_population_comparison(features):
    comparison = {}
    for field, mean in DATASET_MEANS.items():
        val = float(features.get(field, mean) or mean)
        pct = round(min(99, max(1, (val / mean) * 50)), 1)
        direction = "above" if val > mean else "below"
        comparison[field] = {
            "value"    : round(val, 2),
            "mean"     : mean,
            "direction": direction,
            "percentile": pct
        }
    return comparison


# ═══════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════

# ─────────────────────────────────────────
# ROUTE 1: HEALTH CHECK
# ─────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "running", "message": "Diabetes Prediction API is live"})


# ─────────────────────────────────────────
# ROUTE 2: CHECK IF PATIENT EXISTS
# Frontend calls this first to decide
# whether to show new/revisit form
# ─────────────────────────────────────────
@app.route("/patient/check/<int:patient_id>", methods=["GET"])
def check_patient(patient_id):
    exists = patient_exists(patient_id)
    if exists:
        visits     = get_all_visits(patient_id)
        patient    = get_patient(patient_id)
        return jsonify({
            "exists"      : True,
            "visit_count" : len(visits),
            "patient"     : patient
        })
    return jsonify({"exists": False})


# ─────────────────────────────────────────
# ROUTE 3: PREDICT
# Main endpoint — handles both new patients
# and revisits. Returns full prediction
# output including SHAP, risk, alerts etc.
# ─────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data       = request.get_json()
    patient_id = int(data.get("PatientID"))
    visit_date = data.get("visit_date", datetime.today().strftime("%Y-%m-%d"))
    is_revisit = patient_exists(patient_id)

    # ── Static patient fields ──
    static_fields = [
        "Age", "Gender", "Ethnicity", "SocioeconomicStatus",
        "EducationLevel", "Smoking", "FamilyHistoryDiabetes",
        "GestationalDiabetes", "PolycysticOvarySyndrome",
        "PreviousPreDiabetes", "HeavyMetalsExposure",
        "OccupationalExposureChemicals", "WaterQuality", "HealthLiteracy"
    ]

    # ── Save static info if new patient ──
    if not is_revisit:
        patient_data = {"PatientID": patient_id}
        for f in static_fields:
            patient_data[f] = data.get(f, 0)
        save_new_patient(patient_data)

    # ── Compute features ──
    if is_revisit:
        past_visits = get_last_n_visits(patient_id, n=3)
        features    = compute_lag_features(data, past_visits)
    else:
        # New patient — no past visits, compute with current data only
        features = compute_lag_features(data, [])

    # ── Build model input ──
    X = build_model_input(features)

    # ── Predict ──
    probability        = float(model.predict_proba(X)[0][1])
    predicted_diagnosis= int(model.predict(X)[0])
    risk_category      = get_risk_category(probability)

    # ── SHAP explanation ──
    shap_vals = explainer.shap_values(X)[0]
    shap_df   = pd.DataFrame({
        "feature": feature_names,
        "shap"   : shap_vals,
        "value"  : X[0]
    }).sort_values("shap", key=abs, ascending=False).head(10)

    top_factors = shap_df.to_dict(orient="records")

    # ── Recommended actions based on top risk factors ──
    recommendations = []
    top_feature = shap_df.iloc[0]["feature"] if len(shap_df) > 0 else ""
    if "HbA1c" in top_feature:
        recommendations.append("Retest HbA1c in 3 months")
    if "FastingBloodSugar" in top_feature:
        recommendations.append("Monitor fasting glucose daily")
    if "BMI" in top_feature:
        recommendations.append("Consider dietary consultation")
    if "SystolicBP" in top_feature:
        recommendations.append("Monitor blood pressure weekly")
    if not recommendations:
        recommendations.append("Schedule a follow-up in 6 months")

    # ── Population comparison ──
    population_comparison = get_population_comparison(features)

    # ── Trend alerts (revisit only) ──
    trend_alerts = []
    if is_revisit:
        all_visits   = get_all_visits(patient_id)
        trend_alerts = get_trend_alerts(all_visits)

    # ── Save this visit to DB ──
    visit_record = {
        "PatientID"              : patient_id,
        "visit_date"             : visit_date,
        "FastingBloodSugar"      : data.get("FastingBloodSugar"),
        "HbA1c"                  : data.get("HbA1c"),
        "PostPrandialGlucose"    : data.get("PostPrandialGlucose"),
        "Insulin"                : data.get("Insulin"),
        "BMI"                    : data.get("BMI"),
        "SystolicBP"             : data.get("SystolicBP"),
        "DiastolicBP"            : data.get("DiastolicBP"),
        "CholesterolTotal"       : data.get("CholesterolTotal"),
        "CholesterolLDL"         : data.get("CholesterolLDL"),
        "CholesterolHDL"         : data.get("CholesterolHDL"),
        "CholesterolTriglycerides": data.get("CholesterolTriglycerides"),
        "SerumCreatinine"        : data.get("SerumCreatinine"),
        "BUNLevels"              : data.get("BUNLevels"),
        "WaistCircumference"     : data.get("WaistCircumference"),
        "CarbohydrateIntake"     : data.get("CarbohydrateIntake"),
        "CalorieIntake"          : data.get("CalorieIntake"),
        "PhysicalActivity"       : data.get("PhysicalActivity"),
        "SleepQuality"           : data.get("SleepQuality"),
        "DietQuality"            : data.get("DietQuality"),
        "AlcoholConsumption"     : data.get("AlcoholConsumption"),
        "StressLevel"            : data.get("StressLevel"),
        "MedicationAdherence"    : data.get("MedicationAdherence"),
        "AntidiabeticMedications"    : data.get("AntidiabeticMedications", 0),
        "AntihypertensiveMedications": data.get("AntihypertensiveMedications", 0),
        "Statins"                : data.get("Statins", 0),
        "FrequentUrination"      : data.get("FrequentUrination", 0),
        "ExcessiveThirst"        : data.get("ExcessiveThirst", 0),
        "UnexplainedWeightLoss"  : data.get("UnexplainedWeightLoss", 0),
        "BlurredVision"          : data.get("BlurredVision", 0),
        "SlowHealingSores"       : data.get("SlowHealingSores", 0),
        "TinglingHandsFeet"      : data.get("TinglingHandsFeet", 0),
        "Hypertension"           : data.get("Hypertension", 0),
        "MedicalCheckupsFrequency": data.get("MedicalCheckupsFrequency"),
        "QualityOfLifeScore"     : data.get("QualityOfLifeScore"),
        "FatigueLevels"          : data.get("FatigueLevels"),
        "Diagnosis"              : data.get("Diagnosis", predicted_diagnosis),
        "DoctorNotes"            : data.get("DoctorNotes", ""),
        "PredictedProbability"   : round(probability, 4),
        "PredictedDiagnosis"     : predicted_diagnosis,
        "RiskCategory"           : risk_category,
    }
    save_visit(visit_record)

    # ── Return full response ──
    return jsonify({
        "patient_id"           : patient_id,
        "is_revisit"           : is_revisit,
        "visit_date"           : visit_date,
        "prediction": {
            "diagnosis"        : "Diabetic" if predicted_diagnosis == 1 else "Non-Diabetic",
            "probability"      : round(probability * 100, 2),
            "risk_category"    : risk_category,
        },
        "shap_explanation"     : top_factors,
        "recommendations"      : recommendations,
        "population_comparison": population_comparison,
        "trend_alerts"         : trend_alerts,
    })


# ─────────────────────────────────────────
# ROUTE 4: PATIENT VISIT HISTORY
# Returns all visits for a patient —
# used for the risk trajectory chart
# ─────────────────────────────────────────
@app.route("/patient/<int:patient_id>/history", methods=["GET"])
def patient_history(patient_id):
    if not patient_exists(patient_id):
        return jsonify({"error": "Patient not found"}), 404

    visits  = get_all_visits(patient_id)
    patient = get_patient(patient_id)

    history = []
    for v in visits:
        history.append({
            "visit_date"         : v["visit_date"],
            "PredictedProbability": v["PredictedProbability"],
            "RiskCategory"       : v["RiskCategory"],
            "HbA1c"              : v["HbA1c"],
            "FastingBloodSugar"  : v["FastingBloodSugar"],
            "BMI"                : v["BMI"],
            "Diagnosis"          : v["Diagnosis"],
            "DoctorNotes"        : v["DoctorNotes"],
        })

    return jsonify({
        "patient" : patient,
        "history" : history,
        "total_visits": len(visits)
    })


# ─────────────────────────────────────────
# ROUTE 5: POPULATION DASHBOARD
# Aggregate stats across all patients
# ─────────────────────────────────────────
@app.route("/dashboard", methods=["GET"])
def dashboard():
    from database.setup_database import get_connection
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total FROM patients")
    total_patients = cursor.fetchone()["total"]

    cursor.execute("""
        SELECT RiskCategory, COUNT(*) as count
        FROM visits
        WHERE VisitID IN (
            SELECT MAX(VisitID) FROM visits GROUP BY PatientID
        )
        GROUP BY RiskCategory
    """)
    risk_distribution = {row["RiskCategory"]: row["count"] for row in cursor.fetchall()}

    cursor.execute("""
        SELECT AVG(HbA1c) as avg_hba1c,
               AVG(FastingBloodSugar) as avg_fbs,
               AVG(BMI) as avg_bmi
        FROM visits
    """)
    row  = cursor.fetchone()
    avgs = {
        "avg_hba1c": round(row["avg_hba1c"] or 0, 2),
        "avg_fbs"  : round(row["avg_fbs"] or 0, 2),
        "avg_bmi"  : round(row["avg_bmi"] or 0, 2),
    }

    cursor.execute("""
        SELECT COUNT(*) as diabetic_count
        FROM visits
        WHERE PredictedDiagnosis = 1
        AND VisitID IN (
            SELECT MAX(VisitID) FROM visits GROUP BY PatientID
        )
    """)
    diabetic_count = cursor.fetchone()["diabetic_count"]
    conn.close()

    return jsonify({
        "total_patients"   : total_patients,
        "diabetic_count"   : diabetic_count,
        "risk_distribution": risk_distribution,
        "averages"         : avgs,
    })


# ─────────────────────────────────────────
# ROUTE 6: CONFIRM DIAGNOSIS
# Doctor confirms or corrects the prediction
# Stores verified label for retraining
# ─────────────────────────────────────────
@app.route("/confirm", methods=["POST"])
def confirm():
    data          = request.get_json()
    patient_id    = int(data.get("patient_id"))
    visit_date    = data.get("visit_date")
    actual_diagnosis = int(data.get("actual_diagnosis"))
    was_correct   = bool(data.get("was_correct"))

    from database.setup_database import get_connection
    conn   = get_connection()
    cursor = conn.cursor()

    # Add confirmed_diagnosis and was_correct columns if not exist
    try:
        cursor.execute("ALTER TABLE visits ADD COLUMN confirmed_diagnosis INTEGER DEFAULT NULL")
        cursor.execute("ALTER TABLE visits ADD COLUMN was_correct INTEGER DEFAULT NULL")
        conn.commit()
    except:
        pass  # columns already exist

    # Update the most recent visit for this patient
    cursor.execute("""
        UPDATE visits
        SET confirmed_diagnosis = ?,
            was_correct = ?
        WHERE PatientID = ?
        AND visit_date = ?
    """, (actual_diagnosis, 1 if was_correct else 0, patient_id, visit_date))
    conn.commit()

    # Count how many confirmed cases we have now
    cursor.execute("""
        SELECT COUNT(*) as total
        FROM visits
        WHERE confirmed_diagnosis IS NOT NULL
    """)
    total_confirmed = cursor.fetchone()["total"]
    conn.close()

    # Check if we hit the retraining threshold
    should_retrain = total_confirmed > 0 and total_confirmed % 10 == 0

    return jsonify({
        "message"         : "Diagnosis confirmed, thank you!",
        "total_confirmed" : total_confirmed,
        "should_retrain"  : should_retrain,
        "retrain_message" : f"You have {total_confirmed} confirmed cases. Retraining recommended!" if should_retrain else f"{total_confirmed} confirmed cases collected. Retraining triggers at next multiple of 10."
    })

# ─────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\nStarting Diabetes Prediction API...")
    print("  URL: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop\n")
    app.run(debug=True, port=5000)