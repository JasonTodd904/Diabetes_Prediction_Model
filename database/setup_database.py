# ============================================================
# Step 2: Diabetes Prediction Model - Database Setup
# File   : database/setup_database.py
# ============================================================

import sqlite3
import os

# ─────────────────────────────────────────
# 1. DATABASE PATH
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "database", "diabetes.db")

def get_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # allows dict-like access to rows
    return conn

def setup_database():
    print("Setting up database...")
    conn = get_connection()
    cursor = conn.cursor()

    # ─────────────────────────────────────────
    # 2. PATIENTS TABLE
    # Stores static info that doesn't change
    # across visits (demographics, risk flags)
    # ─────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            PatientID               INTEGER PRIMARY KEY,
            Age                     INTEGER,
            Gender                  INTEGER,
            Ethnicity               INTEGER,
            SocioeconomicStatus     INTEGER,
            EducationLevel          INTEGER,
            Smoking                 INTEGER,
            FamilyHistoryDiabetes   INTEGER,
            GestationalDiabetes     INTEGER,
            PolycysticOvarySyndrome INTEGER,
            PreviousPreDiabetes     INTEGER,
            HeavyMetalsExposure     INTEGER,
            OccupationalExposureChemicals INTEGER,
            WaterQuality            INTEGER,
            HealthLiteracy          INTEGER,
            created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("  Created table: patients")

    # ─────────────────────────────────────────
    # 3. VISITS TABLE
    # Stores every visit's clinical readings.
    # Multiple rows per patient allowed.
    # Lag/rolling features are recomputed
    # dynamically from this table on revisits.
    # ─────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visits (
            VisitID                 INTEGER PRIMARY KEY AUTOINCREMENT,
            PatientID               INTEGER NOT NULL,
            visit_date              TEXT NOT NULL,

            -- Clinical measurements
            FastingBloodSugar       REAL,
            HbA1c                   REAL,
            PostPrandialGlucose     REAL,
            Insulin                 REAL,
            BMI                     REAL,
            SystolicBP              INTEGER,
            DiastolicBP             INTEGER,
            CholesterolTotal        REAL,
            CholesterolLDL          REAL,
            CholesterolHDL          REAL,
            CholesterolTriglycerides REAL,
            SerumCreatinine         REAL,
            BUNLevels               REAL,
            WaistCircumference      REAL,
            CarbohydrateIntake      REAL,
            CalorieIntake           REAL,
            Insulin_value           REAL,

            -- Lifestyle
            PhysicalActivity        REAL,
            SleepQuality            REAL,
            DietQuality             REAL,
            AlcoholConsumption      REAL,
            StressLevel             REAL,
            MedicationAdherence     REAL,

            -- Medications
            AntidiabeticMedications     INTEGER,
            AntihypertensiveMedications INTEGER,
            Statins                     INTEGER,

            -- Symptoms
            FrequentUrination       INTEGER,
            ExcessiveThirst         INTEGER,
            UnexplainedWeightLoss   INTEGER,
            BlurredVision           INTEGER,
            SlowHealingSores        INTEGER,
            TinglingHandsFeet       INTEGER,

            -- Other static-ish fields
            Hypertension            INTEGER,
            MedicalCheckupsFrequency REAL,
            QualityOfLifeScore      REAL,
            FatigueLevels           REAL,

            -- Diagnosis for this visit
            Diagnosis               INTEGER,

            -- Doctor notes
            DoctorNotes             TEXT,

            -- Prediction outputs stored per visit
            PredictedProbability    REAL,
            PredictedDiagnosis      INTEGER,
            RiskCategory            TEXT,

            visit_timestamp         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (PatientID) REFERENCES patients(PatientID)
        )
    """)
    print("  Created table: visits")

    # ─────────────────────────────────────────
    # 4. INDEX FOR FAST PATIENT LOOKUPS
    # ─────────────────────────────────────────
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_visits_patient
        ON visits(PatientID, visit_date)
    """)
    print("  Created index: idx_visits_patient")

    conn.commit()
    conn.close()
    print(f"\nDatabase ready at: database/diabetes.db")
    print("Step 2 complete!")


# ─────────────────────────────────────────
# 5. HELPER FUNCTIONS
# Used by the backend later
# ─────────────────────────────────────────

def patient_exists(patient_id):
    """Check if a patient already exists in the DB."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT PatientID FROM patients WHERE PatientID = ?", (patient_id,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def get_patient(patient_id):
    """Get static patient info."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE PatientID = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_last_n_visits(patient_id, n=3):
    """Get the last n visits for a patient, ordered by date descending."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM visits
        WHERE PatientID = ?
        ORDER BY visit_date DESC
        LIMIT ?
    """, (patient_id, n))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_visits(patient_id):
    """Get all visits for a patient ordered by date ascending (for charts)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM visits
        WHERE PatientID = ?
        ORDER BY visit_date ASC
    """, (patient_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_new_patient(patient_data):
    """Insert a new patient's static info."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO patients (
            PatientID, Age, Gender, Ethnicity, SocioeconomicStatus,
            EducationLevel, Smoking, FamilyHistoryDiabetes,
            GestationalDiabetes, PolycysticOvarySyndrome,
            PreviousPreDiabetes, HeavyMetalsExposure,
            OccupationalExposureChemicals, WaterQuality, HealthLiteracy
        ) VALUES (
            :PatientID, :Age, :Gender, :Ethnicity, :SocioeconomicStatus,
            :EducationLevel, :Smoking, :FamilyHistoryDiabetes,
            :GestationalDiabetes, :PolycysticOvarySyndrome,
            :PreviousPreDiabetes, :HeavyMetalsExposure,
            :OccupationalExposureChemicals, :WaterQuality, :HealthLiteracy
        )
    """, patient_data)
    conn.commit()
    conn.close()


def save_visit(visit_data):
    """Insert a new visit record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO visits (
            PatientID, visit_date,
            FastingBloodSugar, HbA1c, PostPrandialGlucose, Insulin,
            BMI, SystolicBP, DiastolicBP, CholesterolTotal,
            CholesterolLDL, CholesterolHDL, CholesterolTriglycerides,
            SerumCreatinine, BUNLevels, WaistCircumference,
            CarbohydrateIntake, CalorieIntake,
            PhysicalActivity, SleepQuality, DietQuality,
            AlcoholConsumption, StressLevel, MedicationAdherence,
            AntidiabeticMedications, AntihypertensiveMedications, Statins,
            FrequentUrination, ExcessiveThirst, UnexplainedWeightLoss,
            BlurredVision, SlowHealingSores, TinglingHandsFeet,
            Hypertension, MedicalCheckupsFrequency, QualityOfLifeScore,
            FatigueLevels, Diagnosis, DoctorNotes,
            PredictedProbability, PredictedDiagnosis, RiskCategory
        ) VALUES (
            :PatientID, :visit_date,
            :FastingBloodSugar, :HbA1c, :PostPrandialGlucose, :Insulin,
            :BMI, :SystolicBP, :DiastolicBP, :CholesterolTotal,
            :CholesterolLDL, :CholesterolHDL, :CholesterolTriglycerides,
            :SerumCreatinine, :BUNLevels, :WaistCircumference,
            :CarbohydrateIntake, :CalorieIntake,
            :PhysicalActivity, :SleepQuality, :DietQuality,
            :AlcoholConsumption, :StressLevel, :MedicationAdherence,
            :AntidiabeticMedications, :AntihypertensiveMedications, :Statins,
            :FrequentUrination, :ExcessiveThirst, :UnexplainedWeightLoss,
            :BlurredVision, :SlowHealingSores, :TinglingHandsFeet,
            :Hypertension, :MedicalCheckupsFrequency, :QualityOfLifeScore,
            :FatigueLevels, :Diagnosis, :DoctorNotes,
            :PredictedProbability, :PredictedDiagnosis, :RiskCategory
        )
    """, visit_data)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    setup_database()