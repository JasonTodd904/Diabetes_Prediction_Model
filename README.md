# Diabetes_Prediction_Model
Using Machine Learing algorithm to figure out the trends which can lead to Diabetes and giving recommendation based on prediction

#**YOU CAN DOWNLOAD A READMe FILE FROM THE DOCUMENTS FOLDER AS WELL along with A PRESENTATION VIEW !!!!

# Diabetes Prediction System — Time Series ML Project

A machine learning web application that predicts diabetes risk for patients using clinical measurements. The system supports **new patient predictions** and **revisit tracking** — computing temporal features like lag values, rolling averages and trends across visits to detect deteriorating health patterns over time.

Built as a college mini project using Python, XGBoost, Flask and SQLite.

---

## What it does

- Accepts patient clinical data via a web form or CSV upload
- Detects whether a patient is new or returning
- For returning patients, automatically computes time series features (lag values, rolling averages, trends) from their visit history
- Predicts diabetes probability using a trained XGBoost model
- Explains the prediction using SHAP values — showing which features drove the result
- Shows a risk trajectory chart across visits for returning patients
- Fires trend alerts when key metrics like HbA1c or fasting blood sugar are worsening
- Allows doctors to confirm or correct predictions — feeding real verified data back into periodic model retraining

---

## Demo

### Single patient prediction
Enter patient data → get diagnosis, probability score, risk category, top SHAP factors and recommendations

### Revisit patient
Same patient, second visit with worsened values → lag features computed automatically → higher risk score → trend alerts fire → trajectory chart shows progression

### Example results
```
Patient 6053 (borderline non-diabetic):  36%  — Moderate Risk
Patient 6072 Visit 1 (borderline):       89%  — High Risk
Patient 6072 Visit 2 (worsened values):  99%  — Critical Risk
```

---

## Project structure

```
diabetes_project/
├── model/
│   ├── train_model.py                  # Initial model training
│   ├── retrain_model.py                # Periodic retraining with confirmed cases
│   ├── retrain_reduced_features.py     # Feature reduction + real data retraining
│   ├── retrain_with_real_data.py       # Kaggle data integration script
│   ├── xgb_model.pkl                   # Trained model (active)
│   ├── xgb_model_backup.pkl            # Previous model backup
│   ├── feature_names.pkl               # Active feature list
│   └── plots/                          # Evaluation plots
│       ├── confusion_matrix.png
│       ├── feature_importance.png
│       ├── shap_summary.png
│       ├── precision_recall.png
│       └── calibration_curve.png
├── database/
│   ├── setup_database.py               # DB setup + helper functions
│   └── diabetes.db                     # SQLite database (auto-created)
├── backend/
│   └── app.py                          # Flask API (6 routes)
├── frontend/
│   └── index.html                      # Web interface
└── data/
    ├── diabetes_timeseries_v3.csv      # Synthetic time series dataset
    └── diabetes_prediction_dataset.csv # Real Kaggle dataset (Mohammed Mustafa)
```

---

## Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| ML Model | XGBoost | Classification + regression |
| Explainability | SHAP | Feature importance per prediction |
| Data | pandas, numpy | Feature engineering |
| Database | SQLite (built-in Python) | Patient + visit storage |
| Backend | Flask + flask-cors | REST API |
| Model persistence | joblib | Save/load model |
| Visualisation | matplotlib, seaborn | Evaluation plots |
| Frontend | HTML + Bootstrap 5 + Chart.js | Web interface |

---

## Running the project

Start the backend server:
```bash
python backend/app.py
```
Backend runs at `http://127.0.0.1:5000`

Open the frontend by right clicking `frontend/index.html` → Open with → Chrome or Edge. Do not use Live Server as it causes page reloads.

---

## How to run a prediction

1. Enter a Patient ID and visit date
2. Click **Check Patient** — system detects new vs returning
3. Fill in clinical values or upload a CSV file
4. Click **Run Prediction**
5. View diagnosis, probability, SHAP factors, recommendations
6. For returning patients — trajectory chart and trend alerts appear automatically
7. Doctor confirms or corrects the prediction using the verification card

---

## Dataset

### Primary dataset — Synthetic time series
- 1,879 patients, 114 columns, 0 null values
- Originally a standard diabetes dataset, extended with engineered time series features
- Lag features: `HbA1c_lag1`, `HbA1c_lag2`, `HbA1c_lag3`
- Rolling averages: `HbA1c_RollingAvg3`, `FastingBloodSugar_RollingAvg3`
- Trend features: `HbA1c_Trend`, `SystolicBP_Trend`
- Target: `Diagnosis` (0 = Non-Diabetic, 1 = Diabetic)
- Class distribution: 1,127 non-diabetic, 752 diabetic

### Secondary dataset — Real world data (Kaggle)
- Source: [Diabetes Prediction Dataset by Mohammed Mustafa](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- 100,000 patients, 9 columns
- Genuine columns used: `HbA1c_level`, `blood_glucose_level`, `bmi`, `age`, `hypertension`, `smoking_history`, `gender`
- Balanced to 16,836 rows (8,418 diabetic, 8,418 non-diabetic)
- Combined with synthetic data for retraining (real data given 2x weight)

### Known limitation
> The synthetic dataset was generated with mathematically clean feature boundaries. This causes the model to be highly confident on clearly diabetic or non-diabetic values. Adding real Kaggle data and reducing to 50 core features improved probability spread — borderline patients now receive 30-65% probabilities rather than binary 0% or 97% outputs. For a production clinical tool, larger real-world datasets with natural noise would be required for proper probability calibration.

---

## Model architecture

### Algorithm — XGBoost Classifier
XGBoost (Extreme Gradient Boosting) was chosen over alternatives for the following reasons:

- Handles tabular data with mixed feature types natively
- Works well on small to medium datasets (1,879 rows)
- Built-in handling for class imbalance via `scale_pos_weight`
- Compatible with SHAP for explainability
- Fast training (seconds vs hours for neural networks)
- Does not require feature scaling

### Hyperparameters
```python
n_estimators    = 200
max_depth       = 5
learning_rate   = 0.05
subsample       = 0.8
colsample_bytree= 0.8
scale_pos_weight= 1.02  # handles class imbalance
```

### Feature set — 50 core features
Reduced from 110 to 50 to prevent overfitting on the small synthetic dataset:

- Clinical measurements: HbA1c, FastingBloodSugar, PostPrandialGlucose, BMI, Insulin, BP, Cholesterol
- Demographics: Age, Gender, Smoking, Hypertension, FamilyHistory
- Lifestyle: PhysicalActivity, DietQuality, StressLevel, MedicationAdherence
- Symptoms: FrequentUrination, ExcessiveThirst, BlurredVision, TinglingHandsFeet
- Time series: lag1/lag2/lag3, RollingAvg3, Trend for key clinical measurements
- Engineered: InsulinResistanceScore, GlucoseToInsulinRatio, GlucoseVariability

### Training data
```
Synthetic dataset  :  1,879 rows
Real Kaggle data   : 16,836 rows × 2 (weighted) = 33,672 rows
Combined total     : 35,551 rows
Train/test split   : 80% / 20%
```

### Evaluation results
```
AUC-ROC  : 0.9770
F1 Score : 0.9069
Accuracy : 91.38%
CV AUC   : 0.9773 ± 0.0011 (5-fold)

Probability distribution:
  0-30%  : 3,029 patients  (clearly non-diabetic)
  30-60% :   826 patients  (borderline uncertain)
  60-80% :   587 patients  (moderate risk)
  80-95% :   282 patients  (high risk)
  95%+   : 2,387 patients  (clearly diabetic)
```

---

## Time series & revisit logic

The system implements time series awareness at **inference time** — not just training time.

### How it works

```
New patient visit
        ↓
Save to database
        ↓
Run model on current values only
        ↓
Return prediction

Returning patient visit
        ↓
Pull last 3 visits from database
        ↓
Compute lag features:
    HbA1c_lag1 = previous visit HbA1c
    HbA1c_lag2 = 2 visits ago HbA1c
    HbA1c_RollingAvg3 = mean of last 3
    HbA1c_Trend = current - oldest
        ↓
Feed enriched feature vector to model
        ↓
Return prediction + trend alerts + trajectory chart
```

### Trend alerts
Fires when a key metric has increased across 3 or more consecutive visits:
- HbA1c rising
- Fasting Blood Sugar rising
- BMI rising
- Systolic BP rising

### Risk trajectory chart
Plots the predicted diabetes probability across all visits for a patient — visually showing if risk is stable, improving or deteriorating.

---

## Retraining feedback loop

The system supports continuous improvement through doctor-verified data.

### Flow
```
Doctor runs prediction
        ↓
Doctor confirms: "Correct" or "Wrong — actual is X"
        ↓
Confirmed label saved to database
        ↓
Every 10 confirmations → system recommends retraining
        ↓
Run: python model/retrain_model.py
        ↓
Script combines original data + confirmed cases (3x weight)
        ↓
Retrains XGBoost → backs up old model → saves new model
        ↓
Restart backend → new model active
```

### Why 3x weight for confirmed cases
With only 10-50 confirmed cases against 35,000 training rows, giving each confirmed case 3x weight ensures real verified data has meaningful influence on the model rather than being statistically insignificant.

### Retraining scripts
```
retrain_model.py              — periodic retraining with confirmed DB cases
retrain_reduced_features.py   — one-time script used to reduce to 50 features
retrain_with_real_data.py     — one-time script used to add Kaggle data
```

---

## Output reference

### Per prediction
- Diagnosis label (Diabetic / Non-Diabetic)
- Probability score (0-100%)
- Risk category (Low / Moderate / High / Critical)
- Top 6 SHAP factors with direction (↑ risk / ↓ risk)
- Recommended actions based on top risk factor
- Population comparison (patient vs dataset averages)

### For revisit patients (additional)
- Risk trajectory chart across all visits
- Trend alerts for worsening metrics
- Visit badge showing lag features were computed

### Model evaluation (training phase)
- AUC-ROC, F1, Accuracy, Precision, Recall
- Confusion matrix
- Feature importance chart
- SHAP summary plot
- Precision-recall curve
- Calibration curve
- 5-fold cross validation scores

---

## Future improvements

- Add more real clinical datasets to improve probability calibration
- Implement what-if simulator (adjust BMI/HbA1c and see risk change)
- Add PDF export for patient summary cards
- Deploy backend to cloud (Render, Railway, or AWS)
- Add authentication for multi-doctor use
- Extend to regression output — predicting next glucose level
- Replace SQLite with PostgreSQL for production scale

---

## Acknowledgements

- Dataset: [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) by Mohammed Mustafa on Kaggle
- XGBoost: Chen & Guestrin, 2016
- SHAP: Lundberg & Lee, 2017
