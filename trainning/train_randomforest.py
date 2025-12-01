import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ========================================
# 1. PATH
# ========================================
DATA_PATH = r"..\nhan\water_labeled_both_species.csv"
MODEL_DIR = r"..\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ========================================
# 2. LOAD DATA
# ========================================
df = pd.read_csv(DATA_PATH)

# ========================================
# 3. CHỌN CỘT TRAIN
# ========================================
feature_cols = [
    "volatile_suspended_solids_mg_l",
    "unionised_ammonia_mg_l",
    "turbidity_ntu",
    "total_phosphorus_mg_l",
    "total_nitrogen_mg_l",
    "total_kjeldahl_nitrogen_mg_l",
    "total_inorganic_nitrogen_mg_l",
    "temperature_°c",
    "suspended_solids_mg_l",
    "silica_mg_l",
    "secchi_disc_depth_m",
    "salinity_psu",
    "phaeo_pigments_μg_l",
    "ph",
    "orthophosphate_phosphorus_mg_l",
    "nitrite_nitrogen_mg_l",
    "nitrate_nitrogen_mg_l",
    "faecal_coliforms_cfu_100ml",
    "e._coli_cfu_100ml",
    "dissolved_oxygen_mg_l",
    "dissolved_oxygen_pctsaturation",
    "chlorophyll_a_μg_l",
    "ammonia_nitrogen_mg_l",
    "5_day_biochemical_oxygen_demand_mg_l",
    "year",
    "month",
    "day"
]

label_cagio = "Label3_cagio"
label_hau = "Label3_hau"

# ========================================
# 4. TẠO X VÀ Y
# ========================================
X = df[feature_cols]
y_cagio = df[label_cagio]
y_hau = df[label_hau]

# ========================================
# 5. HÀM TRAIN + SCALER + LƯU MODEL
# ========================================
def train_save_model(X, y, model_name):

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ===== SCALER =====
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RandomForest
    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)

    # Predict
    y_pred = clf.predict(X_test_scaled)

    # Evaluation
    print(f"\n===== {model_name} =====")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model + scaler
    joblib.dump(clf, os.path.join(MODEL_DIR, f"{model_name}.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{model_name}.pkl"))

    print(f"Đã lưu mô hình {model_name} và scaler kèm theo!\n")

# ========================================
# 6. TRAIN 2 MÔ HÌNH
# ========================================
train_save_model(X, y_cagio, "rf_cagio")
train_save_model(X, y_hau, "rf_hau")
