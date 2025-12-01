import pandas as pd
import os
import joblib

# ========================================
# 1. PATH
# ========================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "new_water_data.csv")  # dữ liệu mới
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

# ========================================
# 2. LOAD DATA
# ========================================
df_new = pd.read_csv(DATA_PATH)

# Các feature giống như khi train
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

X_new = df_new[feature_cols]

# ========================================
# 3. LOAD MODELS + SCALERS
# ========================================
models = {
    "rf_cagio": joblib.load(os.path.join(MODEL_DIR, "rf_cagio.pkl")),
    "xgb_cagio": joblib.load(os.path.join(MODEL_DIR, "xgb_cagio.pkl")),
    "rf_hau": joblib.load(os.path.join(MODEL_DIR, "rf_hau.pkl")),
    "xgb_hau": joblib.load(os.path.join(MODEL_DIR, "xgb_hau.pkl"))
}

scalers = {
    "rf_cagio": joblib.load(os.path.join(MODEL_DIR, "scaler_rf_cagio.pkl")),
    "xgb_cagio": joblib.load(os.path.join(MODEL_DIR, "scaler_xgb_cagio.pkl")),
    "rf_hau": joblib.load(os.path.join(MODEL_DIR, "scaler_rf_hau.pkl")),
    "xgb_hau": joblib.load(os.path.join(MODEL_DIR, "scaler_xgb_hau.pkl"))
}

# ========================================
# 4. SCALE DỮ LIỆU
# ========================================
X_scaled = {}
for key in scalers:
    X_scaled[key] = scalers[key].transform(X_new)

# ========================================
# 5. DỰ BÁO
# ========================================
predictions = {
    "rf_cagio": models["rf_cagio"].predict(X_scaled["rf_cagio"]),
    "xgb_cagio": models["xgb_cagio"].predict(X_scaled["xgb_cagio"]),
    "rf_hau": models["rf_hau"].predict(X_scaled["rf_hau"]),
    "xgb_hau": models["xgb_hau"].predict(X_scaled["xgb_hau"])
}

# ========================================
# 6. IN KẾT QUẢ RA TERMINAL
# ========================================
for i in range(len(df_new)):
    print(f"Mẫu {i+1}:")
    print(f"  Cá Giò - RF: {predictions['rf_cagio'][i]}, XGB: {predictions['xgb_cagio'][i]}")
    print(f"  Hàu   - RF: {predictions['rf_hau'][i]}, XGB: {predictions['xgb_hau'][i]}")
    print("-"*40)
