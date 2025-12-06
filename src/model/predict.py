import pandas as pd
import os
import joblib

# ========================================
# 1. PATH
# ========================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..","..", "data", "data_1.csv")  # dữ liệu mới
MODEL_DIR = os.path.join(BASE_DIR,"..", "..", "models")

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
# ========================================
# 6. IN KẾT QUẢ RA TERMINAL VỚI NHẬN XÉT
# ========================================
for i in range(len(df_new)):
    rf_cagio = predictions['rf_cagio'][i]
    xgb_cagio = predictions['xgb_cagio'][i]
    rf_hau = predictions['rf_hau'][i]
    xgb_hau = predictions['xgb_hau'][i]

    # Xác định mức độ phù hợp dựa trên số lượng nhãn risk (2) / warning (1)
    def assess_level(rf_pred, xgb_pred):
        risk_count = sum([rf_pred == 2, xgb_pred == 2])
        warning_count = sum([rf_pred == 1, xgb_pred == 1])

        if risk_count >= 1:
            return "Low", "Không phù hợp, nguy cơ cao"
        elif warning_count >= 1:
            return "Medium", "Cẩn thận, mức độ phù hợp trung bình"
        else:
            return "High", "Phù hợp, an toàn"

    level_cagio, comment_cagio = assess_level(rf_cagio, xgb_cagio)
    level_hau, comment_hau = assess_level(rf_hau, xgb_hau)

    print(f"Mẫu {i+1}:")
    print(f"  Cá Giò - RF: {rf_cagio}, XGB: {xgb_cagio} | Mức độ: {level_cagio} | Nhận xét: {comment_cagio}")
    print(f"  Hàu   - RF: {rf_hau}, XGB: {xgb_hau} | Mức độ: {level_hau} | Nhận xét: {comment_hau}")
    print("-"*60)

