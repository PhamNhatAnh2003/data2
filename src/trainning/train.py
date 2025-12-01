import os
import pandas as pd
from train_randomforest import train_save_rf
from train_xgboost import train_save_xgb

def main():
    # Đường dẫn data và thư mục lưu model
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # DATA_PATH = os.path.join(BASE_DIR, 'data', 'water_labeled_both_species.csv')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')

    # Load dữ liệu
    df = pd.read_csv("..\\nhan\\water_labeled_both_species.csv")

    # Các biến đầu vào
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

    X = df[feature_cols]
    y_cagio = df[label_cagio]
    y_hau = df[label_hau]

    # Train Random Forest
    print("=== Training Random Forest ===")
    train_save_rf(X, y_cagio, "cagio", MODEL_DIR)
    train_save_rf(X, y_hau, "hau", MODEL_DIR)

    # Train XGBoost
    print("=== Training XGBoost ===")
    train_save_xgb(X, y_cagio, "cagio", MODEL_DIR)
    train_save_xgb(X, y_hau, "hau", MODEL_DIR)

if __name__ == "__main__":
    main()
