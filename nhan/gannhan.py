import pandas as pd
import numpy as np

# Load dữ liệu nước đã clean
df = pd.read_csv(r"..\data\preprocessing\cleaned\marine_water_quality_clean.csv")

# Chuyển các cột số về numeric
num_cols = [
    "dissolved_oxygen_mg_l", "temperature_°c", "ph", "salinity_psu",
    "alkalinity_mg_l", "secchi_disc_depth_m", "ammonia_nitrogen_mg_l", "hydrogen_sulfide_mg_l"
]

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)

# Hàm gán nhãn 3 lớp chung
def label3_species(row, species="cagio"):
    risk = 0
    warning = 0
    
    if species.lower() == "cagio":
        # Cá giò
        # DO
        if row['dissolved_oxygen_mg_l'] < 6:
            risk += 1
        # Temperature
        if row['temperature_°c'] < 24 or row['temperature_°c'] > 33:
            risk += 1
        elif 24 <= row['temperature_°c'] < 26 or 28 < row['temperature_°c'] <= 33:
            warning += 1
        # pH
        if row['ph'] < 7.5 or row['ph'] > 8.5:
            risk += 1
        # Salinity
        if row['salinity_psu'] < 27 or row['salinity_psu'] > 33:
            risk += 1
        # NH3
        if row['ammonia_nitrogen_mg_l'] >= 0.1:
            risk += 1

    elif species.lower() == "hau":
        # Hàu
        if row['dissolved_oxygen_mg_l'] < 5:
            risk += 1
        # Temperature
        if row['temperature_°c'] < 18 or row['temperature_°c'] > 33:
            risk += 1
        elif 18 <= row['temperature_°c'] < 20 or 28 < row['temperature_°c'] <= 33:
            warning += 1
        # pH
        if row['ph'] < 7.5 or row['ph'] > 8:
            risk += 1
        # Salinity
        if row['salinity_psu'] < 20 or row['salinity_psu'] > 25:
            risk += 1
        # Alkalinity
        if 'alkalinity_mg_l' in row and (row['alkalinity_mg_l'] < 60 or row['alkalinity_mg_l'] > 180):
            risk += 1
        # Secchi depth
        if 'secchi_disc_depth_m' in row and (row['secchi_disc_depth_m'] < 0.2 or row['secchi_disc_depth_m'] > 0.5):
            risk += 1
        # NH3
        if row['ammonia_nitrogen_mg_l'] >= 0.3:
            risk += 1
        # H2S
        if 'hydrogen_sulfide_mg_l' in row and row['hydrogen_sulfide_mg_l'] >= 0.05:
            risk += 1

    # Quy tắc gộp nhãn
    if risk > 0:
        return 2
    elif warning > 0:
        return 1
    else:
        return 0

# Áp dụng cho cá giò
df['Label3_cagio'] = df.apply(lambda row: label3_species(row, species="cagio"), axis=1)

# Áp dụng cho hàu
df['Label3_hau'] = df.apply(lambda row: label3_species(row, species="hau"), axis=1)

# Kiểm tra phân bố nhãn
print("Cá giò:\n", df['Label3_cagio'].value_counts())
print("Hàu:\n", df['Label3_hau'].value_counts())

# Lưu file gộp nhãn
df.to_csv("water_labeled_both_species.csv", index=False)
