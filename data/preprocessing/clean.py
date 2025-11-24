import pandas as pd

# ======== 1. Đọc dữ liệu ========
# Thay đường dẫn bằng file CSV của bạn
water_file = "data_2.csv"  # marine_water_quality.csv
df = pd.read_csv(r"C:\Users\Admin\Desktop\Data_science\data\raw\data_1.csv", on_bad_lines='skip')
print(df.columns.tolist())

sediment_file = "data_1.csv"  # marine_sedi_quality.csv

# Đọc CSV (bỏ các dòng lỗi nếu có)
df_water = pd.read_csv(water_file, sep=",", quotechar='"', on_bad_lines='skip')
df_sediment = pd.read_csv(sediment_file, sep=",", quotechar='"', on_bad_lines='skip')

# ======== 2. Chuẩn hóa kiểu dữ liệu ========
numeric_cols_water = [
    'Dissolved Oxygen (mg/L)', 'pH', 'Temperature (°C)', 'Salinity (psu)',
    'TSS (mg/L)', 'Chlorophyll-a (μg/L)'
]

numeric_cols_sediment = [col for col in df_sediment.columns if col not in ['Water Control Zone','Station','Dates','Sample No']]

for col in numeric_cols_water:
    if col in df_water.columns:
        df_water[col] = pd.to_numeric(df_water[col], errors='coerce')

for col in numeric_cols_sediment:
    if col in df_sediment.columns:
        df_sediment[col] = pd.to_numeric(df_sediment[col], errors='coerce')

# ======== 3. Kiểm tra khoảng giá trị hợp lý ========
df_water['Dissolved Oxygen (mg/L)'] = df_water['Dissolved Oxygen (mg/L)'].where(
    (df_water['Dissolved Oxygen (mg/L)'] >= 0) & (df_water['Dissolved Oxygen (mg/L)'] <= 20)
)
df_water['Temperature (°C)'] = df_water['Temperature (°C)'].where(
    (df_water['Temperature (°C)'] >= -2) & (df_water['Temperature (°C)'] <= 40)
)
df_water['Salinity (psu)'] = df_water['Salinity (psu)'].where(
    (df_water['Salinity (psu)'] >= 0) & (df_water['Salinity (psu)'] <= 42)
)

# ======== 4. Loại cột thiếu >85% ========
threshold = 0.85
df_water = df_water.loc[:, df_water.isna().mean() < threshold]
df_sediment = df_sediment.loc[:, df_sediment.isna().mean() < threshold]

# ======== 5. Điền giá trị thiếu theo nhóm (Station × Month) ========
for df in [df_water, df_sediment]:
    df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
    df['Month'] = df['Dates'].dt.month

    group_cols = ['Station', 'Month']
    numeric_cols = [col for col in df.columns if df[col].dtype in ['float64','int64']]
    
    for col in numeric_cols:
        df[col] = df.groupby(group_cols)[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

# ======== 6. Forward-fill / Backward-fill theo nhóm thời gian ========
def ffill_bfill_group(df, max_gap_days=14):
    df = df.sort_values(['Station','Dates'])
    filled = []
    for station, group in df.groupby('Station'):
        group = group.set_index('Dates')
        group_diff = group.index.to_series().diff().fillna(pd.Timedelta(seconds=0))
        # Chỉ ffill/bfill khi gap <= 14 ngày
        group = group.ffill(limit=None)
        group = group.bfill(limit=None)
        filled.append(group.reset_index())
    return pd.concat(filled).sort_values(['Station','Dates'])

df_water_cleaned = ffill_bfill_group(df_water)
df_sediment_cleaned = ffill_bfill_group(df_sediment)

# ======== 7. Merge water & sediment ========
key_cols = ['Water Control Zone','Station','Dates','Sample No']
df_merged = pd.merge(df_water_cleaned, df_sediment_cleaned, on=key_cols, how='outer', suffixes=('_water','_sediment'))

# ======== 8. Lưu file kết quả ========
output_path = r"C:\Users\Admin\Desktop\Data_science\data\raw\marine_merged_cleaned.csv"
df_merged.to_csv(output_path, index=False)
print(f"Đã lưu file sạch và merge: {output_path}")

