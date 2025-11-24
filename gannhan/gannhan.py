import pandas as pd

# Đọc file
input_file = r"C:\Users\Admin\Desktop\Data_science\data\raw\marine_merged_cleaned.csv"
df = pd.read_csv(input_file, sep=",", low_memory=False)

# Chuyển các cột quan trọng sang float, lỗi -> NaN
cols_to_numeric = ['Dissolved Oxygen (mg/L)', 'pH', 'Temperature (°C)',
                   'Salinity (psu)', 'Secchi Disc Depth (M)',
                   'Unionised Ammonia (mg/L)', 'Total Sulphide (mg/kg)']

for col in cols_to_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Hàm gán nhãn 3 lớp
def label_species(row, species='Fish'):
    score = 0

    do = row.get('Dissolved Oxygen (mg/L)')
    if pd.notna(do):
        score += 1 if (do >= 6 if species=='Fish' else do >= 5) else 0

    pH = row.get('pH')
    if pd.notna(pH):
        score += 1 if 7 <= pH <= 8.5 else 0

    temp = row.get('Temperature (°C)')
    if pd.notna(temp):
        score += 1 if 15 <= temp <= 30 else 0

    sal = row.get('Salinity (psu)')
    if pd.notna(sal):
        if species=='Fish' and 5 <= sal <= 35:
            score += 1
        elif species=='Oyster' and 10 <= sal <= 30:
            score += 1

    secchi = row.get('Secchi Disc Depth (M)')
    if pd.notna(secchi) and secchi >= 1:
        score += 1

    nh3 = row.get('Unionised Ammonia (mg/L)')
    if pd.notna(nh3) and nh3 <= 0.5:
        score += 1

    h2s = row.get('Total Sulphide (mg/kg)')
    if pd.notna(h2s) and h2s <= 0.1:
        score += 1

    # Gán nhãn 3 lớp
    if score >= 6:
        return 'Good'
    elif score >= 4:
        return 'Moderate'
    else:
        return 'Poor'

# Áp dụng
df['Fish_Label'] = df.apply(lambda row: label_species(row, 'Fish'), axis=1)
df['Oyster_Label'] = df.apply(lambda row: label_species(row, 'Oyster'), axis=1)

# Lưu file
output_file = r"C:\Users\Admin\Desktop\Data_science\data\raw\marine_labeled_multi2.csv"
df.to_csv(output_file, index=False)

print("Đã tạo file gán nhãn nhiều chỉ tiêu:", output_file)
