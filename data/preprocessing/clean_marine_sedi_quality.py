import pandas as pd
import numpy as np
import re

# -----------------------------
# 1. Đọc dữ liệu
# -----------------------------
df = pd.read_csv(r"..\raw\data_1.csv")

# -----------------------------
# 2. Chuẩn hóa tên cột
# -----------------------------
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace("(", "")
              .str.replace(")", "")
              .str.replace("%", "pct")
              .str.replace("/", "_")
)

# -----------------------------
# 3. Chuẩn hóa ngày
# -----------------------------
df["dates"] = pd.to_datetime(df["dates"], errors="coerce")
df["year"] = df["dates"].dt.year
df["month"] = df["dates"].dt.month
df["day"] = df["dates"].dt.day

# -----------------------------
# 4. Xử lý giá trị đặc biệt
# -----------------------------
def clean_special_values(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip()
        if x.upper() in ["N/A", "NA", "", "NULL"]:
            return np.nan
        if x.startswith("<"):
            try:
                val = float(re.sub("[^0-9.]", "", x))
                return val / 2
            except:
                return np.nan
        if x.startswith(">"):
            try:
                val = float(re.sub("[^0-9.]", "", x))
                return val
            except:
                return np.nan
    try:
        return float(x)
    except:
        return np.nan

# Áp dụng cho tất cả cột numeric
for col in df.columns:
    if col not in ["water_control_zone", "station", "type", "dates"]:
        df[col] = df[col].map(clean_special_values)

# -----------------------------
# 5. Loại bỏ cột missing > 85%
# -----------------------------
null_rate = df.isna().mean()
drop_cols = null_rate[null_rate > 0.85].index.tolist()
df = df.drop(columns=drop_cols)
print("Các cột bị loại:", drop_cols)

# -----------------------------
# 6. Tạo cột nhóm zone_station
# -----------------------------
df["zone_station"] = df["water_control_zone"].astype(str) + "_" + df["station"].astype(str)

# -----------------------------
# 7. Median theo nhóm zone_station × month
# -----------------------------
num_cols = df.select_dtypes(include=np.number).columns.tolist()
df = df.sort_values(["zone_station", "dates"])

for col in num_cols:
    df[col] = df.groupby(["zone_station", "month"])[col] \
                 .transform(lambda x: x.fillna(x.median(skipna=True)))

# -----------------------------
# 8. Điền median toàn cục
# -----------------------------
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# -----------------------------
# 9. FFill/BFill theo nhóm zone_station
# -----------------------------
for col in num_cols:
    df[col] = df.groupby("zone_station")[col].transform(lambda x: x.ffill().bfill())

# -----------------------------
# 10. Loại bỏ các cột không cần
# -----------------------------
remove_cols = [
    "sample_no", "type", "particle_size_fraction_<63_micrometer_pctw_w", "dry_wet_ratio"
]
df = df.drop(columns=[c for c in remove_cols if c in df.columns])

# -----------------------------
# 11. Kiểm tra dữ liệu
# -----------------------------
print("Số NaN còn lại:", df.isna().sum().sum())
print("Shape dataset:", df.shape)
print("Một số dòng đầu tiên:")
print(df.head())

# -----------------------------
# 12. Lưu file sạch
# -----------------------------
df.to_csv("cleaned\marine_sedi_quality.csv", index=False)
print(">>> Lưu xong file: marine_sedi_quality.csv")
