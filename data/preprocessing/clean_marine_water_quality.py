import pandas as pd
import numpy as np
import re

# -----------------------------
# 1. Đọc dữ liệu
# -----------------------------
df = pd.read_csv(r"..\raw\data_2.csv")

# -----------------------------
# 2. Chuẩn hóa tên cột
# -----------------------------
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace("(", "")
              .str.replace(")", "")
              .str.replace("-", "_")
              .str.replace("/", "_")
              .str.replace("%", "pct")
)

# -----------------------------
# 3. Chuẩn hóa ngày
# -----------------------------
df["dates"] = pd.to_datetime(df["dates"], errors="coerce", dayfirst=False)
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

# Áp dụng cho tất cả cột numeric (ngoại trừ identifier)
exclude_cols = ["water_control_zone", "station", "sample_no", "depth", "dates", "type"]
for col in df.columns:
    if col not in exclude_cols:
        df[col] = df[col].map(clean_special_values)

# -----------------------------
# 5. Loại bỏ cột missing > 85%
# -----------------------------
null_rate = df.isna().mean()
drop_cols = null_rate[null_rate > 0.85].index.tolist()
df = df.drop(columns=drop_cols)
print("Các cột bị loại:", drop_cols)

# -----------------------------
# 6. Tạo cột nhóm: station_depth
# -----------------------------
df["station_depth"] = df["water_control_zone"].astype(str) + "_" + df["station"].astype(str) + "_" + df["depth"].astype(str)

# -----------------------------
# 7. Median theo nhóm station_depth × month
# -----------------------------
num_cols = df.select_dtypes(include=np.number).columns.tolist()
df = df.sort_values(["station_depth", "dates"])

for col in num_cols:
    df[col] = df.groupby(["station_depth", "month"])[col].transform(lambda x: x.fillna(x.median(skipna=True)))

# -----------------------------
# 8. Điền median toàn cục
# -----------------------------
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# -----------------------------
# 9. FFill/BFill theo nhóm station_depth
# -----------------------------
for col in num_cols:
    df[col] = df.groupby("station_depth")[col].transform(lambda x: x.ffill().bfill())

# -----------------------------
# 10. Loại bỏ các cột không cần
# -----------------------------
remove_cols = ["sample_no"]  # hoặc thêm nếu muốn
df = df.drop(columns=[c for c in remove_cols if c in df.columns])

# -----------------------------
# 11. Kiểm tra dữ liệu
# -----------------------------
print("Số NaN còn lại:", df.isna().sum().sum())
print("Shape dataset:", df.shape)
print("Một số dòng đầu:")
print(df.head())

# -----------------------------
# 12. Lưu file sạch
# -----------------------------
df.to_csv("cleaned\marine_water_quality_clean.csv", index=False)
print(">>> Lưu xong file: marine_water_quality_clean.csv")
