import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# ========================================
# 1. LOAD DATA
# ========================================
DATA_PATH = "..\\nhan\\water_labeled_both_species.csv"  # đổi path nếu cần
df = pd.read_csv(DATA_PATH)

# ========================================
# 2. CHỌN BIẾN CỤ THỂ
# ========================================
selected_features = [
    "temperature_°c",
    "salinity_psu",
    "ph",
    "chlorophyll_a_μg_l",
    "dissolved_oxygen_mg_l"
]

# ========================================
# 3. HISTOGRAM – Selected Features
# ========================================
plt.figure(figsize=(14, 10))
df[selected_features].hist(bins=30, figsize=(14, 10))
plt.suptitle("Histogram – Selected Features", fontsize=16)
plt.tight_layout()
plt.show()

# ========================================
# 4. BOXPLOT – Selected Features
# ========================================
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[selected_features])
plt.xticks(rotation=45)
plt.title("Boxplot – Selected Features")
plt.show()

# ========================================
# 5. HEATMAP – Correlation Selected Features
# ========================================
plt.figure(figsize=(10, 8))
sns.heatmap(df[selected_features].corr(), cmap="coolwarm", annot=True)
plt.title("Correlation Heatmap – Selected Features")
plt.show()

# ========================================
# 6. SCATTER MATRIX – Selected Features
# ========================================
scatter_matrix(df[selected_features], figsize=(10, 10), diagonal='kde')
plt.suptitle("Scatter Matrix – Selected Features")
plt.show()

# ========================================
# 7. COUNT PLOT LABEL
# ========================================
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
sns.countplot(x=df["Label3_cagio"])
plt.title("Label Count - Cá Giò")

plt.subplot(1, 2, 2)
sns.countplot(x=df["Label3_hau"])
plt.title("Label Count - Hàu")

plt.tight_layout()
plt.show()

# ========================================
# 8. VISUALIZE THEO ĐỊA ĐIỂM HOẶC ZONE
# ========================================
# Ví dụ lọc theo station
station_name = "ST03"  # đổi theo file của bạn
df_loc = df[df["station"] == station_name]

plt.figure(figsize=(14, 6))
df_loc[selected_features].hist(bins=20, figsize=(14, 6))
plt.suptitle(f"Histogram – Selected Features (Station {station_name})", fontsize=16)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_loc[selected_features])
plt.xticks(rotation=45)
plt.title(f"Boxplot – Selected Features (Station {station_name})")
plt.show()

# ========================================
# 9. VISUALIZE THEO THỜI GIAN
# ========================================
# Lineplot: Temperature theo năm
plt.figure(figsize=(10,5))
sns.lineplot(data=df, x="year", y="temperature_°c", marker="o")
plt.title("Nhiệt độ theo năm")
plt.show()

# Boxplot: Salinity theo tháng
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x="month", y="salinity_psu")
plt.title("Độ mặn theo tháng")
plt.show()
