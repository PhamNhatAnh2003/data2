import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATA
df = pd.read_csv("..\\nhan\\water_labeled_both_species.csv")

# 2. CHỌN BIẾN QUAN TRỌNG
features = [
    "temperature_°c",
    "salinity_psu",
    "ph",
    "dissolved_oxygen_mg_l",
    "total_nitrogen_mg_l",
    "total_phosphorus_mg_l",
    "suspended_solids_mg_l",
    "chlorophyll_a_μg_l",
    "faecal_coliforms_cfu_100ml"
]

# 3. HISTOGRAM – Phân bố dữ liệu
plt.figure(figsize=(14,6))
df[features].hist(bins=25, figsize=(14,6))
plt.suptitle("Histogram – Important Environmental Variables", fontsize=16)
plt.tight_layout()
plt.show()

# 4. BOXPLOT – Phát hiện outlier
plt.figure(figsize=(12,5))
sns.boxplot(data=df[features])
plt.xticks(rotation=45)
plt.title("Boxplot – Important Environmental Variables")
plt.show()

# 5. CORRELATION HEATMAP
plt.figure(figsize=(10,8))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap – Important Environmental Variables")
plt.show()

# 6. COUNT PLOT cho nhãn
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.countplot(x=df["Label3_cagio"])
plt.title("Label Count - Cá Giò")

plt.subplot(1,2,2)
sns.countplot(x=df["Label3_hau"])
plt.title("Label Count - Hàu")
plt.tight_layout()
plt.show()
