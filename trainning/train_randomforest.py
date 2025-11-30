import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# --------- Cấu hình đường dẫn ----------
DATA_PATH = r"C:\Users\Admin\Desktop\Data-science\data2\nhan\water_labeled_both_species.csv"
MODEL_DIR = r"C:\Users\Admin\Desktop\Data-science\data2\models"

os.makedirs(MODEL_DIR, exist_ok=True)

# --------- Load dữ liệu ----------
df = pd.read_csv(DATA_PATH)

# --------- Encode toàn bộ cột OBJECT ----------
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# --------- Features & Labels ----------
drop_cols = ['water_control_zone', 'station', 'dates', 'year', 'month', 'day', 'station_depth']
features = [col for col in df_encoded.columns if col not in drop_cols + ['Label3_cagio', 'Label3_hau']]
X = df_encoded[features]

# --------- Hàm train + lưu mô hình ---------
def train_save_model(X, y, model_name):
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Chuẩn hóa
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RandomForest
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)
    
    # Đánh giá
    print(f"\n=== {model_name} ===")
    print(confusion_matrix(y_test, y_pred := clf.predict(X_test_scaled)))
    print(classification_report(y_test, y_pred))
    
    # Lưu model + scaler
    joblib.dump(clf, os.path.join(MODEL_DIR, f"{model_name}.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{model_name}.pkl"))
    print(f"Đã lưu mô hình {model_name} và scaler!\n")

# --------- Train & Save Cá Giò ---------
train_save_model(X, df_encoded['Label3_cagio'], "rf_cagio")

# --------- Train & Save Hàu ---------
train_save_model(X, df_encoded['Label3_hau'], "rf_hau")
