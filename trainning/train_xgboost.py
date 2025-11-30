import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import xgboost as xgb
import numpy as np

# --------- Cấu hình đường dẫn ----------
DATA_PATH = r"C:\Users\Admin\Desktop\Data-science\data2\nhan\marine_water_combined.csv"
MODEL_DIR = r"C:\Users\Admin\Desktop\Data-science\data2\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --------- Load dữ liệu ----------
df = pd.read_csv(DATA_PATH)

# --------- Features & Labels ----------
drop_cols = ['water_control_zone','station','dates','year','month','day','station_depth']
features = [col for col in df.columns if col not in drop_cols + ['Label3_cagio','Label3_hau']]
X = df[features]

# --------- Hàm train + lưu mô hình XGBoost ---------
def train_save_xgb(X, y, model_name):
    # Chia train/test stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode label (XGBoost multi-class cần integer labels)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Tính class weights để cân bằng
    classes = np.unique(y_train_enc)
    counts = np.bincount(y_train_enc)
    class_weights = {i: sum(counts)/counts[i] for i in classes}
    sample_weight = np.array([class_weights[i] for i in y_train_enc])
    
    # Train XGBoost
    clf = xgb.XGBClassifier(
        objective='multi:softmax',  # multi-class classification
        num_class=len(classes),
        n_estimators=200,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    clf.fit(X_train_scaled, y_train_enc, sample_weight=sample_weight)
    
    # Dự đoán
    y_pred_enc = clf.predict(X_test_scaled)
    y_pred = le.inverse_transform(y_pred_enc)
    
    print(f"=== {model_name} ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Lưu model + scaler + label encoder
    joblib.dump(clf, os.path.join(MODEL_DIR, f"{model_name}.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{model_name}.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, f"labelencoder_{model_name}.pkl"))
    print(f"Đã lưu mô hình {model_name}, scaler và label encoder!\n")

# --------- Train & save Cá Giò ---------
train_save_xgb(X, df['Label3_cagio'], "xgb_cagio")

# --------- Train & save Hàu ---------
train_save_xgb(X, df['Label3_hau'], "xgb_hau")
