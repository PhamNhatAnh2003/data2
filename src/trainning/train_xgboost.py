import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_save_xgb(X, y, model_name, model_dir):
    """
    Train XGBoost, chuẩn hóa dữ liệu, lưu model + scaler
    """
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Chuẩn hóa
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # XGBoost
    clf = XGBClassifier(
        n_estimators=300,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)

    # Đánh giá
    y_pred = clf.predict(X_test_scaled)
    print(f"\n===== {model_name} (XGBoost) =====")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Lưu model và scaler
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, f"xgb_{model_name}.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, f"scaler_xgb_{model_name}.pkl"))
    print(f"Đã lưu mô hình XGBoost {model_name} và scaler.\n")
