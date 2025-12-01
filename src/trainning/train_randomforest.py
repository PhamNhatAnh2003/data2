import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_save_rf(X, y, model_name, model_dir):
    """
    Train Random Forest, chuẩn hóa dữ liệu, lưu model + scaler
    """
    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Chuẩn hóa
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)

    # Đánh giá
    y_pred = clf.predict(X_test_scaled)
    print(f"\n===== {model_name} (Random Forest) =====")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Lưu model và scaler
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, f"rf_{model_name}.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, f"scaler_rf_{model_name}.pkl"))
    print(f"Đã lưu mô hình Random Forest {model_name} và scaler.\n")
