import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    mean_absolute_error
)
from sklearn.preprocessing import label_binarize


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

    # Dự đoán
    y_pred = clf.predict(X_test_scaled)
    y_pred_prob = clf.predict_proba(X_test_scaled)  # dùng cho AUC

    print(f"\n===== {model_name} (Random Forest) =====")
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # ======= METRIC BỔ SUNG ========
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # ROC-AUC (đa lớp)
    classes = np.unique(y)
    try:
        y_test_binarized = label_binarize(y_test, classes=classes)
        roc_auc = roc_auc_score(
            y_test_binarized,
            y_pred_prob,
            multi_class="ovr"
        )
    except Exception:
        roc_auc = "Không tính được (có thể tập test chỉ có 1 lớp)"

    print("\n=== Metrics nâng cao ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Cohen Kappa: {kappa:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"MAE (phân loại): {mae:.4f}")
    print(f"ROC-AUC (multi-class OVR): {roc_auc}")

    print("\n--------------------------\n")

    # Lưu model và scaler
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, f"rf_{model_name}.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, f"scaler_rf_{model_name}.pkl"))
    print(f"Đã lưu mô hình Random Forest {model_name} và scaler.\n")
