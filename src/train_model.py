import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import xgboost as xgb

from data_preprocessing import load_and_preprocess, split_features_labels

def train_baseline_logistic(df_encoded: pd.DataFrame, save_path: str):
    X_baseline = df_encoded[["age"]]
    y = df_encoded["target"]

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_baseline, y, test_size=0.2, random_state=42, stratify=y
    )

    logreg = LogisticRegression()
    logreg.fit(X_train_b, y_train_b)

    y_pred_proba_b = logreg.predict_proba(X_test_b)[:, 1]
    auc_b = roc_auc_score(y_test_b, y_pred_proba_b)
    acc_b = accuracy_score(y_test_b, (y_pred_proba_b > 0.5).astype(int))
    print(f"[Baseline Logistic] Test AUC: {auc_b:.3f}, Accuracy: {acc_b:.3f}")

    joblib.dump(logreg, save_path)
    print(f"Saved baseline logistic model to {save_path}")

def train_xgboost(df_encoded: pd.DataFrame, save_path: str):
    X_full, y_full = split_features_labels(df_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    xgb_model.fit(X_train, y_train)

    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, (y_pred_proba > 0.5).astype(int))
    print(f"[XGBoost] Test AUC: {auc:.3f}, Accuracy: {acc:.3f}")

    xgb_model.save_model(save_path)
    print(f"Saved XGBoost model to {save_path}")

if __name__ == "__main__":
    df_clean = load_and_preprocess("data/heart.csv")

    train_baseline_logistic(
        df_encoded=df_clean,
        save_path="models/logreg_baseline.joblib"
    )

    train_xgboost(
        df_encoded=df_clean,
        save_path="models/xgb_model.json"
    )
