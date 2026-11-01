import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

FEATURES = ["amount", "oldbalanceOrg", "newbalanceOrig", "tx_velocity_1h", "drain_ratio", "balance_error"]

def generate_transactions(n_rows: int = 5000) -> pd.DataFrame:
    np.random.seed(42)
    is_fraud = np.random.choice([0, 1], size=n_rows, p=[0.96, 0.04])
    old_bal = np.random.uniform(1000, 100000, size=n_rows).round(2)

    # Add realistic overlap so the trees learn non-trivial boundaries
    amount = np.where(
        is_fraud == 1,
        old_bal * np.random.uniform(0.65, 1.0, size=n_rows),
        old_bal * np.random.uniform(0.02, 0.78, size=n_rows)
    ).round(2)

    # 15% of normal transfers also empty an account (e.g., rent payment)
    new_bal = np.where(
        (is_fraud == 1) & (np.random.rand(n_rows) > 0.2),
        0.0,
        np.maximum(0.0, old_bal - amount)
    ).round(2)
    tx_per_hour = np.where(is_fraud == 1, np.random.randint(4, 15, n_rows), np.random.randint(1, 7, n_rows))

    df = pd.DataFrame({
        "amount": amount,
        "oldbalanceOrg": old_bal,
        "newbalanceOrig": new_bal,
        "tx_velocity_1h": tx_per_hour,
        "isFraud": is_fraud
    })
    df["drain_ratio"] = (df["amount"] / (df["oldbalanceOrg"] + 1.0)).round(4)
    df["balance_error"] = ((df["oldbalanceOrg"] - df["amount"]) - df["newbalanceOrig"]).abs().round(2)
    return df

def train_fraud_engine():
    df = generate_transactions()
    X, y = df[FEATURES], df["isFraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 1. Unsupervised Anomaly Detector (Isolation Forest)
    iso = IsolationForest(n_estimators=100, contamination=0.04, random_state=42)
    iso.fit(X_train)

    # 2. Supervised Gradient Boosted Decision Trees (LightGBM)
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1)
    clf.fit(X_train, y_train)

    # Evaluate Precision-Recall AUC on unseen test transactions
    probs = clf.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, probs)
    return clf, iso, round(pr_auc, 4)

MODEL_LGB, MODEL_ISO, METRIC_PRAUC = train_fraud_engine()

def score_transaction(amount: float, old_bal: float, new_bal: float, velocity_1h: int) -> dict:
    drain_ratio = round(amount / (old_bal + 1.0), 4)
    balance_error = round(abs((old_bal - amount) - new_bal), 2)
    row = pd.DataFrame([[amount, old_bal, new_bal, velocity_1h, drain_ratio, balance_error]], columns=FEATURES)

    lgb_prob = float(MODEL_LGB.predict_proba(row)[0][1])
    iso_anomaly = int(MODEL_ISO.predict(row)[0] == -1)  # -1 means outlier in Isolation Forest
    risk_score = round(min(1.0, lgb_prob * 0.85 + iso_anomaly * 0.15), 4)

    return {
        "risk_score": risk_score,
        "is_flagged": risk_score >= 0.70,
        "drain_ratio": drain_ratio,
        "balance_error": balance_error
    }

if __name__ == "__main__":
    print(f"Model Trained | Test Set PR-AUC: {METRIC_PRAUC}")
    normal_tx = score_transaction(amount=2500, old_bal=45000, new_bal=42500, velocity_1h=2)
    scam_tx = score_transaction(amount=48500, old_bal=49000, new_bal=0.0, velocity_1h=9)
    print("Normal UPI Transfer ->", normal_tx)
    print("Mule Scam Transfer  ->", scam_tx)