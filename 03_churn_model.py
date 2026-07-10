"""
03_churn_model.py
------------------
Binary churn classifier: will a customer become inactive (no purchase in the
next 90 days from the snapshot date) based on their behavioral features?

Label definition:
  churned = 1 if customer's last order was more than 90 days before snapshot
            AND customer had signed up more than 90 days before snapshot
            (so we don't unfairly label brand-new customers as churned)

Features engineered from RFM + transaction history:
  recency, frequency, monetary, avg_order_value, tenure_days,
  distinct_categories, avg_discount_used, region (one-hot)

Model: Random Forest (robust to feature scale, gives interpretable importances)
Evaluated with train/test split, ROC-AUC, precision/recall, confusion matrix.
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, accuracy_score)

customers = pd.read_csv("customers.csv", parse_dates=["signup_date"])
tx = pd.read_csv("transactions.csv", parse_dates=["order_date"])
rfm = pd.read_csv("rfm_output.csv", parse_dates=["last_order"])

SNAPSHOT_DATE = tx["order_date"].max() + pd.Timedelta(days=1)

orders = tx.groupby("order_id").agg(
    customer_id=("customer_id", "first"),
    order_date=("order_date", "first"),
    order_value=("revenue", "sum"),
    n_categories=("category", "nunique"),
    avg_discount=("discount_pct", "mean"),
).reset_index()

cust_features = orders.groupby("customer_id").agg(
    frequency=("order_id", "count"),
    monetary=("order_value", "sum"),
    avg_order_value=("order_value", "mean"),
    distinct_categories=("n_categories", "max"),
    avg_discount_used=("avg_discount", "mean"),
    last_order=("order_date", "max"),
).reset_index()

cust_features = cust_features.merge(customers[["customer_id", "region", "signup_date"]], on="customer_id")
cust_features["recency"] = (SNAPSHOT_DATE - cust_features["last_order"]).dt.days
cust_features["tenure_days"] = (SNAPSHOT_DATE - cust_features["signup_date"]).dt.days

# Only include customers with enough tenure to fairly judge churn (>=90 days)
eligible = cust_features[cust_features["tenure_days"] >= 90].copy()
eligible["churned"] = (eligible["recency"] > 90).astype(int)

print(f"Eligible customers for churn modeling: {len(eligible)}")
print(f"Churn rate: {eligible['churned'].mean():.2%}")

feature_cols = ["frequency", "monetary", "avg_order_value", "distinct_categories",
                 "avg_discount_used", "tenure_days"]

le = LabelEncoder()
eligible["region_enc"] = le.fit_transform(eligible["region"])
feature_cols_full = feature_cols + ["region_enc"]

X = eligible[feature_cols_full]
y = eligible["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=5,
    class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1": round(f1_score(y_test, y_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
}

cm = confusion_matrix(y_test, y_pred).tolist()
fpr, tpr, _ = roc_curve(y_test, y_proba)
# downsample ROC curve points for a clean chart
idx = np.linspace(0, len(fpr) - 1, min(50, len(fpr))).astype(int)

feature_importance = sorted(
    zip(feature_cols_full, model.feature_importances_.round(4).tolist()),
    key=lambda x: -x[1]
)

print("\nModel performance:")
for k, v in metrics.items():
    print(f"  {k}: {v}")
print("\nFeature importances:")
for f, imp in feature_importance:
    print(f"  {f}: {imp}")

churn_output = {
    "metrics": metrics,
    "confusion_matrix": cm,
    "roc_curve": {
        "fpr": fpr[idx].round(4).tolist(),
        "tpr": tpr[idx].round(4).tolist(),
    },
    "feature_importance": {
        "labels": [f for f, _ in feature_importance],
        "values": [v for _, v in feature_importance],
    },
    "churn_rate_overall": round(eligible["churned"].mean(), 4),
    "n_eligible": int(len(eligible)),
}

with open("churn_model_output.json", "w") as f:
    json.dump(churn_output, f)

# merge into main dashboard json
with open("dashboard_data.json") as f:
    dashboard = json.load(f)
dashboard["churn"] = churn_output
with open("dashboard_data.json", "w") as f:
    json.dump(dashboard, f)

print("\nSaved churn_model_output.json and merged into dashboard_data.json")
