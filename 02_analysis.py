"""
02_analysis.py
--------------
Core analytics:
1. EDA - revenue trends, category/region performance
2. RFM (Recency, Frequency, Monetary) customer segmentation
3. Cohort retention analysis (monthly signup cohorts)
Outputs a single dashboard_data.json consumed by the HTML dashboard.
"""

import pandas as pd
import numpy as np
import json

customers = pd.read_csv("customers.csv", parse_dates=["signup_date"])
tx = pd.read_csv("transactions.csv", parse_dates=["order_date"])

SNAPSHOT_DATE = tx["order_date"].max() + pd.Timedelta(days=1)

# ---------------------------------------------------------------
# 1. EDA
# ---------------------------------------------------------------
orders = tx.groupby("order_id").agg(
    customer_id=("customer_id", "first"),
    order_date=("order_date", "first"),
    region=("region", "first"),
    order_value=("revenue", "sum")
).reset_index()

monthly_revenue = tx.set_index("order_date").resample("ME")["revenue"].sum()
monthly_orders = orders.set_index("order_date").resample("ME")["order_id"].count()

category_revenue = tx.groupby("category")["revenue"].sum().sort_values(ascending=False)
category_margin = (tx.groupby("category").apply(
    lambda d: (d["revenue"] * d["margin_pct"]).sum(), include_groups=False
)).sort_values(ascending=False)

region_revenue = tx.groupby("region")["revenue"].sum().sort_values(ascending=False)
region_orders = orders.groupby("region")["order_id"].count().sort_values(ascending=False)

avg_order_value = orders["order_value"].mean()
total_revenue = tx["revenue"].sum()
total_orders = orders["order_id"].nunique()
total_customers = customers["customer_id"].nunique()

# ---------------------------------------------------------------
# 2. RFM Segmentation
# ---------------------------------------------------------------
rfm = orders.groupby("customer_id").agg(
    last_order=("order_date", "max"),
    frequency=("order_id", "count"),
    monetary=("order_value", "sum")
).reset_index()
rfm["recency"] = (SNAPSHOT_DATE - rfm["last_order"]).dt.days

# Score 1-5 (5 = best) using quantiles
rfm["R_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["RFM_sum"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

def segment_customer(row):
    r, f, m = row["R_score"], row["F_score"], row["M_score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "New / Promising"
    elif r <= 2 and f >= 3:
        return "At Risk"
    elif r <= 2 and f <= 2 and m <= 2:
        return "Lost"
    else:
        return "Needs Attention"

rfm["segment"] = rfm.apply(segment_customer, axis=1)
rfm = rfm.merge(customers[["customer_id", "region", "segment_true"]], on="customer_id", how="left")

segment_summary = rfm.groupby("segment").agg(
    customers=("customer_id", "count"),
    avg_monetary=("monetary", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_recency=("recency", "mean"),
).reset_index().sort_values("avg_monetary", ascending=False)

rfm.to_csv("rfm_output.csv", index=False)

# ---------------------------------------------------------------
# 3. Cohort Retention Analysis
# ---------------------------------------------------------------
orders_c = orders.merge(customers[["customer_id", "signup_date"]], on="customer_id")
orders_c["cohort_month"] = orders_c["signup_date"].dt.to_period("M")
orders_c["order_month"] = orders_c["order_date"].dt.to_period("M")
orders_c["cohort_index"] = (
    (orders_c["order_month"].dt.year - orders_c["cohort_month"].dt.year) * 12 +
    (orders_c["order_month"].dt.month - orders_c["cohort_month"].dt.month)
)

cohort_data = orders_c.groupby(["cohort_month", "cohort_index"])["customer_id"].nunique().reset_index()
cohort_pivot = cohort_data.pivot(index="cohort_month", columns="cohort_index", values="customer_id")
cohort_sizes = cohort_pivot.iloc[:, 0]
retention = cohort_pivot.divide(cohort_sizes, axis=0).round(3)

# limit to cohorts with enough history & first 6 months for a clean viz
retention_display = retention.iloc[:12, :7]

cohort_json = {
    "cohorts": [str(c) for c in retention_display.index],
    "months": [int(c) for c in retention_display.columns],
    "matrix": retention_display.fillna(0).values.round(3).tolist(),
}

# ---------------------------------------------------------------
# Assemble dashboard JSON
# ---------------------------------------------------------------
dashboard = {
    "kpis": {
        "total_revenue": round(total_revenue, 2),
        "total_orders": int(total_orders),
        "total_customers": int(total_customers),
        "avg_order_value": round(avg_order_value, 2),
    },
    "monthly_revenue": {
        "labels": [d.strftime("%b %Y") for d in monthly_revenue.index],
        "values": monthly_revenue.round(0).tolist(),
    },
    "monthly_orders": {
        "labels": [d.strftime("%b %Y") for d in monthly_orders.index],
        "values": monthly_orders.tolist(),
    },
    "category_revenue": {
        "labels": category_revenue.index.tolist(),
        "values": category_revenue.round(0).tolist(),
    },
    "category_margin": {
        "labels": category_margin.index.tolist(),
        "values": category_margin.round(0).tolist(),
    },
    "region_revenue": {
        "labels": region_revenue.index.tolist(),
        "values": region_revenue.round(0).tolist(),
    },
    "rfm_segments": {
        "labels": segment_summary["segment"].tolist(),
        "customers": segment_summary["customers"].tolist(),
        "avg_monetary": segment_summary["avg_monetary"].round(0).tolist(),
        "avg_frequency": segment_summary["avg_frequency"].round(2).tolist(),
        "avg_recency": segment_summary["avg_recency"].round(1).tolist(),
    },
    "cohort_retention": cohort_json,
}

with open("dashboard_data.json", "w") as f:
    json.dump(dashboard, f)

print("EDA + RFM + Cohort analysis complete.")
print("\nRFM Segment breakdown:")
print(segment_summary.to_string(index=False))
print(f"\nTotal revenue: ₹{total_revenue:,.0f} | Orders: {total_orders:,} | AOV: ₹{avg_order_value:,.0f}")
