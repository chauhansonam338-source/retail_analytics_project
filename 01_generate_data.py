"""
01_generate_data.py
--------------------
Generates a realistic 2-year retail transactions dataset for a multi-category
e-commerce business operating across Indian metro regions.

Why synthetic data (and why it's still a legitimate analytics project):
Real transactional datasets of this kind are proprietary. To build a portfolio
project with full control over ground truth (so churn/segmentation results can
be validated), we simulate data with realistic statistical structure:
- Seasonality (festive season spikes: Oct-Nov, New Year)
- Category-wise price/margin differences
- Customer heterogeneity (loyal vs one-time vs lapsing)
- Regional demand differences
- Natural churn behavior baked into purchase recency patterns

This mirrors how a real analyst would validate a pipeline before pointing it
at production data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

N_CUSTOMERS = 3000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 12, 31)
N_DAYS = (END_DATE - START_DATE).days

REGIONS = ["Mumbai", "Delhi NCR", "Bengaluru", "Pune", "Hyderabad", "Chennai", "Kolkata"]
REGION_WEIGHTS = [0.22, 0.20, 0.18, 0.12, 0.11, 0.10, 0.07]

CATEGORIES = {
    "Electronics":     {"price_range": (799, 45000), "margin": 0.12, "weight": 0.20},
    "Fashion":         {"price_range": (299, 4500),   "margin": 0.35, "weight": 0.25},
    "Home & Kitchen":  {"price_range": (199, 12000),  "margin": 0.22, "weight": 0.18},
    "Beauty & Personal Care": {"price_range": (149, 3500), "margin": 0.30, "weight": 0.15},
    "Groceries":       {"price_range": (49, 1800),    "margin": 0.08, "weight": 0.12},
    "Sports & Fitness":{"price_range": (299, 15000),  "margin": 0.18, "weight": 0.10},
}
cat_names = list(CATEGORIES.keys())
cat_weights = [CATEGORIES[c]["weight"] for c in cat_names]

SEGMENT_PROFILE = {
    # name: (fraction of base, avg orders/year multiplier, churn tendency)
    "Loyal":      dict(frac=0.15, freq_mult=3.2, recency_bias=0.05),
    "Regular":    dict(frac=0.35, freq_mult=1.6, recency_bias=0.20),
    "Occasional": dict(frac=0.30, freq_mult=0.8, recency_bias=0.45),
    "One-time":   dict(frac=0.20, freq_mult=0.25, recency_bias=0.85),
}

def seasonal_multiplier(day_of_year):
    # Festive season boost (Diwali/New Year window ~ Oct 5 - Nov 15) and Jan New Year sales
    base = 1.0 + 0.15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # mild yearly wave
    festive = 1.0
    if 278 <= day_of_year <= 319:  # ~Oct5-Nov15
        festive = 2.4
    elif day_of_year <= 10:
        festive = 1.6
    return base * festive

# ---- Generate customers ----
customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, N_CUSTOMERS + 1)]
segments = rng.choice(
    list(SEGMENT_PROFILE.keys()),
    size=N_CUSTOMERS,
    p=[SEGMENT_PROFILE[s]["frac"] for s in SEGMENT_PROFILE]
)
regions_for_customers = rng.choice(REGIONS, size=N_CUSTOMERS, p=REGION_WEIGHTS)
signup_offsets = rng.integers(0, int(N_DAYS * 0.6), size=N_CUSTOMERS)  # most signed up in first 60% of window
signup_dates = [START_DATE + timedelta(days=int(o)) for o in signup_offsets]

customers_df = pd.DataFrame({
    "customer_id": customer_ids,
    "segment_true": segments,
    "region": regions_for_customers,
    "signup_date": signup_dates,
})

# ---- Generate transactions per customer ----
records = []
order_id_counter = 100000

for idx, row in customers_df.iterrows():
    cust_id = row["customer_id"]
    seg = row["segment_true"]
    profile = SEGMENT_PROFILE[seg]
    signup = row["signup_date"]
    active_days = (END_DATE - signup).days
    if active_days <= 0:
        continue

    base_annual_orders = 6 * profile["freq_mult"]
    expected_orders = max(1, int(rng.poisson(base_annual_orders * (active_days / 365))))

    # churn: occasional/one-time customers stop purchasing partway through
    churn_point = active_days
    if rng.random() < profile["recency_bias"]:
        churn_point = int(active_days * rng.uniform(0.15, 0.75))

    order_days = sorted(rng.integers(0, max(churn_point, 1), size=expected_orders))

    for od in order_days:
        order_date = signup + timedelta(days=int(od))
        if order_date > END_DATE:
            continue
        doy = order_date.timetuple().tm_yday
        mult = seasonal_multiplier(doy)
        # number of line items in this order
        n_items = rng.choice([1, 2, 3, 4], p=[0.45, 0.30, 0.16, 0.09])
        for _ in range(n_items):
            cat = rng.choice(cat_names, p=cat_weights)
            lo, hi = CATEGORIES[cat]["price_range"]
            price = float(np.round(rng.uniform(lo, hi) * min(mult, 1.8) / max(mult * 0.3, 1) + rng.uniform(lo, hi)*0.3, 2))
            price = float(np.clip(price, lo * 0.8, hi * 1.3))
            qty = int(rng.choice([1, 1, 1, 2, 3], p=[0.55, 0.15, 0.15, 0.1, 0.05]))
            discount_pct = float(np.round(rng.choice([0, 0, 0.05, 0.10, 0.15, 0.20],
                                                        p=[0.5, 0.15, 0.15, 0.1, 0.06, 0.04]), 2)
                                  if doy < 278 or doy > 319 else rng.choice([0.10, 0.20, 0.30, 0.40]))
            revenue = round(price * qty * (1 - discount_pct), 2)
            records.append({
                "order_id": f"ORD{order_id_counter}",
                "customer_id": cust_id,
                "order_date": order_date.strftime("%Y-%m-%d"),
                "region": row["region"],
                "category": cat,
                "unit_price": round(price, 2),
                "quantity": qty,
                "discount_pct": discount_pct,
                "revenue": revenue,
                "margin_pct": CATEGORIES[cat]["margin"],
            })
        order_id_counter += 1

transactions_df = pd.DataFrame(records)
transactions_df["order_date"] = pd.to_datetime(transactions_df["order_date"])
transactions_df = transactions_df.sort_values("order_date").reset_index(drop=True)

print(f"Customers: {len(customers_df)}")
print(f"Transactions (line items): {len(transactions_df)}")
print(f"Unique orders: {transactions_df['order_id'].nunique()}")
print(f"Date range: {transactions_df['order_date'].min()} to {transactions_df['order_date'].max()}")
print(f"Total revenue: ₹{transactions_df['revenue'].sum():,.0f}")

customers_df.to_csv("customers.csv", index=False)
transactions_df.to_csv("transactions.csv", index=False)
print("\nSaved customers.csv and transactions.csv")
