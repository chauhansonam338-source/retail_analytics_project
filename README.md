# Retail Sales & Customer Intelligence Platform

An end-to-end analytics project: data engineering → exploratory analysis → customer segmentation → cohort retention → machine learning (churn prediction) → interactive dashboard.

## What it does

Analyzes 2 years of e-commerce transactions (3,000 customers, 33.7K orders, 6 product categories, 7 Indian metro regions) to answer four business questions:

1. **Where is revenue coming from, and when?** (seasonality, category mix, regional performance, margin vs. revenue — Electronics is the top revenue category but not the top *margin* category, which is the kind of insight that makes a dashboard useful rather than decorative)
2. **Who are the most valuable customers?** (RFM segmentation into 6 groups: Champions, Loyal, At Risk, Needs Attention, New/Promising, Lost)
3. **Are we retaining customers over time?** (monthly cohort retention analysis, tracked 6 months post-signup)
4. **Which customers are about to churn, and why?** (Random Forest classifier, 0.85 ROC-AUC, with feature importance showing purchase *frequency* and *monetary value* are the strongest churn predictors — more so than region or category breadth)

## Files

| File | Purpose |
|---|---|
| `01_generate_data.py` | Generates the synthetic-but-realistic transactions dataset (see note below) |
| `02_analysis.py` | EDA, RFM scoring, cohort retention computation |
| `03_churn_model.py` | Feature engineering + Random Forest churn classifier + evaluation |
| `customers.csv`, `transactions.csv` | The generated dataset |
| `rfm_output.csv` | Per-customer RFM scores and segment labels |
| `dashboard_data.json` | All computed metrics, pre-aggregated for the dashboard |
| `dashboard_final.html` | **The interactive dashboard** — open this in a browser |

## Why synthetic data (be upfront about this in interviews)

Real transactional data of this kind is proprietary to companies, so this project generates a synthetic dataset with **realistic statistical structure** rather than random noise:
- Festive-season demand spikes (Oct–Nov, New Year)
- Category-specific pricing and margins
- Four underlying customer behavior types (loyal / regular / occasional / one-time) that drive realistic churn patterns
- Regional demand weighting

This is a legitimate and common technique (as long as you disclose it) because it lets you **validate your pipeline against known ground truth** before ever pointing it at real data — you know what segments and churn patterns *should* emerge, so you can sanity-check that your RFM logic and ML model are actually working correctly. If asked "is this real data," say exactly this.

**To make it a "real" project later:** swap `01_generate_data.py`'s output for any real transactional CSV (Kaggle has several public e-commerce datasets — "Online Retail II," "Brazilian E-Commerce / Olist," or "Superstore Sales" all have the same customer/order/product shape) and re-run `02_analysis.py` and `03_churn_model.py` unchanged — they operate on the shape of the data (customer_id, order_date, revenue, category, region), not the specific values.

## How to talk about this project in an interview

- **The pipeline, not just the dashboard.** Be ready to explain RFM scoring (quantile-based, 1–5 scale on each of Recency/Frequency/Monetary), why cohort retention is computed month-over-month relative to signup date, and why churn is defined as "no purchase in 90 days" (a business-decided threshold, not universal).
- **The ML choice.** Random Forest was used over logistic regression because feature relationships here aren't linear and it gives interpretable feature importances without heavy tuning — good default for a first churn model. Mention you could also try XGBoost/LightGBM as a stated next step.
- **The metric choice.** ROC-AUC (0.85) was prioritized over raw accuracy because churn classes are imbalanced-ish and a business cares more about ranking customers by risk than a single threshold.
- **The "so what."** Each analysis ties to an action: At-Risk segment → win-back campaign; Lost segment → excluded from paid retargeting spend; churn-flagged customers → proactive discount/retention email.

## Next steps if you want to extend it

- Swap in a real Kaggle dataset (see above) to make it a live case study
- Add a sales forecasting model (Prophet or simple SARIMA) for next-quarter revenue
- Deploy the churn model as a small Flask/FastAPI endpoint (you already know this stack)
- Turn `dashboard_final.html` into a live Streamlit or Plotly Dash app if you want to show code-driven interactivity rather than a static export
