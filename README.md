# 📱 Cell2Cell Telecom — Customer Churn Prediction & BI Dashboard

## Project Overview

This project is an end-to-end business intelligence solution for predicting and analysing customer churn for **Cell2Cell Telecom**, using the Cell2Cell Customer Churn dataset from Kaggle. It combines machine learning (scikit-learn) with an interactive dashboard (Streamlit + Plotly) to give business stakeholders a clear view of churn risk, key drivers, and actionable retention recommendations.

**Key features:**
- Comprehensive data cleaning: missing values, placeholder strings (`"Unknown"` in HandsetPrice, Yes/No text columns), outlier capping, mostly-empty column removal
- Derived features: Tenure Band, Care Call Band, Revenue Band
- Two ML models: Logistic Regression and Random Forest with automatic best-model selection
- Class-imbalance handling via `class_weight="balanced"` (dataset has ~29% churn)
- All results cached with `@st.cache_data` for fast re-renders on 51,000 rows
- Three-section interactive dashboard: Executive Overview, Customer & Segment Analysis, Risk/Opportunity/Action
- All code in a single file: `churn_dashboard.py`

---

## Dataset

**Cell2Cell Telecom Customer Churn Dataset** — available on Kaggle:  
[https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom)

Download **`cell2celltrain.csv`** (~51,000 rows, ~58 columns). If a separate holdout file (with a blank Churn column) is included, ignore it — it has no labels and cannot be used for training or evaluation.

Place `cell2celltrain.csv` in the **same directory** as `churn_dashboard.py`.

---

## How to Run the Project

### 1. Prerequisites
- Python 3.9 or later
- pip

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset file

Place `cell2celltrain.csv` in the same folder as `churn_dashboard.py`.

### 4. Launch the dashboard

```bash
streamlit run churn_dashboard.py
```

The app will open in your browser at `http://localhost:8501`.  
**First load takes ~30 seconds** while the model trains on 51,000 rows; subsequent navigation is instant thanks to Streamlit's cache.

---

## Model Used

| Model | Description |
|---|---|
| **Logistic Regression** | Baseline linear classifier inside a `StandardScaler` pipeline; regularisation strength `C=0.1` to prevent overfitting on the large one-hot-encoded feature space |
| **Random Forest** | Ensemble of 300 trees, `max_depth=12`, `min_samples_leaf=20`; typically the stronger performer |

Both models use `class_weight="balanced"` to handle the ~29% churn minority class.  
The model with the higher **ROC-AUC** on a stratified 20% holdout is automatically selected.

> **Honest note on performance:** The Cell2Cell dataset is inherently noisy — customer behaviour is hard to predict from billing and usage data alone. A ROC-AUC of 0.70–0.80 is realistic and is reported honestly. Accuracy alone is a misleading metric here because predicting "No churn" for every customer would yield ~71% accuracy without learning anything useful.

Feature importances are always drawn from the Random Forest regardless of which model wins.

---

## Retention Column Decision

The columns **`RetentionCalls`**, **`RetentionOffersAccepted`**, and **`MadeCallToRetentionTeam`** are **excluded from all model features**.

**Reason:** These variables are post-hoc — a customer must already be signalling intent to leave before the retention team contacts them. Including these features would cause **data leakage**: the model would learn to predict churn almost perfectly on training data, but it would fail in production (when these columns are zero or unknown for future customers who haven't yet called to cancel). They are retained in the dataframe for KPI display only.

---

## Data Cleaning Summary

| Issue | Decision |
|---|---|
| `HandsetPrice` contains `"Unknown"` strings | Strip non-numeric characters → NaN → impute with median |
| Yes/No text columns | Map `Yes→1`, `No→0` |
| `Churn` = `Yes`/`No` | Map to `1`/`0` |
| Columns >60% missing | Dropped entirely |
| Remaining numeric NaN | Filled with column median |
| Remaining categorical NaN | Filled with `"Unknown"` category |
| Revenue, minutes, overage, roaming extremes | Winsorised at 1st–99th percentile |
| `ServiceArea` (high cardinality) | Kept top 10 areas; rest → `"Other"` |
| Duplicates | Removed |

---

## Dashboard Sections

| Section | Contents |
|---|---|
| **Executive Overview** | 4 KPI cards, churn by tenure band & care call band, model comparison table, churn probability histogram |
| **Customer & Segment Analysis** | Churn by credit rating & PrizmCode, tenure/care-call pie charts, dropped-vs-blocked scatter, handset age histogram, top 15 feature importances, correlation heatmap |
| **Risk, Opportunity & Action** | Highest-risk segment card, growth opportunity, recommended action, churn by service area, high-risk customer table (prob ≥ 0.75), tenure × revenue scatter |

---

## Project Structure

```
.
├── churn_dashboard.py      ← All code: cleaning, ML, dashboard
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── churn_report.docx       ← Project report (Word)
└── cell2celltrain.csv      ← Dataset (download from Kaggle)
```
