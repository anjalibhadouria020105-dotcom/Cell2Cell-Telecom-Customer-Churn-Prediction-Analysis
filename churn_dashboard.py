"""
Cell2Cell Telecom Customer Churn Prediction & BI Dashboard
Dataset: cell2celltrain.csv (~51,000 rows, ~58 columns)
Run:     streamlit run churn_dashboard.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# NOTE ON RETENTION COLUMNS
# RetentionCalls, RetentionOffersAccepted, MadeCallToRetentionTeam
# are EXCLUDED from model features because they are post-hoc signals:
# a customer must already be expressing intent to leave before these
# events occur.  Including them would cause data leakage and produce
# an artificially inflated ROC-AUC. They are kept in the dataframe
# for KPI display only.
# ─────────────────────────────────────────────────────────────

RETENTION_COLS = [
    "RetentionCalls", "RetentionOffersAccepted", "MadeCallToRetentionTeam"
]

# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    candidates = ["cell2celltrain.csv", "Cell2CellTrain.csv",
                  "cell2cell_train.csv", "cell2cell.csv"]
    for f in candidates:
        if os.path.exists(f):
            return pd.read_csv(f, low_memory=False)
    # Try any CSV in the current directory
    csvs = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
    if csvs:
        return pd.read_csv(csvs[0], low_memory=False)
    st.error(
        "No CSV file found. Place cell2celltrain.csv in the same "
        "directory as churn_dashboard.py and restart."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────
# 2. DATA CLEANING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def clean_and_engineer(df_raw: pd.DataFrame):
    df = df_raw.copy()
    log = {}

    # ── 2a. Normalise column names ────────────────────────────
    df.columns = [c.strip() for c in df.columns]

    # ── 2b. Drop rows where Churn is missing (holdout rows) ──
    if "Churn" in df.columns:
        before = len(df)
        df = df[df["Churn"].notna() & (df["Churn"].astype(str).str.strip() != "")]
        log["churn_blank_rows_dropped"] = before - len(df)
    else:
        st.error("Column 'Churn' not found in the CSV.")
        st.stop()

    # ── 2c. Churn → 0/1 ──────────────────────────────────────
    df["Churn"] = df["Churn"].astype(str).str.strip().str.upper()
    df["Churn"] = df["Churn"].map({"YES": 1, "NO": 0, "1": 1, "0": 0,
                                    "TRUE": 1, "FALSE": 0}).fillna(0).astype(int)

    # ── 2d. Remove exact duplicates ──────────────────────────
    dupes = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    log["duplicates_dropped"] = int(dupes)

    # ── 2e. Yes/No columns → 0/1 ─────────────────────────────
    yn_candidates = [
        "TruckOwner", "RVOwner", "Homeowner",
        "HandsetWebCapable", "BuysViaMailOrder",
        "RespondsToMailOffers", "OptOutMailings",
        "NonUSTravel", "OwnsComputer", "HasCreditCard",
        "NewCellphoneUser", "NotNewCellphoneUser",
        "MadeCallToRetentionTeam",
    ]
    for c in yn_candidates:
        if c in df.columns:
            df[c] = (
                df[c].astype(str).str.strip().str.upper()
                .map({"YES": 1, "NO": 0, "Y": 1, "N": 0,
                      "1": 1, "0": 0, "TRUE": 1, "FALSE": 0})
                .fillna(0).astype(int)
            )

    # ── 2f. HandsetPrice: "Unknown" → NaN, then coerce numeric ─
    if "HandsetPrice" in df.columns:
        df["HandsetPrice"] = (
            df["HandsetPrice"].astype(str).str.replace("[^0-9.]", "", regex=True)
            .replace("", np.nan)
        )
        df["HandsetPrice"] = pd.to_numeric(df["HandsetPrice"], errors="coerce")

    # ── 2g. Columns that are mostly empty (>60 % missing) ────
    mostly_empty = [c for c in df.columns
                    if df[c].isnull().mean() > 0.60]
    df.drop(columns=mostly_empty, inplace=True)
    log["mostly_empty_columns_dropped"] = mostly_empty

    # ── 2h. Fill numeric NaN with column median ──────────────
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for c in num_cols:
        if df[c].isnull().any():
            df[c].fillna(df[c].median(), inplace=True)

    # ── 2i. Fill categorical NaN with "Unknown" ─────────────
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        df[c].fillna("Unknown", inplace=True)
        df[c] = df[c].astype(str).str.strip()

    # ── 2j. Cap extreme outliers (1 %–99 % winsorisation) ────
    cap_cols = ["MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge",
                "OverageMinutes", "RoamingCalls"]
    for c in cap_cols:
        if c in df.columns:
            lo, hi = df[c].quantile(0.01), df[c].quantile(0.99)
            df[c] = df[c].clip(lo, hi)
    log["winsorised_columns"] = [c for c in cap_cols if c in df.columns]

    # ── 2k. ServiceArea: keep top 10, rest → "Other" ─────────
    if "ServiceArea" in df.columns:
        top10 = df["ServiceArea"].value_counts().nlargest(10).index.tolist()
        df["ServiceArea"] = df["ServiceArea"].where(
            df["ServiceArea"].isin(top10), other="Other"
        )

    # ── 2l. Derived: Tenure Band ─────────────────────────────
    if "MonthsInService" in df.columns:
        df["Tenure Band"] = pd.cut(
            df["MonthsInService"],
            bins=[-1, 6, 12, 24, 9999],
            labels=["0–6 mo", "7–12 mo", "13–24 mo", "25+ mo"],
            right=True,
        )

    # ── 2m. Derived: Care Call Band ──────────────────────────
    if "CustomerCareCalls" in df.columns:
        df["Care Call Band"] = pd.cut(
            df["CustomerCareCalls"],
            bins=[-1, 0, 2, 9999],
            labels=["0", "1–2", "3+"],
            right=True,
        )

    # ── 2n. Derived: Revenue Band ────────────────────────────
    if "MonthlyRevenue" in df.columns:
        df["Revenue Band"] = pd.qcut(
            df["MonthlyRevenue"], q=3,
            labels=["Low", "Medium", "High"],
            duplicates="drop",
        )

    log["rows_final"] = len(df)
    log["columns_final"] = df.shape[1]
    return df, log


# ─────────────────────────────────────────────────────────────
# 3. ENCODING FOR ML
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def encode_for_model(df: pd.DataFrame):
    df_m = df.copy()

    # Drop leakage / ID / derived band columns (not predictive for model)
    drop_always = (
        ["CustomerID", "Churn", "Tenure Band", "Care Call Band", "Revenue Band"]
        + RETENTION_COLS
    )
    df_m.drop(columns=[c for c in drop_always if c in df_m.columns], inplace=True)

    # One-hot encode high-cardinality categoricals
    ohe_cols = [c for c in ["CreditRating", "PrizmCode", "Occupation",
                             "MaritalStatus", "ServiceArea"]
                if c in df_m.columns]
    df_m = pd.get_dummies(df_m, columns=ohe_cols, drop_first=False, dtype=int)

    # Drop any remaining object columns (shouldn't be many)
    remaining_obj = df_m.select_dtypes(include="object").columns.tolist()
    df_m.drop(columns=remaining_obj, inplace=True)

    # Final safety: fill any stray NaN
    df_m.fillna(0, inplace=True)

    feature_cols = df_m.columns.tolist()
    X = df_m[feature_cols]
    y = df["Churn"].values
    return X, y, feature_cols


# ─────────────────────────────────────────────────────────────
# 4. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def train_models(df: pd.DataFrame):
    X, y, feature_cols = encode_for_model(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Logistic Regression ──────────────────────────────────
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42, C=0.1
        )),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    lr_prob = lr_pipe.predict_proba(X_test)[:, 1]

    # ── Random Forest ────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        max_depth=12, min_samples_leaf=20,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    def metrics(y_t, y_p, y_pr):
        return {
            "Accuracy":  round(accuracy_score(y_t, y_p), 4),
            "Precision": round(precision_score(y_t, y_p, zero_division=0), 4),
            "Recall":    round(recall_score(y_t, y_p, zero_division=0), 4),
            "F1":        round(f1_score(y_t, y_p, zero_division=0), 4),
            "ROC-AUC":   round(roc_auc_score(y_t, y_pr), 4),
        }

    lr_metrics = metrics(y_test, lr_pred, lr_prob)
    rf_metrics = metrics(y_test, rf_pred, rf_prob)

    best_model = rf if rf_metrics["ROC-AUC"] >= lr_metrics["ROC-AUC"] else lr_pipe
    best_name  = "Random Forest" if rf_metrics["ROC-AUC"] >= lr_metrics["ROC-AUC"] else "Logistic Regression"
    best_metrics = rf_metrics if best_name == "Random Forest" else lr_metrics

    # Feature importances always from RF
    importances = (
        pd.Series(rf.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
    )

    # Score all customers
    X_all, _, _ = encode_for_model(df)
    churn_prob = best_model.predict_proba(X_all)[:, 1]

    return (
        best_name, best_metrics, lr_metrics, rf_metrics,
        importances, churn_prob,
    )


# ─────────────────────────────────────────────────────────────
# 5. KPI COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame, churn_prob: np.ndarray) -> dict:
    df = df.copy()
    df["Churn_Prob"]      = churn_prob
    df["Predicted_Churn"] = (churn_prob >= 0.5).astype(int)

    overall_churn_rate = df["Churn"].mean() * 100
    avg_tenure = df["MonthsInService"].mean() if "MonthsInService" in df.columns else 0

    # MRR at risk: predicted churners who have NOT yet churned in labels
    at_risk = df[(df["Predicted_Churn"] == 1) & (df["Churn"] == 0)]
    mrr_at_risk = at_risk["MonthlyRevenue"].sum() if "MonthlyRevenue" in df.columns else 0

    # ── Churn by CreditRating ─────────────────────────────────
    credit_churn = pd.DataFrame()
    if "CreditRating" in df.columns:
        credit_churn = (
            df.groupby("CreditRating")["Churn"].mean()
            .reset_index()
            .rename(columns={"Churn": "Churn Rate"})
        )
        credit_churn["Churn Rate %"] = (credit_churn["Churn Rate"] * 100).round(1)

    # ── Churn by PrizmCode (area type) ───────────────────────
    prizm_churn = pd.DataFrame()
    if "PrizmCode" in df.columns:
        prizm_churn = (
            df.groupby("PrizmCode")["Churn"].mean()
            .reset_index()
            .rename(columns={"Churn": "Churn Rate"})
        )
        prizm_churn["Churn Rate %"] = (prizm_churn["Churn Rate"] * 100).round(1)

    # ── Churn by ServiceArea (top 10 + Other) ────────────────
    area_churn = pd.DataFrame()
    if "ServiceArea" in df.columns:
        area_churn = (
            df.groupby("ServiceArea")["Churn"].mean()
            .reset_index()
            .rename(columns={"Churn": "Churn Rate"})
            .sort_values("Churn Rate", ascending=False)
            .head(10)
        )
        area_churn["Churn Rate %"] = (area_churn["Churn Rate"] * 100).round(1)

    # ── Churn by Tenure Band ──────────────────────────────────
    tenure_churn = pd.DataFrame()
    if "Tenure Band" in df.columns:
        tenure_churn = (
            df.groupby("Tenure Band", observed=True)["Churn"].mean()
            .reset_index()
            .rename(columns={"Churn": "Churn Rate"})
        )
        tenure_churn["Churn Rate %"] = (tenure_churn["Churn Rate"] * 100).round(1)

    # ── Churn by Care Call Band ───────────────────────────────
    care_churn = pd.DataFrame()
    if "Care Call Band" in df.columns:
        care_churn = (
            df.groupby("Care Call Band", observed=True)["Churn"].mean()
            .reset_index()
            .rename(columns={"Churn": "Churn Rate"})
        )
        care_churn["Churn Rate %"] = (care_churn["Churn Rate"] * 100).round(1)

    # ── Highest-risk segment (CreditRating × Care Call Band) ─
    risk_segment = {}
    if "CreditRating" in df.columns and "Care Call Band" in df.columns:
        seg = df.groupby(["CreditRating", "Care Call Band"], observed=True).agg(
            Churn_Rate=("Churn", "mean"),
            MRR_at_Risk=("MonthlyRevenue",
                         lambda x: x[df.loc[x.index, "Predicted_Churn"] == 1].sum()),
            Count=("Churn", "count"),
        ).reset_index()
        top = seg.sort_values("Churn_Rate", ascending=False).iloc[0]
        risk_segment = {
            "label": f"Credit Rating: {top['CreditRating']} | Care Calls: {top['Care Call Band']}",
            "churn_rate": round(top["Churn_Rate"] * 100, 1),
            "mrr_at_risk": round(top["MRR_at_Risk"], 2),
            "count": int(top["Count"]),
        }

    # ── Growth opportunity: high-revenue, low-churn-risk ─────
    growth_opportunity = ""
    if "Revenue Band" in df.columns:
        rev_seg = df.groupby("Revenue Band", observed=True).agg(
            Churn_Rate=("Churn", "mean"),
            Avg_Revenue=("MonthlyRevenue", "mean"),
            Count=("Churn", "count"),
        ).reset_index()
        # High-revenue customers with below-average churn
        avg_cr = df["Churn"].mean()
        opp = rev_seg[
            (rev_seg["Revenue Band"] == "High") |
            (rev_seg["Churn_Rate"] < avg_cr)
        ].sort_values("Avg_Revenue", ascending=False).iloc[0]
        growth_opportunity = (
            f"'{opp['Revenue Band']}'-revenue customers churn at only "
            f"{round(opp['Churn_Rate']*100,1)}% — below the fleet average of "
            f"{round(avg_cr*100,1)}%. "
            "Deepening engagement (loyalty rewards, premium add-ons) with this "
            "segment can grow ARPU while maintaining low churn."
        )

    # ── Recommended retention action ─────────────────────────
    retention_action = ""
    if not care_churn.empty:
        worst = care_churn.sort_values("Churn Rate %", ascending=False).iloc[0]
        retention_action = (
            f"Customers in the '{worst['Care Call Band']}' customer-care-call "
            f"band churn at {worst['Churn Rate %']}%. "
            "Trigger an automatic case-escalation workflow when a customer's "
            "cumulative care calls reach 3, routing them to a dedicated "
            "retention specialist within 24 hours."
        )

    return {
        "overall_churn_rate": round(overall_churn_rate, 2),
        "avg_tenure": round(avg_tenure, 1),
        "mrr_at_risk": round(mrr_at_risk, 2),
        "credit_churn": credit_churn,
        "prizm_churn": prizm_churn,
        "area_churn": area_churn,
        "tenure_churn": tenure_churn,
        "care_churn": care_churn,
        "risk_segment": risk_segment,
        "growth_opportunity": growth_opportunity,
        "retention_action": retention_action,
        "df_scored": df,
    }


# ─────────────────────────────────────────────────────────────
# 6. STREAMLIT APP
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Cell2Cell Churn Intelligence",
        page_icon="📱",
        layout="wide",
    )

    st.markdown("""
    <style>
      .kpi-card{background:#f7f8fa;border-radius:10px;padding:18px 22px;
                border-left:5px solid #3b82d4;margin-bottom:8px;}
      .kpi-label{font-size:13px;color:#57606a;font-weight:500;}
      .kpi-value{font-size:28px;font-weight:700;color:#1f2328;}
      .kpi-sub{font-size:12px;color:#57606a;}
      .risk-box{background:#fff3cd;border-left:5px solid #f0ad4e;
                border-radius:8px;padding:16px;margin-bottom:12px;}
      .opp-box{background:#d4edda;border-left:5px solid #28a745;
               border-radius:8px;padding:16px;margin-bottom:12px;}
      .action-box{background:#cce5ff;border-left:5px solid #3b82d4;
                  border-radius:8px;padding:16px;}
      .note-box{background:#f8d7da;border-left:5px solid #dc3545;
                border-radius:8px;padding:12px;margin-bottom:12px;
                font-size:12px;}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("📱 Cell2Cell Churn")
        st.markdown("**Cell2Cell Telecom — Churn BI**")
        st.markdown("---")
        section = st.radio(
            "Navigate",
            ["Executive Overview",
             "Customer & Segment Analysis",
             "Risk, Opportunity & Action"],
        )
        st.markdown("---")
        st.caption("Model: Random Forest (class_weight='balanced')")
        st.caption("Dataset: Cell2Cell (~51,000 rows, ~58 cols)")
        st.markdown("""
        <div class="note-box">
        ⚠️ <b>Leakage guard:</b> RetentionCalls, RetentionOffersAccepted,
        MadeCallToRetentionTeam are <u>excluded</u> from model features
        (post-hoc signals). See README for details.
        </div>
        """, unsafe_allow_html=True)

    # ── Load, clean, model ────────────────────────────────────
    with st.spinner("Loading and processing ~51,000 rows — first run may take ~30 s…"):
        df_raw = load_raw()
        df, cleaning_log = clean_and_engineer(df_raw)
        (best_name, best_metrics, lr_metrics, rf_metrics,
         importances, churn_prob) = train_models(df)
        kpis = compute_kpis(df, churn_prob)
        df_scored = kpis["df_scored"]

    # ════════════════════════════════════════════════════════
    # A – EXECUTIVE OVERVIEW
    # ════════════════════════════════════════════════════════
    if section == "Executive Overview":
        st.header("📊 Executive Overview")

        c1, c2, c3, c4 = st.columns(4)
        pred_ch = int(df_scored["Predicted_Churn"].sum())
        total   = len(df_scored)

        with c1:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Overall Churn Rate</div>
              <div class="kpi-value">{kpis['overall_churn_rate']}%</div>
              <div class="kpi-sub">Actual labelled data</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">MRR at Risk</div>
              <div class="kpi-value">${kpis['mrr_at_risk']:,.0f}</div>
              <div class="kpi-sub">Non-churned, predicted high-risk</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Avg Tenure</div>
              <div class="kpi-value">{kpis['avg_tenure']} mo</div>
              <div class="kpi-sub">MonthsInService mean</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">Predicted Churners</div>
              <div class="kpi-value">{pred_ch:,}</div>
              <div class="kpi-sub">of {total:,} customers (prob ≥ 0.5)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Churn Rate by Tenure Band")
            if not kpis["tenure_churn"].empty:
                fig = px.bar(
                    kpis["tenure_churn"], x="Tenure Band", y="Churn Rate %",
                    color="Churn Rate %", color_continuous_scale="Blues",
                    text="Churn Rate %",
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(coloraxis_showscale=False, height=360)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Churn Rate by Customer Care Call Band")
            if not kpis["care_churn"].empty:
                fig2 = px.bar(
                    kpis["care_churn"], x="Care Call Band", y="Churn Rate %",
                    color="Churn Rate %", color_continuous_scale="Reds",
                    text="Churn Rate %",
                )
                fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig2.update_layout(coloraxis_showscale=False, height=360)
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("Model Comparison")
        model_df = pd.DataFrame({
            "Metric": list(lr_metrics.keys()),
            "Logistic Regression": list(lr_metrics.values()),
            "Random Forest": list(rf_metrics.values()),
        })
        st.dataframe(
            model_df.set_index("Metric").style.highlight_max(axis=1, color="#d4edda"),
            use_container_width=True,
        )
        st.caption(
            f"✅ Best model: **{best_name}** (highest ROC-AUC). "
            "Note: on this noisy dataset a ROC-AUC of 0.70–0.80 is realistic and honestly reported."
        )

        st.subheader("Churn Probability Distribution")
        fig3 = px.histogram(
            df_scored, x="Churn_Prob", nbins=50,
            color_discrete_sequence=["#3b82d4"],
            labels={"Churn_Prob": "Churn Probability"},
        )
        fig3.add_vline(x=0.5, line_dash="dash", line_color="red",
                       annotation_text="Threshold 0.5")
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # B – CUSTOMER & SEGMENT ANALYSIS
    # ════════════════════════════════════════════════════════
    elif section == "Customer & Segment Analysis":
        st.header("🔍 Customer & Segment Analysis")

        # Credit Rating
        if not kpis["credit_churn"].empty:
            st.subheader("Churn Rate by Credit Rating")
            fig_cr = px.bar(
                kpis["credit_churn"].sort_values("Churn Rate %", ascending=False),
                x="CreditRating", y="Churn Rate %",
                color="Churn Rate %", text="Churn Rate %",
                color_continuous_scale="Oranges",
            )
            fig_cr.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_cr.update_layout(coloraxis_showscale=False, height=360)
            st.plotly_chart(fig_cr, use_container_width=True)

        # PrizmCode
        if not kpis["prizm_churn"].empty:
            st.subheader("Churn Rate by Area Type (PrizmCode)")
            fig_pz = px.bar(
                kpis["prizm_churn"].sort_values("Churn Rate %", ascending=False),
                x="PrizmCode", y="Churn Rate %",
                color="Churn Rate %", text="Churn Rate %",
                color_continuous_scale="Purples",
            )
            fig_pz.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_pz.update_layout(coloraxis_showscale=False, height=360)
            st.plotly_chart(fig_pz, use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Churn by Tenure Band")
            if not kpis["tenure_churn"].empty:
                fig_tb = px.pie(
                    kpis["tenure_churn"],
                    names="Tenure Band", values="Churn Rate %",
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                )
                fig_tb.update_layout(height=340)
                st.plotly_chart(fig_tb, use_container_width=True)

        with col2:
            st.subheader("Churn by Care Call Band")
            if not kpis["care_churn"].empty:
                fig_cc = px.pie(
                    kpis["care_churn"],
                    names="Care Call Band", values="Churn Rate %",
                    hole=0.4,
                    color_discrete_sequence=["#28a745", "#ffc107", "#dc3545"],
                )
                fig_cc.update_layout(height=340)
                st.plotly_chart(fig_cc, use_container_width=True)

        st.markdown("---")

        # Dropped/Blocked calls scatter vs Churn Prob
        if "DroppedCalls" in df_scored.columns and "BlockedCalls" in df_scored.columns:
            st.subheader("Dropped vs Blocked Calls (Colour = Churn Probability)")
            sample = df_scored.sample(min(5000, len(df_scored)), random_state=1)
            fig_sc = px.scatter(
                sample, x="DroppedCalls", y="BlockedCalls",
                color="Churn_Prob",
                color_continuous_scale="RdYlGn_r",
                opacity=0.5,
                hover_data=["MonthlyRevenue", "MonthsInService", "Churn"],
                labels={"Churn_Prob": "Churn Prob"},
            )
            fig_sc.update_layout(height=400)
            st.plotly_chart(fig_sc, use_container_width=True)

        # Handset age vs Churn
        if "CurrentEquipmentDays" in df_scored.columns:
            st.subheader("Handset Age Distribution by Churn Status")
            fig_hd = px.histogram(
                df_scored, x="CurrentEquipmentDays",
                color=df_scored["Churn"].map({1: "Churned", 0: "Retained"}),
                barmode="overlay", nbins=40, opacity=0.7,
                color_discrete_map={"Churned": "#dc3545", "Retained": "#28a745"},
                labels={"CurrentEquipmentDays": "Equipment Age (days)"},
            )
            fig_hd.update_layout(height=360)
            st.plotly_chart(fig_hd, use_container_width=True)

        st.markdown("---")
        # Top feature importances
        st.subheader("Top Feature Importances (Random Forest)")
        top_feat = importances.head(15).reset_index()
        top_feat.columns = ["Feature", "Importance"]
        fig_fi = px.bar(
            top_feat.sort_values("Importance"),
            x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Teal",
        )
        fig_fi.update_layout(coloraxis_showscale=False, height=480)
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("---")
        # Correlation heatmap (numeric)
        st.subheader("Correlation Matrix (numeric columns)")
        num_df = df_scored.select_dtypes(include=np.number).drop(
            columns=["Churn_Prob", "Predicted_Churn"] + RETENTION_COLS,
            errors="ignore",
        )
        # Limit to 20 cols for readability
        if num_df.shape[1] > 20:
            top_corr_cols = (
                num_df.corr()["Churn"].abs()
                .sort_values(ascending=False)
                .head(20).index.tolist()
            )
            num_df = num_df[top_corr_cols]
        corr = num_df.corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # C – RISK, OPPORTUNITY & ACTION
    # ════════════════════════════════════════════════════════
    elif section == "Risk, Opportunity & Action":
        st.header("⚠️ Risk, Opportunity & Action")

        rs = kpis.get("risk_segment", {})
        if rs:
            st.subheader("Highest-Risk Customer Segment")
            st.markdown(f"""
            <div class="risk-box">
              <strong>Segment:</strong> {rs['label']}<br>
              <strong>Churn Rate:</strong> {rs['churn_rate']}%&nbsp;&nbsp;
              <strong>Customers:</strong> {rs['count']:,}&nbsp;&nbsp;
              <strong>MRR at Risk:</strong> ${rs['mrr_at_risk']:,.2f}
            </div>
            """, unsafe_allow_html=True)

        if kpis["growth_opportunity"]:
            st.subheader("Growth Opportunity")
            st.markdown(f"""
            <div class="opp-box">{kpis['growth_opportunity']}</div>
            """, unsafe_allow_html=True)

        if kpis["retention_action"]:
            st.subheader("Recommended Retention Action")
            st.markdown(f"""
            <div class="action-box">{kpis['retention_action']}</div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        # Service Area churn
        if not kpis["area_churn"].empty:
            st.subheader("Churn Rate by Service Area (Top 10)")
            fig_sa = px.bar(
                kpis["area_churn"].sort_values("Churn Rate %"),
                x="Churn Rate %", y="ServiceArea",
                orientation="h", text="Churn Rate %",
                color="Churn Rate %", color_continuous_scale="Reds",
            )
            fig_sa.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_sa.update_layout(coloraxis_showscale=False, height=420)
            st.plotly_chart(fig_sa, use_container_width=True)

        st.markdown("---")
        # High-risk table
        st.subheader("High-Risk Customer Records (Churn Probability ≥ 0.75)")
        display_cols = [c for c in [
            "CustomerID", "MonthsInService", "MonthlyRevenue",
            "CreditRating", "PrizmCode", "ServiceArea",
            "CustomerCareCalls", "DroppedCalls",
            "CurrentEquipmentDays", "Churn_Prob", "Churn",
        ] if c in df_scored.columns]
        high_risk = (
            df_scored[df_scored["Churn_Prob"] >= 0.75][display_cols]
            .sort_values("Churn_Prob", ascending=False)
            .head(100)
        )
        high_risk["Churn_Prob"] = high_risk["Churn_Prob"].round(3)
        st.dataframe(high_risk.reset_index(drop=True), use_container_width=True)

        st.markdown("---")
        # Tenure vs Revenue scatter
        if "MonthsInService" in df_scored.columns and "MonthlyRevenue" in df_scored.columns:
            st.subheader("Tenure vs Monthly Revenue (Colour = Churn Probability)")
            sample = df_scored.sample(min(6000, len(df_scored)), random_state=42)
            fig_rv = px.scatter(
                sample, x="MonthsInService", y="MonthlyRevenue",
                color="Churn_Prob",
                color_continuous_scale="RdYlGn_r",
                opacity=0.55,
                hover_data=["CreditRating", "CustomerCareCalls", "Churn"],
                labels={
                    "MonthsInService": "Tenure (months)",
                    "MonthlyRevenue": "Monthly Revenue ($)",
                    "Churn_Prob": "Churn Prob.",
                },
            )
            fig_rv.update_layout(height=440)
            st.plotly_chart(fig_rv, use_container_width=True)

        # Cleaning log
        with st.expander("📋 Data Cleaning Log"):
            st.json(cleaning_log)
            st.caption(
                "• Rows without Churn labels dropped (holdout rows). "
                "• Duplicates removed. "
                "• Yes/No columns mapped to 1/0. "
                "• HandsetPrice 'Unknown' → NaN → median imputed. "
                "• Columns >60% missing dropped. "
                "• Remaining numeric NaN → median; categorical NaN → 'Unknown'. "
                "• Revenue, minutes, overage, roaming capped at 1st–99th percentile. "
                "• ServiceArea grouped into top 10 + 'Other'. "
                "• Derived columns: Tenure Band, Care Call Band, Revenue Band. "
                "• RetentionCalls, RetentionOffersAccepted, MadeCallToRetentionTeam "
                "  excluded from model features (post-hoc leakage risk)."
            )


if __name__ == "__main__":
    main()
