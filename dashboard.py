import os
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

px.defaults.template = "plotly_dark"

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="API Abuse - AI Cyber Resilience Dashboard",
    page_icon="",
    layout="wide",
)

# =========================
# THEME COLORS (Black + Purple/Blue)
# =========================
BG = "#020617"
CARD = "#0b1220"
BORDER = "#1e293b"
TEXT = "#e5e7eb"
MUTED = "#94a3b8"

RISK_COLORS = {
    "High": "#a855f7",    # purple
    "Medium": "#3b82f6",  # blue
    "Low": "#22d3ee",     # cyan
}

# =========================
# STYLING
# =========================
st.markdown(
    f"""
    <style>
      body {{ background-color: {BG}; }}
      .block-container {{ padding-top: 1.35rem; padding-bottom: 2.5rem; }}
      div[data-testid="stMetric"] {{
        background: {CARD};
        border: 1px solid {BORDER};
        padding: 16px;
        border-radius: 16px;
      }}
      .title {{
        font-size: 2.15rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        margin-bottom: .1rem;
        color: {TEXT};
      }}
      .subtitle {{ color: {MUTED}; margin-top: -6px; }}
      .card {{
        background: {CARD};
        border: 1px solid {BORDER};
        padding: 18px;
        border-radius: 18px;
        margin-bottom: 16px;
      }}
      .pill {{
        display:inline-block;
        padding:.35rem .7rem;
        border-radius:999px;
        background:{BG};
        border:1px solid #334155;
        color:{TEXT};
        font-size:.85rem;
      }}
      .muted {{ color: {MUTED}; }}
      hr {{ border: none; border-top: 1px solid {BORDER}; margin: 14px 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# HEADER
# =========================
st.markdown('<div class="title">API Abuse - AI Cyber Resilience Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Behaviour metrics and call-graph intelligence with Isolation Forest risk scoring</p>',
    unsafe_allow_html=True,
)

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

SUP_FILE = os.path.join(OUTPUTS_DIR, "supervised_scored.csv")
REM_FILE = os.path.join(OUTPUTS_DIR, "remaining_scored.csv")

DETECTOR_SCRIPT = os.path.join(BASE_DIR, "api_abuse_anomaly_detector.py")

# =========================
# HELPERS
# =========================
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def safe_cols(df: pd.DataFrame, cols):
    return [c for c in cols if c in df.columns]

def apply_dark_layout(fig, height=320):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color=TEXT,
        legend_title_text="",
    )
    return fig

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### SOC Controls")
    run_detector = st.button("Run Detection Pipeline", use_container_width=True)

    st.markdown("---")
    dataset_choice = st.radio(
        "Dataset",
        ["Remaining (Operational View)", "Supervised (Validation View)"],
        index=0,
    )

    st.markdown("---")
    top_n = st.slider("Top Suspicious Sessions", 5, 300, 25)
    score_threshold = st.slider("Anomaly Score Threshold (Filter)", 0.0, 1.0, 0.0, 0.01)

    st.markdown("---")
    st.markdown("### Risk Tier Settings")
    high_q = st.slider("High risk percentile", 0.85, 0.99, 0.95, 0.01)
    med_q = st.slider("Medium risk percentile", 0.50, 0.95, 0.80, 0.01)

    st.markdown("---")
    st.markdown("### Filters")
    filter_ip = st.checkbox("Filter by ip_type", value=False)
    filter_source = st.checkbox("Filter by source", value=False)

# =========================
# RUN PIPELINE
# =========================
if run_detector:
    with st.spinner("Running anomaly detection pipeline..."):
        result = subprocess.run(
            ["python", DETECTOR_SCRIPT],
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        st.error("Pipeline execution failed")
        st.code(result.stderr)
        st.stop()
    else:
        st.success("Pipeline executed successfully")
        with st.expander("View pipeline logs"):
            st.code(result.stdout)

# =========================
# LOAD DATA
# =========================
sup = load_csv(SUP_FILE)
rem = load_csv(REM_FILE)
df_raw = rem if dataset_choice.startswith("Remaining") else sup

if df_raw.empty:
    st.warning("No output CSVs found. Run the detector first to generate outputs/*.csv")
    st.stop()

required = ["_id", "anomaly_score", "is_anomaly"]
missing_req = [c for c in required if c not in df_raw.columns]
if missing_req:
    st.error(f"Missing required columns in CSV: {missing_req}")
    st.stop()

# Keep a copy of the unfiltered dataset for KPI integrity
df_all = df_raw.copy()

# Apply threshold filter for analysis views (but KPI anomaly rate will show both)
df = df_raw.copy()
if score_threshold > 0:
    df = df[df["anomaly_score"] >= score_threshold].copy()

# Optional categorical filters
if filter_ip and "ip_type" in df.columns:
    ip_selected = st.sidebar.multiselect(
        "Select ip_type",
        sorted(df["ip_type"].dropna().unique().tolist())
    )
    if ip_selected:
        df = df[df["ip_type"].isin(ip_selected)].copy()

if filter_source and "source" in df.columns:
    source_selected = st.sidebar.multiselect(
        "Select source",
        sorted(df["source"].dropna().unique().tolist())
    )
    if source_selected:
        df = df[df["source"].isin(source_selected)].copy()

# =========================
# RISK TIERS (PERCENTILE-BASED, DATA-DRIVEN)
# =========================
# Compute quantile cutoffs from the (filtered) df if possible; fallback to df_all
base_for_quantiles = df if len(df) >= 20 else df_all

p_med = float(base_for_quantiles["anomaly_score"].quantile(med_q))
p_high = float(base_for_quantiles["anomaly_score"].quantile(high_q))

def risk_tier(score: float) -> str:
    if score >= p_high:
        return "High"
    elif score >= p_med:
        return "Medium"
    return "Low"

df["risk_tier"] = df["anomaly_score"].apply(risk_tier)
df_all["risk_tier"] = df_all["anomaly_score"].apply(risk_tier)

df = df.sort_values("anomaly_score", ascending=False)

# =========================
# KPIs
# =========================
# IMPORTANT: Anomaly rate in KPI is based on df_all (unfiltered) so it reflects the dataset view
total_sessions = len(df_all)
flagged = int(df_all["is_anomaly"].sum())
anomaly_rate = (df_all["is_anomaly"].mean() * 100.0) if total_sessions else 0.0
max_score = float(df_all["anomaly_score"].max()) if total_sessions else 0.0
high_risk_count = int((df_all["risk_tier"] == "High").sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Sessions", f"{total_sessions:,}")
k2.metric("Flagged Anomalies", f"{flagged:,}")
k3.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
k4.metric("Max Score", f"{max_score:.4f}")
k5.metric("High Risk Count", f"{high_risk_count:,}")

st.markdown("<hr/>", unsafe_allow_html=True)

# =========================
# POSTURE OVERVIEW
# =========================
st.markdown("## Posture Overview")

row1a, row1b, row1c = st.columns([1.2, 1, 1])

with row1a:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Score Distribution")
    fig = px.histogram(
        df,
        x="anomaly_score",
        nbins=60,
        color="risk_tier",
        color_discrete_map=RISK_COLORS,
    )
    st.plotly_chart(apply_dark_layout(fig, 300), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row1b:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Risk Tiers")
    pie = px.pie(
        df,
        names="risk_tier",
        hole=0.55,
        color="risk_tier",
        color_discrete_map=RISK_COLORS,
    )
    st.plotly_chart(apply_dark_layout(pie, 300), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row1c:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Flagged vs Not Flagged")
    tmp = df.copy()
    tmp["flag_status"] = np.where(tmp["is_anomaly"] == 1, "Flagged", "Not Flagged")
    pie2 = px.pie(tmp, names="flag_status", hole=0.55)
    st.plotly_chart(apply_dark_layout(pie2, 300), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

row2a, row2b = st.columns([1, 1])

with row2a:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Score by Risk Tier")
    box = px.box(df, x="risk_tier", y="anomaly_score", color="risk_tier", color_discrete_map=RISK_COLORS)
    st.plotly_chart(apply_dark_layout(box, 300), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row2b:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Origin Breakdown")
    sub = st.columns(2)

    if "source" in df.columns:
        src = df.groupby(["source", "risk_tier"]).size().reset_index(name="count")
        fig_src = px.bar(
            src,
            x="source",
            y="count",
            color="risk_tier",
            barmode="stack",
            color_discrete_map=RISK_COLORS,
        )
        sub[0].plotly_chart(apply_dark_layout(fig_src, 290), use_container_width=True)
    else:
        sub[0].info("Column 'source' not found.")

    if "ip_type" in df.columns:
        ip = df.groupby(["ip_type", "risk_tier"]).size().reset_index(name="count")
        fig_ip = px.bar(
            ip,
            x="ip_type",
            y="count",
            color="risk_tier",
            barmode="stack",
            color_discrete_map=RISK_COLORS,
        )
        sub[1].plotly_chart(apply_dark_layout(fig_ip, 290), use_container_width=True)
    else:
        sub[1].info("Column 'ip_type' not found.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# =========================
# DRIVER ANALYSIS
# =========================
st.markdown("## Driver Analysis")

candidate_features = [
    "num_unique_apis",
    "sequence_length(count)",
    "vsession_duration(min)",
    "inter_api_access_duration(sec)",
    "api_access_uniqueness",
    "graph_num_nodes",
    "graph_num_edges",
    "graph_density",
    "graph_out_deg_mean",
    "graph_out_deg_max",
    "graph_in_deg_mean",
    "graph_in_deg_max",
]
available_numeric = [c for c in candidate_features if c in df.columns]

r3a, r3b = st.columns([1.15, 1])

with r3a:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Behaviour vs Graph Complexity")
    if "num_unique_apis" in df.columns and "graph_num_edges" in df.columns:
        bubble = px.scatter(
            df.head(2000),
            x="num_unique_apis",
            y="graph_num_edges",
            size="anomaly_score",
            color="risk_tier",
            color_discrete_map=RISK_COLORS,
            hover_data=safe_cols(df, ["_id", "anomaly_score", "sequence_length(count)", "graph_num_nodes", "source", "ip_type"]),
        )
        st.plotly_chart(apply_dark_layout(bubble, 340), use_container_width=True)
    else:
        st.info("Need 'num_unique_apis' and 'graph_num_edges' for this chart.")
    st.markdown("</div>", unsafe_allow_html=True)

with r3b:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Feature vs Anomaly Score")
    if available_numeric:
        feat = st.selectbox("Feature", available_numeric, index=0)
        scatter = px.scatter(
            df.head(3000),
            x=feat,
            y="anomaly_score",
            color="risk_tier",
            color_discrete_map=RISK_COLORS,
            hover_data=safe_cols(df, ["_id", "is_anomaly", "source", "ip_type"]),
        )
        st.plotly_chart(apply_dark_layout(scatter, 340), use_container_width=True)
    else:
        st.info("No numeric feature columns found.")
    st.markdown("</div>", unsafe_allow_html=True)

r4a, r4b = st.columns([1.15, 1])

with r4a:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Correlation Heatmap")
    if available_numeric:
        corr_df = df[available_numeric + ["anomaly_score"]].corr(numeric_only=True)
        heat = px.imshow(corr_df, aspect="auto")
        st.plotly_chart(apply_dark_layout(heat, 360), use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation.")
    st.markdown("</div>", unsafe_allow_html=True)

with r4b:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Distributions by Risk Tier")
    if available_numeric:
        feat2 = st.selectbox("Distribution feature", available_numeric, index=min(2, len(available_numeric) - 1), key="feat2")
        viol = px.violin(
            df,
            x="risk_tier",
            y=feat2,
            color="risk_tier",
            color_discrete_map=RISK_COLORS,
            box=True,
            points="outliers",
        )
        st.plotly_chart(apply_dark_layout(viol, 360), use_container_width=True)
    else:
        st.info("No numeric feature columns found.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# =========================
# INVESTIGATION WORKBENCH
# =========================
st.markdown("## Investigation Workbench")

top = df.head(top_n).copy()
lcol, rcol = st.columns([1.35, 1])

with lcol:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Top Suspicious Sessions")
    show_cols = safe_cols(
        top,
        [
            "_id",
            "anomaly_score",
            "risk_tier",
            "is_anomaly",
            "num_unique_apis",
            "sequence_length(count)",
            "vsession_duration(min)",
            "graph_num_nodes",
            "graph_num_edges",
            "graph_density",
            "source",
            "ip_type",
        ],
    )
    st.dataframe(top[show_cols], use_container_width=True, height=430)
    st.markdown("</div>", unsafe_allow_html=True)

with rcol:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Session Drill-Down")
    selected_id = st.selectbox("Select _id", top["_id"].tolist())
    row = df[df["_id"] == selected_id].iloc[0].to_dict()

    st.markdown(f"<span class='pill'>_id: {selected_id}</span>", unsafe_allow_html=True)
    st.write("")

    st.markdown("SOC Summary")
    st.write(f"Risk tier: {row.get('risk_tier')}")
    st.write(f"Anomaly score: {row.get('anomaly_score')}")
    st.write(f"Flagged: {row.get('is_anomaly')}")
    if "source" in row:
        st.write(f"Source: {row.get('source')}")
    if "ip_type" in row:
        st.write(f"IP type: {row.get('ip_type')}")

    st.markdown("Key drivers (available fields)")
    for k in [
        "num_unique_apis",
        "sequence_length(count)",
        "vsession_duration(min)",
        "api_access_uniqueness",
        "inter_api_access_duration(sec)",
        "graph_num_nodes",
        "graph_num_edges",
        "graph_density",
        "graph_out_deg_mean",
        "graph_out_deg_max",
        "graph_in_deg_mean",
        "graph_in_deg_max",
    ]:
        if k in row:
            st.write(f"- {k}: {row[k]}")

    with st.expander("Show full raw record"):
        st.json(row)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# SUPERVISED VALIDATION VIEW
# =========================
if dataset_choice.startswith("Supervised") and "classification" in sup.columns:
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("## Validation View")

    tmp = sup.copy()
    tmp["label_outlier"] = (tmp["classification"].astype(str).str.lower() == "outlier").astype(int)

    v1, v2 = st.columns([1, 1])

    with v1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Label vs Predicted")
        cm = pd.crosstab(tmp["label_outlier"], tmp["is_anomaly"], rownames=["Label outlier"], colnames=["Pred anomaly"])
        st.dataframe(cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with v2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Score by Label")
        lab = tmp.copy()
        lab["label"] = np.where(lab["label_outlier"] == 1, "Outlier", "Normal")
        box2 = px.box(lab, x="label", y="anomaly_score", color="label")
        st.plotly_chart(apply_dark_layout(box2, 300), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
