import os
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.colors as pc

px.defaults.template = "plotly_dark"

st.set_page_config(
    page_title="API Anomaly - AI Cyber Resilience Dashboard",
    page_icon="",
    layout="wide",
)

BG = "#020617"
CARD = "#0b1220"
BORDER = "#1e293b"
TEXT = "#e5e7eb"
MUTED = "#94a3b8"

RISK_COLORS = {
    "High": "#a855f7",
    "Medium": "#3b82f6",
    "Low": "#22d3ee",
}

st.markdown(
    f"""
    <style>
      body {{ background-color: {BG}; }}
      .block-container {{ padding-top: 3.2rem; padding-bottom: 2.5rem; }}
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

st.markdown('<div class="title">API Abuse - AI Cyber Resilience Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Behaviour metrics and call-graph intelligence with Isolation Forest risk scoring</p>',
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

SUP_FILE = os.path.join(OUTPUTS_DIR, "supervised_scored.csv")
REM_FILE = os.path.join(OUTPUTS_DIR, "remaining_scored.csv")

DETECTOR_SCRIPT = os.path.join(BASE_DIR, "api_abuse_anomaly_detector.py")
COMPLIANCE_FILE = os.path.join(BASE_DIR, "compliance_mapping.csv")


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_cols(df: pd.DataFrame, cols):
    return [c for c in cols if c in df.columns]


def apply_dark_layout(fig, height=320):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color=TEXT,
        legend_title_text="",
    )
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    return fig


def add_bubble_size(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    score = pd.to_numeric(out.get("anomaly_score", 0.0), errors="coerce").fillna(0.0)
    score_pos = score.clip(lower=0.0)

    smin, smax = float(score_pos.min()), float(score_pos.max())
    if smax > smin:
        norm = (score_pos - smin) / (smax - smin)
    else:
        norm = score_pos * 0.0

    out["bubble_size"] = 8.0 + (norm * 37.0)
    return out


# ✅ FIXED: supports your CSV headers:
# framework, article_or_ref, requirement, dataset_fields_used, evidence_artifact, how_to_show_in_dashboard
def load_compliance(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    cdf = pd.read_csv(path)
    if cdf.empty:
        return cdf

    rename_map = {}
    for col in cdf.columns:
        lc = str(col).strip().lower()

        if lc in ["framework", "standard"]:
            rename_map[col] = "framework"

        # Your sheet uses article_or_ref for control identifier
        elif lc in ["control_id", "article", "clause", "requirement_id", "article_or_ref", "article_or_ref."]:
            rename_map[col] = "control_id"

        # Your sheet uses requirement for control title/requirement text
        elif lc in ["control_title", "requirement", "control", "title"]:
            rename_map[col] = "control_title"

        # Your sheet uses dataset_fields_used (comma-separated features)
        elif lc in ["dataset_fields_used", "dataset_fields", "dataset_field", "fields_used", "dataset_features_used"]:
            rename_map[col] = "dataset_fields_used"

        elif lc in ["evidence_artifact", "evidence", "artifact"]:
            rename_map[col] = "evidence_artifact"

        elif lc in ["how_to_show_in_dashboard", "how_to_show", "dashboard_display", "show_in_dashboard"]:
            rename_map[col] = "how_to_show_in_dashboard"

        elif lc in ["rationale", "mapping_rationale", "notes", "justification"]:
            rename_map[col] = "mapping_rationale"

    cdf = cdf.rename(columns=rename_map)

    # Ensure optional cols exist
    for must in ["evidence_artifact", "how_to_show_in_dashboard", "mapping_rationale", "dataset_fields_used"]:
        if must not in cdf.columns:
            cdf[must] = ""

    # Build mapping_rationale from how_to_show + evidence (if mapping_rationale not provided)
    if "mapping_rationale" in cdf.columns:
        base = cdf["mapping_rationale"].fillna("").astype(str).str.strip()
    else:
        base = ""

    how = cdf["how_to_show_in_dashboard"].fillna("").astype(str).str.strip()
    evd = cdf["evidence_artifact"].fillna("").astype(str).str.strip()

    combined = base.copy() if isinstance(base, pd.Series) else pd.Series([""] * len(cdf))
    combined = combined.where(combined != "", how)
    combined = combined.where(combined != "", evd)
    cdf["mapping_rationale"] = combined.fillna("")

    # ✅ KEY: create feature_name by splitting dataset_fields_used
    # If dataset_fields_used has "a,b,c" -> 3 rows with feature_name=a / b / c
    fields = cdf["dataset_fields_used"].fillna("").astype(str)
    cdf["feature_name"] = fields.apply(
        lambda x: [f.strip() for f in x.split(",") if f.strip()] if x.strip() else [""]
    )
    cdf = cdf.explode("feature_name", ignore_index=True)

    # Normalize text fields
    for col in ["framework", "feature_name", "control_id", "control_title", "mapping_rationale", "evidence_artifact", "how_to_show_in_dashboard"]:
        if col in cdf.columns:
            cdf[col] = cdf[col].astype(str).fillna("").str.strip()

    return cdf


def short_label(s: str, n: int = 64) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else (s[: n - 3] + "...")


def wrap_label(s: str, width: int = 38) -> str:
    s = "" if s is None else str(s)
    if len(s) <= width:
        return s
    words = s.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "<br>".join(lines[:3]) + ("<br>..." if len(lines) > 3 else "")


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

df_all = df_raw.copy()
df = df_raw.copy()

df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce")
df_all["anomaly_score"] = pd.to_numeric(df_all["anomaly_score"], errors="coerce")

if score_threshold > 0:
    df = df[df["anomaly_score"] >= score_threshold].copy()

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

df = add_bubble_size(df)
df_all = add_bubble_size(df_all)

base_for_quantiles = df if len(df) >= 20 else df_all
p_med = float(base_for_quantiles["anomaly_score"].quantile(med_q))
p_high = float(base_for_quantiles["anomaly_score"].quantile(high_q))


def risk_tier(score: float) -> str:
    if pd.isna(score):
        return "Low"
    if score >= p_high:
        return "High"
    if score >= p_med:
        return "Medium"
    return "Low"


df["risk_tier"] = df["anomaly_score"].apply(risk_tier)
df_all["risk_tier"] = df_all["anomaly_score"].apply(risk_tier)

df = df.sort_values("anomaly_score", ascending=False)

total_sessions = len(df_all)
flagged = int(pd.to_numeric(df_all["is_anomaly"], errors="coerce").fillna(0).sum())
anomaly_rate = (pd.to_numeric(df_all["is_anomaly"], errors="coerce").fillna(0).mean() * 100.0) if total_sessions else 0.0
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
        df.dropna(subset=["anomaly_score"]),
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
    tmp["flag_status"] = np.where(pd.to_numeric(tmp["is_anomaly"], errors="coerce").fillna(0) == 1, "Flagged", "Not Flagged")
    pie2 = px.pie(tmp, names="flag_status", hole=0.55)
    st.plotly_chart(apply_dark_layout(pie2, 300), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

row2a, row2b = st.columns([1, 1])

with row2a:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Score by Risk Tier")
    box = px.box(df.dropna(subset=["anomaly_score"]), x="risk_tier", y="anomaly_score", color="risk_tier",
                 color_discrete_map=RISK_COLORS)
    st.plotly_chart(apply_dark_layout(box, 300), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with row2b:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Origin Breakdown")
    sub = st.columns(2)

    if "source" in df.columns:
        src = df.groupby(["source", "risk_tier"]).size().reset_index(name="count")
        fig_src = px.bar(src, x="source", y="count", color="risk_tier", barmode="stack", color_discrete_map=RISK_COLORS)
        sub[0].plotly_chart(apply_dark_layout(fig_src, 290), use_container_width=True)
    else:
        sub[0].info("Column 'source' not found.")

    if "ip_type" in df.columns:
        ip = df.groupby(["ip_type", "risk_tier"]).size().reset_index(name="count")
        fig_ip = px.bar(ip, x="ip_type", y="count", color="risk_tier", barmode="stack", color_discrete_map=RISK_COLORS)
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
            size="bubble_size",
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
            df.head(3000).dropna(subset=["anomaly_score"]),
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
        feat2 = st.selectbox("Distribution feature", available_numeric,
                             index=min(2, len(available_numeric) - 1), key="feat2")
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

selected_id = None

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

    if len(top) == 0:
        st.info("No records available with current filters.")
    else:
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
# GOVERNANCE / COMPLIANCE MAPPING
# =========================
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("## Governance and Compliance Mapping")

compliance_df = load_compliance(COMPLIANCE_FILE)

if compliance_df.empty:
    st.info("compliance_mapping.csv not found (or empty). Place it next to dashboard.py to enable this section.")
else:
    required_cols = ["framework", "feature_name", "control_id", "control_title"]
    missing_cols = [c for c in required_cols if c not in compliance_df.columns]
    if missing_cols:
        st.error(f"compliance_mapping.csv missing required columns (after normalization): {missing_cols}")
        st.stop()

    present_features = set(df.columns)
    compliance_df = compliance_df.copy()
    compliance_df["feature_present_in_view"] = compliance_df["feature_name"].isin(present_features)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Compliance Filters")

    frameworks = sorted(compliance_df["framework"].dropna().astype(str).unique().tolist())
    frameworks = [fw for fw in frameworks if "owasp" not in fw.lower()]
    fw_selected = st.sidebar.multiselect("Framework", frameworks, default=frameworks)

    show_only_present = st.sidebar.checkbox("Show only features present in current view", value=True)

    risk_selected = st.sidebar.multiselect(
        "Risk tiers for compliance metrics",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )

    metric_mode = st.sidebar.radio(
        "Compliance metric mode",
        ["Control Impact (Trigger-based)", "Coverage (Feature Present)"],
        index=0
    )

    cview = compliance_df.copy()
    if fw_selected:
        cview = cview[cview["framework"].astype(str).isin(fw_selected)].copy()
    if show_only_present:
        cview = cview[cview["feature_present_in_view"] == True].copy()

    df_risk_scope = df[df["risk_tier"].isin(risk_selected)].copy() if "risk_tier" in df.columns else df.copy()

    palette = (pc.qualitative.Set2 + pc.qualitative.Pastel + pc.qualitative.Dark24)
    framework_color_map = {fw: palette[i % len(palette)] for i, fw in enumerate(frameworks)}

    def feature_coverage_count(feature: str) -> int:
        if feature not in df.columns:
            return 0
        return int(df[feature].notna().sum())

    def feature_coverage_in_risk(feature: str) -> int:
        if feature not in df_risk_scope.columns:
            return 0
        return int(df_risk_scope[feature].notna().sum())

    def control_impact_count(row: pd.Series) -> int:
        cid = str(row.get("control_id", "")).lower()
        title = str(row.get("control_title", "")).lower()
        key = cid + " " + title

        if ("logging" in key) or ("art. 12" in key) or ("record-keeping" in key):
            if "is_anomaly" in df_risk_scope.columns:
                return int((pd.to_numeric(df_risk_scope["is_anomaly"], errors="coerce").fillna(0) == 1).sum())
            return 0

        if ("risk management" in key) or ("art. 9" in key):
            if "risk_tier" in df_risk_scope.columns:
                return int((df_risk_scope["risk_tier"] == "High").sum())
            return 0

        if ("post-market" in key) or ("art. 72" in key) or ("monitoring" in key):
            if "anomaly_score" in df_risk_scope.columns:
                return int((pd.to_numeric(df_risk_scope["anomaly_score"], errors="coerce") >= p_high).sum())
            return 0

        if ("quality management" in key) or ("art. 17" in key) or ("validation" in key):
            if ("is_anomaly" in df_risk_scope.columns) and ("risk_tier" in df_risk_scope.columns):
                return int(
                    ((pd.to_numeric(df_risk_scope["is_anomaly"], errors="coerce").fillna(0) == 1) &
                     (df_risk_scope["risk_tier"] != "Low")).sum()
                )
            return 0

        f = str(row.get("feature_name", ""))
        return feature_coverage_in_risk(f)

    cview["sessions_with_feature"] = cview["feature_name"].apply(feature_coverage_count)

    if metric_mode.startswith("Coverage"):
        cview["sessions_in_selected_risk"] = cview["feature_name"].apply(feature_coverage_in_risk)
    else:
        cview["sessions_in_selected_risk"] = cview.apply(control_impact_count, axis=1)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Mapped rows (filtered)", f"{len(cview):,}")
    g2.metric("Frameworks selected", f"{cview['framework'].nunique():,}")
    g3.metric("Features present in view", f"{int(cview['feature_present_in_view'].sum()):,}")
    g4.metric("Risk scope sessions", f"{len(df_risk_scope):,}")

    tab_overview, tab_top, tab_table, tab_session = st.tabs(
        ["Overview", "Top Controls", "Mapping Table", "Session Drill-Down"]
    )

    with tab_overview:
        c1, c2 = st.columns([1.0, 1.2])

        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Controls by Framework")
            fw_counts = cview.groupby("framework").size().reset_index(name="controls")
            fig_fw = px.bar(
                fw_counts,
                x="framework",
                y="controls",
                color="framework",
                color_discrete_map=framework_color_map,
                text="controls",
            )
            fig_fw.update_traces(textposition="outside")
            st.plotly_chart(apply_dark_layout(fig_fw, 360), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Coverage / Impact by Framework (Selected Risk Scope)")
            fw_cov = cview.groupby("framework")["sessions_in_selected_risk"].sum().reset_index()
            fw_cov = fw_cov.sort_values("sessions_in_selected_risk", ascending=False)
            fig_fw2 = px.bar(
                fw_cov,
                x="framework",
                y="sessions_in_selected_risk",
                color="framework",
                color_discrete_map=framework_color_map,
            )
            st.plotly_chart(apply_dark_layout(fig_fw2, 360), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_top:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Top Controls (Selected Risk Scope)")

        top_controls = cview.sort_values("sessions_in_selected_risk", ascending=False).head(15).copy()
        top_controls["control_label_full"] = top_controls["control_id"].astype(str) + " - " + top_controls["control_title"].astype(str)
        top_controls["control_label"] = top_controls["control_label_full"].apply(lambda x: wrap_label(x, 42))

        fig_cov = px.bar(
            top_controls,
            x="sessions_in_selected_risk",
            y="control_label",
            orientation="h",
            color="framework",
            color_discrete_map=framework_color_map,
            hover_data={
                "framework": True,
                "sessions_in_selected_risk": True,
                "control_label_full": True,
                "control_label": False,
            },
        )
        fig_cov.update_layout(
            yaxis_title="",
            xaxis_title="sessions_in_selected_risk" if metric_mode.startswith("Coverage") else "impact_count_in_selected_risk",
            legend_title_text="Framework",
        )
        st.plotly_chart(apply_dark_layout(fig_cov, 520), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_table:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Compliance Mapping Table (Filtered)")

        cshow = cview.copy()
        cshow["control"] = cshow["control_id"].astype(str) + " - " + cshow["control_title"].astype(str)
        cshow["control"] = cshow["control"].apply(lambda x: short_label(x, 90))

        show_cols = [
            "framework",
            "control",
            "feature_name",
            "sessions_in_selected_risk",
            "sessions_with_feature",
            "evidence_artifact",
            "how_to_show_in_dashboard",
        ]
        show_cols = [c for c in show_cols if c in cshow.columns]

        st.dataframe(cshow[show_cols], use_container_width=True, height=520)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_session:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Session-to-Control Drill-Down")

        if selected_id is None:
            st.info("Select a session in Investigation Workbench to enable this drill-down.")
        else:
            session_row = df[df["_id"] == selected_id]
            if session_row.empty:
                st.info("Selected session not found in the current filtered view. Adjust filters or select another session.")
            else:
                session_dict = session_row.iloc[0].to_dict()
                mapped_features = set(cview["feature_name"].tolist())

                active_features = []
                for f in mapped_features:
                    if f and (f in session_dict) and pd.notna(session_dict[f]):
                        active_features.append(f)

                st.write(f"Session ID: {selected_id}")
                st.write(f"Risk tier: {session_dict.get('risk_tier')}")
                st.write("Active mapped features found:", ", ".join(active_features) if active_features else "None")

                session_controls = cview[cview["feature_name"].isin(active_features)].copy()
                if session_controls.empty:
                    st.info("No mapped controls matched active features for this session (in current compliance filters).")
                else:
                    session_controls = session_controls.sort_values("sessions_in_selected_risk", ascending=False)
                    st.dataframe(
                        session_controls[["framework", "control_id", "control_title", "feature_name", "sessions_in_selected_risk",
                                         "evidence_artifact", "how_to_show_in_dashboard"]],
                        use_container_width=True,
                        height=420
                    )

        st.markdown("</div>", unsafe_allow_html=True)

