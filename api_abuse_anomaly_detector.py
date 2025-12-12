#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API Abuse Anomaly Detector using Kaggle API behavior dataset

Files (as provided in the archive):
- supervised_dataset.csv          : behavior metrics + label ("classification")
- supervised_call_graphs.json     : call graphs (edges) for supervised rows
- remaining_behavior_ext.csv      : behavior metrics for larger unlabeled set
- remaining_call_graphs.json      : call graphs (edges) for remaining set

Core idea:
- Join behavior metrics with call-graph-derived metrics via `_id`.
- Train an Isolation Forest on "normal" behavior (semi-supervised anomaly detection).
- Evaluate on the labeled supervised set.
- Score the larger remaining dataset and list the most suspicious sessions.

This script is fully self-contained. Just ensure the 4 dataset files
are present in DATA_DIR.
"""

import json
import os
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG – update this if your files are in a different folder
# ============================================================

DATA_DIR = "data"  # folder containing the 4 files

SUPERVISED_CSV = os.path.join(DATA_DIR, "supervised_dataset.csv")
SUPERVISED_JSON = os.path.join(DATA_DIR, "supervised_call_graphs.json")
REMAINING_CSV = os.path.join(DATA_DIR, "remaining_behavior_ext.csv")
REMAINING_JSON = os.path.join(DATA_DIR, "remaining_call_graphs.json")


# Behavior metric columns
BASE_NUMERIC_COLS = [
    "inter_api_access_duration(sec)",
    "api_access_uniqueness",
    "sequence_length(count)",
    "vsession_duration(min)",
    "num_sessions",
    "num_users",
    "num_unique_apis",
]

# Graph-derived numeric columns (we will compute these)
GRAPH_NUMERIC_COLS = [
    "graph_num_nodes",
    "graph_num_edges",
    "graph_self_loops",
    "graph_density",
    "graph_out_deg_mean",
    "graph_out_deg_max",
    "graph_out_deg_std",
    "graph_in_deg_mean",
    "graph_in_deg_max",
    "graph_in_deg_std",
]

# Categorical columns to one-hot encode
CATEGORICAL_COLS = ["ip_type", "source"]


# ============================================================
# GRAPH STATS
# ============================================================

def compute_graph_stats(call_graph: List[dict]) -> dict:
    """
    Compute simple graph statistics from a call graph.

    call_graph: list of edges, each like {"fromId": "...", "toId": "..."}

    Returns a dict with keys in GRAPH_NUMERIC_COLS.
    """
    n_edges = len(call_graph)
    nodes = set()
    out_deg = Counter()
    in_deg = Counter()
    self_loops = 0

    for edge in call_graph:
        f = edge.get("fromId")
        t = edge.get("toId")
        if f is not None:
            nodes.add(f)
            out_deg[f] += 1
        if t is not None:
            nodes.add(t)
            in_deg[t] += 1
        if f == t:
            self_loops += 1

    n_nodes = len(nodes)
    density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

    def stats(counter: Counter) -> Tuple[float, float, float]:
        if not counter:
            return 0.0, 0.0, 0.0
        vals = np.fromiter(counter.values(), dtype=float)
        return float(vals.mean()), float(vals.max()), float(vals.std())

    out_mean, out_max, out_std = stats(out_deg)
    in_mean, in_max, in_std = stats(in_deg)

    return {
        "graph_num_nodes": n_nodes,
        "graph_num_edges": n_edges,
        "graph_self_loops": self_loops,
        "graph_density": density,
        "graph_out_deg_mean": out_mean,
        "graph_out_deg_max": out_max,
        "graph_out_deg_std": out_std,
        "graph_in_deg_mean": in_mean,
        "graph_in_deg_max": in_max,
        "graph_in_deg_std": in_std,
    }


def build_graph_stats_df(graphs: List[dict]) -> pd.DataFrame:
    """
    Build a DataFrame of graph statistics:

    Input:
        graphs: list of {"_id": ..., "call_graph": [...]}

    Output:
        DataFrame with one row per _id and columns:
        GRAPH_NUMERIC_COLS + ["_id"]
    """
    rows = []
    for item in graphs:
        gid = item["_id"]
        stats = compute_graph_stats(item["call_graph"])
        stats["_id"] = gid
        rows.append(stats)
    return pd.DataFrame(rows)


# ============================================================
# DATA LOADING & MERGING
# ============================================================

def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_and_merge_behavior_and_graph(
    behavior_df: pd.DataFrame, graphs: List[dict]
) -> pd.DataFrame:
    """
    Join behavior metrics CSV with graph stats JSON via `_id`.
    Deduplicates behavior rows on `_id` before merging.
    """
    stats_df = build_graph_stats_df(graphs)

    if "_id" not in behavior_df.columns:
        raise ValueError("Behavior DataFrame does not contain '_id' column.")

    behavior_dedup = behavior_df.drop_duplicates(subset=["_id"])
    merged = behavior_dedup.merge(stats_df, on="_id", how="inner")

    return merged


# ============================================================
# FEATURE BUILDING
# ============================================================

def build_feature_frame(
    df: pd.DataFrame, feature_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the feature matrix from merged DataFrame.

    - Selects BASE_NUMERIC_COLS + GRAPH_NUMERIC_COLS + CATEGORICAL_COLS.
    - One-hot encodes the categorical columns.
    - If feature_columns is None: returns X_enc and learned column order.
    - If feature_columns provided: reindexes to that column set, filling missing
      columns with 0 (for scoring on new data).

    Returns:
        X_enc (DataFrame), feature_columns (list)
    """
    required_cols = BASE_NUMERIC_COLS + GRAPH_NUMERIC_COLS + CATEGORICAL_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[required_cols].copy()
    X = X.dropna()

    # One-hot encode categories
    X_enc = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)

    if feature_columns is None:
        feature_columns = list(X_enc.columns)
        return X_enc, feature_columns
    else:
        # Align to existing feature columns
        X_enc = X_enc.reindex(columns=feature_columns, fill_value=0)
        return X_enc, feature_columns


# ============================================================
# MODEL TRAINING & SCORING
# ============================================================

def train_anomaly_model(
    X: pd.DataFrame, y: pd.Series, contamination: float = 0.35
) -> Tuple[Pipeline, float]:
    """
    Train an Isolation Forest anomaly detector using only normal samples.

    Args:
        X: feature DataFrame
        y: labels (0 = normal, 1 = outlier)
        contamination: expected fraction of anomalies in the data

    Returns:
        model_pipeline: sklearn Pipeline (StandardScaler + IsolationForest)
        threshold: anomaly score threshold (higher => more anomalous)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train only on normal samples (semi-supervised)
    normal_mask = (y_train == 0)
    X_train_norm = X_train[normal_mask]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("iforest", IsolationForest(
                n_estimators=200,
                contamination=contamination,
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )

    model.fit(X_train_norm)

    # Scores on test set (lower score = more normal in IsolationForest)
    scores_test = model.decision_function(X_test)
    anomaly_score_test = -scores_test

    # Threshold based on normal train distribution (95th percentile)
    train_scores = -model.decision_function(X_train_norm)
    threshold = float(np.percentile(train_scores, 95))

    # Hard predictions for evaluation
    y_pred = (anomaly_score_test > threshold).astype(int)

    print("\n=== Evaluation on supervised test set ===")
    print("ROC-AUC:", roc_auc_score(y_test, anomaly_score_test))
    print(classification_report(y_test, y_pred, digits=3))

    return model, threshold


def score_behaviors(
    merged_df: pd.DataFrame,
    model: Pipeline,
    threshold: float,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Score behaviors for anomaly.

    Returns a DataFrame with:
      - original merged_df columns (for rows used)
      - anomaly_score
      - is_anomaly (1/0)
    """
    X_enc, _ = build_feature_frame(merged_df, feature_columns=feature_columns)
    scores = model.decision_function(X_enc)
    anomaly_score = -scores
    is_anomaly = (anomaly_score > threshold).astype(int)

    out = merged_df.loc[X_enc.index].copy()
    out["anomaly_score"] = anomaly_score
    out["is_anomaly"] = is_anomaly

    return out


# ============================================================
# MAIN
# ============================================================

def main():
    # ---------- Load supervised behavior + graphs ----------
    print("Loading supervised datasets...")
    sup_behavior = pd.read_csv(SUPERVISED_CSV)
    sup_graphs = load_json(SUPERVISED_JSON)

    sup_merged = load_and_merge_behavior_and_graph(sup_behavior, sup_graphs)
    print("Supervised merged shape:", sup_merged.shape)

    # Map classification to numeric labels
    if "classification" not in sup_merged.columns:
        raise ValueError("Supervised dataset missing 'classification' column.")

    label_map = {"normal": 0, "outlier": 1}
    y_all = sup_merged["classification"].map(label_map)

    # Build feature matrix on supervised data (also returns feature_columns)
    X_all, feature_columns = build_feature_frame(sup_merged)
    # Align labels with X index
    y_all = y_all.loc[X_all.index]

    print("Supervised feature matrix shape:", X_all.shape)
    print("Label distribution:\n", y_all.value_counts())

    # ---------- Train anomaly detector ----------
    model, threshold = train_anomaly_model(X_all, y_all, contamination=0.35)
    print("\nChosen anomaly threshold (higher score => more anomalous):", threshold)

    # ---------- Score supervised set for inspection ----------
    sup_scored = score_behaviors(sup_merged, model, threshold, feature_columns)
    print("\nSample of supervised scored data:")
    print(
        sup_scored[
            ["_id", "classification", "anomaly_score", "is_anomaly"]
        ]
        .head(10)
        .to_string(index=False)
    )

    # ---------- Load remaining (larger) behavior + graphs ----------
    print("\nLoading remaining datasets...")
    rem_behavior = pd.read_csv(REMAINING_CSV)
    rem_graphs = load_json(REMAINING_JSON)

    rem_merged = load_and_merge_behavior_and_graph(rem_behavior, rem_graphs)
    print("Remaining merged shape:", rem_merged.shape)

    # ---------- Score remaining set ----------
    rem_scored = score_behaviors(rem_merged, model, threshold, feature_columns)

    print("\nTop 10 most suspicious sessions in remaining dataset:")
    top_suspicious = rem_scored.sort_values("anomaly_score", ascending=False).head(10)
    cols_to_show = [
        "_id",
        "anomaly_score",
        "is_anomaly",
        "inter_api_access_duration(sec)",
        "api_access_uniqueness",
        "sequence_length(count)",
        "vsession_duration(min)",
        "num_sessions",
        "num_users",
        "num_unique_apis",
        "graph_num_nodes",
        "graph_num_edges",
    ]
    cols_to_show = [c for c in cols_to_show if c in top_suspicious.columns]
    print(top_suspicious[cols_to_show].to_string(index=False))

    # ============================================================
    # SAVE OUTPUTS FOR DASHBOARD
    # ============================================================
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    sup_out_path = os.path.join(out_dir, "supervised_scored.csv")
    rem_out_path = os.path.join(out_dir, "remaining_scored.csv")

    sup_scored.to_csv(sup_out_path, index=False)
    rem_scored.to_csv(rem_out_path, index=False)

    print(f"\nSaved outputs:\n- {sup_out_path}\n- {rem_out_path}")


if __name__ == "__main__":
    main()
