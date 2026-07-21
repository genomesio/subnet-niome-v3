import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from niome_subnet.genomics.model import Stage4Result

from niome_subnet.utils.settings import STAGE3_DATASET, VALID_EXPERIMENTS_PATH


# =====================================================
# LOAD JSON
# =====================================================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# =====================================================
# FLATTEN STAGE 12 (CRITICAL FIX)
# =====================================================
def flatten_stage12(data):

    rows = []

    for item in data:

        exp = item["experiment"]
        feat = item["features"]

        rows.append({
            "experiment_id": exp["experiment_id"],
            "mutation": exp["mutation"],
            "cas_system": exp["cas_system"],
            "guideRNA": exp["guideRNA"],
            "start": exp["target_alignment_start"],

            # IMPORTANT: flatten features
            "gc": feat["gc"],
            "distance": feat["distance_to_mutation"],
            "gc_score": feat["gc_score"],
            "dist_score": feat["dist_score"],
            "consistency": feat["consistency"],

            # optional stage2 signal
            "stage2_score": item["stage2"]["structural_score"]
        })

    return pd.DataFrame(rows)


# =====================================================
# FLATTEN STAGE 3
# =====================================================
def flatten_stage3(data):

    rows = []

    for item in data:

        rows.append({
            "experiment_id": item["experiment_id"],
            "mutation": item["mutation"],
            "cas_system": item["cas"],

            "gc": item["features"]["gc"],
            "distance": item["features"]["distance"],
            "gc_score": item["features"]["gc_score"],
            "dist_score": item["features"]["dist_score"],
            "consistency": item["features"]["consistency"],

            "energy": item["energy"],
            "mh": int(item["mh"]),

            "outcome": item["outcome"],
            "indel_length": item["indel_length"]
        })

    return pd.DataFrame(rows)


# =====================================================
# FEATURE MATRIX
# =====================================================
def build_X(df):

    required = [
        "gc",
        "distance",
        "gc_score",
        "dist_score",
        "consistency",
        "energy",
        "mh"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in merged dataset: {missing}")

    return df[required]


# =====================================================
# TARGETS
# =====================================================
def build_y(df):

    return pd.DataFrame({
        "is_cut": (df["outcome"] != "no_cut").astype(int),
        "is_hdr": (df["outcome"] == "HDR").astype(int),
        "indel_length": df["indel_length"]
    })


# =====================================================
# MODEL
# =====================================================
def evaluate(X, y):

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=12
    )

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "residual_std": float(np.std(y_test - pred))
    }


def run_stage4():
    stage3 = flatten_stage3(load_json(STAGE3_DATASET))
    stage12 = flatten_stage12(load_json(VALID_EXPERIMENTS_PATH))
    stage12_slim = stage12[["experiment_id", "guideRNA", "start", "stage2_score"]]

    df = stage3.merge(
        stage12_slim,
        on="experiment_id",
        how="inner"
    )

    if len(df) == 0:
        raise ValueError("Merge failed: no matching experiment_id between Stage 3 and Stage 12")

    X = build_X(df)
    y = build_y(df)

    results = {}

    for col in y.columns:
        results[col] = evaluate(X, y[col])

    avg_r2 = np.mean([v["r2"] for v in results.values()])
    avg_mae = np.mean([v["mae"] for v in results.values()])

    consistency_score = 0.7 * max(avg_r2, 0) + 0.3 * (1 - avg_mae)
    
    return Stage4Result(consistency_score=consistency_score)
