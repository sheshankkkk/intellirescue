import os
import json
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

from xgboost import XGBClassifier  # always use XGBoost


# ---------------- CONFIG ----------------
IMPACT_COL = "Total Deaths"
CSV_PATH_FULL = "data/disasters.csv"
CSV_PATH_SAMPLE = "data/disasters_sample.csv"
HIGH_IMPACT_QUANTILE = 0.75  # top 25% by deaths → HighImpact

MODELS_DIR = Path("models")
PIPELINE_PATH = MODELS_DIR / "disaster_risk_xgb.pkl"
METRICS_PATH = MODELS_DIR / "disaster_risk_metrics.json"


def get_data_path():
    if os.path.exists(CSV_PATH_FULL):
        print(f"[BACKEND] Using FULL dataset at {CSV_PATH_FULL}")
        return CSV_PATH_FULL
    elif os.path.exists(CSV_PATH_SAMPLE):
        print(f"[BACKEND] Using SAMPLE dataset at {CSV_PATH_SAMPLE}")
        return CSV_PATH_SAMPLE
    else:
        raise FileNotFoundError(
            f"Neither {CSV_PATH_FULL} nor {CSV_PATH_SAMPLE} found."
        )


def load_dataset():
    path = get_data_path()
    df = pd.read_csv(path)

    # Drop rows with missing impact
    df = df.dropna(subset=[IMPACT_COL])

    # Create HighImpact label
    threshold = df[IMPACT_COL].quantile(HIGH_IMPACT_QUANTILE)
    df["HighImpact"] = (df[IMPACT_COL] >= threshold).astype(int)

    print(f"[BACKEND] HighImpact threshold on {IMPACT_COL}: {threshold:.2f}")
    print(f"[BACKEND] Total rows after cleaning: {len(df)}")
    print(f"[BACKEND] HighImpact rate: {df['HighImpact'].mean():.3f}")

    return df, float(threshold)


def build_model_pipeline():
    """
    Build an XGBoost risk model using:
      - Year (numeric)
      - Region (categorical)
      - Disaster Type (categorical)

    with an explicit 80/20 train–test split.
    """
    df, threshold = load_dataset()

    # Only keep rows where all 3 features are present
    model_df = df[["Year", "Region", "Disaster Type", "HighImpact"]].dropna()
    print(f"[BACKEND] Rows used for modeling: {len(model_df)}")

    X_raw = model_df[["Year", "Region", "Disaster Type"]]
    y = model_df["HighImpact"]

    numeric_features = ["Year"]
    categorical_features = ["Region", "Disaster Type"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # ---- XGBoost model ----
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.2,
        n_jobs=-1,
        random_state=42,
    )
    model_type = "XGBoost"

    # Full pipeline: preprocessing + model
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # === Explicit 80/20 split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y,
        train_size=0.8,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"[BACKEND] Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    print("\n[BACKEND] === XGBOOST RISK MODEL EVALUATION (HELD-OUT 20%) ===")
    print(f"Model type      : {model_type}")
    print(f"Accuracy        : {acc:.3f}")
    print(f"ROC-AUC         : {auc:.3f}")
    print(f"Brier Score     : {brier:.3f}")
    print(f"Positive rate train: {y_train.mean():.3f}")
    print(f"Positive rate test : {y_test.mean():.3f}")
    print("==================================================\n")

    metrics = {
        "model_type": model_type,
        "threshold": threshold,
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "brier": float(brier),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "n_samples": int(len(model_df)),
        "high_impact_quantile": HIGH_IMPACT_QUANTILE,
        "features": ["Year", "Region", "Disaster Type"],
    }

    return pipeline, metrics


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline, metrics = build_model_pipeline()

    # Save pipeline
    import joblib

    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"[BACKEND] Saved model pipeline → {PIPELINE_PATH}")

    # Save metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[BACKEND] Saved metrics → {METRICS_PATH}")


if __name__ == "__main__":
    main()
