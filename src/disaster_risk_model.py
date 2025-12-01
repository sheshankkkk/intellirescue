import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt


# ===========================
# CONFIG — MATCHES YOUR CSV
# ===========================
IMPACT_COL = "Total Deaths"

NON_FEATURE_COLS = [
    IMPACT_COL,
    "Dis No",
    "Event Name",
    "Location",
    "Geo Locations",
    "Glide"
]

CSV_PATH = "data/disasters.csv"
LOG_DIR = "logs/disaster_risk"
# ===========================


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop rows missing the impact column
    df = df.dropna(subset=[IMPACT_COL])

    # Replace NaNs in numeric impact-type columns with 0
    numeric_fill = [
        "No Injured", "No Affected", "No Homeless",
        "Total Affected", "Total Damages ('000 US$)"
    ]
    for col in numeric_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def create_high_impact_label(df: pd.DataFrame, quantile: float = 0.75):
    """Convert impact into binary label."""
    threshold = df[IMPACT_COL].quantile(quantile)
    df["HighImpact"] = (df[IMPACT_COL] >= threshold).astype(int)
    print(f"HighImpact threshold = {threshold:.2f}")
    return df, threshold


def prepare_features(df: pd.DataFrame):
    df, threshold = create_high_impact_label(df)

    cols_to_drop = NON_FEATURE_COLS + ["HighImpact"]
    feature_cols = [c for c in df.columns if c not in cols_to_drop]

    # Separate features + label
    X_raw = df[feature_cols]
    y = df["HighImpact"]

    # One-hot encode all categorical columns
    X = pd.get_dummies(X_raw, drop_first=True)

    return X, y, feature_cols, threshold


def train_disaster_risk_model():
    df = load_data(CSV_PATH)
    X, y, feature_cols, threshold = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Base model
    base_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    # Calibrated classifier for probability quality
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)

    model.fit(X_train, y_train)

    # Predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    print("\n=== INTELLIRESCUE: MULTI-DISASTER RISK MODEL ===")
    print(f"Impact Column     : {IMPACT_COL} (threshold={threshold:.2f})")
    print(f"Features Used     : {len(feature_cols)}")
    print(f"Accuracy          : {acc:.3f}")
    print(f"ROC-AUC           : {auc:.3f}")
    print(f"Brier Score       : {brier:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Logging outputs
    os.makedirs(LOG_DIR, exist_ok=True)
    pred_path = f"{LOG_DIR}/disaster_risk_predictions.csv"
    pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba
    }).to_csv(pred_path, index=False)
    print(f"Saved predictions → {pred_path}")

    # Calibration curve
    try:
        from sklearn.calibration import calibration_curve
        frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)

        os.makedirs("logs/plots", exist_ok=True)
        plot_path = "logs/plots/disaster_calibration_curve.png"

        plt.figure()
        plt.plot(mean_pred, frac_pos, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k--", label="Ideal")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Curve: High-Impact Disaster")
        plt.legend()
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        print(f"Saved calibration plot → {plot_path}")
    except Exception as e:
        print(f"Calibration plot failed: {e}")

    return model


if __name__ == "__main__":
    train_disaster_risk_model()
