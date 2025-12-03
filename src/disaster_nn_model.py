import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    classification_report,
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ---------------- CONFIG (aligned with other scripts) ----------------
IMPACT_COL = "Total Deaths"

NON_FEATURE_COLS = [
    IMPACT_COL,
    "Dis No",
    "Event Name",
    "Location",
    "Geo Locations",
    "Glide",
]

CSV_PATH_FULL = "data/disasters.csv"
CSV_PATH_SAMPLE = "data/disasters_sample.csv"
HIGH_IMPACT_QUANTILE = 0.75  # top 25% by deaths -> HighImpact
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "disaster_nn_model.keras")
# --------------------------------------------------------------------


def get_data_path():
    if os.path.exists(CSV_PATH_FULL):
        print(f"[NN] Using FULL dataset at {CSV_PATH_FULL}")
        return CSV_PATH_FULL
    elif os.path.exists(CSV_PATH_SAMPLE):
        print(f"[NN] Using SAMPLE dataset at {CSV_PATH_SAMPLE}")
        return CSV_PATH_SAMPLE
    else:
        raise FileNotFoundError(
            f"Neither {CSV_PATH_FULL} nor {CSV_PATH_SAMPLE} found."
        )


def load_and_prepare_data():
    path = get_data_path()
    df = pd.read_csv(path)

    # Drop rows with missing impact
    df = df.dropna(subset=[IMPACT_COL])

    # Create HighImpact label
    threshold = df[IMPACT_COL].quantile(HIGH_IMPACT_QUANTILE)
    df["HighImpact"] = (df[IMPACT_COL] >= threshold).astype(int)
    print(f"[NN] HighImpact threshold on {IMPACT_COL}: {threshold:.2f}")

    # Fill some important numeric columns with 0 (if present)
    numeric_fill = [
        "No Injured",
        "No Affected",
        "No Homeless",
        "Total Affected",
        "Total Damages ('000 US$)",
    ]
    for col in numeric_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Features: drop non-feature & label
    cols_to_drop = NON_FEATURE_COLS + ["HighImpact"]
    feature_cols_raw = [c for c in df.columns if c not in cols_to_drop]

    X_raw = df[feature_cols_raw]
    y = df["HighImpact"].values

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X_raw, drop_first=True)

    # 🔥 IMPORTANT FIX: force everything to numeric and fill NaNs
    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"[NN] Feature dimension after encoding: {X_encoded.shape[1]}")
    print(f"[NN] Any NaNs after cleaning? {X_encoded.isna().any().any()}")

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Scale features (important for NN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Quick sanity check
    print(
        "[NN] Any NaNs in X_train_scaled?",
        np.isnan(X_train_scaled).any(),
        "| X_val_scaled?",
        np.isnan(X_val_scaled).any(),
        "| X_test_scaled?",
        np.isnan(X_test_scaled).any(),
    )

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        threshold,
        X_encoded.columns.tolist(),
        scaler,
    )


def build_model(input_dim: int) -> tf.keras.Model:
    """
    Simple feed-forward neural network for binary classification.
    """
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def train_nn_model(epochs: int = 50, batch_size: int = 256):
    # Reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        threshold,
        feature_names,
        scaler,
    ) = load_and_prepare_data()

    input_dim = X_train.shape[1]
    model = build_model(input_dim)

    print(f"[NN] Training NN model with input_dim={input_dim}")
    print(f"[NN] Epochs: {epochs}, Batch size: {batch_size}")

    es = callbacks.EarlyStopping(
        monitor="val_auc",
        patience=8,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1,
    )

    # Evaluate on test set
    y_proba = model.predict(X_test).ravel()
    # DEBUG: check for NaNs
    print("[NN] Any NaNs in y_proba?", np.isnan(y_proba).any())

    # Guard in case something is still wrong
    if np.isnan(y_proba).any():
        # Replace NaNs with mean probability to avoid metric crash
        mean_proba = np.nanmean(y_proba)
        y_proba = np.where(np.isnan(y_proba), mean_proba, y_proba)

    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)

    print("\n=== INTELLIRESCUE: NEURAL DISASTER RISK MODEL ===")
    print(f"Impact column     : {IMPACT_COL} (threshold={threshold:.2f})")
    print(f"Epochs (attempted): {epochs}")
    print(f"Test Accuracy     : {acc:.3f}")
    print(f"Test ROC-AUC      : {auc:.3f}")
    print(f"Test Brier Score  : {brier:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model + scaler metadata
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[NN] Saved Keras model → {MODEL_PATH}")

    meta = {
        "feature_names": feature_names,
        "threshold": float(threshold),
        "high_impact_quantile": float(HIGH_IMPACT_QUANTILE),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    meta_path = os.path.join(MODEL_DIR, "disaster_nn_meta.json")
    pd.Series(meta).to_json(meta_path)
    print(f"[NN] Saved model metadata → {meta_path}")

    hist_df = pd.DataFrame(history.history)
    hist_path = os.path.join(MODEL_DIR, "disaster_nn_history.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"[NN] Saved training history → {hist_path}")


if __name__ == "__main__":
    # You can increase epochs to 100/200 later if stable
    train_nn_model(epochs=50, batch_size=256)
