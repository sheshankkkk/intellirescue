import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier


def load_data(path: str) -> pd.DataFrame:
    """
    Loads the flood dataset from a CSV file.

    Expected to have:
      - feature columns (e.g., Rainfall, Temperature, etc.)
      - target column named 'flood' (0/1)
    """
    df = pd.read_csv(path)
    return df


def train_risk_model(csv_path: str = "data/flood_data.csv"):
    # Load dataset
    df = load_data(csv_path)

    # TODO: replace with real column names once dataset is added
    feature_cols = [c for c in df.columns if c not in ["flood"]]
    X = df[feature_cols]
    y = df["flood"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("=== IntelliRescue Risk Model (Phase 1) ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC-AUC : {auc:.3f}")

    return model


if __name__ == "__main__":
    print("Risk model skeleton ready. Plug in dataset next phase.")
    # Later: train_risk_model()

