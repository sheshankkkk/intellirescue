import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Path to your CSV (the one you put in data/)
CSV_PATH = "data/flood_data.csv"  # this is actually your traffic dataset


def load_and_featurize(path: str) -> pd.DataFrame:
    """
    Loads the traffic dataset and creates time-based features from DateTime.
    Expected columns: DateTime, Junction, Vehicles, ID
    """
    df = pd.read_csv(path)

    # Parse DateTime column
    df["DateTime"] = pd.to_datetime(df["DateTime"])

    # Time-based features
    df["hour"] = df["DateTime"].dt.hour
    df["dayofweek"] = df["DateTime"].dt.dayofweek  # 0=Mon
    df["month"] = df["DateTime"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)  # Sat/Sun

    # Drop columns we don't want as features
    df = df.drop(columns=["DateTime", "ID"])

    return df


def train_traffic_model(csv_path: str = CSV_PATH):
    """
    Trains a Random Forest regression model to predict Vehicles
    (traffic volume) based on time and junction features.
    """
    df = load_and_featurize(csv_path)

    # Target: Vehicles
    y = df["Vehicles"]
    X = df.drop(columns=["Vehicles"])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("=== Traffic Flow Prediction Model (Random Forest) ===")
    print(f"Features used   : {list(X.columns)}")
    print(f"MAE             : {mae:.3f} vehicles")
    print(f"RMSE            : {rmse:.3f} vehicles")
    print(f"R^2             : {r2:.3f}")

    return model


if __name__ == "__main__":
    train_traffic_model()
