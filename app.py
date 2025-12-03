import os

from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------------- CONFIG (MATCH YOUR EXISTING SETUP) ----------------
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
HIGH_IMPACT_QUANTILE = 0.75
# -------------------------------------------------------------------


def get_data_path():
    if os.path.exists(CSV_PATH_FULL):
        print(f"[APP] Using FULL dataset at {CSV_PATH_FULL}")
        return CSV_PATH_FULL
    elif os.path.exists(CSV_PATH_SAMPLE):
        print(f"[APP] Using SAMPLE dataset at {CSV_PATH_SAMPLE}")
        return CSV_PATH_SAMPLE
    else:
        raise FileNotFoundError(
            f"Neither {CSV_PATH_FULL} nor {CSV_PATH_SAMPLE} found."
        )


def load_and_prepare_for_app():
    """
    Load the EOSDIS disasters dataset, create HighImpact label,
    prepare features, and train a RandomForest model.
    Returns:
      - trained model
      - list of feature columns used by the model
      - metrics dict
      - region risk table
      - feature default values (for building inputs from the form)
      - lists of available regions and disaster types for dropdowns
    """
    path = get_data_path()
    df = pd.read_csv(path)

    # Drop rows with missing impact
    df = df.dropna(subset=[IMPACT_COL])

    # Create HighImpact label based on quantile
    threshold = df[IMPACT_COL].quantile(HIGH_IMPACT_QUANTILE)
    df["HighImpact"] = (df[IMPACT_COL] >= threshold).astype(int)

    # Compute region-level risk for display
    df_region = df.dropna(subset=["Region"])
    grouped = df_region.groupby("Region")
    counts = grouped["HighImpact"].count()
    high_counts = grouped["HighImpact"].sum()
    risk_rate = high_counts / counts
    region_risk = (
        pd.DataFrame(
            {
                "Region": counts.index,
                "events": counts.values,
                "high_impact": high_counts.values,
                "risk_rate": risk_rate.values,
            }
        )
        .sort_values("risk_rate", ascending=False)
        .reset_index(drop=True)
    )

    # Prepare features
    cols_to_drop = NON_FEATURE_COLS + ["HighImpact"]
    feature_cols_raw = [c for c in df.columns if c not in cols_to_drop]

    X_raw = df[feature_cols_raw]
    y = df["HighImpact"]

    # One-hot encode categoricals
    X = pd.get_dummies(X_raw, drop_first=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Basic metrics for display
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "n_samples": int(len(df)),
    }

    # Default values for building an input row from partial user input
    default_values = {}
    for col in feature_cols_raw:
        if pd.api.types.is_numeric_dtype(df[col]):
            default_values[col] = float(df[col].median())
        else:
            default_values[col] = str(df[col].mode().iloc[0])

    # Collect dropdown options
    regions = sorted(df_region["Region"].dropna().unique().tolist())
    disaster_types = sorted(df["Disaster Type"].dropna().unique().tolist())

    return (
        model,
        list(X.columns),
        metrics,
        region_risk,
        default_values,
        feature_cols_raw,
        regions,
        disaster_types,
    )


# Train model and prepare helpers ONCE when the app starts
(
    MODEL,
    FEATURE_COLUMNS,
    MODEL_METRICS,
    REGION_RISK,
    DEFAULT_VALUES,
    FEATURE_COLS_RAW,
    REGIONS,
    DISASTER_TYPES,
) = load_and_prepare_for_app()


def build_feature_vector_from_form(form_data):
    """
    Build a single-row DataFrame with the same columns as the training data
    using form inputs (Year, Region, Disaster Type), and defaults for other features.
    """
    # Start from defaults
    row = {col: DEFAULT_VALUES[col] for col in FEATURE_COLS_RAW}

    # Overwrite with user-provided values (if those columns exist)
    if "Year" in row and form_data.get("year"):
        row["Year"] = int(form_data.get("year"))

    if "Region" in row and form_data.get("region"):
        row["Region"] = form_data.get("region")

    if "Disaster Type" in row and form_data.get("disaster_type"):
        row["Disaster Type"] = form_data.get("disaster_type")

    # Convert to DataFrame
    df_input_raw = pd.DataFrame([row])

    # One-hot encode in the same way as training
    df_input = pd.get_dummies(df_input_raw, drop_first=True)

    # Make sure all training columns exist
    for col in FEATURE_COLUMNS:
        if col not in df_input.columns:
            df_input[col] = 0

    # Keep the same column order as training
    df_input = df_input[FEATURE_COLUMNS]

    return df_input


# ------------------------ FLASK APP ------------------------

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
<head>
    <title>IntelliRescue Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; }
        h1 { margin-bottom: 0; }
        h2 { margin-top: 40px; }
        .metrics, .form-container, .region-table {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 6px;
        }
        table { border-collapse: collapse; width: 100%; margin-top: 10px; }
        th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }
        th { background-color: #f5f5f5; }
        .btn { padding: 8px 16px; border: none; background-color: #007bff;
               color: white; border-radius: 4px; cursor: pointer; }
        .btn:hover { background-color: #0056b3; }
        .result-box { margin-top: 15px; padding: 10px; border-radius: 6px; }
        .low-risk { background-color: #e3f7e3; }
        .high-risk { background-color: #fbe4e4; }
        label { display: inline-block; width: 150px; margin-bottom: 8px; }
        select, input[type=number] { padding: 4px; width: 220px; }
    </style>
</head>
<body>
    <h1>IntelliRescue: Disaster Risk Dashboard</h1>
    <p>Multi-disaster risk model using EOSDIS dataset + agent-based response.</p>

    <div class="metrics">
        <h2>Model Summary</h2>
        <p><b>Impact column:</b> {{ impact_col }} (HighImpact threshold = {{ metrics.threshold | round(2) }})</p>
        <p><b>Training samples:</b> {{ metrics.n_samples }}</p>
        <p><b>Accuracy:</b> {{ (metrics.accuracy * 100) | round(1) }}%</p>
        <p><b>ROC-AUC:</b> {{ metrics.roc_auc | round(3) }}</p>
        <p>HighImpact = 1 if {{ impact_col }} ≥ threshold; else 0.</p>
    </div>

    <div class="region-table">
        <h2>Top High-Risk Regions (by high-impact rate)</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Region</th>
                <th>Events</th>
                <th>High-impact events</th>
                <th>Risk rate</th>
            </tr>
            {% for row in top_regions %}
            <tr>
                <td>{{ loop.index }}</td>
                <td>{{ row.Region }}</td>
                <td>{{ row.events }}</td>
                <td>{{ row.high_impact }}</td>
                <td>{{ "%.3f"|format(row.risk_rate) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="form-container">
        <h2>Predict High-Impact Risk for a Scenario</h2>
        <form method="post" action="/predict">
            <p>
                <label for="year">Year:</label>
                <input type="number" id="year" name="year" value="{{ default_year }}" min="1900" max="2100">
            </p>
            <p>
                <label for="region">Region:</label>
                <select id="region" name="region">
                    {% for r in regions %}
                        <option value="{{ r }}" {% if r == default_region %}selected{% endif %}>{{ r }}</option>
                    {% endfor %}
                </select>
            </p>
            <p>
                <label for="disaster_type">Disaster Type:</label>
                <select id="disaster_type" name="disaster_type">
                    {% for t in disaster_types %}
                        <option value="{{ t }}" {% if t == default_disaster_type %}selected{% endif %}>{{ t }}</option>
                    {% endfor %}
                </select>
            </p>
            <p>
                <button type="submit" class="btn">Predict Risk</button>
            </p>
        </form>

        {% if prediction is not none %}
            <div class="result-box {% if prediction >= 0.5 %}high-risk{% else %}low-risk{% endif %}">
                <p><b>Predicted probability of HIGH-IMPACT disaster:</b>
                   {{ "%.1f"|format(prediction * 100) }}%</p>
                <p>Interpretation:
                    {% if prediction >= 0.5 %}
                        High-risk scenario. Region and hazard type historically associated with many severe events.
                    {% else %}
                        Lower-risk scenario relative to the top 25% most deadly events.
                    {% endif %}
                </p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    # Show top 10 regions by risk rate
    top_regions = REGION_RISK.head(10).to_dict(orient="records")

    # Simple default year
    default_year = 2000

    # Use highest-risk region and first disaster type as defaults
    default_region = REGION_RISK.loc[0, "Region"]
    default_disaster_type = DISASTER_TYPES[0] if DISASTER_TYPES else ""

    return render_template_string(
        INDEX_HTML,
        impact_col=IMPACT_COL,
        metrics=MODEL_METRICS,
        top_regions=top_regions,
        regions=REGIONS,
        disaster_types=DISASTER_TYPES,
        default_year=default_year,
        default_region=default_region,
        default_disaster_type=default_disaster_type,
        prediction=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    X_input = build_feature_vector_from_form(form)
    proba = MODEL.predict_proba(X_input)[0, 1]

    top_regions = REGION_RISK.head(10).to_dict(orient="records")
    default_year = int(form.get("year") or 2000)
    default_region = form.get("region")
    default_disaster_type = form.get("disaster_type")

    return render_template_string(
        INDEX_HTML,
        impact_col=IMPACT_COL,
        metrics=MODEL_METRICS,
        top_regions=top_regions,
        regions=REGIONS,
        disaster_types=DISASTER_TYPES,
        default_year=default_year,
        default_region=default_region,
        default_disaster_type=default_disaster_type,
        prediction=float(proba),
    )


if __name__ == "__main__":
    # Run the Flask development server
    app.run(debug=True)
