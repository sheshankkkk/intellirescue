import os
import numpy as np
import pandas as pd

from flask import Flask, request, render_template_string

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------------- CONFIG ----------------
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

# Features you want to show when user clicks "Show key model inputs"
EXPLAIN_FEATURES = [
    "Year",
    "Seq",
    "Aid Contribution",
    "Dis Mag Value",
    "Start Year",
    "Start Month",
    "Start Day",
    "End Year",
    "End Month",
    "End Day",
    "No Injured",
    "No Affected",
    "No Homeless",
    "Total Affected",
    "Reconstruction Costs ('000 US$')",
]
# ----------------------------------------


def get_data_path():
    """Return path to dataset (full or sample)."""
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
    Load dataset, create HighImpact label, compute region risk,
    train a RandomForest model (used for predictions), and
    prepare defaults & feature names.
    """
    path = get_data_path()
    df = pd.read_csv(path)

    # Drop rows with missing impact
    df = df.dropna(subset=[IMPACT_COL])

    # HighImpact label
    threshold = df[IMPACT_COL].quantile(HIGH_IMPACT_QUANTILE)
    df["HighImpact"] = (df[IMPACT_COL] >= threshold).astype(int)

    # Region-level risk statistics
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

    # Feature columns for model
    cols_to_drop = NON_FEATURE_COLS + ["HighImpact"]
    feature_cols_raw = [c for c in df.columns if c not in cols_to_drop]

    X_raw = df[feature_cols_raw]
    y = df["HighImpact"]

    # One-hot encode categoricals
    X = pd.get_dummies(X_raw, drop_first=True)
    rf_feature_names = list(X.columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # RandomForest model
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf_model.fit(X_train, y_train)

    y_proba = rf_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "n_samples": int(len(df)),
    }

    # Defaults for form fields
    default_values = {}
    for col in feature_cols_raw:
        if pd.api.types.is_numeric_dtype(df[col]):
            default_values[col] = float(df[col].median())
        else:
            default_values[col] = str(df[col].mode().iloc[0])

    regions = sorted(df_region["Region"].dropna().unique().tolist())
    disaster_types = sorted(df["Disaster Type"].dropna().unique().tolist())

    print("[APP] Loaded dataset & trained RandomForest for web app.")
    return (
        metrics,
        region_risk,
        default_values,
        feature_cols_raw,
        regions,
        disaster_types,
        rf_model,
        rf_feature_names,
    )


# ---- Load data + model once at startup ----
(
    MODEL_METRICS,
    REGION_RISK,
    DEFAULT_VALUES,
    FEATURE_COLS_RAW,
    REGIONS,
    DISASTER_TYPES,
    RF_MODEL,
    RF_FEATURE_NAMES,
) = load_and_prepare_for_app()


def build_feature_vector_from_form(form_data):
    """
    Build a single feature vector for the RandomForest model from form data.
    Also return a dictionary of key features with original values for explanation.
    """
    # Start from defaults for all columns
    row = {col: DEFAULT_VALUES[col] for col in FEATURE_COLS_RAW}

    # Overwrite with user inputs
    if "Year" in row and form_data.get("year"):
        row["Year"] = int(form_data.get("year"))

    if "Region" in row and form_data.get("region"):
        row["Region"] = form_data.get("region")

    if "Disaster Type" in row and form_data.get("disaster_type"):
        row["Disaster Type"] = form_data.get("disaster_type")

    # Raw input (original-scale values)
    df_input_raw = pd.DataFrame([row])

    # One-hot encode to match training
    df_input = pd.get_dummies(df_input_raw, drop_first=True)

    # Ensure all RF features exist
    for col in RF_FEATURE_NAMES:
        if col not in df_input.columns:
            df_input[col] = 0

    # Reorder to RF feature order
    df_input = df_input[RF_FEATURE_NAMES]

    X_input = df_input.values.astype(float)

    # Prepare key features (original scale) for explanation
    feature_debug = []
    for name in EXPLAIN_FEATURES:
        if name in df_input_raw.columns:
            val = df_input_raw.loc[0, name]
            # Try to convert to float for numeric columns; keep as str otherwise
            try:
                val_clean = float(val)
            except (TypeError, ValueError):
                val_clean = str(val)
            feature_debug.append({"name": name, "value": val_clean})

    # Logging for debugging
    print("\n[DEBUG] Scenario:")
    print(
        "Year:", form_data.get("year"),
        "| Region:", form_data.get("region"),
        "| Type:", form_data.get("disaster_type")
    )
    print("[DEBUG] Key features:")
    for f in feature_debug:
        print("  ", f["name"], "=", f["value"])
    print("[DEBUG] ---------------------------\n")

    return X_input, feature_debug


def get_region_stats(region_name: str | None):
    """Look up region stats from REGION_RISK."""
    if not region_name:
        return None

    matches = REGION_RISK[REGION_RISK["Region"] == region_name]
    if matches.empty:
        return None

    row = matches.iloc[0]
    return {
        "region": row["Region"],
        "events": int(row["events"]),
        "high_impact": int(row["high_impact"]),
        "risk_rate": float(row["risk_rate"]),
    }


# ------------------------ FLASK APP ------------------------

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
<head>
    <title>INTELLIRESCUE: INTELLIGENT DISASTER MITIGATION SYSTEM</title>
    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 0;
            padding: 0;
            background: radial-gradient(circle at top left, #e0f2fe 0, #f9fafb 45%, #eef2ff 100%);
            color: #0f172a;
        }
        .page {
            max-width: 1150px;
            margin: 0 auto;
            padding: 24px 20px 40px;
        }
        h1 {
            margin-bottom: 4px;
            letter-spacing: 0.04em;
            font-size: 1.4rem;
        }
        .subtitle {
            margin-top: 0;
            color: #4b5563;
            font-size: 0.95rem;
        }
        .section-title {
            margin-top: 24px;
            margin-bottom: 4px;
            font-size: 1.15rem;
        }
        .section-subtitle {
            margin-top: 0;
            color: #6b7280;
            font-size: 0.9rem;
        }
        .header-card {
            margin-top: 10px;
            padding: 16px 18px;
            background: rgba(255,255,255,0.9);
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
            border: 1px solid rgba(148, 163, 184, 0.35);
        }
        .card {
            margin-top: 14px;
            padding: 18px 20px;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
            border: 1px solid #e5e7eb;
            transition: transform 0.18s ease-out, box-shadow 0.18s ease-out;
            animation: fadeUp 0.4s ease-out both;
        }
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.12);
        }
        .card-title {
            margin-top: 0;
            margin-bottom: 4px;
            font-size: 1.05rem;
        }
        .card-subtitle {
            margin-top: 0;
            margin-bottom: 10px;
            color: #6b7280;
            font-size: 0.9rem;
        }

        .layout {
            display: grid;
            grid-template-columns: minmax(0, 3fr) minmax(0, 2.4fr);
            gap: 18px;
        }

        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 10px;
            font-size: 0.9rem;
        }
        th, td {
            border: 1px solid #e5e7eb;
            padding: 6px 8px;
            text-align: left;
        }
        th {
            background-color: #f9fafb;
            font-weight: 600;
        }

        .btn {
            padding: 8px 18px;
            border: none;
            background: linear-gradient(135deg, #2563eb, #22c55e);
            color: white;
            border-radius: 999px;
            cursor: pointer;
            font-weight: 500;
            letter-spacing: 0.01em;
            box-shadow: 0 6px 14px rgba(37, 99, 235, 0.35);
            transition: transform 0.15s ease-out, box-shadow 0.15s ease-out;
        }
        .btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.45);
        }
        .btn-secondary {
            background: #4b5563;
            box-shadow: 0 4px 10px rgba(75, 85, 99, 0.4);
        }
        .btn-secondary:hover {
            box-shadow: 0 8px 18px rgba(75, 85, 99, 0.5);
        }

        .result-box {
            margin-top: 12px;
            padding: 12px 14px;
            border-radius: 12px;
            border-left: 4px solid;
            font-size: 0.93rem;
        }
        .low-risk {
            background-color: #ecfdf3;
            border-color: #16a34a;
        }
        .medium-risk {
            background-color: #fffbeb;
            border-color: #f97316;
        }
        .high-risk {
            background-color: #fef2f2;
            border-color: #dc2626;
        }

        label {
            display: inline-block;
            width: 120px;
            margin-bottom: 6px;
            font-size: 0.92rem;
        }
        select, input[type=number] {
            padding: 6px 8px;
            width: 220px;
            border-radius: 6px;
            border: 1px solid #d1d5db;
            font-size: 0.92rem;
        }

        .risk-chip {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.8em;
            color: #fff;
            margin-left: 6px;
        }
        .risk-high { background-color: #dc2626; }
        .risk-medium { background-color: #f97316; }
        .risk-low { background-color: #16a34a; }

        .metrics-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }
        .metric-pill {
            padding: 8px 10px;
            border-radius: 999px;
            background-color: #eff6ff;
            font-size: 0.85rem;
        }
        .metric-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6b7280;
        }
        .metric-value {
            font-weight: 600;
        }

        @keyframes fadeUp {
            from {
                opacity: 0;
                transform: translateY(6px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 800px) {
            .page {
                padding: 16px 12px 28px;
            }
            .layout {
                grid-template-columns: minmax(0,1fr);
            }
            label {
                width: 110px;
            }
        }
    </style>
</head>
<body>
    <div class="page">
        <div class="header-card">
            <h1>INTELLIRESCUE: INTELLIGENT DISASTER MITIGATION SYSTEM</h1>
            <p class="subtitle">
                Use historical multi-disaster data to estimate high-impact risk and explore vulnerable regions.
            </p>
        </div>

        <!-- MAIN TWO-COLUMN LAYOUT -->
        <div class="layout">
            <!-- LEFT COLUMN: FUTURE SCENARIO PREDICTION -->
            <div>
                <h2 class="section-title">Future Scenario Prediction</h2>
                <p class="section-subtitle">
                    Configure a hypothetical disaster scenario (including future years) and estimate its probability of being high-impact.
                </p>

                <div class="card">
                    <h3 class="card-title">Scenario Risk Forecast</h3>
                    <p class="card-subtitle">
                        Choose a year, region, and disaster type. IntelliRescue uses a RandomForest model trained on
                        multi-decade EOSDIS/EM-DAT style data to classify whether a scenario is likely to be high-impact
                        (top {{ (100 - HIGH_IMPACT_QUANTILE * 100) | round(0) }}% by deaths).
                    </p>

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
                        <p style="margin-top: 12px;">
                            <button type="submit" class="btn">Predict risk</button>
                        </p>
                    </form>

                    {% if prediction is not none %}
                        {# classify risk level based on probability #}
                        {% if prediction < 0.33 %}
                            {% set risk_label = "Mild / Low risk" %}
                            {% set risk_expl = "This scenario has relatively low likelihood of becoming a high-impact disaster, based on historical patterns." %}
                            {% set risk_class = "low-risk" %}
                        {% elif prediction < 0.66 %}
                            {% set risk_label = "Moderate risk" %}
                            {% set risk_expl = "This scenario has a moderate chance of becoming high-impact. Preparedness measures are recommended." %}
                            {% set risk_class = "medium-risk" %}
                        {% else %}
                            {% set risk_label = "High risk / disaster-prone" %}
                            {% set risk_expl = "This scenario is highly likely to become a high-impact disaster if it occurs. Strong preparedness and response capacity are critical." %}
                            {% set risk_class = "high-risk" %}
                        {% endif %}

                        <div class="result-box {{ risk_class }}">
                            <p><b>Predicted probability of HIGH-IMPACT disaster:</b>
                               {{ "%.2f"|format(prediction * 100) }}%</p>
                            <p><b>Risk level:</b> {{ risk_label }}</p>
                            <p>{{ risk_expl }}</p>
                            <p style="font-size: 0.85rem; color: #4b5563;">
                                This is a <b>scenario forecast</b>: you can choose future years (e.g. 2035) to estimate how prone a region is
                                to severe disasters of a given type.
                            </p>
                        </div>

                        {% if feature_debug %}
                            <p style="margin-top: 14px;">
                                <button type="button"
                                        id="toggle-features-btn"
                                        class="btn btn-secondary"
                                        onclick="toggleSection('model-input-details','toggle-features-btn','Show key model inputs','Hide key model inputs')">
                                    Show key model inputs
                                </button>
                            </p>
                            <div id="model-input-details"
                                 style="margin-top: 10px; padding: 8px 10px; border-radius: 8px; background: #f9fafb; border: 1px dashed #e5e7eb; font-size: 0.85rem; display: none;">
                                <p style="margin-top: 0;"><b>Key input features used for this scenario</b></p>
                                <p>
                                    Original-scale values for a subset of important features that went into the risk model.
                                </p>
                                <table>
                                    <tr>
                                        <th>Feature name</th>
                                        <th>Value (original scale)</th>
                                    </tr>
                                    {% for f in feature_debug %}
                                        <tr>
                                            <td>{{ f.name }}</td>
                                            <td>{{ f.value }}</td>
                                        </tr>
                                    {% endfor %}
                                </table>
                            </div>
                        {% endif %}
                    {% endif %}
                </div>
            </div>

            <!-- RIGHT COLUMN: HISTORICAL DETAILS -->
            <div>
                <h2 class="section-title">Historical Risk & Region Insights</h2>
                <p class="section-subtitle">
                    Understand how often regions have experienced high-impact disasters and how the model performs overall.
                </p>

                <div class="card">
                    <h3 class="card-title">Selected Region History</h3>
                    <p class="card-subtitle">
                        Summary of recorded disasters in the currently selected region.
                    </p>
                    {% if region_stats %}
                        {% set rr = region_stats.risk_rate %}
                        {% if rr >= 0.4 %}
                            {% set chip_class = "risk-high" %}
                            {% set chip_label = "High risk" %}
                        {% elif rr >= 0.2 %}
                            {% set chip_class = "risk-medium" %}
                            {% set chip_label = "Moderate risk" %}
                        {% else %}
                            {% set chip_class = "risk-low" %}
                            {% set chip_label = "Lower risk" %}
                        {% endif %}
                        <p>
                            <b>Region:</b> {{ region_stats.region }}
                            <span class="risk-chip {{ chip_class }}">{{ chip_label }}</span>
                        </p>
                        <p><b>Total recorded disasters:</b> {{ region_stats.events }}</p>
                        <p><b>High-impact disasters:</b> {{ region_stats.high_impact }}</p>
                        <p><b>High-impact rate:</b> {{ "%.1f"|format(region_stats.risk_rate * 100) }}%</p>
                        <p style="font-size: 0.86rem; color: #6b7280;">
                            Based on historical records across all years and hazard types for this region.
                        </p>
                    {% else %}
                        <p>No historical stats available for this region.</p>
                    {% endif %}
                </div>

                <div class="card">
                    <h3 class="card-title">High-Risk Regions Explorer</h3>
                    <p class="card-subtitle">
                        See which regions have the highest fraction of high-impact disasters.
                    </p>
                    <p>
                        <button type="button"
                                id="toggle-top-regions-btn"
                                class="btn btn-secondary"
                                onclick="toggleSection('top-regions-section','toggle-top-regions-btn','Show top regions','Hide top regions')">
                            Show top regions
                        </button>
                    </p>
                    <div id="top-regions-section" style="display: none;">
                        <table>
                            <tr>
                                <th>Rank</th>
                                <th>Region</th>
                                <th>Events</th>
                                <th>High-impact events</th>
                                <th>High-impact rate</th>
                            </tr>
                            {% for row in top_regions %}
                            <tr>
                                <td>{{ loop.index }}</td>
                                <td>{{ row.Region }}</td>
                                <td>{{ row.events }}</td>
                                <td>{{ row.high_impact }}</td>
                                <td>{{ "%.1f"|format(row.risk_rate * 100) }}%</td>
                            </tr>
                            {% endfor %}
                        </table>
                    </div>
                </div>

                <div class="card">
                    <h3 class="card-title">Dataset & Model Summary</h3>
                    <p class="card-subtitle">
                        How we define "high-impact" and how well the RandomForest performs on held-out data.
                    </p>
                    <div class="metrics-grid">
                        <div class="metric-pill">
                            <div class="metric-label">Impact column</div>
                            <div class="metric-value">{{ impact_col }}</div>
                        </div>
                        <div class="metric-pill">
                            <div class="metric-label">HighImpact threshold</div>
                            <div class="metric-value">≥ {{ metrics.threshold | round(1) }} deaths</div>
                        </div>
                        <div class="metric-pill">
                            <div class="metric-label">Samples</div>
                            <div class="metric-value">{{ metrics.n_samples }}</div>
                        </div>
                        <div class="metric-pill">
                            <div class="metric-label">RF Accuracy</div>
                            <div class="metric-value">{{ (metrics.accuracy * 100) | round(1) }}%</div>
                        </div>
                        <div class="metric-pill">
                            <div class="metric-label">RF ROC-AUC</div>
                            <div class="metric-value">{{ metrics.roc_auc | round(3) }}</div>
                        </div>
                    </div>
                    <p style="margin-top: 10px; font-size: 0.86rem; color: #6b7280;">
                        HighImpact = 1 if <b>{{ impact_col }}</b> is above the
                        {{ (HIGH_IMPACT_QUANTILE * 100) | round(0) }}th percentile of historical events
                        (representing the most severe disasters in the dataset).
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        function toggleSection(sectionId, buttonId, showLabel, hideLabel) {
            const section = document.getElementById(sectionId);
            const button = document.getElementById(buttonId);
            if (!section) return;

            const currentlyHidden = (section.style.display === "none" || section.style.display === "");
            if (currentlyHidden) {
                section.style.display = "block";
                if (button && hideLabel) button.textContent = hideLabel;
            } else {
                section.style.display = "none";
                if (button && showLabel) button.textContent = showLabel;
            }
        }
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    top_regions = REGION_RISK.head(10).to_dict(orient="records")
    default_year = 2000
    default_region = REGION_RISK.loc[0, "Region"]
    default_disaster_type = DISASTER_TYPES[0] if DISASTER_TYPES else ""
    region_stats = get_region_stats(default_region)

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
        feature_debug=None,
        region_stats=region_stats,
        HIGH_IMPACT_QUANTILE=HIGH_IMPACT_QUANTILE,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if RF_MODEL is None:
        return "RandomForest model not loaded.", 500

    form = request.form
    X_input, feature_debug = build_feature_vector_from_form(form)
    proba = float(RF_MODEL.predict_proba(X_input)[0, 1])

    # Debug log in terminal
    print(
        "[PREDICT] year=",
        form.get("year"),
        "region=",
        form.get("region"),
        "type=",
        form.get("disaster_type"),
        "-> proba=",
        proba,
    )

    top_regions = REGION_RISK.head(10).to_dict(orient="records")
    default_year = int(form.get("year") or 2000)
    default_region = form.get("region")
    default_disaster_type = form.get("disaster_type")
    region_stats = get_region_stats(default_region)

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
        prediction=proba,
        feature_debug=feature_debug,
        region_stats=region_stats,
        HIGH_IMPACT_QUANTILE=HIGH_IMPACT_QUANTILE,
    )


if __name__ == "__main__":
    app.run(debug=True)
