import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

# ---------------- CONFIG ----------------
IMPACT_COL = "Total Deaths"
CSV_PATH_FULL = "data/disasters.csv"
CSV_PATH_SAMPLE = "data/disasters_sample.csv"

PIPELINE_PATH = Path("models/disaster_risk_xgb.pkl")
METRICS_PATH = Path("models/disaster_risk_metrics.json")


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


def load_dataset_for_ui():
    """
    Load dataset for:
      - region risk table
      - dropdown options (regions, disaster types)
      - default form values
    """
    path = get_data_path()
    df = pd.read_csv(path)

    # Drop rows without impact
    df = df.dropna(subset=[IMPACT_COL])

    # HighImpact label (same rule as backend)
    threshold = 50.0  # your backend uses 50 deaths as threshold
    df["HighImpact"] = (df[IMPACT_COL] >= threshold).astype(int)

    # Region-level risk
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

    regions = sorted(df_region["Region"].dropna().unique().tolist())
    disaster_types = sorted(df["Disaster Type"].dropna().unique().tolist())

    default_values = {
        "Year": int(df["Year"].median()),
        "Region": df_region["Region"].mode().iloc[0],
        "Disaster Type": df["Disaster Type"].mode().iloc[0],
    }

    print("[APP] Dataset loaded for UI elements.")
    return region_risk, regions, disaster_types, default_values, int(len(df))


def load_model_pipeline():
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"{PIPELINE_PATH} not found. Run the backend training script first."
        )
    pipeline = joblib.load(PIPELINE_PATH)
    print(f"[APP] Loaded model pipeline from {PIPELINE_PATH}")
    return pipeline


def load_metrics():
    if not METRICS_PATH.exists():
        print("[APP] Metrics file not found; using empty defaults.")
        return {}
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    print(f"[APP] Loaded metrics from {METRICS_PATH}")
    return metrics


def get_region_stats(region_name: str | None, region_risk_df: pd.DataFrame):
    if not region_name:
        return None
    matches = region_risk_df[region_risk_df["Region"] == region_name]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return {
        "region": row["Region"],
        "events": int(row["events"]),
        "high_impact": int(row["high_impact"]),
        "risk_rate": float(row["risk_rate"]),
    }


def interpret_risk(prob: float):
    if prob >= 0.70:
        return "Severe / very high risk", "high-risk"
    elif prob >= 0.40:
        return "Elevated / moderate risk", "medium-risk"
    else:
        return "Mild / low risk", "low-risk"


# ---------- Load everything at startup ----------
REGION_RISK, REGIONS, DISASTER_TYPES, DEFAULT_VALUES, N_SAMPLES_UI = load_dataset_for_ui()
PIPELINE = load_model_pipeline()
MODEL_METRICS = load_metrics()

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
            background: radial-gradient(circle at top left, #111827 0, #020617 40%, #1f2937 100%);
            color: #e5e7eb;
        }
        .page {
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px 18px 40px;
        }
        .header {
            background: linear-gradient(90deg, #7f1d1d, #b91c1c, #f97316);
            border-radius: 0 0 18px 18px;
            padding: 18px 20px 22px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.65);
            margin-bottom: 16px;
        }
        .title {
            margin: 0;
            font-size: 1.6rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .subtitle {
            margin-top: 6px;
            color: #fee2e2;
            font-size: 0.9rem;
        }
        .badge-live {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(15,23,42,0.85);
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 10px;
        }
        .badge-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #f97316;
            box-shadow: 0 0 8px #fecaca;
        }
        .layout {
            display: grid;
            grid-template-columns: minmax(0, 3.2fr) minmax(0, 2.6fr);
            gap: 18px;
        }
        .card {
            background: radial-gradient(circle at top left, #111827, #020617);
            border-radius: 14px;
            padding: 18px 20px;
            border: 1px solid rgba(248, 113, 113, 0.25);
            box-shadow: 0 8px 26px rgba(0,0,0,0.7);
            position: relative;
            overflow: hidden;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }
        .card::before {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at top right, rgba(248,113,113,0.20), transparent 60%);
            opacity: 0.7;
            pointer-events: none;
        }
        .card:hover {
            transform: translateY(-2px);
            border-color: rgba(248, 113, 113, 0.6);
            box-shadow: 0 14px 32px rgba(0,0,0,0.85);
        }
        .card-title {
            margin: 0;
            font-size: 1.05rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }
        .card-subtitle {
            margin-top: 6px;
            margin-bottom: 12px;
            font-size: 0.9rem;
            color: #9ca3af;
        }
        label {
            display: block;
            margin-bottom: 4px;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
        }
        select, input[type=number] {
            width: 100%;
            padding: 8px 10px;
            border-radius: 8px;
            border: 1px solid #4b5563;
            background: #020617;
            color: #e5e7eb;
            font-size: 0.92rem;
            margin-bottom: 12px;
        }
        select:focus, input[type=number]:focus {
            outline: none;
            border-color: #f97316;
            box-shadow: 0 0 0 1px #f97316;
        }
        .btn {
            padding: 9px 18px;
            border-radius: 999px;
            border: none;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: linear-gradient(135deg, #ef4444, #f97316);
            color: #0b1120;
            box-shadow: 0 10px 24px rgba(248,113,113,0.6);
            transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
        }
        .btn:hover {
            transform: translateY(-1px);
            filter: brightness(1.05);
            box-shadow: 0 14px 32px rgba(248,113,113,0.8);
        }
        .btn-secondary {
            background: #111827;
            color: #f9fafb;
            border: 1px solid #4b5563;
            box-shadow: 0 6px 18px rgba(15,23,42,0.9);
        }
        .btn-secondary:hover {
            border-color: #f97316;
            box-shadow: 0 8px 22px rgba(15,23,42,1);
        }
        .btn-icon {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            background: #fecaca;
        }
        .result-box {
            margin-top: 14px;
            padding: 14px 14px;
            border-radius: 12px;
            border-left: 4px solid;
            background: rgba(15,23,42,0.95);
            font-size: 0.93rem;
        }
        .low-risk {
            border-color: #22c55e;
        }
        .medium-risk {
            border-color: #facc15;
        }
        .high-risk {
            border-color: #f97316;
        }
        .result-title {
            font-size: 1.0rem;
            margin: 0 0 4px 0;
        }
        .highlight {
            color: #f97316;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 8px;
            font-size: 0.85rem;
        }
        th, td {
            border: 1px solid #4b5563;
            padding: 6px 8px;
            text-align: left;
        }
        th {
            background: #020617;
            color: #f9fafb;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.75rem;
        }
        .risk-chip {
            display: inline-flex;
            align-items: center;
            padding: 2px 9px;
            border-radius: 999px;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .chip-high {
            background: rgba(248,113,113,0.18);
            color: #fecaca;
            border: 1px solid rgba(248,113,113,0.6);
        }
        .chip-med {
            background: rgba(250,204,21,0.1);
            color: #feecc7;
            border: 1px solid rgba(250,204,21,0.6);
        }
        .chip-low {
            background: rgba(34,197,94,0.12);
            color: #bbf7d0;
            border: 1px solid rgba(34,197,94,0.6);
        }
        .metrics-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .metric-pill {
            padding: 7px 10px;
            border-radius: 999px;
            background: #020617;
            border: 1px solid #374151;
            font-size: 0.8rem;
        }
        .metric-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #9ca3af;
        }
        .metric-value {
            font-weight: 600;
            color: #e5e7eb;
        }
        @media (max-width: 900px) {
            .layout { grid-template-columns: minmax(0,1fr); }
            .page { padding: 18px 12px 32px; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1 class="title">INTELLIRESCUE: INTELLIGENT DISASTER MITIGATION SYSTEM</h1>
        <p class="subtitle">
            AI-driven early warning and emergency resource planning using historical multi-disaster data.
        </p>
        <div class="badge-live">
            <div class="badge-dot"></div>
            LIVE RISK SIMULATOR
        </div>
    </div>

    <div class="page">
        <div class="layout">
            <!-- LEFT: Future scenario prediction -->
            <div>
                <div class="card">
                    <h2 class="card-title">SCENARIO FORECAST</h2>
                    <p class="card-subtitle">
                        Configure a hypothetical disaster scenario (including future years). IntelliRescue uses a
                        gradient-boosted XGBoost risk model trained on historical disasters to estimate whether the
                        event is likely to become <span class="highlight">high-impact</span> (top events by deaths).
                    </p>

                    <form method="post" action="/predict">
                        <label for="year">Year</label>
                        <input type="number" id="year" name="year" value="{{ default_year }}" min="1970" max="2100">

                        <label for="region">Region</label>
                        <select id="region" name="region">
                            {% for r in regions %}
                                <option value="{{ r }}" {% if r == default_region %}selected{% endif %}>{{ r }}</option>
                            {% endfor %}
                        </select>

                        <label for="disaster_type">Disaster Type</label>
                        <select id="disaster_type" name="disaster_type">
                            {% for t in disaster_types %}
                                <option value="{{ t }}" {% if t == default_disaster_type %}selected{% endif %}>{{ t }}</option>
                            {% endfor %}
                        </select>

                        <button type="submit" class="btn">
                            <span class="btn-icon"></span>
                            RUN RISK CHECK
                        </button>
                    </form>

                    {% if prediction is not none %}
                        <div class="result-box {{ risk_class }}">
                            <p class="result-title">
                                Predicted probability of <span class="highlight">HIGH-IMPACT</span> disaster:
                                <b>{{ "%.2f"|format(prediction * 100) }}%</b>
                            </p>
                            <p><b>Risk level:</b> {{ risk_label }}</p>
                            <p style="margin-top: 6px; color: #d1d5db;">
                                This scenario has a {{ risk_label|lower }} of escalating into a high-impact disaster,
                                based on similar historical events in the selected region and hazard type.
                            </p>
                        </div>

                        <p style="margin-top: 14px;">
                            <button type="button"
                                    id="toggle-scenario-btn"
                                    class="btn btn-secondary"
                                    onclick="toggleSection('scenario-details','toggle-scenario-btn','Show scenario details','Hide scenario details')">
                                Show scenario details
                            </button>
                        </p>

                        <div id="scenario-details"
                             style="display:none; margin-top: 10px; padding: 9px 10px; border-radius: 10px; background:#020617; border: 1px dashed #4b5563; font-size: 0.85rem;">
                            <p style="margin-top:0; margin-bottom:6px; color:#9ca3af;">
                                Scenario inputs used by the model:
                            </p>
                            <table>
                                <tr><th>Feature</th><th>Value</th></tr>
                                <tr><td>Year</td><td>{{ detail_year }}</td></tr>
                                <tr><td>Region</td><td>{{ detail_region }}</td></tr>
                                <tr><td>Disaster Type</td><td>{{ detail_disaster_type }}</td></tr>
                            </table>
                        </div>
                    {% endif %}
                </div>
            </div>

            <!-- RIGHT: Historical insights -->
            <div>
                <div class="card">
                    <h2 class="card-title">REGION RISK SNAPSHOT</h2>
                    <p class="card-subtitle">
                        Historical footprint of recorded disasters in the selected region. High-impact events are those
                        exceeding a severity threshold based on {{ impact_col }}.
                    </p>

                    {% if region_stats %}
                        {% set rr = region_stats.risk_rate %}
                        {% if rr >= 0.4 %}
                            {% set chip_class = "chip-high" %}
                            {% set chip_label = "HIGH RISK REGION" %}
                        {% elif rr >= 0.2 %}
                            {% set chip_class = "chip-med" %}
                            {% set chip_label = "ELEVATED RISK" %}
                        {% else %}
                            {% set chip_class = "chip-low" %}
                            {% set chip_label = "LOWER RISK" %}
                        {% endif %}
                        <p>
                            <b>Region:</b> {{ region_stats.region }}
                            <span class="risk-chip {{ chip_class }}">{{ chip_label }}</span>
                        </p>
                        <p><b>Total recorded disasters:</b> {{ region_stats.events }}</p>
                        <p><b>High-impact disasters:</b> {{ region_stats.high_impact }}</p>
                        <p><b>High-impact rate:</b> {{ "%.1f"|format(region_stats.risk_rate * 100) }}%</p>
                    {% else %}
                        <p>No historical stats available for this region.</p>
                    {% endif %}
                </div>

                <div class="card" style="margin-top:14px;">
                    <h2 class="card-title">TOP HIGH-RISK REGIONS</h2>
                    <p class="card-subtitle">
                        Ranked by fraction of disasters that historically became high-impact.
                    </p>
                    <p>
                        <button type="button"
                                id="toggle-top-btn"
                                class="btn btn-secondary"
                                onclick="toggleSection('top-regions','toggle-top-btn','Show high-risk regions','Hide high-risk regions')">
                            Show high-risk regions
                        </button>
                    </p>
                    <div id="top-regions" style="display:none;">
                        <table>
                            <tr>
                                <th>Rank</th>
                                <th>Region</th>
                                <th>Events</th>
                                <th>High-impact</th>
                                <th>Rate</th>
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

                <div class="card" style="margin-top:14px;">
                    <h2 class="card-title">MODEL STATUS</h2>
                    <p class="card-subtitle">
                        XGBoost-style gradient boosted classifier trained offline, then served here for fast risk checks.
                    </p>
                    <div class="metrics-grid">
                        <div class="metric-pill">
                            <div class="metric-label">Impact column</div>
                            <div class="metric-value">{{ impact_col }}</div>
                        </div>
                        {% if metrics.threshold is not none %}
                        <div class="metric-pill">
                            <div class="metric-label">High-impact threshold</div>
                            <div class="metric-value">≥ {{ metrics.threshold | round(1) }} deaths</div>
                        </div>
                        {% endif %}
                        {% if metrics.n_samples is not none %}
                        <div class="metric-pill">
                            <div class="metric-label">Samples used</div>
                            <div class="metric-value">{{ metrics.n_samples }}</div>
                        </div>
                        {% endif %}
                        {% if metrics.accuracy is not none %}
                        <div class="metric-pill">
                            <div class="metric-label">Test accuracy</div>
                            <div class="metric-value">{{ (metrics.accuracy * 100) | round(1) }}%</div>
                        </div>
                        {% endif %}
                        {% if metrics.roc_auc is not none %}
                        <div class="metric-pill">
                            <div class="metric-label">Test ROC-AUC</div>
                            <div class="metric-value">{{ metrics.roc_auc | round(3) }}</div>
                        </div>
                        {% endif %}
                        {% if metrics.model_type %}
                        <div class="metric-pill">
                            <div class="metric-label">Model</div>
                            <div class="metric-value">{{ metrics.model_type }}</div>
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function toggleSection(sectionId, buttonId, showLabel, hideLabel) {
            const section = document.getElementById(sectionId);
            const button = document.getElementById(buttonId);
            if (!section) return;
            const hidden = (section.style.display === "none" || section.style.display === "");
            if (hidden) {
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
    default_year = DEFAULT_VALUES["Year"]
    default_region = DEFAULT_VALUES["Region"]
    default_disaster_type = DEFAULT_VALUES["Disaster Type"]
    region_stats = get_region_stats(default_region, REGION_RISK)

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
        risk_label=None,
        risk_class="low-risk",
        detail_year=default_year,
        detail_region=default_region,
        detail_disaster_type=default_disaster_type,
        region_stats=region_stats,
    )


@app.route("/predict", methods=["POST"])
def predict():
    year_val = int(request.form.get("year") or DEFAULT_VALUES["Year"])
    region_val = request.form.get("region") or DEFAULT_VALUES["Region"]
    dtype_val = request.form.get("disaster_type") or DEFAULT_VALUES["Disaster Type"]

    X_input = pd.DataFrame(
        [{"Year": year_val, "Region": region_val, "Disaster Type": dtype_val}]
    )

    proba = float(PIPELINE.predict_proba(X_input)[0, 1])
    risk_label, risk_class = interpret_risk(proba)

    print(
        "[PREDICT]",
        "Year=", year_val,
        "| Region=", region_val,
        "| Type=", dtype_val,
        "-> proba=", proba,
    )

    top_regions = REGION_RISK.head(10).to_dict(orient="records")
    region_stats = get_region_stats(region_val, REGION_RISK)

    return render_template_string(
        INDEX_HTML,
        impact_col=IMPACT_COL,
        metrics=MODEL_METRICS,
        top_regions=top_regions,
        regions=REGIONS,
        disaster_types=DISASTER_TYPES,
        default_year=year_val,
        default_region=region_val,
        default_disaster_type=dtype_val,
        prediction=proba,
        risk_label=risk_label,
        risk_class=risk_class,
        detail_year=year_val,
        detail_region=region_val,
        detail_disaster_type=dtype_val,
        region_stats=region_stats,
    )


if __name__ == "__main__":
    print("[APP] Starting IntelliRescue emergency-response dashboard on http://127.0.0.1:5000")
    app.run(debug=True)
