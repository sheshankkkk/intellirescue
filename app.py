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
      - region risk summary
      - dropdown options (regions, disaster types)
      - default form values
      - scenario-level statistics
      - approximate geo coordinates per region
    """
    path = get_data_path()
    df = pd.read_csv(path)

    # Keep a copy for scenario stats
    base_df = df.copy()

    # Drop rows without impact
    df = df.dropna(subset=[IMPACT_COL])

    # HighImpact label (same logical threshold as backend)
    threshold = 50.0  # >= 50 deaths is high-impact
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

    # ---- approximate geo-coordinates per region (for map) ----
    df_geo = df_region.copy()
    df_geo["Latitude_num"] = pd.to_numeric(df_geo.get("Latitude"), errors="coerce")
    df_geo["Longitude_num"] = pd.to_numeric(df_geo.get("Longitude"), errors="coerce")

    geo_group = (
        df_geo.groupby("Region")[["Latitude_num", "Longitude_num"]]
        .mean()
        .reset_index()
    )

    region_risk = region_risk.merge(geo_group, on="Region", how="left")
    region_risk.rename(
        columns={"Latitude_num": "lat", "Longitude_num": "lon"}, inplace=True
    )

    regions = sorted(df_region["Region"].dropna().unique().tolist())
    disaster_types = sorted(df["Disaster Type"].dropna().unique().tolist())

    default_values = {
        "Year": int(df["Year"].median()),
        "Region": df_region["Region"].mode().iloc[0],
        "Disaster Type": df["Disaster Type"].mode().iloc[0],
    }

    print("[APP] Dataset loaded for UI elements.")
    return region_risk, regions, disaster_types, default_values, int(len(df)), base_df


def load_model_pipeline():
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"{PIPELINE_PATH} not found. Run backend training first."
        )
    pipeline = joblib.load(PIPELINE_PATH)
    print(f"[APP] Loaded model pipeline from {PIPELINE_PATH}")
    return pipeline


def load_metrics():
    if not METRICS_PATH.exists():
        print("[APP] Metrics file not found; using empty defaults.")
        return {
            "model_type": None,
            "threshold": None,
            "accuracy": None,
            "roc_auc": None,
            "n_samples": None,
        }
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


def get_scenario_stats(year: int, region: str, dtype: str, df: pd.DataFrame):
    """
    Compute simple, understandable statistics from the dataset for the chosen scenario:
      - Median Total Deaths
      - Median No Injured
      - Median Total Affected
    using all historical events with the same Region and Disaster Type.
    """
    if "Region" not in df.columns or "Disaster Type" not in df.columns:
        return None

    subset = df.copy()
    subset = subset[subset["Region"] == region]
    subset = subset[subset["Disaster Type"] == dtype]

    if subset.empty:
        return None

    stats = {"n_events": int(len(subset))}

    def median_if(col):
        if col in subset.columns:
            val = subset[col].dropna().median()
            return None if pd.isna(val) else float(val)
        return None

    stats["median_deaths"] = median_if("Total Deaths")
    stats["median_injured"] = median_if("No Injured")
    stats["median_affected"] = median_if("Total Affected")

    return stats


def interpret_risk(prob: float):
    if prob >= 0.70:
        return "Severe / very high risk", "high-risk"
    elif prob >= 0.40:
        return "Elevated / moderate risk", "medium-risk"
    else:
        return "Mild / low risk", "low-risk"


def run_allocation_sim(region_risk_df: pd.DataFrame, teams: int, years: int, seed: int = 42):
    """
    Very lightweight Monte Carlo simulation inspired by emergency_simulation.py.
    """
    rng = np.random.default_rng(seed)

    regions = region_risk_df["Region"].values
    events = region_risk_df["events"].values.astype(float)
    risk_rates = region_risk_df["risk_rate"].values.astype(float)

    n_regions = len(regions)
    teams = max(1, min(teams, n_regions))
    years = max(1, years)

    # sample synthetic events
    weights = events / events.sum()
    n_events = years * 50  # arbitrary scaling

    region_idx = rng.choice(n_regions, size=n_events, p=weights)
    high_flags = rng.random(n_events) < risk_rates[region_idx]

    if high_flags.sum() == 0:
        return {
            "teams": teams,
            "years": years,
            "total_high": 0,
            "naive_covered": 0,
            "risk_aware_covered": 0,
            "naive_rate": 0.0,
            "risk_aware_rate": 0.0,
        }

    # naive strategy: random assignment of teams
    naive_assigned = rng.choice(n_regions, size=teams, replace=False)
    naive_covered_mask = high_flags & np.isin(region_idx, naive_assigned)
    naive_covered = int(naive_covered_mask.sum())

    # risk-aware: assign teams to top risk regions
    top_indices = np.argsort(risk_rates)[::-1][:teams]
    aware_covered_mask = high_flags & np.isin(region_idx, top_indices)
    aware_covered = int(aware_covered_mask.sum())

    total_high = int(high_flags.sum())

    return {
        "teams": teams,
        "years": years,
        "total_high": total_high,
        "naive_covered": naive_covered,
        "risk_aware_covered": aware_covered,
        "naive_rate": naive_covered / total_high if total_high > 0 else 0.0,
        "risk_aware_rate": aware_covered / total_high if total_high > 0 else 0.0,
    }


# ---------- Load everything at startup ----------
(
    REGION_RISK,
    REGIONS,
    DISASTER_TYPES,
    DEFAULT_VALUES,
    N_SAMPLES_UI,
    DISASTER_DF,
) = load_dataset_for_ui()
PIPELINE = load_model_pipeline()
MODEL_METRICS = load_metrics()

# ------------------------ FLASK APP ------------------------
app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
<head>
    <title>INTELLIRESCUE: INTELLIGENT DISASTER MITIGATION SYSTEM</title>
    <!-- Plotly for geographic map -->
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 0;
            padding: 0;
            background: radial-gradient(circle at top left, #0f172a 0, #020617 40%, #020617 100%);
            color: #e5e7eb;
        }
        .page {
            max-width: 1240px;
            margin: 0 auto;
            padding: 20px 18px 40px;
        }
        .header {
            background: linear-gradient(90deg, #7f1d1d, #b91c1c, #f97316);
            border-radius: 0 0 18px 18px;
            padding: 18px 20px 22px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.65);
            margin-bottom: 10px;
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
            margin-top: 8px;
        }
        .badge-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #f97316;
            box-shadow: 0 0 8px #fecaca;
        }

        .layout-row {
            display: grid;
            grid-template-columns: minmax(0, 1.5fr) minmax(0, 1.4fr);
            gap: 18px;
            margin-top: 12px;
        }
        .layout-row-bottom {
            display: grid;
            grid-template-columns: minmax(0, 1.4fr) minmax(0, 1.6fr);
            gap: 18px;
            margin-top: 18px;
        }

        .card {
            background: radial-gradient(circle at top left, #111827, #020617);
            border-radius: 14px;
            padding: 18px 20px;
            border: 1px solid rgba(148,163,184,0.35);
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
            opacity: 0.6;
            pointer-events: none;
        }
        .card:hover {
            transform: translateY(-2px);
            border-color: rgba(248, 113, 113, 0.6);
            box-shadow: 0 14px 32px rgba(0,0,0,0.85);
        }
        .card-title {
            margin: 0;
            font-size: 1.02rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }
        .card-subtitle {
            margin-top: 6px;
            margin-bottom: 12px;
            font-size: 0.88rem;
            color: #9ca3af;
        }

        label {
            display: block;
            margin-bottom: 4px;
            font-size: 0.8rem;
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
            margin-bottom: 10px;
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
            font-size: 0.86rem;
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
        .low-risk { border-color: #22c55e; }
        .medium-risk { border-color: #facc15; }
        .high-risk { border-color: #f97316; }

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

        /* map container */
        #risk-map {
            width: 100%;
            height: 360px;
        }

        @media (max-width: 960px) {
            .layout-row,
            .layout-row-bottom {
                grid-template-columns: minmax(0, 1fr);
            }
            .page { padding: 18px 12px 32px; }
        }

        a {
            color: #f97316;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
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
        <!-- ROW 1: Scenario (left) + Region risk + model status (right) -->
        <div class="layout-row">
            <!-- LEFT: Scenario forecast -->
            <div>
                <div class="card">
                    <h2 class="card-title">SCENARIO FORECAST</h2>
                    <p class="card-subtitle">
                        Configure a hypothetical or future disaster scenario. IntelliRescue estimates the probability
                        that it becomes <span class="highlight">high-impact</span> based on similar historical events.
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

                        <button type="submit" class="btn" style="margin-top:10px;">
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

                        <p style="margin-top: 12px;">
                            <button type="button"
                                    id="toggle-scenario-btn"
                                    class="btn btn-secondary"
                                    onclick="toggleSection('scenario-details','toggle-scenario-btn','Show scenario details','Hide scenario details')">
                                Show scenario details
                            </button>
                        </p>

                        <div id="scenario-details"
                             style="display:none; margin-top: 8px; padding: 9px 10px; border-radius: 10px; background:#020617; border: 1px dashed #4b5563; font-size: 0.85rem;">
                            <p style="margin-top:0; margin-bottom:6px; color:#9ca3af;">
                                Scenario inputs and typical historical impact for similar events:
                            </p>
                            <table>
                                <tr><th>Feature</th><th>Value</th></tr>
                                <tr><td>Scenario Year</td><td>{{ detail_year }}</td></tr>
                                <tr><td>Region</td><td>{{ detail_region }}</td></tr>
                                <tr><td>Disaster Type</td><td>{{ detail_disaster_type }}</td></tr>

                                {% if scenario_stats %}
                                    <tr>
                                        <td>Historical events (same region &amp; type)</td>
                                        <td>{{ scenario_stats.n_events }}</td>
                                    </tr>
                                    {% if scenario_stats.median_deaths is not none %}
                                    <tr>
                                        <td>Typical total deaths (median)</td>
                                        <td>{{ scenario_stats.median_deaths | round(1) }}</td>
                                    </tr>
                                    {% endif %}
                                    {% if scenario_stats.median_injured is not none %}
                                    <tr>
                                        <td>Typical total injured (median)</td>
                                        <td>{{ scenario_stats.median_injured | round(1) }}</td>
                                    </tr>
                                    {% endif %}
                                    {% if scenario_stats.median_affected is not none %}
                                    <tr>
                                        <td>Typical people affected / displaced (median)</td>
                                        <td>{{ scenario_stats.median_affected | round(1) }}</td>
                                    </tr>
                                    {% endif %}
                                {% else %}
                                    <tr>
                                        <td colspan="2">
                                            No matching historical events found in the dataset for this region
                                            and disaster type.
                                        </td>
                                    </tr>
                                {% endif %}
                            </table>
                        </div>
                    {% endif %}
                </div>
            </div>

            <!-- RIGHT: Region risk snapshot + model status stacked -->
            <div style="display:flex; flex-direction:column; gap:14px;">
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

                <div class="card">
                    <h2 class="card-title">MODEL STATUS</h2>
                    <p class="card-subtitle">
                        XGBoost classifier trained offline (80% train / 20% test), then deployed here for
                        real-time risk checks and simulations.
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

        <!-- ROW 2: Simulation + resources (left) / Map (right) -->
        <div class="layout-row-bottom">
            <!-- LEFT: simulation + resources -->
            <div style="display:flex; flex-direction:column; gap:14px;">
                <div class="card">
                    <h2 class="card-title">EMERGENCY TEAM ALLOCATION (SIMULATION)</h2>
                    <p class="card-subtitle">
                        Compare a <span class="highlight">naive</span> vs
                        <span class="highlight">risk-aware</span> team deployment strategy using a Monte Carlo simulation
                        on the historical risk map.
                    </p>

                    <form method="post" action="/simulate">
                        <label for="sim_teams">Number of teams</label>
                        <input type="number" id="sim_teams" name="sim_teams" value="{{ sim_teams }}" min="1" max="25">

                        <label for="sim_years">Simulation years</label>
                        <input type="number" id="sim_years" name="sim_years" value="{{ sim_years }}" min="1" max="100">

                        <button type="submit" class="btn" style="margin-top:8px;">
                            <span class="btn-icon"></span>
                            RUN ALLOCATION SIM
                        </button>
                    </form>

                    {% if sim_results %}
                        <div class="result-box medium-risk" style="margin-top: 12px;">
                            <p class="result-title">Simulation summary</p>
                            <p style="margin: 4px 0;">
                                Teams: <b>{{ sim_results.teams }}</b> |
                                Years simulated: <b>{{ sim_results.years }}</b> |
                                High-impact events (simulated): <b>{{ sim_results.total_high }}</b>
                            </p>
                            <table>
                                <tr>
                                    <th>Strategy</th>
                                    <th>Covered high-impact events</th>
                                    <th>Coverage rate</th>
                                </tr>
                                <tr>
                                    <td>Naive allocation</td>
                                    <td>{{ sim_results.naive_covered }}</td>
                                    <td>{{ "%.1f"|format(sim_results.naive_rate * 100) }}%</td>
                                </tr>
                                <tr>
                                    <td>Risk-aware allocation</td>
                                    <td>{{ sim_results.risk_aware_covered }}</td>
                                    <td>{{ "%.1f"|format(sim_results.risk_aware_rate * 100) }}%</td>
                                </tr>
                            </table>
                            <p style="margin-top: 6px; font-size: 0.83rem; color:#e5e7eb;">
                                In this run, assigning teams to historically high-risk regions
                                <span class="highlight">increased coverage of severe events</span>
                                compared to a naive, random deployment.
                            </p>
                        </div>
                    {% endif %}
                </div>

                <div class="card">
                    <h2 class="card-title">EMERGENCY RESPONSE RESOURCES</h2>
                    <p class="card-subtitle">
                        Practical guidance that emergency planners and citizens can use alongside IntelliRescue’s
                        predictions. Adapt this section to your local city / campus guidelines.
                    </p>

                    <button type="button"
                            id="toggle-resources-btn"
                            class="btn btn-secondary"
                            onclick="toggleSection('resources-content','toggle-resources-btn','Show emergency response resources','Hide emergency response resources')">
                        Show emergency response resources
                    </button>

                    <div id="resources-content" style="display:none; margin-top:10px;">
                        <h3 style="margin-top:4px; margin-bottom:4px; font-size:0.9rem;">Before a disaster (Preparedness)</h3>
                        <ul style="margin-top:4px; padding-left:18px; font-size:0.86rem;">
                            <li>Create a basic emergency kit: water, non-perishable food, torch, power bank, first-aid box, essential medicines, copies of ID cards.</li>
                            <li>Keep a small “grab-and-go” bag ready near the door for fast evacuation.</li>
                            <li>Agree on a common family / team meeting point if phones stop working.</li>
                            <li>Save local emergency numbers and nearby hospital numbers on your phone.</li>
                        </ul>

                        <h3 style="margin-top:8px; margin-bottom:4px; font-size:0.9rem;">During a disaster (Immediate response)</h3>
                        <ul style="margin-top:4px; padding-left:18px; font-size:0.86rem;">
                            <li>Follow official warnings and evacuation orders – do not wait “to confirm” if the area is marked high risk.</li>
                            <li>Flooding: avoid walking or driving through water; even shallow fast water can sweep vehicles away.</li>
                            <li>Earthquakes: <b>Drop, Cover, Hold On</b> – drop low, cover your head and neck, hold onto sturdy furniture away from windows.</li>
                            <li>Fire / smoke: stay low, cover your nose and mouth with cloth, exit by stairs (never lifts), and close doors behind you.</li>
                        </ul>

                        <h3 style="margin-top:8px; margin-bottom:4px; font-size:0.9rem;">After a disaster</h3>
                        <ul style="margin-top:4px; padding-left:18px; font-size:0.86rem;">
                            <li>Check yourself and others for injuries; call emergency services for serious injuries.</li>
                            <li>Stay away from damaged buildings, fallen electric lines, and unstable bridges.</li>
                            <li>Use SMS / data messages instead of phone calls to avoid overloading networks.</li>
                            <li>Listen to local authorities for information about safe shelters, medical camps, and relief distribution.</li>
                        </ul>

                        <h3 style="margin-top:8px; margin-bottom:4px; font-size:0.9rem;">Important contacts (to customise)</h3>
                        <ul style="margin-top:4px; padding-left:18px; font-size:0.86rem;">
                            <li><b>Local emergency number:</b> 1 0 X (example – replace with your country’s number).</li>
                            <li><b>City disaster control room:</b> add phone and email for your city / campus.</li>
                            <li><b>Nearest major hospital:</b> add at least 2–3 hospital contacts.</li>
                            <li><b>College / organisation helpline:</b> add your institute’s emergency contact.</li>
                        </ul>

                        <p style="margin-top:8px; font-size:0.8rem; color:#9ca3af;">
                            IntelliRescue can highlight <span class="highlight">where</span> high-impact events are likely.
                            These emergency response guidelines help decide <span class="highlight">how people and teams should react</span>
                            when a warning is issued.
                        </p>
                    </div>
                </div>
            </div>

            <!-- RIGHT: Global risk map -->
            <div>
                <div class="card" style="height:100%; display:flex; flex-direction:column;">
                    <h2 class="card-title">GLOBAL RISK MAP</h2>
                    <p class="card-subtitle">
                        Approximate world map showing historical regions coloured by high-impact rate.
                        Larger, brighter points indicate regions with more frequent severe disasters.
                    </p>
                    <div id="risk-map" style="flex:1;"></div>
                    <p style="margin-top:6px; font-size:0.78rem; color:#9ca3af;">
                        Note: Coordinates are approximated from recorded event centroids per region.
                        This visual is intended for communication / education, not precise navigation.
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
            const hidden = (section.style.display === "none" || section.style.display === "");
            if (hidden) {
                section.style.display = "block";
                if (button && hideLabel) button.textContent = hideLabel;
            } else {
                section.style.display = "none";
                if (button && showLabel) button.textContent = showLabel;
            }
        }

        // ---- GLOBAL RISK MAP ----
        const REGION_POINTS = {{ region_points_json | safe }};

        document.addEventListener("DOMContentLoaded", function () {
            const pts = REGION_POINTS.filter(p => p.lat !== null && p.lon !== null);
            if (pts.length === 0) {
                return;
            }

            const lats = pts.map(p => p.lat);
            const lons = pts.map(p => p.lon);
            const texts = pts.map(p =>
                `${p.Region}<br>` +
                `High-impact rate: ${(p.risk_rate * 100).toFixed(1)}%<br>` +
                `Events: ${p.events} | High-impact: ${p.high_impact}`
            );
            const sizes = pts.map(p => 8 + 25 * p.risk_rate);
            const colors = pts.map(p => p.risk_rate);

            const data = [{
                type: "scattergeo",
                mode: "markers",
                lat: lats,
                lon: lons,
                text: texts,
                hoverinfo: "text",
                marker: {
                    size: sizes,
                    color: colors,
                    colorscale: "Reds",
                    cmin: 0,
                    cmax: 0.5,
                    colorbar: {
                        title: "High-impact rate",
                        outlinewidth: 0
                    },
                    line: {
                        color: "rgba(15,23,42,0.9)",
                        width: 0.6
                    },
                    opacity: 0.9
                }
            }];

            const layout = {
                geo: {
                    scope: "world",
                    projection: { type: "natural earth" },
                    showland: true,
                    landcolor: "rgb(15,23,42)",
                    oceancolor: "rgb(15,23,42)",
                    showocean: true,
                    coastlinecolor: "rgb(55,65,81)",
                    countrycolor: "rgb(55,65,81)",
                    lataxis: { showgrid: true, gridwidth: 0.2, range: [-60, 80], dtick: 20 },
                    lonaxis: { showgrid: true, gridwidth: 0.2, range: [-180, 180], dtick: 60 }
                },
                margin: { t: 10, b: 0, l: 0, r: 0 },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
            };

            Plotly.newPlot("risk-map", data, layout, {displayModeBar: false});
        });
    </script>
</body>
</html>
"""


def _render_index(
    prediction=None,
    risk_label=None,
    risk_class="low-risk",
    year_val=None,
    region_val=None,
    dtype_val=None,
    sim_results=None,
    sim_teams=10,
    sim_years=50,
    scenario_stats=None,
):
    top_regions = REGION_RISK.head(10).to_dict(orient="records")

    if year_val is None:
        year_val = DEFAULT_VALUES["Year"]
    if region_val is None:
        region_val = DEFAULT_VALUES["Region"]
    if dtype_val is None:
        dtype_val = DEFAULT_VALUES["Disaster Type"]

    region_stats = get_region_stats(region_val, REGION_RISK)

    region_points_df = REGION_RISK[
        ["Region", "risk_rate", "lat", "lon", "events", "high_impact"]
    ]
    region_points = json.dumps(region_points_df.to_dict(orient="records"))

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
        prediction=prediction,
        risk_label=risk_label,
        risk_class=risk_class,
        detail_year=year_val,
        detail_region=region_val,
        detail_disaster_type=dtype_val,
        scenario_stats=scenario_stats,
        region_stats=region_stats,
        sim_results=sim_results,
        sim_teams=sim_teams,
        sim_years=sim_years,
        region_points_json=region_points,
    )


@app.route("/", methods=["GET"])
def index():
    return _render_index()


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

    scenario_stats = get_scenario_stats(year_val, region_val, dtype_val, DISASTER_DF)

    print(
        "[PREDICT]",
        "Year=", year_val,
        "| Region=", region_val,
        "| Type=", dtype_val,
        "-> proba=", proba,
        "| scenario_stats=", scenario_stats,
    )

    return _render_index(
        prediction=proba,
        risk_label=risk_label,
        risk_class=risk_class,
        year_val=year_val,
        region_val=region_val,
        dtype_val=dtype_val,
        scenario_stats=scenario_stats,
    )


@app.route("/simulate", methods=["POST"])
def simulate():
    sim_teams = int(request.form.get("sim_teams") or 10)
    sim_years = int(request.form.get("sim_years") or 50)

    sim_results = run_allocation_sim(REGION_RISK, sim_teams, sim_years)

    print(
        "[SIMULATE] teams=", sim_teams,
        "| years=", sim_years,
        "| total_high=", sim_results["total_high"],
        "| naive_rate=", sim_results["naive_rate"],
        "| risk_aware_rate=", sim_results["risk_aware_rate"],
    )

    return _render_index(
        prediction=None,
        risk_label=None,
        risk_class="low-risk",
        year_val=DEFAULT_VALUES["Year"],
        region_val=DEFAULT_VALUES["Region"],
        dtype_val=DEFAULT_VALUES["Disaster Type"],
        sim_results=sim_results,
        sim_teams=sim_teams,
        sim_years=sim_years,
        scenario_stats=None,
    )


if __name__ == "__main__":
    print("[APP] Starting IntelliRescue dashboard on http://127.0.0.1:5000")
    app.run(debug=True)
