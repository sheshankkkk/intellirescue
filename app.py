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
    elif os.path.exists(CV_PATH_SAMPLE := CSV_PATH_SAMPLE):
        print(f"[APP] Using SAMPLE dataset at {CV_PATH_SAMPLE}")
        return CV_PATH_SAMPLE
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
    """
    path = get_data_path()
    df = pd.read_csv(path)

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

    - Sample synthetic disaster events by region, proportional to historical event counts.
    - Each event becomes high-impact with probability equal to region's risk_rate.
    - Compare:
        * naive: teams placed on random regions
        * risk_aware: teams placed on highest-risk regions
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
    n_events = years * 50  # arbitrary scaling: 50 synthetic events per year

    region_idx = rng.choice(n_regions, size=n_events, p=weights)
    # whether each event becomes high-impact
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
        <div class="layout">
            <!-- LEFT: Future scenario prediction -->
            <div>
                <div class="card">
                    <h2 class="card-title">SCENARIO FORECAST</h2>
                    <p class="card-subtitle">
                        Configure a hypothetical disaster scenario (including future years). IntelliRescue uses
                        an XGBoost risk model trained on historical disasters to estimate whether the event is likely
                        to become <span class="highlight">high-impact</span> (top events by deaths).
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

            <!-- RIGHT: Historical insights + simulation + emergency resources -->
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

                        <button type="submit" class="btn">
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

                <div class="card" style="margin-top:14px;">
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

                <div class="card" style="margin-top:14px;">
                    <h2 class="card-title">EMERGENCY RESPONSE RESOURCES</h2>
                    <p class="card-subtitle">
                        Practical guidance that emergency planners and citizens can use alongside IntelliRescue’s
                        predictions. This content is generic – it should be adapted to your city / campus guidelines.
                    </p>

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
):
    top_regions = REGION_RISK.head(10).to_dict(orient="records")

    if year_val is None:
        year_val = DEFAULT_VALUES["Year"]
    if region_val is None:
        region_val = DEFAULT_VALUES["Region"]
    if dtype_val is None:
        dtype_val = DEFAULT_VALUES["Disaster Type"]

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
        prediction=prediction,
        risk_label=risk_label,
        risk_class=risk_class,
        detail_year=year_val,
        detail_region=region_val,
        detail_disaster_type=dtype_val,
        region_stats=region_stats,
        sim_results=sim_results,
        sim_teams=sim_teams,
        sim_years=sim_years,
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

    print(
        "[PREDICT]",
        "Year=", year_val,
        "| Region=", region_val,
        "| Type=", dtype_val,
        "-> proba=", proba,
    )

    return _render_index(
        prediction=proba,
        risk_label=risk_label,
        risk_class=risk_class,
        year_val=year_val,
        region_val=region_val,
        dtype_val=dtype_val,
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

    # keep scenario inputs at defaults on simulation route
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
    )


if __name__ == "__main__":
    print("[APP] Starting IntelliRescue dashboard on http://127.0.0.1:5000")
    app.run(debug=True)
