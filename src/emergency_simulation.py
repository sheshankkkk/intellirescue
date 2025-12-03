import os
import random
from collections import Counter

import numpy as np
import pandas as pd


# === CONFIG (MATCHES YOUR DISASTER DATASET) ==================
IMPACT_COL = "Total Deaths"
CSV_PATH_FULL = "data/disasters.csv"
CSV_PATH_SAMPLE = "data/disasters_sample.csv"
HIGH_IMPACT_QUANTILE = 0.75  # top 25% by deaths -> HighImpact
# ============================================================


def get_data_path():
    """Prefer full dataset; fall back to sample."""
    if os.path.exists(CSV_PATH_FULL):
        print(f"Using FULL dataset at {CSV_PATH_FULL}")
        return CSV_PATH_FULL
    elif os.path.exists(CSV_PATH_SAMPLE):
        print(f"Using SAMPLE dataset at {CSV_PATH_SAMPLE}")
        return CSV_PATH_SAMPLE
    else:
        raise FileNotFoundError(
            f"Neither {CSV_PATH_FULL} nor {CSV_PATH_SAMPLE} found."
        )


def load_and_label_disasters():
    """Load dataset, create HighImpact label, return DataFrame."""
    path = get_data_path()
    df = pd.read_csv(path)

    # Drop rows without impact info
    df = df.dropna(subset=[IMPACT_COL])

    # Compute threshold for high impact
    threshold = df[IMPACT_COL].quantile(HIGH_IMPACT_QUANTILE)
    df["HighImpact"] = (df[IMPACT_COL] >= threshold).astype(int)

    print(f"HighImpact threshold on {IMPACT_COL}: {threshold:.2f}")
    print(f"Total rows after cleaning: {len(df)}")
    print(f"HighImpact events: {df['HighImpact'].sum()}")

    return df, threshold


def compute_region_risk(df: pd.DataFrame, region_col: str = "Region"):
    """
    Compute high-impact disaster rate per region:
    risk(region) = P(HighImpact=1 | Region)
    """
    # Filter rows that have region info
    df_region = df.dropna(subset=[region_col])

    grouped = df_region.groupby(region_col)
    counts = grouped["HighImpact"].count()
    high_counts = grouped["HighImpact"].sum()

    risk = (high_counts / counts).sort_values(ascending=False)
    risk_df = pd.DataFrame(
        {
            "events": counts,
            "high_impact": high_counts,
            "risk_rate": risk,
        }
    ).sort_values("risk_rate", ascending=False)

    print("\n=== Top 10 regions by high-impact risk ===")
    print(risk_df.head(10))

    return risk_df


def simulate_strategy(
    df: pd.DataFrame,
    region_risk: pd.DataFrame,
    num_teams: int = 10,
    years: int = 50,
    events_per_year: int = 50,
    strategy: str = "naive",
    random_seed: int = 42,
):
    """
    Simple agent-based simulation:
    - Each 'event' is a sampled disaster from the historical dataset.
    - Each event belongs to a region.
    - Teams are allocated to regions based on strategy:
        - naive: spread uniformly over regions
        - risk_aware: prioritize high-risk regions
    - A high-impact event is 'covered' if its region has at least one team.
    """

    random.seed(random_seed)
    np.random.seed(random_seed)

    # Only consider rows with Region info
    df_region = df.dropna(subset=["Region"])
    regions = df_region["Region"].unique().tolist()

    if len(regions) == 0:
        raise ValueError("No region information available in dataset.")

    # Team allocation based on strategy
    if strategy == "naive":
        # Spread teams as evenly as possible across all regions
        assigned_regions = []
        for i in range(num_teams):
            assigned_regions.append(regions[i % len(regions)])
        print(f"\n[NAIVE] Assigned teams to {len(set(assigned_regions))} regions.")
    elif strategy == "risk_aware":
        # Assign teams to top-K high-risk regions
        top_regions = region_risk.index.tolist()
        if len(top_regions) == 0:
            raise ValueError("Region risk table is empty.")

        assigned_regions = []
        # Fill teams by cycling through highest-risk regions first
        for i in range(num_teams):
            assigned_regions.append(top_regions[i % len(top_regions)])
        print(
            f"\n[RISK-AWARE] Assigned teams to high-risk regions: "
            f"{sorted(set(assigned_regions))}"
        )
    else:
        raise ValueError("strategy must be 'naive' or 'risk_aware'")

    team_locations = Counter(assigned_regions)
    print("Team distribution:", dict(team_locations))

    total_events = 0
    total_high_impact = 0
    covered_high_impact = 0

    # Simulation loop
    for year in range(years):
        # Sample events for this year
        sampled_events = df_region.sample(n=events_per_year, replace=True)

        for _, event in sampled_events.iterrows():
            total_events += 1
            region = event["Region"]
            is_high = int(event["HighImpact"])

            if is_high:
                total_high_impact += 1
                # Event is covered if there's at least one team in its region
                if team_locations.get(region, 0) > 0:
                    covered_high_impact += 1

    coverage_rate = (
        covered_high_impact / total_high_impact if total_high_impact > 0 else 0.0
    )

    return {
        "strategy": strategy,
        "num_teams": num_teams,
        "years": years,
        "events_per_year": events_per_year,
        "total_events": total_events,
        "total_high_impact": total_high_impact,
        "covered_high_impact": covered_high_impact,
        "coverage_rate": coverage_rate,
    }


def run_simulation():
    # 1) Load + label disasters
    df, threshold = load_and_label_disasters()

    # 2) Compute risk per region
    region_risk = compute_region_risk(df, region_col="Region")

    # 3) Run naive strategy
    naive_result = simulate_strategy(
        df,
        region_risk,
        num_teams=10,
        years=50,
        events_per_year=50,
        strategy="naive",
        random_seed=42,
    )

    # 4) Run risk-aware strategy
    risk_aware_result = simulate_strategy(
        df,
        region_risk,
        num_teams=10,
        years=50,
        events_per_year=50,
        strategy="risk_aware",
        random_seed=42,
    )

    print("\n=== Simulation Results ===")
    for result in [naive_result, risk_aware_result]:
        print(
            f"\nStrategy       : {result['strategy']}"
            f"\nTeams          : {result['num_teams']}"
            f"\nYears simulated: {result['years']}"
            f"\nHigh-impact events        : {result['total_high_impact']}"
            f"\nCovered high-impact events: {result['covered_high_impact']}"
            f"\nCoverage rate             : {result['coverage_rate']:.3f}"
        )

        os.makedirs("logs/simulation", exist_ok=True)
    results_df = pd.DataFrame([naive_result, risk_aware_result])
    results_path = "logs/simulation/emergency_simulation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved simulation results → {results_path}")


if __name__ == "__main__":
    run_simulation()
