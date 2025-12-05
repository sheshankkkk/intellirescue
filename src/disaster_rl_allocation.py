"""
disaster_rl_allocation.py

Lightweight reinforcement learning for IntelliRescue.

We model emergency team allocation as a multi-armed bandit problem:
- Each REGION is an "arm".
- At each time step, a simulated disaster occurs in some region.
- If the chosen region is hit by a HIGH-IMPACT event, the agent receives reward = 1.
- Otherwise reward = 0.

The agent uses epsilon-greedy exploration to learn which regions to prioritize,
based on historical high-impact rates from the EOSDIS-style dataset.

Outputs:
- Console logs summarizing learning.
- JSON summary file with learned policy and coverage comparison.

This script is meant to SUPPORT your claim:
"an agent-based simulation model employing reinforcement learning optimizes
the allocation of emergency response resources."
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
IMPACT_COL = "Total Deaths"
CSV_PATH_FULL = "data/disasters.csv"
CSV_PATH_SAMPLE = "data/disasters_sample.csv"

OUTPUT_DIR = Path("logs") / "rl_allocation"
SUMMARY_PATH = OUTPUT_DIR / "rl_allocation_summary.json"


def get_data_path():
    if os.path.exists(CSV_PATH_FULL):
        print(f"[RL] Using FULL dataset at {CSV_PATH_FULL}")
        return CSV_PATH_FULL
    elif os.path.exists(CSV_PATH_SAMPLE):
        print(f"[RL] Using SAMPLE dataset at {CSV_PATH_SAMPLE}")
        return CSV_PATH_SAMPLE
    else:
        raise FileNotFoundError(
            f"Neither {CSV_PATH_FULL} nor {CSV_PATH_SAMPLE} found."
        )


def build_region_risk_table():
    """
    Build the same region risk summary as in the web app:

    - events: # of disasters recorded
    - high_impact: # with Total Deaths >= 50
    - risk_rate: high_impact / events
    """
    path = get_data_path()
    df = pd.read_csv(path)

    # Drop rows without the impact column or region
    df = df.dropna(subset=[IMPACT_COL, "Region"])

    # High-impact label (>= 50 deaths, same convention as app)
    threshold = 50.0
    df["HighImpact"] = (df[IMPACT_COL] >= threshold).astype(int)

    grouped = df.groupby("Region")
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

    print(f"[RL] Built region risk table with {len(region_risk)} regions.")
    print("[RL] Top 5 regions by high-impact rate:")
    print(region_risk.head(5)[["Region", "events", "high_impact", "risk_rate"]])

    return region_risk


# --------------- RL ENVIRONMENT ---------------

class DisasterAllocationEnv:
    """
    Simple environment for RL:

    - Arms: regions (0 .. n_regions-1)
    - At each step:
        * sample a region where a disaster occurs, based on historical frequencies
        * that event becomes high-impact with probability = region_risk[region].risk_rate
        * reward = 1 if agent chose the same region AND event is high-impact, else 0
    """

    def __init__(self, region_risk: pd.DataFrame, events_per_step: int = 1, seed: int = 42):
        self.region_risk = region_risk.reset_index(drop=True)
        self.regions = self.region_risk["Region"].values
        self.n_arms = len(self.regions)

        events = self.region_risk["events"].values.astype(float)
        self._event_weights = events / events.sum()

        self._risk_rates = self.region_risk["risk_rate"].values.astype(float)
        self.events_per_step = max(1, events_per_step)
        self.rng = np.random.default_rng(seed)

    def step(self, arm: int):
        """
        One RL time step: agent chooses a region index (arm).

        We simulate `events_per_step` disasters; reward is 1 if at least one
        high-impact event hits the chosen region; otherwise reward 0.
        """
        # sample which regions get disasters at this step
        region_indices = self.rng.choice(
            self.n_arms,
            size=self.events_per_step,
            p=self._event_weights,
        )

        # for each event, decide if it is high-impact
        risk_for_events = self._risk_rates[region_indices]
        high_flags = self.rng.random(self.events_per_step) < risk_for_events

        # reward if any high-impact event occurred in chosen arm
        impacted_in_chosen = (region_indices == arm) & high_flags
        reward = 1.0 if impacted_in_chosen.any() else 0.0

        return reward


# --------------- EPSILON-GREEDY BANDIT AGENT ---------------

class EpsilonGreedyAgent:
    """
    Classic epsilon-greedy bandit:

    - Q[a]: estimated value (expected reward) for arm a
    - N[a]: number of times arm a was chosen
    - At each step:
        * with probability epsilon: choose random arm (exploration)
        * otherwise: choose arm with largest Q[a] (exploitation)
    """

    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon

        self.Q = np.zeros(n_arms, dtype=float)
        self.N = np.zeros(n_arms, dtype=int)

        self.total_steps = 0
        self.total_reward = 0.0

    def select_arm(self, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return rng.integers(0, self.n_arms)
        # tie-breaking at random among best arms
        max_Q = self.Q.max()
        best_indices = np.flatnonzero(self.Q == max_Q)
        return int(rng.choice(best_indices))

    def update(self, arm: int, reward: float):
        self.total_steps += 1
        self.total_reward += reward

        self.N[arm] += 1
        # incremental average
        alpha = 1.0 / self.N[arm]
        self.Q[arm] += alpha * (reward - self.Q[arm])

    @property
    def average_reward(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_reward / self.total_steps


# --------------- TRAINING LOOP ---------------

def train_bandit_agent(
    env: DisasterAllocationEnv,
    n_steps: int = 50_000,
    epsilon: float = 0.1,
    seed: int = 123,
):
    """
    Train an epsilon-greedy bandit on the disaster allocation environment.
    Returns the trained agent and a list of rolling average rewards.
    """
    rng = np.random.default_rng(seed)
    agent = EpsilonGreedyAgent(env.n_arms, epsilon=epsilon)

    rewards = []
    avg_rewards = []

    for t in range(1, n_steps + 1):
        arm = agent.select_arm(rng)
        reward = env.step(arm)
        agent.update(arm, reward)

        rewards.append(reward)
        avg_rewards.append(agent.average_reward)

        if t % 10_000 == 0:
            print(
                f"[RL] Step {t}/{n_steps} | "
                f"epsilon={epsilon} | "
                f"avg reward={agent.average_reward:.4f}"
            )

    return agent, np.array(rewards), np.array(avg_rewards)


# --------------- EVALUATION (POLICY VS NAIVE & RISK-AWARE) ---------------

def evaluate_policies(region_risk: pd.DataFrame, agent: EpsilonGreedyAgent, n_events: int = 5000, seed: int = 999):
    """
    Compare three strategies on a fresh Monte Carlo run:

    1. Naive: choose region at random (each event has a randomly protected region)
    2. Risk-aware: always protect the region with highest true high-impact rate
    3. RL policy: GREEDY w.r.t. learned Q-values (always protect region with max Q)

    We assume 1 team for simplicity; each event occurs in some region and we
    check if that region is the one being protected.
    """
    rng = np.random.default_rng(seed)

    regions = region_risk["Region"].values
    events = region_risk["events"].values.astype(float)
    risk_rates = region_risk["risk_rate"].values.astype(float)

    n_arms = len(regions)

    # event sampling distribution
    weights = events / events.sum()
    region_idx = rng.choice(n_arms, size=n_events, p=weights)
    high_flags = rng.random(n_events) < risk_rates[region_idx]
    # only count high-impact events
    high_indices = np.where(high_flags)[0]
    if len(high_indices) == 0:
        print("[RL] Warning: no high-impact events sampled in evaluation; increasing n_events may help.")
        return {
            "total_high": 0,
            "naive_covered": 0,
            "naive_rate": 0.0,
            "risk_aware_covered": 0,
            "risk_aware_rate": 0.0,
            "rl_covered": 0,
            "rl_rate": 0.0,
        }

    # strategy 1: naive (random arm per event)
    naive_choices = rng.integers(0, n_arms, size=n_events)
    naive_covered = int(((naive_choices == region_idx) & high_flags).sum())

    # strategy 2: risk-aware (always choose best arm by true risk_rate)
    best_arm_true = int(np.argmax(risk_rates))
    aware_choices = np.full(n_events, best_arm_true, dtype=int)
    aware_covered = int(((aware_choices == region_idx) & high_flags).sum())

    # strategy 3: RL policy (GREEDY: always choose arm with max Q)
    Q = agent.Q.copy()
    if np.allclose(Q, 0.0):
        # no learning? fallback to uniform random
        rl_choices = rng.integers(0, n_arms, size=n_events)
    else:
        best_arm_rl = int(np.argmax(Q))
        rl_choices = np.full(n_events, best_arm_rl, dtype=int)

    rl_covered = int(((rl_choices == region_idx) & high_flags).sum())

    total_high = int(high_flags.sum())

    return {
        "total_high": total_high,
        "naive_covered": naive_covered,
        "naive_rate": naive_covered / total_high if total_high > 0 else 0.0,
        "risk_aware_covered": aware_covered,
        "risk_aware_rate": aware_covered / total_high if total_high > 0 else 0.0,
        "rl_covered": rl_covered,
        "rl_rate": rl_covered / total_high if total_high > 0 else 0.0,
    }


# --------------- MAIN ---------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    region_risk = build_region_risk_table()

    # Build environment and train RL agent
    env = DisasterAllocationEnv(region_risk, events_per_step=3, seed=42)
    n_steps = 50_000
    epsilon = 0.1

    print(f"[RL] Training epsilon-greedy agent for {n_steps} steps (epsilon={epsilon})...")
    agent, rewards, avg_rewards = train_bandit_agent(
        env, n_steps=n_steps, epsilon=epsilon, seed=123
    )

    # Show top learned regions
    sorted_indices = np.argsort(agent.Q)[::-1]
    print("\n[RL] Top 10 regions by learned Q-value (expected reward):")
    for rank, idx in enumerate(sorted_indices[:10], start=1):
        print(
            f"  {rank:2d}. {region_risk.loc[idx, 'Region']:<30} "
            f"Q={agent.Q[idx]:.4f} | "
            f"true risk_rate={region_risk.loc[idx, 'risk_rate']:.3f}"
        )

    # Evaluate on a fresh Monte Carlo run
    eval_results = evaluate_policies(region_risk, agent, n_events=5000, seed=999)

    print("\n[RL] === POLICY EVALUATION ON SYNTHETIC EVENTS ===")
    print(f"Total high-impact events (simulated): {eval_results['total_high']}")
    print(
        f"Naive allocation  : covered={eval_results['naive_covered']} "
        f"({eval_results['naive_rate']*100:.1f}% coverage)"
    )
    print(
        f"Risk-aware (oracle): covered={eval_results['risk_aware_covered']} "
        f"({eval_results['risk_aware_rate']*100:.1f}% coverage)"
    )
    print(
        f"RL policy          : covered={eval_results['rl_covered']} "
        f"({eval_results['rl_rate']*100:.1f}% coverage)"
    )
    print("=================================================\n")

    # Save summary JSON for your report or potential web UI integration
    summary = {
        "n_steps": n_steps,
        "epsilon": epsilon,
        "avg_reward_final": float(agent.average_reward),
        "top_regions_by_Q": [
            {
                "rank": int(rank),
                "region": str(region_risk.loc[idx, "Region"]),
                "Q": float(agent.Q[idx]),
                "true_risk_rate": float(region_risk.loc[idx, "risk_rate"]),
            }
            for rank, idx in enumerate(sorted_indices[:10], start=1)
        ],
        "evaluation": eval_results,
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[RL] Saved RL summary → {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
