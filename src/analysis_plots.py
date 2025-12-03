import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

PRED_PATH = "logs/disaster_risk/disaster_risk_predictions.csv"
SIM_PATH = "logs/simulation/emergency_simulation_results.csv"
PLOTS_DIR = "logs/plots"


def ensure_paths():
    if not os.path.exists(PRED_PATH):
        raise FileNotFoundError(f"Missing predictions file: {PRED_PATH}")
    if not os.path.exists(SIM_PATH):
        raise FileNotFoundError(f"Missing simulation results file: {SIM_PATH}")
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_risk_score_distribution():
    df = pd.read_csv(PRED_PATH)

    plt.figure()
    df[df["y_true"] == 0]["y_proba"].hist(bins=20, alpha=0.6, label="Low impact (0)")
    df[df["y_true"] == 1]["y_proba"].hist(bins=20, alpha=0.6, label="High impact (1)")
    plt.xlabel("Predicted probability of HighImpact")
    plt.ylabel("Count")
    plt.title("Distribution of Predicted Risk Scores")
    plt.legend()

    out_path = os.path.join(PLOTS_DIR, "risk_score_distribution.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved risk score distribution → {out_path}")


def plot_calibration_from_predictions():
    df = pd.read_csv(PRED_PATH)

    y_true = df["y_true"].values
    y_proba = df["y_proba"].values

    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)

    plt.figure()
    plt.plot(mean_pred, frac_pos, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "--", label="Ideal")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve (from predictions)")
    plt.legend()

    out_path = os.path.join(PLOTS_DIR, "disaster_calibration_curve_from_preds.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved calibration curve (from preds) → {out_path}")


def plot_simulation_coverage():
    df_sim = pd.read_csv(SIM_PATH)

    plt.figure()
    plt.bar(df_sim["strategy"], df_sim["coverage_rate"])
    plt.ylabel("Coverage rate of high-impact events")
    plt.xlabel("Strategy")
    plt.title("Emergency Team Coverage: Naive vs Risk-Aware")

    out_path = os.path.join(PLOTS_DIR, "simulation_coverage_comparison.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved simulation coverage comparison → {out_path}")


def main():
    ensure_paths()
    plot_risk_score_distribution()
    plot_calibration_from_predictions()
    plot_simulation_coverage()


if __name__ == "__main__":
    main()
