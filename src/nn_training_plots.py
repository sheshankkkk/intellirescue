import os
import pandas as pd
import matplotlib.pyplot as plt

HISTORY_PATH = "models/disaster_nn_history.csv"
PLOTS_DIR = "logs/plots"


def ensure_paths():
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError(f"Missing NN history file: {HISTORY_PATH}")
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_loss(history_df: pd.DataFrame):
    plt.figure()
    if "loss" in history_df.columns:
        plt.plot(history_df["loss"], label="Train loss")
    if "val_loss" in history_df.columns:
        plt.plot(history_df["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Neural Network Training: Loss vs Epoch")
    plt.legend()
    out_path = os.path.join(PLOTS_DIR, "nn_loss_curve.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve → {out_path}")


def plot_accuracy(history_df: pd.DataFrame):
    # Depending on tf version, accuracy column may be "accuracy"
    # and val_accuracy -> "val_accuracy"
    if "accuracy" not in history_df.columns:
        print("No 'accuracy' column in history; skipping accuracy plot.")
        return

    plt.figure()
    plt.plot(history_df["accuracy"], label="Train accuracy")
    if "val_accuracy" in history_df.columns:
        plt.plot(history_df["val_accuracy"], label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Neural Network Training: Accuracy vs Epoch")
    plt.legend()
    out_path = os.path.join(PLOTS_DIR, "nn_accuracy_curve.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy curve → {out_path}")


def plot_auc(history_df: pd.DataFrame):
    # AUC metric was named "auc" and "val_auc" in our model
    if "auc" not in history_df.columns:
        print("No 'auc' column in history; skipping AUC plot.")
        return

    plt.figure()
    plt.plot(history_df["auc"], label="Train AUC")
    if "val_auc" in history_df.columns:
        plt.plot(history_df["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Neural Network Training: AUC vs Epoch")
    plt.legend()
    out_path = os.path.join(PLOTS_DIR, "nn_auc_curve.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved AUC curve → {out_path}")


def main():
    ensure_paths()
    history_df = pd.read_csv(HISTORY_PATH)
    print("History columns:", list(history_df.columns))

    plot_loss(history_df)
    plot_accuracy(history_df)
    plot_auc(history_df)


if __name__ == "__main__":
    main()
