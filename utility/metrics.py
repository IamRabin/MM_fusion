import torch
import shutil
import numpy as np
import pandas as pd
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy.stats


def compute_metrics(y_pred, y_true):
    """Compute MSE, R2, and MAE for regression."""
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return mse, r2, mae


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    torch.save(state, f_path)

    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer defined in previous training
    """

    # load checkpoint
    checkpoint = torch.load(checkpoint_fpath)

    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint["state_dict"])

    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint["optimizer"])

    # initialize valid_r2_max from checkpoint
    valid_r2_max = checkpoint["valid_r2_max"]

    # return model, optimizer, epoch value, min validation loss
    return model, optimizer, checkpoint["epoch"], valid_r2_max


def plot_training_validation_stats(logs, save_dir="logs"):
    """
    Plots training and validation metrics (R2, RMSE, MSE) and loss over epochs.

    Parameters:
    - metric_stats (dict): Dictionary containing training and validation R2, RMSE, and MSE.
    - loss_stats (dict): Dictionary containing training and validation loss.
    - save_dir (str): Directory to save the generated plots.
    """

    # Number of epochs
    epochs = range(1, len(logs["train"]["R2"]) + 1)

    # Plot Training and Validation R2 Score
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, logs["train"]["R2"], label="Training R2", marker="o")
    plt.plot(epochs, logs["val"]["R2"], label="Validation R2", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("R2 Score")
    plt.title("Training and Validation R2 Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/r2_plot.png")
    plt.close()

    # Plot Training and Validation MAE
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, logs["train"]["MAE"], label="Training MAE", marker="o")
    plt.plot(epochs, logs["val"]["MAE"], label="Validation MAE", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.title("Training and Validation MAE")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/mae_plot.png")
    plt.close()

    # Plot Training and Validation MSE
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, logs["train"]["MSE"], label="Training MSE", marker="o")
    plt.plot(epochs, logs["val"]["MSE"], label="Validation MSE", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("Training and Validation MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/mse_plot.png")
    plt.close()

    # Plot Training and Validation Loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, logs["train"]["loss"], label="Training Loss", marker="o")
    plt.plot(epochs, logs["val"]["loss"], label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss_plot.png")
    plt.close()


def evaluate_model(model, data_loader, device, save_path=None):

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for (eeg_input, tabular_input), labels in data_loader:

            eeg_input = eeg_input.to(device)
            tabular_input = tabular_input.to(device)
            labels = labels.to(device)

            labels = labels.unsqueeze(-1)

            predictions = model(eeg_data=eeg_input, tabular_data=tabular_input)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    metrics = {"MSE": mse, "MAE": mae, "R2": r2}

    # Save results to a file if save_path is provided
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "metrics": metrics,
                    "ground_truth": all_labels,
                    "predictions": all_preds,
                },
                f,
            )

    return metrics, all_labels, all_preds


def visualize_ground_truth_vs_predicted(
    ground_truth, predictions, mae, mse, r2, save_path=None
):

    plt.figure(figsize=(8, 8))
    sns.set(style="whitegrid", font_scale=1.2)

    plt.scatter(
        ground_truth, predictions, alpha=0.6, color="#0072B2", s=30, label="Predicted"
    )

    max_val = max(max(ground_truth), max(predictions))
    min_val = min(min(ground_truth), min(predictions))

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="#D55E00",
        linestyle="--",
        linewidth=2,
        label="Ideal (y = x)",
    )

    plt.title("True vs Predicted Values", fontsize=16, weight="bold")
    plt.xlabel("True p-tau217 level (Normalized)", fontsize=14)
    plt.ylabel("Predicted p-tau217 level (Normalized)", fontsize=14)

    metrics_text = f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nR²: {r2:.2f}"
    plt.text(
        0.05,
        0.95,
        metrics_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    plt.legend(
        loc="upper right", fontsize=12, frameon=True, fancybox=True, framealpha=0.9
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
    else:
        plt.show()
