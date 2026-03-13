import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Subset
import random

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay

import json
import os
import time

from dataset import MultiModalEEGDataset, AddExtraDimension, SubsetEEG
from utility.metrics import (
    compute_metrics,
    save_ckp,
    plot_training_validation_stats,
    evaluate_model,
    visualize_ground_truth_vs_predicted,
)
from module import AdaptiveBlock, EEGEncoder, TabularEncoder
from model import Multimodalmodel
from utility.utils import init_model, get_dataloader


def train(
    n_epochs,
    val_r2_max_input,
    model,
    optimizer,
    criterion,
    scheduler,
    train_loader,
    val_loader,
    checkpoint_path,
    best_model_path,
    log_dir="./logs",
    start_epoch=1,
):

    os.makedirs(log_dir, exist_ok=True)

    val_r2_max = val_r2_max_input  # Best validation R2 score

    logs = {
        "train": {"R2": [], "MAE": [], "MSE": [], "loss": []},
        "val": {"R2": [], "MAE": [], "MSE": [], "loss": []},
    }

    model.to(device)

    for e in tqdm(range(start_epoch, n_epochs + 1)):

        # TRAINING
        train_epoch_loss = 0
        train_r2_sum, train_mae_sum, train_mse_sum = 0, 0, 0

        model.train()

        for i, ((X_eeg_batch, X_tabular_batch), y_train_batch) in enumerate(
            train_loader
        ):

            X_eeg_batch, X_tabular_batch, y_train_batch = (
                X_eeg_batch.to(device),
                X_tabular_batch.to(device),
                y_train_batch.to(device),
            )

            y_train_batch = y_train_batch.unsqueeze(-1)

            optimizer.zero_grad()

            y_train_pred = model(eeg_data=X_eeg_batch, tabular_data=X_tabular_batch)

            train_loss = criterion(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            scheduler.step(n_epochs + i / len(train_loader))

            train_epoch_loss += train_loss.item()

            # Compute regression metrics
            mse, r2, mae = compute_metrics(y_train_pred, y_train_batch)

            train_r2_sum += r2
            train_mae_sum += mae
            train_mse_sum += mse

        # Average metrics across training batches
        logs["train"]["loss"].append(train_epoch_loss / len(train_loader))
        logs["train"]["R2"].append(train_r2_sum / len(train_loader))
        logs["train"]["MAE"].append(train_mae_sum / len(train_loader))
        logs["train"]["MSE"].append(train_mse_sum / len(train_loader))

        # VALIDATION
        val_epoch_loss = 0
        val_r2_sum, val_mae_sum, val_mse_sum = 0, 0, 0

        model.eval()

        with torch.no_grad():

            for (X_val_eeg_batch, X_val_tabular_batch), y_val_batch in val_loader:

                X_val_eeg_batch, X_val_tabular_batch, y_val_batch = (
                    X_val_eeg_batch.to(device),
                    X_val_tabular_batch.to(device),
                    y_val_batch.to(device),
                )

                y_val_batch = y_val_batch.unsqueeze(-1)

                y_val_pred = model(
                    eeg_data=X_val_eeg_batch, tabular_data=X_val_tabular_batch
                )

                val_loss = criterion(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()

                # Compute regression metrics
                mse, r2, mae = compute_metrics(y_val_pred, y_val_batch)

                val_r2_sum += r2
                val_mae_sum += mae
                val_mse_sum += mse
        # Average metrics across validation batches
        logs["val"]["loss"].append(val_epoch_loss / len(val_loader))
        logs["val"]["R2"].append(val_r2_sum / len(val_loader))
        logs["val"]["MAE"].append(val_mae_sum / len(val_loader))
        logs["val"]["MSE"].append(val_mse_sum / len(val_loader))

        # Log metrics
        print(
            f"Epoch {e:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | "
            f"Val Loss: {val_epoch_loss / len(val_loader):.5f} | "
            f"Train R2: {train_r2_sum / len(train_loader):.3f} | "
            f"Val R2: {val_r2_sum / len(val_loader):.3f} | "
            f"Train MAE: {train_mae_sum / len(train_loader):.3f} | "
            f"Val MAE: {val_mae_sum / len(val_loader):.3f}"
        )

        # Learning rate scheduler step
        scheduler.step(val_epoch_loss / len(val_loader))

        # Checkpointing: Save the model if validation R2 improves
        current_val_r2 = val_r2_sum / len(val_loader)

        checkpoint = {
            "epoch": e + 1,
            "valid_r2_max": current_val_r2,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        if current_val_r2 > val_r2_max:
            print(
                f"Validation R2 increased ({val_r2_max:.6f} --> {current_val_r2:.6f}). "
                "Saving model ..."
            )
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            val_r2_max = current_val_r2

    # Save logs to a JSON file at the end of training
    with open("./logs/training_logs.json", "w") as json_file:
        json.dump(logs, json_file, indent=4)

    return model, logs


if __name__ == "__main__":

    # source_channels = ['AF3', 'AF4', 'AF7', 'AF8', 'AFF1h', 'AFF2', 'AFF5h', 'AFF6h', 'AFp3h', 'AFp4h',
    #                    'AFz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1h', 'CCP2h', 'CCP3h', 'CCP4h',
    #                    'CCP5h', 'CCP6h', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPP1h', 'CPP2h',
    #                    'CPP3h', 'CPP4h', 'CPP5h', 'CPP6h', 'Cz', 'F1', 'F10', 'F2', 'F3', 'F4', 'F5',
    #                    'F6', 'F7', 'F8', 'F9', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCC1h', 'FCC2h',
    #                    'FCC3h', 'FCC4h', 'FCC5h', 'FCC6h', 'FCz', 'FFC1h', 'FFC2h', 'FFC3h', 'FFC4h',
    #                    'FFC5h', 'FFC6h', 'FFT7h', 'FFT8h', 'FFT9h', 'FT7', 'FT8', 'FT9', 'FTT10h',
    #                    'FTT7h', 'FTT8h', 'FTT9h', 'FP1', 'FP2', 'Fpz', 'Fz', 'I1', 'I2', 'Iz', 'M1', 'M2',
    #                    'O1', 'O2', 'OI1h', 'OI2h', 'P1', 'P10', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
    #                    'P9', 'PO10', 'PO3', 'PO4', 'PO7', 'PO8', 'PO9', 'POO10h', 'POO3h', 'POO4h',
    #                    'POO9h', 'POz', 'PPO1', 'PPO10h', 'PPO2', 'PPO5h', 'PPO6h', 'PPO9h', 'Pz',
    #                    'T7', 'T8', 'TP7', 'TP8', 'TPP10h', 'TPP7h', 'TPP8h', 'TPP9h', 'TTP7h', 'TTP8h']

    # channel_indices = [source_channels.index(ch) for ch in
    #                    ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'FP1', 'Cz', 'FP2', 'Fz',
    #                     'O1', 'O2', 'P3', 'P4', 'Pz', 'T7', 'T8', 'P7', 'P8']]

    # ########################################
    # ## Set Up dataset
    # ########################################

    data_fpath = "/rabindrk/Documents/rk_dev/normalized_merged_data_log_transformed.csv"

    eeg_column = "FPath"
    target_column = "Tau"

    exclude_cols = [
        "session_id",
        "site_idName",
        "site",
        "PALTE12",
        "PALTE2",
        "PALTE28",
        "PALTE4",
        "PALTE6",
        "IDX",
        "Participant_ID_x",
        "neuropsyc_med",
    ]

    dataset = MultiModalEEGDataset(
        annotations_file=data_fpath,
        eeg_column=eeg_column,
        target_column=target_column,
        exclude_cols=exclude_cols,
        eeg_transform=transforms.Compose(
            [
                transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float)),
                # SubsetEEG(channel_indices),
                AddExtraDimension(),
            ]
        ),
    )

    # ########################
    # ## Loading the Dataset
    # ########################

    participant_ids = dataset.data["Participant_ID_x"].tolist()

    dataset.data["APOE"] = dataset.data[
        [
            "APOE_E2E2",
            "APOE_E2E2",
            "APOE_E2E3",
            "APOE_E2E4",
            "APOE_E3E3",
            "APOE_E3E4",
            "APOE_E4E4",
        ]
    ].idxmax(axis=1)

    dataset.data["APOE"] = dataset.data["APOE"].replace(
        {
            "APOE_E2E2": "protective",
            "APOE_E2E4": "Risk",
            "APOE_E2E3": "protective",
            "APOE_E3E3": "Neutral",
            "APOE_E3E4": "Risk",
            "APOE_E4E4": "Risk",
        }
    )

    apoe_status = dataset.data["APOE"].tolist()

    train_loader, val_loader, test_loader = get_dataloader(
        dataset,
        participant_ids,
        apoe_status,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=16,
        seed=25,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ###################################
    # ## Initializing the Model
    # ###################################

    pretrained_path = "/rabindrk/Documents/rk_dev/VJEPA/jepa-best.pth.tar"
    model_name = "vit_small"
    patch_size = 16
    tubelet_size = 4
    pretrain_frames_per_clip = 32
    uniform_power = True
    checkpoint_key = "target_encoder"
    use_SiLU = False
    tight_SiLU = False
    use_sdpa = True

    pretrained_enc = init_model(
        crop_size=224,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa,
    ).to(device)

    eeg_encoder = EEGEncoder(
        encoder=pretrained_enc,
        new_shape=(88, 126, 800),
        target_shape=(118, 19, 500),
        output_dim=128,
        freeze_encoder=True,
    )

    tabular_encoder = TabularEncoder(input_dim=11, output_dim=64)

    # model = MultimodalModel(eeg_encoder, tabular_encoder, combined_dim=64+32, fusion_dim=256, output_dim=1)

    model = Multimodalmodel(
        tabular_encoder,
        eeg_encoder,
        combined_dim=128 + 64,
        fusion_dim=512,
        output_dim=1,
        mode="multimodal",
    )

    # ## Uni-EEG only:
    # model = Multimodalmodel(
    #     tabular_encoder,
    #     eeg_encoder,
    #     combined_dim = None,      # unused in this mode
    #     eeg_dim      = 128,
    #     fusion_dim   = 512,
    #     output_dim   = 1,
    #     mode         = 'eeg'
    # )

    # Uni-Tabular only:
    # model = Multimodalmodel(
    #     tabular_encoder,
    #     eeg_encoder,
    #     combined_dim = None,      # unused in this mode
    #     tab_dim      = 64,
    #     fusion_dim   = 512,
    #     output_dim   = 1,
    #     mode         = 'tabular'
    # )

    # ########################################
    # ## Optimizer
    # ########################################

    # optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=0.01)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=10, verbose=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=2e-6
    )

    # ########################################
    # ## Loss criterion
    # ########################################

    criterion = nn.HuberLoss(delta=1.0)

    # ########################################
    # ## Training and Validation
    # ########################################

    # val_r2_max_input = 0.0
    # trained_model, logs = train(
    #     200,
    #     val_r2_max_input,
    #     model,
    #     optimizer,
    #     criterion,
    #     scheduler,
    #     train_loader,
    #     val_loader,
    #     "./logs/current_checkpoint.pt",
    #     "./logs/best_model.pt",
    #     log_dir="./logs",
    #     start_epoch=1
    # )

    # ########################################
    # ## Load the best model
    # ########################################

    checkpoint_path = "/logs/best_model.pt"
    # checkpoint_path = "/tsd/p1504/cluster/evpss/logs/current_checkpoint_une.pt"

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])

    # ########################################
    # ## Evaluate Model
    # ########################################

    metrics, ground_truth, predictions = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        save_path="/logs/evaluation_results_cyclic.pkl",
    )

    # ########################################
    # ## Plot logs
    # ########################################

    # mae = metrics["MAE"]
    # mse = metrics["MSE"]
    # r2 = metrics["R2"]

    # plot_training_validation_stats(logs, save_dir='/tsd/p1504/cluster/evpss/logs')

    # visualize_ground_truth_vs_predicted(
    #     ground_truth,
    #     predictions,
    #     mae,
    #     mse,
    #     r2,
    #     save_path="/tsd/p1504/cluster/evpss/logs/ground_truth_vs_predicted.png"
# )
