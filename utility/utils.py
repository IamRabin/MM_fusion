import torch
import torch.nn as nn
import logging

from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split

import sys
import numpy as np
import pandas as pd
import os

sys.path.append(os.path.abspath("/Developer/pretrained/VJEPA"))
import src.models.vision_transformer as vit


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pretrained(encoder, pretrained, checkpoint_key="target_encoder"):

    logger.info(f"Loading pretrained model from {pretrained}")
    checkpoint = torch.load(pretrained, map_location="cpu")

    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {
        k.replace("backbone.", ""): v for k, v in pretrained_dict.items()
    }

    for k, v in encoder.state_dict().items():

        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')

        elif pretrained_dict[k].shape != v.shape:
            logger.info(
                f'key "{k}" is of different shape in model and loaded state dict'
            )

    msg = encoder.load_state_dict(pretrained_dict, strict=False)

    logger.info(f"loaded pretrained model with msg: {msg}")
    logger.info(
        f'loaded pretrained encoder from epoch: {checkpoint["epoch"]} \n path: {pretrained}'
    )

    del checkpoint

    return encoder


def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    # Video specific parameters
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key="target_encoder",
):

    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )

    encoder.to(device)
    encoder = load_pretrained(
        encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key
    )

    return encoder


def simple_dataloader(
    dataset,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    batch_size=32,
    num_workers=0,
    seed=42,
):

    # Ensure the ratios add up to 1
    assert (
        train_ratio + val_ratio + test_ratio == 1
    ), "Train, val, and test ratios must sum to 1."

    torch.manual_seed(seed)

    total_len = len(dataset)

    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_len, val_len, test_len]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def get_dataloader(
    dataset,
    participant_ids,
    apoe_categories,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    batch_size=32,
    num_workers=0,
    seed=25,
):

    assert (
        train_ratio + val_ratio + test_ratio == 1
    ), "Train, val, and test ratios must sum to 1."

    ###################
    target_col = "Tau"
    apoe_col = "APOE"
    df = dataset.data.copy()

    # ---- Outlier removal (optional, commented) ----
    # outlier_idx = df["Tau"].dropna().idxmax()
    # outlier_pid = df.loc[outlier_idx, "Participant ID_x"]
    # print(f"[Split] Excluding participant {outlier_pid} with max Tau={df.loc[outlier_idx,target_col]}")

    # df_nool = df[df["Participant ID_x"] != outlier_pid]

    # unique_participants_df = (
    #     df_nool[["Participant ID_x", apoe_col]]
    #     .drop_duplicates(subset=["Participant ID_x"])
    #     .reset_index(drop=True)
    # )

    ###################
    unique_participants_df = pd.DataFrame(
        {"Participant ID_x": participant_ids, "APOE": apoe_categories}
    ).drop_duplicates()

    train_participants, temp_participants = train_test_split(
        unique_participants_df,
        test_size=(val_ratio + test_ratio),
        stratify=unique_participants_df[apoe_col],
        random_state=seed,
    )

    train_unique_ids = train_participants["Participant ID_x"].values

    val_participants, test_participants = train_test_split(
        temp_participants,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_participants[apoe_col],
        random_state=seed,
    )

    val_unique_ids = val_participants["Participant ID_x"].values
    test_unique_ids = test_participants["Participant ID_x"].values

    train_indices = [
        i for i, pid in enumerate(participant_ids) if pid in train_unique_ids
    ]

    val_indices = [i for i, pid in enumerate(participant_ids) if pid in val_unique_ids]

    test_indices = [
        i for i, pid in enumerate(participant_ids) if pid in test_unique_ids
    ]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # --- Optional: remove high-leverage samples ---
    # high_lev_idx = np.load("/tsd/p1504/home/p1504-rabindrk/Documents/rk_dev/high_leverage_indices.npy")
    # n = len(test_dataset)
    # all_idx_subset_space = np.arange(n)
    # keep_mask = np.ones(n, dtype=bool)
    # keep_mask[high_lev_idx] = False
    # keep_idx = all_idx_subset_space[keep_mask]
    # test_dataset = Subset(test_dataset, keep_idx)

    ### Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # total = len(train_unique_ids) + len(val_unique_ids) + len(test_unique_ids)

    print(len(train_dataset))
    print("val_Dataset:", len(val_dataset))
    print("test_dataset:", len(test_dataset))

    # train_apoe_status = [apoe_categories[i] for i in train_indices]
    # val_apoe_status = [apoe_categories[i] for i in val_indices]
    # test_apoe_status = [apoe_categories[i] for i in test_indices]

    # print("APOE status distribution in Training Set:\n", pd.Series(train_apoe_status).value_counts())
    # print("APOE status distribution in Validation Set:\n", pd.Series(val_apoe_status).value_counts())
    # print("APOE status distribution in Test Set:\n", pd.Series(test_apoe_status).value_counts())

    return train_loader, val_loader, test_loader


class FiLM(nn.Module):
    def __init__(self, d_eeg: int, d_tab: int, hidden: int = 256, p: float = 0.1):
        super().__init__()

        self.mod = nn.Sequential(
            nn.LayerNorm(d_tab),
            nn.Linear(d_tab, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, 2 * d_eeg),
        )

    def forward(self, eeg_vec: torch.Tensor, tab_vec: torch.Tensor):
        # eeg_vec: (B, D_eeg), tab_vec: (B, D_tab)

        gamma, beta = self.mod(tab_vec).chunk(2, dim=-1)

        # stabilize: bound gamma to [-1,1] so (1+gamma) ∈ [0,2]
        gamma = torch.tanh(gamma)

        return (1 + gamma) * eeg_vec + beta


if __name__ == "__main__":

    pass
