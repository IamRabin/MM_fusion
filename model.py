import torch
import torch.nn as nn
import torch.nn.functional as F

from module import AdaptiveBlock, EEGEncoder, TabularEncoder
from utility.utils import init_model, get_dataloader, FiLM

from dataset import MultiModalEEGDataset, AddExtraDimension, SubsetEEG
from torchvision import transforms


class RegressionModel(nn.Module):

    def __init__(
        self,
        encoder,
        encoder_output_dim,
        new_shape,
        target_shape,
        num_outputs=1,
        freeze_encoder=True,
    ):
        super(RegressionModel, self).__init__()

        self.preprocess = AdaptiveBlock(
            input_shape=new_shape, target_shape=target_shape, in_channels=1
        )

        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.token_pooling = nn.AdaptiveAvgPool1d(1)

        self.regression_head = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_output_dim // 2, num_outputs),
        )

    def forward(self, x):

        x = self.preprocess(x)

        with torch.no_grad():
            features = self.encoder(x)

        pooled_features = self.token_pooling(features.transpose(1, 2)).squeeze(
            -1
        )  # Shape: (batch_size, encoder_output_dim)

        output = self.regression_head(pooled_features)

        return output


class ResMLPBlock(nn.Module):
    """Pre-norm residual MLP: x + MLP(LN(x))"""

    def __init__(self, dim: int, expansion: int = 4, p: float = 0.3):
        super().__init__()
        hidden = dim * expansion
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop1(y)
        y = self.fc2(y)
        y = self.drop2(y)
        return x + y


class Multimodalmodel(nn.Module):

    def __init__(
        self,
        tabular_encoder: nn.Module,
        eeg_encoder: nn.Module,
        # for multimodal, this should be tab_dim + eeg_dim
        combined_dim: int,
        fusion_dim: int = 512,
        output_dim: int = 1,
        dropout_rate: float = 0.3,
        mode: str = "multimodal",
        tab_dim: int = None,
        eeg_dim: int = None,
    ):
        super().__init__()

        self.mode = mode

        self.tabular_encoder = tabular_encoder
        self.eeg_encoder = eeg_encoder

        self.film = FiLM(d_eeg=128, d_tab=64)

        # figure out what input-dim goes into the fusion head
        if mode == "multimodal":
            fusion_in = combined_dim
        elif mode == "tabular":
            assert tab_dim is not None, "Please pass tab_dim for uni-tabular mode"
            fusion_in = tab_dim
        elif mode == "eeg":
            assert eeg_dim is not None, "Please pass eeg_dim for uni-eeg mode"
            fusion_in = eeg_dim
        else:
            raise ValueError(f"Unknown mode {mode!r}")

        self.fuse_in_to_width = nn.Sequential(
            nn.Linear(fusion_in, fusion_dim), nn.BatchNorm1d(fusion_dim)
        )

        # self.fc_fusion = nn.Linear(fusion_in, fusion_dim)
        # self.batch_norm = nn.BatchNorm1d(fusion_dim)
        # self.dropout = nn.Dropout(dropout_rate)
        # self.fc_out = nn.Linear(fusion_dim, output_dim)

        depth = 2

        self.res_blocks = nn.ModuleList(
            [
                ResMLPBlock(dim=fusion_dim, expansion=4, p=dropout_rate)
                for _ in range(depth)
            ]
        )

        # projection head
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, output_dim),
        )

        def forward(self, tabular_data=None, eeg_data=None):

            print(f"[MultimodalModel] Running in mode: {self.mode!r}")

            if self.mode in ["multimodal", "eeg"]:
                eeg_emb = self.eeg_encoder(eeg_data)  # (B, D_eeg)

            if self.mode in ["multimodal", "tabular"]:
                tab_emb = self.tabular_encoder(tabular_data)  # (B, D_tab)

            # combine
            if self.mode == "multimodal":
                eeg_emb = self.film(eeg_emb, tab_emb)
                x = torch.cat((eeg_emb, tab_emb), dim=1)  # (B, D_eeg + D_tab)

            elif self.mode == "eeg":
                x = eeg_emb
            else:  # tabular only
                x = tab_emb

            x = self.fuse_in_to_width(x)

            for block in self.res_blocks:
                x = block(x)
            out = self.head(x)

            return out
