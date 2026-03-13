import umap
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Multimodalmodel
from module import AdaptiveBlock, EEGEncoder, TabularEncoder
from utility.utils import init_model, get_dataloader

from dataset import MultiModalEEGDataset, AddExtraDimension
from torchvision import transforms


def extract_features(dataloader, model, device):

    model.eval()
    model.to(device)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for (eeg_input, tabular_input), labels in dataloader:

            eeg_input = eeg_input.to(device)
            tabular_input = tabular_input.to(device)
            labels = labels.to(device)

            labels = labels.unsqueeze(-1)

            features = model.eeg_encoder(eeg_input)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    all_features = torch.cat(all_features, dim=0)  # Shape: (num_samples, feature_dim)
    all_labels = torch.cat(all_labels, dim=0)  # Shape: (num_samples,)

    return all_features, all_labels


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_path = "/rk_dev/VJEPA/jepa-best.pth.tar"
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

    eeg_column = "FPath"
    target_column = "APOE_status"

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
        annotations_file="/rk_dev/normalized_merged_data_status.csv",
        eeg_column=eeg_column,
        target_column=target_column,
        exclude_cols=exclude_cols,
        eeg_transform=transforms.Compose(
            [
                transforms.Lambda(
                    lambda x: torch.tensor(x, dtype=torch.float)
                ),  # Convert to tensor
                AddExtraDimension(),
            ]
        ),
    )

    participant_ids = dataset.data["Participant_ID_x"].tolist()

    dataset.data["APOE"] = dataset.data[
        ["APOE_E2E2", "APOE_E2E3", "APOE_E2E4", "APOE_E3E3", "APOE_E3E4", "APOE_E4E4"]
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

    checkpoint_path = "/logs_str/best_model.pt"

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])

    print("Extracting Features......")

    features, labels = extract_features(test_loader, model, device)

    print("Extracting features Completed...")

    features_np = features.numpy()
    labels_np = labels.numpy()

    np.save("features_apoe.npy", features_np)
    np.save("labels_apoe.npy", labels_np)

    print("Features Saved...")

    reducer = umap.UMAP(
        n_neighbors=5, min_dist=0.2, metric="correlation", random_state=42, verbose=True
    )

    reduced_features = reducer.fit_transform(features_np)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=labels_np,
        cmap="viridis",
        alpha=0.7,
    )

    plt.colorbar(scatter, label="Labels")
    plt.title("UMAP Visualization of Extracted Features")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)
    plt.savefig("umap_features_apoe.png")
