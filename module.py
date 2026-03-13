import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveBlock(nn.Module):

    def __init__(self, input_shape, target_shape, in_channels=1):
        super(AdaptiveBlock, self).__init__()

        kernel_size = self.calculate_kernel_size(input_shape, target_shape)
        stride = (1, 1, 1)

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

        self.padding_depth = max((target_shape[0] - input_shape[0]) // 2, 0)
        self.padding_height = max((target_shape[1] - input_shape[1]) // 2, 0)
        self.padding_width = max((target_shape[2] - input_shape[2]) // 2, 0)

    def calculate_kernel_size(self, new_shape, target_shape):

        kernel_depth = max(new_shape[0] - target_shape[0] + 1, 1)
        kernel_height = max(new_shape[1] - target_shape[1] + 1, 1)
        kernel_width = max(new_shape[2] - target_shape[2] + 1, 1)

        return (kernel_depth, kernel_height, kernel_width)

    def forward(self, x):

        x = self.conv(x)

        x = F.pad(
            x,
            (
                self.padding_width,
                self.padding_width,
                self.padding_height,
                self.padding_height,
                self.padding_depth,
                self.padding_depth,
            ),
        )

        return x


class MultiHeadAttentionPool(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int = 4, out_dim: int = None):
        super().__init__()

        self.num_heads = num_heads

        self.q = nn.Parameter(torch.randn(num_heads, 1, embed_dim))

        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)

        # project concatenated heads back to embed_dim
        self.proj = nn.Linear(embed_dim * num_heads, embed_dim)

        # optional dimensionality reduction AFTER pooling
        self.out_dim = out_dim

        if out_dim is not None and out_dim != embed_dim:
            self.reduce = nn.Sequential(
                nn.LayerNorm(embed_dim), nn.Linear(embed_dim, out_dim), nn.GELU()
            )
        else:
            self.reduce = nn.Identity()

        def forward(self, x, mask=None):  # x: [B, T, D]
            B, T, D = x.shape

            k = self.k(x)
            v = self.v(x)

            outs = []
            scale = D**0.5

            for h in range(self.num_heads):
                q = self.q[h].expand(B, -1, -1)  # [B, 1, D]
                scores = (q @ k.transpose(1, 2)) / scale  # [B, 1, T]

                if mask is not None:
                    scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

                w = scores.softmax(-1)
                outs.append((w @ v).squeeze(1))  # [B, D]

            pooled = torch.cat(outs, dim=-1)  # [B, D * H]
            pooled = self.proj(pooled)  # [B, D]
            pooled = self.reduce(pooled)  # [B, out_dim or D]

            return pooled


class EEGEncoder(nn.Module):

    def __init__(
        self,
        encoder,
        new_shape,
        target_shape,
        output_dim,
        intermediate_dim=128,
        freeze_encoder=True,
        use_attention_pooling=True,
        dropout_rate=0.3,
    ):
        super(EEGEncoder, self).__init__()

        self.encoder = encoder
        self.use_attention_pooling = use_attention_pooling

        self.adaptive_layer = AdaptiveBlock(
            input_shape=new_shape, target_shape=target_shape, in_channels=1
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.use_attention_pooling:
            # self.attention_pool = MultiHeadAttentionPool(
            #     embed_dim=self.encoder.embed_dim,
            #     num_heads=2
            # )
            self.attention_pool = nn.Sequential(
                nn.Linear(self.encoder.embed_dim, 1),  # Attention weights per token
                nn.Softmax(dim=1),
            )

        self.fc1 = nn.Linear(self.encoder.embed_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(intermediate_dim)


def forward(self, x):

    x = self.adaptive_layer(x)

    with (
        torch.no_grad()
        if not any(param.requires_grad for param in self.encoder.parameters())
        else torch.enable_grad()
    ):
        features = self.encoder(x)

    if self.use_attention_pooling:
        attention_weights = self.attention_pool(
            features
        )  # Shape: (batch_size, num_tokens, 1)
        pooled = (features * attention_weights).sum(dim=1)  # Weighted sum
    else:
        pooled = features.mean(dim=1)

    x = self.fc1(pooled)
    x = self.batch_norm(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)

    return x


class TabularEncoder(nn.Module):

    def __init__(
        self, input_dim, output_dim, hidden_layers=[256, 128, 64], dropout_rate=0.2
    ):
        super(TabularEncoder, self).__init__()

        layers = [
            nn.Linear(input_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.Dropout(dropout_rate),
        ]

        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_layers[i]))
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        # combine layers into sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
