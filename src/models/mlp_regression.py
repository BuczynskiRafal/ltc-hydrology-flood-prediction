import torch.nn as nn


class MLPRegression(nn.Module):
    def __init__(
        self,
        input_size=31,
        seq_len=45,
        hidden_dims=None,
        num_depth_outputs=5,
        dropout=0.2,
        use_batch_norm=True,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims or [256, 128, 64])
        flat_size = seq_len * input_size
        layers = []
        in_dim = flat_size

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.depth_head = nn.Linear(hidden_dims[-1], num_depth_outputs)
        self.overflow_head = nn.Sequential(nn.Linear(hidden_dims[-1], 1), nn.Sigmoid())

    def forward(self, x):
        x_flat = x.reshape(x.size(0), -1)
        features = self.mlp(x_flat)
        depths = self.depth_head(features)
        overflow = self.overflow_head(features)
        return depths, overflow
