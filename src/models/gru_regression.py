import torch.nn as nn


class GRURegression(nn.Module):
    def __init__(
        self,
        input_size=31,
        hidden_size=128,
        num_depth_outputs=5,
        num_layers=3,
        dropout=0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.depth_head = nn.Linear(hidden_size, num_depth_outputs)
        self.overflow_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, x):
        _, h_n = self.gru(x)
        hidden = h_n[-1]
        depths = self.depth_head(hidden)
        overflow = self.overflow_head(hidden)
        return depths, overflow
