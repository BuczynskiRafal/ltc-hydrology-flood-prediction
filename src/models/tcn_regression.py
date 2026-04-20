import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return self.dropout(self.relu(out))


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(
            n_channels, n_channels, kernel_size, dilation, dropout
        )
        self.conv2 = CausalConv1d(
            n_channels, n_channels, kernel_size, dilation, dropout
        )

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x


class TCNRegression(nn.Module):
    def __init__(
        self,
        input_size=31,
        hidden_size=128,
        num_depth_outputs=5,
        kernel_size=3,
        num_layers=3,
        dropout=0.2,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.input_projection = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_size, kernel_size, 2**i, dropout)
                for i in range(num_layers)
            ]
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.depth_head = nn.Linear(hidden_size, num_depth_outputs)
        self.overflow_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def get_receptive_field(self):
        return 1 + 2 * (self.kernel_size - 1) * (2**self.num_layers - 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.input_projection(x)
        for block in self.residual_blocks:
            out = block(out)
        out = self.global_pool(out).squeeze(2)
        depths = self.depth_head(out)
        overflow = self.overflow_head(out)
        return depths, overflow
