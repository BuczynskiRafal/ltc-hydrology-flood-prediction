import torch
import torch.nn as nn


class FlashFloodGate(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.change_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.tau_min = 0.01
        self.tau_max = 10.0

    def forward(self, x, x_prev=None):
        if x_prev is None:
            x_prev = x
        dx_dt = x - x_prev
        change_input = torch.cat([x, dx_dt], dim=-1)
        change_score = self.change_detector(change_input)
        tau_modulation = self.tau_max - change_score * (self.tau_max - self.tau_min)
        return change_score, tau_modulation


class SimpleLTCCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h, tau):
        batch_size, seq_len, _ = x.shape
        h_sequence = []
        if tau.dim() != 2 and tau.dim() != 3:
            raise ValueError(
                f"tau must have shape [B, 1] or [B, T, 1], got {tuple(tau.shape)}"
            )
        for t in range(seq_len):
            x_t = x[:, t, :]
            update = torch.tanh(self.W_in(x_t) + self.W_rec(h))
            tau_t = tau[:, t, :] if tau.dim() == 3 else tau
            dh_dt = (-h + update) / tau_t
            h = h + dh_dt
            h_sequence.append(h)
        return torch.stack(h_sequence, dim=1), h


class HierarchicalLTC(nn.Module):
    def __init__(self, input_size, fast_units, slow_units, output_units):
        super().__init__()
        self.fast_units = fast_units
        self.slow_units = slow_units
        self.fast_ltc = SimpleLTCCell(input_size, fast_units)
        self.slow_ltc = SimpleLTCCell(input_size, slow_units)
        self.register_buffer("slow_tau", torch.tensor(5.0))
        self.fusion = nn.Sequential(
            nn.Linear(fast_units + slow_units, output_units), nn.Tanh()
        )

    def forward(self, x, tau_fast=None, use_fast_path=True, use_slow_path=True):
        batch_size = x.size(0)
        h_fast = torch.zeros(batch_size, self.fast_units, device=x.device)
        h_slow = torch.zeros(batch_size, self.slow_units, device=x.device)

        if tau_fast is None:
            tau_fast = torch.ones(batch_size, 1, device=x.device)

        tau_slow = self.slow_tau.expand(batch_size, 1)
        if use_fast_path:
            fast_out, _ = self.fast_ltc(x, h_fast, tau_fast)
        else:
            fast_out = torch.zeros(
                batch_size, x.size(1), self.fast_units, device=x.device
            )
        if use_slow_path:
            slow_out, _ = self.slow_ltc(x, h_slow, tau_slow)
        else:
            slow_out = torch.zeros(
                batch_size, x.size(1), self.slow_units, device=x.device
            )
        combined = torch.cat([fast_out, slow_out], dim=-1)
        return self.fusion(combined), (fast_out, slow_out)


class LNNRegression(nn.Module):
    def __init__(
        self,
        input_size=31,
        fast_units=32,
        slow_units=32,
        hidden_size=64,
        num_depth_outputs=5,
        dropout=0.2,
        tau_mode="stepwise",
        use_fast_path=True,
        use_slow_path=True,
        use_attention=True,
    ):
        super().__init__()
        if tau_mode != "stepwise":
            raise ValueError(
                "tau_mode must be 'stepwise' in this repository, " f"got '{tau_mode}'."
            )
        self.tau_mode = tau_mode
        self.use_fast_path = use_fast_path
        self.use_slow_path = use_slow_path
        self.use_attention = use_attention
        self.flash_gate = FlashFloodGate(input_dim=input_size * 2, hidden_dim=32)
        self.hierarchical_ltc = HierarchicalLTC(
            input_size, fast_units, slow_units, hidden_size
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.Tanh(), nn.Linear(32, 1), nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(dropout)
        self.depth_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_depth_outputs),
        )
        self.overflow_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x, return_attention=False):
        seq_len = x.size(1)
        tau_values = []
        change_scores = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            x_prev = x[:, t - 1, :] if t > 0 else x_t
            change_score, tau_mod = self.flash_gate(x_t, x_prev)
            change_scores.append(change_score)
            tau_values.append(tau_mod)

        tau_sequence = torch.stack(tau_values, dim=1)
        tau_fast = tau_sequence

        ltc_out, _ = self.hierarchical_ltc(
            x,
            tau_fast=tau_fast,
            use_fast_path=self.use_fast_path,
            use_slow_path=self.use_slow_path,
        )
        if self.use_attention:
            attn_weights = self.attention(ltc_out)
            context = (ltc_out * attn_weights).sum(dim=1)
        else:
            attn_weights = torch.full(
                (x.size(0), seq_len, 1),
                fill_value=1.0 / seq_len,
                device=x.device,
            )
            context = ltc_out.mean(dim=1)
        context = self.dropout(context)

        depths = self.depth_head(context)
        overflow = self.overflow_head(context)
        intensity = self.intensity_head(context)

        if return_attention:
            return depths, overflow, intensity, attn_weights, tau_sequence
        return depths, overflow, intensity
