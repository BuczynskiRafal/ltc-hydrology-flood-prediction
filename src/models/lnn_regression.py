import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError(f"slow_tau_init must be positive, got {value}.")
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


class FlashFloodGate(nn.Module):
    def __init__(self, input_dim, hidden_dim, tau_dim=1, tau_min=0.01, tau_max=10.0):
        super().__init__()
        if tau_min <= 0.0:
            raise ValueError(f"tau_min must be positive, got {tau_min}.")
        if tau_max <= tau_min:
            raise ValueError(
                f"tau_max must be greater than tau_min, got tau_min={tau_min}, tau_max={tau_max}."
            )
        self.change_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, tau_dim),
            nn.Sigmoid(),
        )
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

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
        if tau.dim() not in {2, 3}:
            raise ValueError(
                f"tau must have shape [B, H] or [B, T, H], got {tuple(tau.shape)}"
            )
        for t in range(seq_len):
            x_t = x[:, t, :]
            update = torch.tanh(self.W_in(x_t) + self.W_rec(h))
            tau_t = tau[:, t, :] if tau.dim() == 3 else tau

            decay = torch.exp(-1.0 / tau_t)
            h = h * decay + update * (1.0 - decay)

            h_sequence.append(h)
        return torch.stack(h_sequence, dim=1), h


class HierarchicalLTC(nn.Module):
    def __init__(
        self,
        input_size,
        fast_units,
        slow_units,
        output_units,
        *,
        use_learnable_slow_tau=False,
        slow_tau_init=5.0,
        use_path_layer_norm=False,
    ):
        super().__init__()
        self.fast_units = fast_units
        self.slow_units = slow_units
        self.use_learnable_slow_tau = use_learnable_slow_tau
        self.fast_ltc = SimpleLTCCell(input_size, fast_units)
        self.slow_ltc = SimpleLTCCell(input_size, slow_units)
        if use_learnable_slow_tau:
            self.slow_tau = nn.Parameter(
                torch.tensor(
                    _inverse_softplus(float(slow_tau_init)), dtype=torch.float32
                )
            )
        else:
            self.register_buffer(
                "slow_tau",
                torch.tensor(float(slow_tau_init), dtype=torch.float32),
            )
        self.fast_norm = (
            nn.LayerNorm(fast_units) if use_path_layer_norm else nn.Identity()
        )
        self.slow_norm = (
            nn.LayerNorm(slow_units) if use_path_layer_norm else nn.Identity()
        )
        self.fusion = nn.Sequential(
            nn.Linear(fast_units + slow_units, output_units), nn.Tanh()
        )

    def _resolve_slow_tau(self, batch_size, device, dtype):
        tau = (
            F.softplus(self.slow_tau)
            if isinstance(self.slow_tau, nn.Parameter)
            else self.slow_tau
        )
        return tau.to(device=device, dtype=dtype).expand(batch_size, 1)

    def forward(self, x, tau_fast=None, use_fast_path=True, use_slow_path=True):
        batch_size = x.size(0)
        h_fast = torch.zeros(batch_size, self.fast_units, device=x.device)
        h_slow = torch.zeros(batch_size, self.slow_units, device=x.device)

        if tau_fast is None:
            tau_fast = torch.ones(batch_size, 1, device=x.device)

        tau_slow = self._resolve_slow_tau(batch_size, x.device, x.dtype)
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
        combined = torch.cat(
            [self.fast_norm(fast_out), self.slow_norm(slow_out)], dim=-1
        )
        return self.fusion(combined), (fast_out, slow_out)


class LNNRegression(nn.Module):
    def __init__(
        self,
        input_size=31,
        encoder_input_size=None,
        fast_units=32,
        slow_units=32,
        hidden_size=64,
        num_depth_outputs=5,
        dropout=0.2,
        tau_mode="mean_legacy",
        use_fast_path=True,
        use_slow_path=True,
        use_attention=True,
        use_learnable_slow_tau=False,
        slow_tau_init=5.0,
        use_path_layer_norm=False,
        per_neuron_tau=False,
        fast_tau_min=0.01,
        fast_tau_max=10.0,
        use_separate_depth_heads=False,
        depth_head_hidden_size=64,
        pump_head_target_index=None,
        pump_head_feature_indices=None,
        use_pump_branch=False,
        pump_branch_input_indices=None,
        pump_branch_fast_units=16,
        pump_branch_slow_units=16,
        pump_branch_hidden_size=32,
        pump_branch_use_attention=None,
    ):
        super().__init__()
        if tau_mode not in {"mean_legacy", "stepwise"}:
            raise ValueError(
                "tau_mode must be either 'mean_legacy' or 'stepwise', "
                f"got '{tau_mode}'."
            )
        self.tau_mode = tau_mode
        self.use_fast_path = use_fast_path
        self.use_slow_path = use_slow_path
        self.use_attention = use_attention
        self.per_neuron_tau = per_neuron_tau
        self.use_separate_depth_heads = use_separate_depth_heads
        self.num_depth_outputs = num_depth_outputs
        self.input_size = int(input_size)
        self.encoder_input_size = int(
            self.input_size if encoder_input_size is None else encoder_input_size
        )
        if self.encoder_input_size <= 0:
            raise ValueError(
                f"encoder_input_size must be positive, got {self.encoder_input_size}."
            )
        if self.encoder_input_size > self.input_size:
            raise ValueError(
                "encoder_input_size cannot exceed input_size, got "
                f"encoder_input_size={self.encoder_input_size}, input_size={self.input_size}."
            )
        if depth_head_hidden_size <= 0:
            raise ValueError(
                f"depth_head_hidden_size must be positive, got {depth_head_hidden_size}."
            )
        if pump_head_feature_indices is None:
            pump_head_feature_indices = []
        self.pump_head_feature_indices = tuple(
            int(i) for i in pump_head_feature_indices
        )
        if pump_branch_input_indices is None:
            pump_branch_input_indices = []
        self.pump_branch_input_indices = tuple(
            int(i) for i in pump_branch_input_indices
        )
        self.pump_head_target_index = (
            None if pump_head_target_index is None else int(pump_head_target_index)
        )
        self.use_pump_branch = bool(use_pump_branch)
        if self.pump_head_feature_indices:
            if not use_separate_depth_heads:
                raise ValueError(
                    "pump_head_feature_indices requires use_separate_depth_heads=True."
                )
            if self.pump_head_target_index is None:
                raise ValueError(
                    "pump_head_target_index must be set when using pump_head_feature_indices."
                )
            if not (0 <= self.pump_head_target_index < num_depth_outputs):
                raise ValueError(
                    "pump_head_target_index must be within [0, num_depth_outputs), got "
                    f"{self.pump_head_target_index}."
                )
            invalid_indices = [
                feature_index
                for feature_index in self.pump_head_feature_indices
                if feature_index < 0 or feature_index >= self.input_size
            ]
            if invalid_indices:
                raise ValueError(
                    "pump_head_feature_indices contain out-of-range values: "
                    f"{invalid_indices}"
                )
        self.pump_head_aux_dim = len(self.pump_head_feature_indices)
        if self.use_pump_branch:
            if not use_separate_depth_heads:
                raise ValueError(
                    "use_pump_branch requires use_separate_depth_heads=True."
                )
            if self.pump_head_target_index is None:
                raise ValueError(
                    "pump_head_target_index must be set when use_pump_branch=True."
                )
            if not self.pump_branch_input_indices:
                raise ValueError(
                    "pump_branch_input_indices must be provided when use_pump_branch=True."
                )
            invalid_branch_indices = [
                feature_index
                for feature_index in self.pump_branch_input_indices
                if feature_index < 0 or feature_index >= self.input_size
            ]
            if invalid_branch_indices:
                raise ValueError(
                    "pump_branch_input_indices contain out-of-range values: "
                    f"{invalid_branch_indices}"
                )
            for name, value in {
                "pump_branch_fast_units": pump_branch_fast_units,
                "pump_branch_slow_units": pump_branch_slow_units,
                "pump_branch_hidden_size": pump_branch_hidden_size,
            }.items():
                if int(value) <= 0:
                    raise ValueError(f"{name} must be positive, got {value}.")
        self.pump_branch_output_dim = (
            int(pump_branch_hidden_size) if self.use_pump_branch else 0
        )
        tau_dim = fast_units if per_neuron_tau else 1
        self.flash_gate = FlashFloodGate(
            input_dim=self.encoder_input_size * 2,
            hidden_dim=32,
            tau_dim=tau_dim,
            tau_min=fast_tau_min,
            tau_max=fast_tau_max,
        )
        self.hierarchical_ltc = HierarchicalLTC(
            self.encoder_input_size,
            fast_units,
            slow_units,
            hidden_size,
            use_learnable_slow_tau=use_learnable_slow_tau,
            slow_tau_init=slow_tau_init,
            use_path_layer_norm=use_path_layer_norm,
        )
        self.pump_branch_use_attention = (
            self.use_attention
            if pump_branch_use_attention is None
            else bool(pump_branch_use_attention)
        )
        if self.use_pump_branch:
            pump_tau_dim = int(pump_branch_fast_units) if per_neuron_tau else 1
            self.pump_flash_gate = FlashFloodGate(
                input_dim=len(self.pump_branch_input_indices) * 2,
                hidden_dim=16,
                tau_dim=pump_tau_dim,
                tau_min=fast_tau_min,
                tau_max=fast_tau_max,
            )
            self.pump_hierarchical_ltc = HierarchicalLTC(
                len(self.pump_branch_input_indices),
                int(pump_branch_fast_units),
                int(pump_branch_slow_units),
                int(pump_branch_hidden_size),
                use_learnable_slow_tau=use_learnable_slow_tau,
                slow_tau_init=slow_tau_init,
                use_path_layer_norm=use_path_layer_norm,
            )
            self.pump_attention = nn.Sequential(
                nn.Linear(int(pump_branch_hidden_size), 16),
                nn.Tanh(),
                nn.Linear(16, 1),
                nn.Softmax(dim=1),
            )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.Tanh(), nn.Linear(32, 1), nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(dropout)
        if use_separate_depth_heads:
            self.depth_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            hidden_size
                            + (
                                self.pump_head_aux_dim
                                if head_index == self.pump_head_target_index
                                else 0
                            )
                            + (
                                self.pump_branch_output_dim
                                if head_index == self.pump_head_target_index
                                else 0
                            ),
                            depth_head_hidden_size,
                        ),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(depth_head_hidden_size, 1),
                    )
                    for head_index in range(num_depth_outputs)
                ]
            )
        else:
            self.depth_head = nn.Sequential(
                nn.Linear(hidden_size, depth_head_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(depth_head_hidden_size, num_depth_outputs),
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

    def _build_tau_sequence(self, branch_x, flash_gate):
        seq_len = branch_x.size(1)
        tau_values = []
        for t in range(seq_len):
            x_t = branch_x[:, t, :]
            x_prev = branch_x[:, t - 1, :] if t > 0 else x_t
            _, tau_mod = flash_gate(x_t, x_prev)
            tau_values.append(tau_mod)
        tau_sequence = torch.stack(tau_values, dim=1)
        if self.tau_mode == "mean_legacy":
            return tau_sequence.mean(dim=1), tau_sequence
        return tau_sequence, tau_sequence

    def forward(self, x, return_attention=False):
        encoder_x = x[:, :, : self.encoder_input_size]
        seq_len = x.size(1)
        tau_fast, tau_sequence = self._build_tau_sequence(encoder_x, self.flash_gate)

        ltc_out, _ = self.hierarchical_ltc(
            encoder_x,
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

        if self.use_separate_depth_heads:
            pump_head_context = None
            if self.pump_head_feature_indices:
                aux_features = x[:, :, self.pump_head_feature_indices]
                if self.use_attention:
                    pump_head_context = (aux_features * attn_weights).sum(dim=1)
                else:
                    pump_head_context = aux_features.mean(dim=1)
            pump_branch_context = None
            if self.use_pump_branch:
                pump_branch_x = x[:, :, self.pump_branch_input_indices]
                pump_branch_tau, _ = self._build_tau_sequence(
                    pump_branch_x, self.pump_flash_gate
                )
                pump_ltc_out, _ = self.pump_hierarchical_ltc(
                    pump_branch_x,
                    tau_fast=pump_branch_tau,
                    use_fast_path=self.use_fast_path,
                    use_slow_path=self.use_slow_path,
                )
                if self.pump_branch_use_attention:
                    pump_attn = self.pump_attention(pump_ltc_out)
                    pump_branch_context = (pump_ltc_out * pump_attn).sum(dim=1)
                else:
                    pump_branch_context = pump_ltc_out.mean(dim=1)
                pump_branch_context = self.dropout(pump_branch_context)
            depth_outputs = []
            for head_index, head in enumerate(self.depth_heads):
                head_input = context
                if (
                    pump_head_context is not None
                    and head_index == self.pump_head_target_index
                ):
                    head_input = torch.cat([context, pump_head_context], dim=1)
                if (
                    pump_branch_context is not None
                    and head_index == self.pump_head_target_index
                ):
                    head_input = torch.cat([head_input, pump_branch_context], dim=1)
                depth_outputs.append(head(head_input))
            depths = torch.cat(depth_outputs, dim=1)
        else:
            depths = self.depth_head(context)
        overflow = self.overflow_head(context)
        intensity = self.intensity_head(context)

        if return_attention:
            return depths, overflow, intensity, attn_weights, tau_sequence
        return depths, overflow, intensity
