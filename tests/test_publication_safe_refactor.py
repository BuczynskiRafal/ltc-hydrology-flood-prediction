from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

_SKIP_REASON = "PyTorch is not installed"

if HAS_TORCH:
    from experiments.regression_pipeline import (
        load_trained_model,
        set_global_seed,
    )
    from src.models.gru_regression import GRURegression
    from src.models.lnn_regression import LNNRegression
    from src.models.lstm_regression import LSTMRegression
    from src.models.mlp_regression import MLPRegression
    from src.models.tcn_regression import TCNRegression

from src.prepare_windows import (
    EXPECTED_FEATURE_COLUMNS,
    EXPECTED_TABULAR_COLUMNS,
    get_feature_columns,
    validate_required_columns,
    validate_tabular_schema,
    validate_time_schema,
)
from src.project_config import (
    TARGET_SENSORS,
    WINDOW_STRIDE,
    WINDOW_T_IN,
    WINDOW_T_OUT,
)
from src.selected_features import SELECTED_FEATURES
from src.window_utils import (
    create_sequences_with_regression_targets,
    create_sliding_windows,
    split_dataframe_by_time_gaps,
)


def _make_canonical_labeled_frame(start: str, periods: int) -> pd.DataFrame:
    times = pd.date_range(start=start, periods=periods, freq="1min")
    frame = pd.DataFrame({"time": times, "year": times.year})
    for feature_index, feature_name in enumerate(SELECTED_FEATURES):
        frame[feature_name] = np.linspace(
            feature_index,
            feature_index + 1.0,
            periods,
            dtype=np.float32,
        )
    for target_index, sensor_name in enumerate(TARGET_SENSORS):
        frame[sensor_name] = np.linspace(
            target_index,
            target_index + 0.5,
            periods,
            dtype=np.float32,
        )
    frame["flash_flood"] = 0
    frame["target"] = (np.arange(periods) % 2).astype(np.int32)
    return frame[EXPECTED_TABULAR_COLUMNS]


def _empty_flash_floods() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([]),
            "period_end": pd.to_datetime([]),
            "flash_flood": pd.Series(dtype=int),
        }
    )


def _make_tau_schedule(
    batch_size: int, seq_len: int, first_value: float, second_value: float
) -> torch.Tensor:
    first_half = seq_len // 2
    second_half = seq_len - first_half
    first = torch.full((batch_size, first_half, 1), first_value, dtype=torch.float32)
    second = torch.full((batch_size, second_half, 1), second_value, dtype=torch.float32)
    return torch.cat([first, second], dim=1)


if HAS_TORCH:

    class _SequenceFlashGate(torch.nn.Module):
        def __init__(self, tau_schedule: torch.Tensor):
            super().__init__()
            self.register_buffer("tau_schedule", tau_schedule)
            self._step = 0

        def forward(self, x, x_prev=None):
            tau_t = self.tau_schedule[:, self._step, :].to(
                device=x.device, dtype=x.dtype
            )
            self._step += 1
            change_score = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
            return change_score, tau_t

    class _RecordingHierarchicalLTC(torch.nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.hidden_size = hidden_size
            self.last_tau_fast = None

        def forward(
            self, x, tau_fast=None, use_fast_path=True, use_slow_path=True
        ):  # pragma: no cover - invoked through the model
            self.last_tau_fast = tau_fast.detach().clone()
            if tau_fast.dim() == 2:
                per_step = tau_fast.unsqueeze(1).expand(x.size(0), x.size(1), 1)
            elif tau_fast.dim() == 3:
                per_step = tau_fast
            else:
                raise AssertionError(f"Unexpected tau_fast rank: {tau_fast.dim()}")
            latent = per_step.expand(-1, -1, self.hidden_size).contiguous()
            return latent, (latent, latent)


@unittest.skipUnless(HAS_TORCH, _SKIP_REASON)
class TestModelContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.batch_size = 4
        self.seq_len = WINDOW_T_IN
        self.input_size = 31
        self.x = torch.randn(self.batch_size, self.seq_len, self.input_size)

    def test_baseline_model_shapes(self):
        cases = [
            (
                "gru",
                GRURegression(
                    input_size=31,
                    hidden_size=128,
                    num_depth_outputs=5,
                    num_layers=3,
                    dropout=0.2,
                ),
            ),
            (
                "lstm",
                LSTMRegression(
                    input_size=31,
                    hidden_size=128,
                    num_depth_outputs=5,
                    num_layers=2,
                    dropout=0.2,
                ),
            ),
            (
                "tcn",
                TCNRegression(
                    input_size=31,
                    hidden_size=128,
                    num_depth_outputs=5,
                    kernel_size=3,
                    num_layers=3,
                    dropout=0.2,
                ),
            ),
            (
                "mlp",
                MLPRegression(
                    input_size=31,
                    seq_len=45,
                    hidden_dims=[256, 128, 64],
                    num_depth_outputs=5,
                    dropout=0.2,
                    use_batch_norm=True,
                ),
            ),
        ]

        for name, model in cases:
            with self.subTest(model=name):
                model.eval()
                depths, overflow = model(self.x)
                self.assertEqual(tuple(depths.shape), (self.batch_size, 5))
                self.assertEqual(tuple(overflow.shape), (self.batch_size, 1))
                self.assertTrue(torch.all(overflow >= 0.0))
                self.assertTrue(torch.all(overflow <= 1.0))

    def test_lnn_return_attention_contract(self):
        model = LNNRegression(
            input_size=31,
            fast_units=128,
            slow_units=128,
            hidden_size=320,
            num_depth_outputs=5,
            dropout=0.15,
            tau_mode="stepwise",
        )
        model.eval()

        outputs = model(self.x, return_attention=True)
        self.assertEqual(len(outputs), 5)
        depths, overflow, intensity, attn_weights, tau_sequence = outputs

        self.assertEqual(tuple(depths.shape), (self.batch_size, 5))
        self.assertEqual(tuple(overflow.shape), (self.batch_size, 1))
        self.assertEqual(tuple(intensity.shape), (self.batch_size, 1))
        self.assertEqual(tuple(attn_weights.shape), (self.batch_size, self.seq_len, 1))
        self.assertEqual(tuple(tau_sequence.shape), (self.batch_size, self.seq_len, 1))


@unittest.skipUnless(HAS_TORCH, _SKIP_REASON)
class TestLnnTauSemantics(unittest.TestCase):
    def _build_model(self, tau_schedule: torch.Tensor) -> LNNRegression:
        model = LNNRegression(
            input_size=31,
            fast_units=128,
            slow_units=128,
            hidden_size=320,
            num_depth_outputs=5,
            dropout=0.15,
            tau_mode="stepwise",
        )
        model.flash_gate = _SequenceFlashGate(tau_schedule)
        model.hierarchical_ltc = _RecordingHierarchicalLTC(hidden_size=320)
        model.eval()
        return model

    def test_stepwise_tau_changes_behaviour_for_same_mean_profiles(self):
        batch_size = 2
        seq_len = WINDOW_T_IN
        x = torch.randn(batch_size, seq_len, 31)

        tau_schedule_a = _make_tau_schedule(batch_size, seq_len, 0.1, 2.1)
        tau_schedule_b = torch.flip(tau_schedule_a, dims=[1])
        self.assertAlmostEqual(
            tau_schedule_a.mean().item(), tau_schedule_b.mean().item()
        )

        model_a = self._build_model(tau_schedule_a)
        outputs_a = model_a(x, return_attention=True)
        tau_seen_a = model_a.hierarchical_ltc.last_tau_fast

        model_b = self._build_model(tau_schedule_b)
        outputs_b = model_b(x, return_attention=True)
        tau_seen_b = model_b.hierarchical_ltc.last_tau_fast

        self.assertEqual(tuple(tau_seen_a.shape), (batch_size, seq_len, 1))
        self.assertEqual(tuple(tau_seen_b.shape), (batch_size, seq_len, 1))
        self.assertFalse(torch.allclose(outputs_a[0], outputs_b[0]))
        self.assertFalse(torch.allclose(tau_seen_a, tau_seen_b))


@unittest.skipUnless(HAS_TORCH, _SKIP_REASON)
class TestDeterminism(unittest.TestCase):
    def _run_seeded_forward(self, seed: int):
        set_global_seed(seed)
        model = LSTMRegression(
            input_size=31,
            hidden_size=128,
            num_depth_outputs=5,
            num_layers=2,
            dropout=0.2,
        )
        model.eval()
        x = torch.randn(3, WINDOW_T_IN, 31)
        outputs = model(x)
        snapshot = {
            key: value.detach().clone()
            for key, value in list(model.state_dict().items())[:4]
        }
        return outputs, snapshot

    def test_seed_utility_produces_repeatable_initialization_and_forward(self):
        outputs_a, state_a = self._run_seeded_forward(42)
        outputs_b, state_b = self._run_seeded_forward(42)

        for tensor_a, tensor_b in zip(outputs_a, outputs_b):
            self.assertTrue(torch.allclose(tensor_a, tensor_b))

        self.assertEqual(state_a.keys(), state_b.keys())
        for key in state_a:
            self.assertTrue(torch.equal(state_a[key], state_b[key]), msg=key)


class TestSchemaAndWindowing(unittest.TestCase):
    def test_required_columns_and_tabular_schema(self):
        train_df = _make_canonical_labeled_frame("2021-01-01", 80)
        validate_required_columns("train", train_df)
        validate_tabular_schema("train", train_df)
        self.assertEqual(get_feature_columns(train_df), EXPECTED_FEATURE_COLUMNS)

        missing_target_sensor = train_df.drop(columns=[TARGET_SENSORS[0]])
        with self.assertRaises(ValueError):
            validate_required_columns("train", missing_target_sensor)

        extra_feature = train_df.copy(deep=True)
        extra_feature["unexpected_feature"] = np.linspace(
            0.0, 1.0, len(extra_feature), dtype=np.float32
        )
        with self.assertRaises(ValueError):
            validate_tabular_schema("train", extra_feature)

    def test_duplicate_timestamps_are_rejected(self):
        frame = _make_canonical_labeled_frame("2021-01-01", 20)
        duplicated = pd.concat(
            [frame.iloc[:5], frame.iloc[[4]], frame.iloc[5:]],
            ignore_index=True,
        )
        with self.assertRaises(ValueError):
            validate_time_schema("train", duplicated)

    def test_time_gap_segmentation_prevents_cross_gap_windows(self):
        first = _make_canonical_labeled_frame("2021-01-01 00:00", 70)
        second = _make_canonical_labeled_frame("2021-01-01 02:00", 70)
        frame = pd.concat([first, second], ignore_index=True)
        feature_cols = get_feature_columns(frame)

        segments = split_dataframe_by_time_gaps(frame, max_gap_minutes=1)
        self.assertEqual(len(segments), 2)

        total_windows = 0
        for segment in segments:
            X, y_depths, y_overflow, flood_mask = create_sliding_windows(
                df=segment,
                feature_cols=feature_cols,
                target_cols=TARGET_SENSORS,
                target_col="target",
                T_in=WINDOW_T_IN,
                T_out=WINDOW_T_OUT,
                stride=WINDOW_STRIDE,
            )
            self.assertEqual(X.shape[1:], (WINDOW_T_IN, len(feature_cols)))
            self.assertEqual(y_depths.shape[1], len(TARGET_SENSORS))
            self.assertEqual(y_overflow.ndim, 1)
            self.assertEqual(flood_mask.ndim, 1)
            total_windows += len(X)

        X_all, _, _, _ = create_sequences_with_regression_targets(
            frame,
            _empty_flash_floods(),
            feature_cols,
            TARGET_SENSORS,
            gap_minutes=1,
        )
        self.assertEqual(total_windows, len(X_all))
        self.assertGreater(total_windows, 0)

    def test_missing_predictors_fail_fast(self):
        frame = _make_canonical_labeled_frame("2021-01-01", 80)
        frame.loc[10, SELECTED_FEATURES[0]] = np.nan
        with self.assertRaises(ValueError):
            create_sliding_windows(
                df=frame,
                feature_cols=get_feature_columns(frame),
                target_cols=TARGET_SENSORS,
                target_col="target",
                T_in=WINDOW_T_IN,
                T_out=WINDOW_T_OUT,
                stride=WINDOW_STRIDE,
            )


@unittest.skipUnless(HAS_TORCH, _SKIP_REASON)
class TestCheckpointCompatibility(unittest.TestCase):
    def _canonical_config(self) -> dict:
        return {
            "schema_version": 2,
            "runtime": {"seed": 42, "device": "cpu", "deterministic": True},
            "model": {
                "input_size": 31,
                "fast_units": 128,
                "slow_units": 128,
                "hidden_size": 320,
                "num_depth_outputs": 5,
                "dropout": 0.15,
                "tau_mode": "stepwise",
            },
            "training": {
                "batch_size": 256,
                "epochs": 1,
                "gradient_clip": 0.8,
                "num_workers": 0,
                "optimizer": {
                    "type": "adam",
                    "learning_rate": 0.005,
                    "weight_decay": 0.00001,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
                "scheduler": {
                    "type": "cosine_annealing",
                    "eta_min": 0.0001,
                },
                "early_stopping": {"patience": 20, "min_delta": 0.0},
            },
            "loss": {
                "depth_weight": 0.5,
                "overflow_weight": 0.3,
                "intensity_weight": 0.2,
            },
            "data": {
                "use_reduced": True,
            },
            "evaluation": {
                "threshold_artifact": "artifacts/thresholds/lnn_val_threshold.json"
            },
            "output": {"checkpoint_dir": "artifacts/checkpoints/lnn"},
        }

    def test_embedded_checkpoint_config_is_used_directly(self):
        canonical_config = self._canonical_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "lnn_config.yaml"
            checkpoint_path = tmp_path / "best_model.pt"
            with open(config_path, "w") as handle:
                yaml.safe_dump(canonical_config, handle)

            reference_model = LNNRegression(
                input_size=31,
                fast_units=128,
                slow_units=128,
                hidden_size=320,
                num_depth_outputs=5,
                dropout=0.15,
                tau_mode="stepwise",
            )
            torch.save(
                {
                    "epoch": 1,
                    "model_state_dict": reference_model.state_dict(),
                    "optimizer_state_dict": {},
                    "val_loss": 0.0,
                    "val_accuracy": 0.0,
                    "n_params": sum(
                        parameter.numel()
                        for parameter in reference_model.parameters()
                        if parameter.requires_grad
                    ),
                    "config": canonical_config,
                },
                checkpoint_path,
            )

            artifact = load_trained_model(
                "lnn",
                device=torch.device("cpu"),
                config_path=config_path,
                checkpoint_path=checkpoint_path,
            )
            self.assertEqual(artifact["config_source"], "checkpoint")
            self.assertEqual(artifact["runtime_config"], canonical_config)
            sample_x = torch.randn(2, WINDOW_T_IN, 31)
            depths, overflow, intensity = artifact["model"](sample_x)
            self.assertEqual(tuple(depths.shape), (2, 5))
            self.assertEqual(tuple(overflow.shape), (2, 1))
            self.assertEqual(tuple(intensity.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
