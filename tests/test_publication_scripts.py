"""Smoke tests for publication scripts and the training entrypoint.

These tests verify that eval_robustness.py, eval_ig.py, run_ablations.py,
and the training pipeline can execute their core logic on small synthetic
data without real checkpoints or datasets on disk.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

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
    from experiments.eval_utils import evaluate_depths
    from experiments.regression_pipeline import (
        aggregate_prediction_sets,
        build_ensemble_checkpoint_path,
        create_model,
        create_test_loader,
        evaluate_lnn_ensemble,
        train_lnn_ensemble,
        train_model,
    )
    from src.evaluation.uncertainty_analysis import DeltaMethodUncertainty
    from src.evaluation.uncertainty_metrics import compute_all_uncertainty_metrics
    from src.models.lnn_regression import LNNRegression
    from src.project_config import WINDOW_T_IN
else:
    WINDOW_T_IN = 45  # fallback so helpers below can be defined


def _make_lnn_config() -> dict:
    """Return a minimal valid LNN config for testing."""
    return {
        "schema_version": 2,
        "runtime": {"seed": 42, "device": "cpu", "deterministic": True},
        "model": {
            "input_size": 31,
            "fast_units": 16,
            "slow_units": 16,
            "hidden_size": 32,
            "num_depth_outputs": 5,
            "dropout": 0.0,
            "tau_mode": "stepwise",
        },
        "training": {
            "batch_size": 4,
            "epochs": 1,
            "gradient_clip": 1.0,
            "num_workers": 0,
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
            "scheduler": {"type": "cosine_annealing", "eta_min": 0.0001},
            "early_stopping": {"patience": 5, "min_delta": 0.0},
        },
        "loss": {
            "depth_weight": 0.5,
            "overflow_weight": 0.3,
            "intensity_weight": 0.2,
        },
        "data": {"use_reduced": True},
        "evaluation": {"threshold_artifact": "/tmp/threshold.json"},
        "output": {"checkpoint_dir": "/tmp/checkpoints/lnn"},
    }


def _make_baseline_config(model_name: str) -> dict:
    """Return a minimal valid baseline config for testing."""
    base = {
        "schema_version": 2,
        "runtime": {"seed": 42, "device": "cpu", "deterministic": True},
        "training": {
            "batch_size": 4,
            "epochs": 1,
            "gradient_clip": 1.0,
            "num_workers": 0,
            "optimizer": {"type": "adam", "learning_rate": 0.001},
            "scheduler": {
                "type": "reduce_on_plateau",
                "patience": 5,
                "factor": 0.5,
                "min_lr": 1e-6,
            },
            "early_stopping": {"patience": 5, "min_delta": 0.0},
        },
        "loss": {"depth_weight": 0.7, "overflow_weight": 0.3, "flood_weight": 5.0},
        "data": {"use_reduced": True},
        "evaluation": {"threshold_artifact": "/tmp/threshold.json"},
        "output": {"checkpoint_dir": f"/tmp/checkpoints/{model_name}"},
    }

    model_configs = {
        "gru": {
            "input_size": 31,
            "hidden_size": 32,
            "num_depth_outputs": 5,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "lstm": {
            "input_size": 31,
            "hidden_size": 32,
            "num_depth_outputs": 5,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "tcn": {
            "input_size": 31,
            "hidden_size": 32,
            "num_depth_outputs": 5,
            "kernel_size": 3,
            "num_layers": 2,
            "dropout": 0.0,
        },
        "mlp": {
            "input_size": 31,
            "seq_len": WINDOW_T_IN,
            "hidden_dims": [64, 32],
            "num_depth_outputs": 5,
            "dropout": 0.0,
            "use_batch_norm": False,
        },
    }
    base["model"] = model_configs[model_name]
    return base


def _make_synthetic_split(n_samples: int = 16) -> dict:
    """Return synthetic regression data matching the canonical contract."""
    return {
        "X": np.random.randn(n_samples, WINDOW_T_IN, 31).astype(np.float32),
        "y_depths": np.random.rand(n_samples, 5).astype(np.float32),
        "y_overflow": np.random.randint(0, 2, size=n_samples).astype(np.float32),
        "flood_mask": np.ones(n_samples, dtype=np.float32),
    }


@unittest.skipUnless(HAS_TORCH, _SKIP_REASON)
class TestEvalRobustnessSmoke(unittest.TestCase):
    """Verify the core evaluation logic in eval_robustness works on synthetic data."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.config = _make_lnn_config()
        self.config["model"]["dropout"] = 0.1
        self.model = create_model("lnn", self.config)
        self.model.eval()
        self.split_data = _make_synthetic_split(n_samples=8)

    def test_gaussian_transform_changes_output(self):
        """Gaussian noise transform should produce different predictions."""
        from experiments.eval_robustness import (
            _make_gaussian_transform,
            _predict_with_transform,
        )

        loader = create_test_loader(self.config, self.split_data)
        reference_batch = torch.from_numpy(self.split_data["X"][:4]).float()

        y_true_clean, y_pred_clean = _predict_with_transform(
            "lnn", self.model, loader, torch.device("cpu")
        )
        transform = _make_gaussian_transform(reference_batch, snr_db=10.0)
        y_true_noisy, y_pred_noisy = _predict_with_transform(
            "lnn", self.model, loader, torch.device("cpu"), transform=transform
        )

        self.assertEqual(y_true_clean.shape, y_true_noisy.shape)
        self.assertEqual(y_pred_clean.shape, y_pred_noisy.shape)
        np.testing.assert_array_equal(y_true_clean, y_true_noisy)
        self.assertFalse(
            np.allclose(y_pred_clean, y_pred_noisy, atol=1e-6),
            "Gaussian noise should change predictions.",
        )

    def test_predict_with_transform_forces_eval_mode(self):
        """Baseline/noise prediction helper should disable dropout stochasticity."""
        from experiments.eval_robustness import _predict_with_transform

        loader = create_test_loader(self.config, self.split_data)
        self.model.train()
        _, first_pred = _predict_with_transform(
            "lnn",
            self.model,
            loader,
            torch.device("cpu"),
        )

        self.model.train()
        _, second_pred = _predict_with_transform(
            "lnn",
            self.model,
            loader,
            torch.device("cpu"),
        )

        self.assertFalse(self.model.training)
        np.testing.assert_allclose(first_pred, second_pred)

    def test_gaussian_transform_is_deterministic_for_same_seed(self):
        """Gaussian perturbation should be reproducible for a fixed local seed."""
        from experiments.eval_robustness import _make_gaussian_transform

        reference_batch = torch.from_numpy(self.split_data["X"][:4]).float()
        inputs = torch.from_numpy(self.split_data["X"][:2]).float()
        transform_a = _make_gaussian_transform(reference_batch, snr_db=10.0, seed=123)
        transform_b = _make_gaussian_transform(reference_batch, snr_db=10.0, seed=123)

        np.testing.assert_allclose(
            transform_a(inputs).numpy(),
            transform_b(inputs).numpy(),
        )

    def test_fgsm_produces_valid_results(self):
        """FGSM perturbation should return well-shaped arrays."""
        from experiments.eval_robustness import _predict_with_fgsm

        loader = create_test_loader(self.config, self.split_data)
        y_true, y_pred = _predict_with_fgsm(
            "lnn", self.model, loader, torch.device("cpu"), epsilon=0.01
        )
        self.assertEqual(y_true.shape[0], len(self.split_data["X"]))
        self.assertEqual(y_true.shape[1], 5)
        self.assertEqual(y_pred.shape, y_true.shape)

    def test_missing_data_mask_rate_is_close_to_requested_probability(self):
        """Bernoulli masking should approximate the requested element-wise rate."""
        from experiments.eval_robustness import _apply_missing_data_mask

        x = torch.randn(32, WINDOW_T_IN, 31)
        feature_means = torch.zeros(31)
        generator = torch.Generator().manual_seed(123)
        _, mask = _apply_missing_data_mask(
            x,
            mask_probability=0.2,
            feature_means=feature_means,
            generator=generator,
        )

        self.assertAlmostEqual(mask.float().mean().item(), 0.2, delta=0.03)

    def test_missing_data_interpolation_and_mean_fallback(self):
        """Interpolation should fill interior gaps and feature means should fill full gaps."""
        from experiments.eval_robustness import (
            _apply_missing_data_mask,
            _interpolate_masked_series,
        )

        interpolated = _interpolate_masked_series(
            np.array([1.0, 0.0, 0.0, 4.0], dtype=np.float32),
            np.array([True, False, False, True]),
            fill_value=99.0,
        )
        np.testing.assert_allclose(interpolated, np.array([1.0, 2.0, 3.0, 4.0]))

        x = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
        feature_means = torch.tensor([100.0, 200.0])
        masked_x, mask = _apply_missing_data_mask(
            x,
            mask_probability=1.0,
            feature_means=feature_means,
            generator=torch.Generator().manual_seed(7),
        )
        self.assertTrue(mask.all().item())
        np.testing.assert_allclose(
            masked_x.numpy(),
            np.array([[[100.0, 200.0], [100.0, 200.0], [100.0, 200.0]]]),
        )

    def test_evaluate_depths_on_predictions(self):
        """evaluate_depths should return valid metrics on synthetic predictions."""
        y_true = np.random.rand(20, 5).astype(np.float32)
        y_pred = y_true + np.random.randn(20, 5).astype(np.float32) * 0.1
        metrics = evaluate_depths(y_true, y_pred)
        self.assertIn("aggregated", metrics)
        self.assertIn("NSE", metrics["aggregated"])
        self.assertIn("RMSE", metrics["aggregated"])
        self.assertGreater(metrics["aggregated"]["NSE"], -10.0)


@unittest.skipUnless(HAS_TORCH, _SKIP_REASON)
class TestEvalIgSmoke(unittest.TestCase):
    """Verify Integrated Gradients runs on a small LNN model."""

    def setUp(self):
        torch.manual_seed(42)
        self.config = _make_lnn_config()
        self.model = create_model("lnn", self.config)
        self.model.eval()

    def test_ig_attributions_have_correct_shape(self):
        """IG attributions should match the input tensor shape."""
        try:
            from captum.attr import IntegratedGradients
        except ModuleNotFoundError:
            self.skipTest("Captum is not installed; skipping IG smoke test.")

        inputs = torch.randn(4, WINDOW_T_IN, 31)
        baselines = torch.zeros_like(inputs)

        def overflow_forward(x):
            return self.model(x)[1].squeeze(1)

        ig = IntegratedGradients(overflow_forward)
        attributions, delta = ig.attribute(
            inputs, baselines=baselines, return_convergence_delta=True
        )

        self.assertEqual(attributions.shape, inputs.shape)
        self.assertEqual(delta.shape[0], inputs.shape[0])

        per_feature = attributions.abs().mean(dim=(0, 1)).detach()
        per_timestep = attributions.abs().mean(dim=(0, 2)).detach()
        self.assertEqual(per_feature.shape[0], 31)
        self.assertEqual(per_timestep.shape[0], WINDOW_T_IN)

    def test_ig_module_guard_message(self):
        """eval_ig.main should raise informative error when captum is missing."""
        import experiments.eval_ig as eval_ig_module

        original_ig = eval_ig_module.IntegratedGradients
        original_err = eval_ig_module.CAPTUM_IMPORT_ERROR
        try:
            eval_ig_module.IntegratedGradients = None
            eval_ig_module.CAPTUM_IMPORT_ERROR = ModuleNotFoundError("test")

            with self.assertRaises(ModuleNotFoundError) as ctx:
                eval_ig_module.main()
            self.assertIn("Captum", str(ctx.exception))
        finally:
            eval_ig_module.IntegratedGradients = original_ig
            eval_ig_module.CAPTUM_IMPORT_ERROR = original_err


class TestVisualizeUncertaintyViews(unittest.TestCase):
    """Verify the visualization helper accepts the canonical multi-method NPZ shapes."""

    def test_resolve_uncertainty_view_supports_prefixed_mc_keys(self):
        from experiments.visualize_uncertainty import _resolve_uncertainty_view

        results = {
            "mc_depths_mean": np.ones((2, 5), dtype=np.float32),
            "mc_depths_std": np.ones((2, 5), dtype=np.float32),
            "mc_depths_ci_lower": np.zeros((2, 5), dtype=np.float32),
            "mc_depths_ci_upper": np.full((2, 5), 2.0, dtype=np.float32),
            "mc_depths_true": np.ones((2, 5), dtype=np.float32),
            "mc_overflow_mean": np.ones((2, 1), dtype=np.float32),
            "mc_overflow_std": np.ones((2, 1), dtype=np.float32),
            "mc_overflow_ci_lower": np.zeros((2, 1), dtype=np.float32),
            "mc_overflow_ci_upper": np.full((2, 1), 2.0, dtype=np.float32),
            "mc_overflow_true": np.ones(2, dtype=np.float32),
        }

        resolved, method = _resolve_uncertainty_view(results)

        self.assertEqual(method, "mc")
        self.assertIn("depths_mean", resolved)
        np.testing.assert_allclose(resolved["depths_mean"], 1.0)

    def test_resolve_uncertainty_view_rejects_legacy_keys(self):
        from experiments.visualize_uncertainty import _resolve_uncertainty_view

        results = {
            "depths_mean": np.ones((2, 5), dtype=np.float32),
            "depths_std": np.ones((2, 5), dtype=np.float32),
            "depths_ci_lower": np.zeros((2, 5), dtype=np.float32),
            "depths_ci_upper": np.full((2, 5), 2.0, dtype=np.float32),
            "depths_true": np.ones((2, 5), dtype=np.float32),
            "overflow_mean": np.ones((2, 1), dtype=np.float32),
            "overflow_std": np.ones((2, 1), dtype=np.float32),
            "overflow_ci_lower": np.zeros((2, 1), dtype=np.float32),
            "overflow_ci_upper": np.full((2, 1), 2.0, dtype=np.float32),
            "overflow_true": np.ones(2, dtype=np.float32),
        }

        with self.assertRaises(KeyError):
            _resolve_uncertainty_view(results)


@unittest.skipUnless(HAS_TORCH, _SKIP_REASON)
class TestRunAblationsSmoke(unittest.TestCase):
    """Verify ablation config mutation and evaluation on synthetic data."""

    def test_ablation_flags_are_applied_to_config(self):
        """Each ablation entry should correctly mutate the model config."""
        from experiments.run_ablations import ABLATIONS

        base_config = _make_lnn_config()

        for ablation_name, flags in ABLATIONS.items():
            with self.subTest(ablation=ablation_name):
                config = deepcopy(base_config)
                config["model"].update(flags)

                model = create_model("lnn", config)
                self.assertEqual(model.use_fast_path, flags["use_fast_path"])
                self.assertEqual(model.use_slow_path, flags["use_slow_path"])
                self.assertEqual(model.use_attention, flags["use_attention"])

    def test_ablation_forward_pass_produces_valid_output(self):
        """Each ablation variant should produce valid forward-pass outputs."""
        from experiments.run_ablations import ABLATIONS

        base_config = _make_lnn_config()
        x = torch.randn(2, WINDOW_T_IN, 31)

        for ablation_name, flags in ABLATIONS.items():
            with self.subTest(ablation=ablation_name):
                config = deepcopy(base_config)
                config["model"].update(flags)
                model = create_model("lnn", config)
                model.eval()

                depths, overflow, intensity = model(x)
                self.assertEqual(tuple(depths.shape), (2, 5))
                self.assertEqual(tuple(overflow.shape), (2, 1))
                self.assertEqual(tuple(intensity.shape), (2, 1))

    def test_evaluate_ablation_with_synthetic_data(self):
        """evaluate_ablation should return valid metric keys on synthetic data."""
        from experiments.run_ablations import evaluate_ablation

        config = _make_lnn_config()
        model = create_model("lnn", config)
        model.eval()
        split_data = _make_synthetic_split(n_samples=8)

        metrics = evaluate_ablation(
            "lnn", model, config, split_data, torch.device("cpu"), threshold=0.5
        )
        self.assertIn("depth_nse", metrics)
        self.assertIn("depth_rmse", metrics)
        self.assertIn("overflow_f1", metrics)
        self.assertIn("overflow_roc_auc", metrics)


@unittest.skipUnless(HAS_TORCH, _SKIP_REASON)
class TestEnsemblePipelineSmoke(unittest.TestCase):
    """Verify the LNN ensemble helpers train, aggregate, and evaluate on synthetic data."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.config = _make_lnn_config()
        self.seeds = [42, 43, 44, 45, 46]
        self.train_data = _make_synthetic_split(n_samples=16)
        self.val_data = _make_synthetic_split(n_samples=8)
        self.test_data = _make_synthetic_split(n_samples=8)

    def test_aggregate_prediction_sets_computes_means(self):
        """Ensemble aggregation should average continuous predictions member-wise."""
        base_predictions = {
            "true_depths": np.ones((2, 5), dtype=np.float32),
            "true_overflow": np.array([0.0, 1.0], dtype=np.float32),
            "pred_depths": np.full((2, 5), 2.0, dtype=np.float32),
            "pred_overflow": np.array([0.2, 0.8], dtype=np.float32),
            "pred_intensity": np.array([0.1, 0.9], dtype=np.float32),
        }
        shifted_predictions = {
            "true_depths": base_predictions["true_depths"].copy(),
            "true_overflow": base_predictions["true_overflow"].copy(),
            "pred_depths": np.full((2, 5), 4.0, dtype=np.float32),
            "pred_overflow": np.array([0.4, 0.6], dtype=np.float32),
            "pred_intensity": np.array([0.3, 0.7], dtype=np.float32),
        }

        aggregated = aggregate_prediction_sets(
            "lnn", [base_predictions, shifted_predictions]
        )

        np.testing.assert_allclose(aggregated["pred_depths"], 3.0)
        np.testing.assert_allclose(
            aggregated["pred_overflow"],
            np.array([0.3, 0.7], dtype=np.float32),
        )
        np.testing.assert_allclose(
            aggregated["pred_intensity"],
            np.array([0.2, 0.8], dtype=np.float32),
        )

    def test_train_and_evaluate_lnn_ensemble_on_synthetic_data(self):
        """The ensemble helpers should train five members and emit one manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_payload = train_lnn_ensemble(
                config=self.config,
                seeds=self.seeds,
                checkpoint_root=tmpdir,
                device=torch.device("cpu"),
                train_data=self.train_data,
                val_data=self.val_data,
            )

            self.assertEqual(training_payload["ensemble_size"], 5)
            for seed, member in zip(self.seeds, training_payload["members"]):
                expected_path = build_ensemble_checkpoint_path(
                    "lnn",
                    seed,
                    checkpoint_root=tmpdir,
                )
                self.assertEqual(Path(member["checkpoint_path"]), expected_path)
                self.assertTrue(expected_path.exists())

            payload, output_path = evaluate_lnn_ensemble(
                config=self.config,
                seeds=self.seeds,
                checkpoint_root=tmpdir,
                results_dir=Path(tmpdir) / "results",
                device=torch.device("cpu"),
                val_data=self.val_data,
                test_data=self.test_data,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(payload["ensemble_size"], 5)
            self.assertEqual(len(payload["member_metrics"]), 5)
            self.assertIn("ensemble_metrics", payload)
            self.assertIn("overflow_threshold", payload)
            self.assertEqual(
                payload["overflow_threshold_source"],
                "ensemble:val_average_probabilities",
            )
            self.assertIn("dataset_fingerprint", payload["runtime_metadata"])


@unittest.skipUnless(HAS_TORCH, _SKIP_REASON)
class TestUncertaintySmoke(unittest.TestCase):
    """Verify the uncertainty analyzers and saved artifacts cover both methods."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.config = _make_lnn_config()
        self.model = create_model("lnn", self.config)
        self.model.eval()
        self.split_data = _make_synthetic_split(n_samples=8)

    def test_delta_method_produces_expected_shapes(self):
        """Delta method should emit std for 5 depth outputs and 1 overflow output."""
        loader = create_test_loader(self.config, self.split_data)
        analyzer = DeltaMethodUncertainty(model=self.model, device="cpu")
        results = analyzer.predict_batch_with_uncertainty(loader)

        self.assertEqual(results["depths_mean"].shape, (8, 5))
        self.assertEqual(results["depths_std"].shape, (8, 5))
        self.assertEqual(results["overflow_mean"].shape, (8, 1))
        self.assertEqual(results["overflow_std"].shape, (8, 1))
        self.assertTrue(np.all(results["depths_std"] >= 0.0))
        self.assertTrue(np.all(results["overflow_std"] >= 0.0))

    def test_mc_dropout_predict_with_uncertainty_returns_named_fields(self):
        """MC dropout single-call API should return a structured prediction object."""
        from src.evaluation.uncertainty_analysis import MCDropoutUncertainty

        batch = torch.from_numpy(self.split_data["X"][:4]).float()
        prediction = MCDropoutUncertainty(
            model=self.model,
            n_samples=5,
            device="cpu",
        ).predict_with_uncertainty(batch)

        self.assertEqual(prediction.depths_mean.shape, (4, 5))
        self.assertEqual(prediction.depths_samples.shape[0], 5)
        self.assertEqual(prediction.overflow_samples.shape[0], 5)

    def test_delta_method_predict_with_uncertainty_returns_named_fields(self):
        """Delta method single-call API should return a structured prediction object."""
        sample = torch.from_numpy(self.split_data["X"][0]).float()
        prediction = DeltaMethodUncertainty(
            model=self.model,
            device="cpu",
        ).predict_with_uncertainty(sample)

        self.assertEqual(prediction.depths_mean.shape, (5,))
        self.assertEqual(prediction.overflow_mean.shape, (1,))
        self.assertIsNone(prediction.depths_samples)
        self.assertIsNone(prediction.overflow_samples)

    def test_save_results_persists_mc_dropout_and_delta_method(self):
        """Uncertainty artifact JSON should contain both MC Dropout and Delta method blocks."""
        from experiments.evaluate_lnn_uncertainty import (
            MC_DROPOUT_SAMPLES,
            analyze_flood_uncertainty,
            save_results,
        )
        from src.evaluation.uncertainty_analysis import MCDropoutUncertainty

        loader = create_test_loader(self.config, self.split_data)
        mc_results = MCDropoutUncertainty(
            model=self.model,
            n_samples=MC_DROPOUT_SAMPLES,
            device="cpu",
        ).predict_batch_with_uncertainty(loader)
        delta_results = DeltaMethodUncertainty(
            model=self.model,
            device="cpu",
        ).predict_batch_with_uncertainty(loader)

        mc_metrics = compute_all_uncertainty_metrics(mc_results)
        delta_metrics = compute_all_uncertainty_metrics(delta_results)
        mc_flood_analysis = analyze_flood_uncertainty(mc_results)
        delta_flood_analysis = analyze_flood_uncertainty(delta_results)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_results(
                mc_results,
                delta_results,
                mc_metrics,
                delta_metrics,
                mc_flood_analysis,
                delta_flood_analysis,
                self.config,
                "config.yaml",
                "checkpoint.pt",
                self.split_data,
                True,
                "test",
                results_dir=tmpdir,
            )

            metrics_files = sorted(Path(tmpdir).glob("lnn_uncertainty_metrics_*.json"))
            self.assertEqual(len(metrics_files), 1)
            payload = json.loads(metrics_files[0].read_text())
            self.assertIn("mc_dropout", payload)
            self.assertIn("delta_method", payload)
            self.assertEqual(payload["mc_dropout"]["samples"], MC_DROPOUT_SAMPLES)
            self.assertAlmostEqual(
                payload["delta_method"]["relative_input_error"],
                0.05,
            )


@unittest.skipUnless(HAS_TORCH, _SKIP_REASON)
class TestTrainingEntrypointSmoke(unittest.TestCase):
    """Verify that train_model runs for 1 epoch on synthetic data without error."""

    def test_lnn_training_smoke(self):
        """LNN training should complete 1 epoch on synthetic data."""
        config = _make_lnn_config()
        config["training"]["epochs"] = 1
        train_data = _make_synthetic_split(n_samples=16)
        val_data = _make_synthetic_split(n_samples=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "best_model.pt"
            model, summary = train_model(
                "lnn",
                config,
                device=torch.device("cpu"),
                train_data=train_data,
                val_data=val_data,
                checkpoint_path=checkpoint_path,
            )
            self.assertIsInstance(model, LNNRegression)
            self.assertEqual(summary["epochs_ran"], 1)
            self.assertIn("best_val_loss", summary)
            self.assertTrue(checkpoint_path.exists())

    def test_baseline_gru_training_smoke(self):
        """GRU training should complete 1 epoch on synthetic data."""
        config = _make_baseline_config("gru")
        config["training"]["epochs"] = 1
        train_data = _make_synthetic_split(n_samples=16)
        val_data = _make_synthetic_split(n_samples=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "best_model.pt"
            model, summary = train_model(
                "gru",
                config,
                device=torch.device("cpu"),
                train_data=train_data,
                val_data=val_data,
                checkpoint_path=checkpoint_path,
            )
            self.assertEqual(summary["epochs_ran"], 1)
            self.assertTrue(checkpoint_path.exists())


if __name__ == "__main__":
    unittest.main()
