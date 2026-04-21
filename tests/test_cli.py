"""Tests for the unified experiment CLI (experiments/cli.py).

Validates argument parsing, handler dispatch, grid-search profile lookup,
error handling, and README consistency — without heavy ML imports.
"""

from __future__ import annotations

import sys
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cli import GRID_PROFILES, main


class TestTrainDispatch(unittest.TestCase):
    @patch("experiments.cli.handle_train")
    def test_dispatches_lnn_with_defaults(self, mock_handle):
        main(["train", "lnn"])
        args = mock_handle.call_args[0][0]
        self.assertEqual(args.model, "lnn")
        self.assertIsNone(args.config)

    @patch("experiments.cli.handle_train")
    def test_passes_custom_config(self, mock_handle):
        main(["train", "gru", "--config", "custom.yaml"])
        args = mock_handle.call_args[0][0]
        self.assertEqual(args.model, "gru")
        self.assertEqual(args.config, "custom.yaml")

    def test_rejects_unknown_model(self):
        with self.assertRaises(SystemExit):
            main(["train", "transformer"])


class TestEvaluateDispatch(unittest.TestCase):
    @patch("experiments.cli.handle_evaluate")
    def test_dispatches_mlp(self, mock_handle):
        main(["evaluate", "mlp"])
        args = mock_handle.call_args[0][0]
        self.assertEqual(args.model, "mlp")

    @patch("experiments.cli.handle_evaluate")
    def test_passes_custom_config(self, mock_handle):
        main(["evaluate", "lnn", "--config", "my.yaml"])
        args = mock_handle.call_args[0][0]
        self.assertEqual(args.config, "my.yaml")


class TestEvaluateAllDispatch(unittest.TestCase):
    @patch("experiments.cli.handle_evaluate_all")
    def test_dispatches_with_defaults(self, mock_handle):
        main(["evaluate-all"])
        args = mock_handle.call_args[0][0]
        self.assertEqual(args.split, "test")
        self.assertEqual(args.results_dir, "artifacts/results/release")

    @patch("experiments.cli.handle_evaluate_all")
    def test_overrides_split_and_dir(self, mock_handle):
        main(["evaluate-all", "--split", "val", "--results-dir", "/tmp/out"])
        args = mock_handle.call_args[0][0]
        self.assertEqual(args.split, "val")
        self.assertEqual(args.results_dir, "/tmp/out")


class TestGridSearchDispatch(unittest.TestCase):
    @patch("experiments.cli.handle_grid_search")
    def test_dispatches_gru_default(self, mock_handle):
        main(["grid-search", "gru"])
        args = mock_handle.call_args[0][0]
        self.assertEqual(args.model, "gru")
        self.assertEqual(args.profile, "default")

    @patch("experiments.cli.handle_grid_search")
    def test_dispatches_gru_full(self, mock_handle):
        main(["grid-search", "gru", "--profile", "full"])
        args = mock_handle.call_args[0][0]
        self.assertEqual(args.profile, "full")

    @patch("experiments.cli.handle_grid_search")
    def test_max_epochs_override(self, mock_handle):
        main(["grid-search", "tcn", "--max-epochs", "5"])
        args = mock_handle.call_args[0][0]
        self.assertEqual(args.max_epochs, 5)

    def test_no_profile_for_lnn_exits(self):
        """LNN has no grid profiles — handler should call parser.error."""
        with self.assertRaises(SystemExit) as ctx:
            main(["grid-search", "lnn"])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_unknown_profile_exits(self):
        """Unknown profile name should cause parser.error."""
        with self.assertRaises(SystemExit) as ctx:
            main(["grid-search", "lstm", "--profile", "full"])
        self.assertNotEqual(ctx.exception.code, 0)


class TestEnsembleDispatch(unittest.TestCase):
    @patch("experiments.cli.handle_ensemble_train")
    def test_dispatches_ensemble_train_defaults(self, mock_handle):
        main(["ensemble-train", "lnn"])
        args = mock_handle.call_args[0][0]
        self.assertEqual(args.model, "lnn")
        self.assertEqual(args.seeds, [42, 43, 44, 45, 46])
        self.assertEqual(args.checkpoint_root, "artifacts/checkpoints/ensemble")

    @patch("experiments.cli.handle_ensemble_evaluate")
    def test_dispatches_ensemble_evaluate_overrides(self, mock_handle):
        main(
            [
                "ensemble-evaluate",
                "lnn",
                "--seeds",
                "7",
                "8",
                "--checkpoint-root",
                "/tmp/ckpts",
                "--results-dir",
                "/tmp/results",
            ]
        )
        args = mock_handle.call_args[0][0]
        self.assertEqual(args.model, "lnn")
        self.assertEqual(args.seeds, [7, 8])
        self.assertEqual(args.checkpoint_root, "/tmp/ckpts")
        self.assertEqual(args.results_dir, "/tmp/results")


class TestGridProfiles(unittest.TestCase):
    def test_every_profile_has_required_keys(self):
        for model, profiles in GRID_PROFILES.items():
            for name, profile in profiles.items():
                with self.subTest(model=model, profile=name):
                    self.assertIn("grid", profile)
                    self.assertIn("max_epochs", profile)
                    self.assertIn("header", profile)
                    self.assertIsInstance(profile["grid"], dict)
                    self.assertIsInstance(profile["max_epochs"], int)

    def test_gru_has_default_and_full(self):
        self.assertCountEqual(GRID_PROFILES["gru"].keys(), ["default", "full"])

    def test_full_gru_grid_has_more_combinations(self):
        default = GRID_PROFILES["gru"]["default"]["grid"]
        full = GRID_PROFILES["gru"]["full"]["grid"]
        default_n = len(list(product(*default.values())))
        full_n = len(list(product(*full.values())))
        self.assertGreater(full_n, default_n)


class TestParserHelp(unittest.TestCase):
    def _assert_help_exits_zero(self, argv: list[str]):
        with self.assertRaises(SystemExit) as ctx:
            main(argv)
        self.assertEqual(ctx.exception.code, 0)

    def test_top_level(self):
        self._assert_help_exits_zero(["--help"])

    def test_train(self):
        self._assert_help_exits_zero(["train", "--help"])

    def test_evaluate(self):
        self._assert_help_exits_zero(["evaluate", "--help"])

    def test_evaluate_all(self):
        self._assert_help_exits_zero(["evaluate-all", "--help"])

    def test_grid_search(self):
        self._assert_help_exits_zero(["grid-search", "--help"])

    def test_ensemble_train(self):
        self._assert_help_exits_zero(["ensemble-train", "--help"])

    def test_ensemble_evaluate(self):
        self._assert_help_exits_zero(["ensemble-evaluate", "--help"])


_DELETED_WRAPPERS = [
    "train_gru.py",
    "train_lstm.py",
    "train_tcn.py",
    "train_mlp.py",
    "train_lnn.py",
    "evaluate_gru.py",
    "evaluate_lstm.py",
    "evaluate_tcn.py",
    "evaluate_mlp.py",
    "evaluate_lnn.py",
    "grid_search_gru.py",
    "grid_search_lstm.py",
    "grid_search_mlp.py",
    "grid_search_tcn.py",
    "evaluate_all_models.py",
]


class TestReadmeConsistency(unittest.TestCase):
    """Verify README does not reference deleted wrapper files."""

    def test_no_stale_wrapper_references(self):
        content = (PROJECT_ROOT / "README.md").read_text()
        for wrapper in _DELETED_WRAPPERS:
            self.assertNotIn(
                wrapper,
                content,
                f"README.md still references deleted wrapper: {wrapper}",
            )


if __name__ == "__main__":
    unittest.main()
