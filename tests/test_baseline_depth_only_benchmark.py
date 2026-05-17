from pathlib import Path

import yaml

BASELINE_CONFIGS = {
    "gru": Path("configs/gru_regression_config.yaml"),
    "lstm": Path("configs/lstm_regression_config.yaml"),
    "tcn": Path("configs/tcn_regression_config.yaml"),
    "mlp": Path("configs/mlp_regression_config.yaml"),
}
DEPTH_ONLY_CONFIGS = {
    "gru": Path("configs/baseline_depth_only/gru_depth_only_config.yaml"),
    "lstm": Path("configs/baseline_depth_only/lstm_depth_only_config.yaml"),
    "tcn": Path("configs/baseline_depth_only/tcn_depth_only_config.yaml"),
    "mlp": Path("configs/baseline_depth_only/mlp_depth_only_config.yaml"),
}


def load_yaml(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def test_depth_only_configs_keep_baseline_architecture_and_training_protocol():
    for model_name, baseline_path in BASELINE_CONFIGS.items():
        baseline = load_yaml(baseline_path)
        depth_only = load_yaml(DEPTH_ONLY_CONFIGS[model_name])

        assert depth_only["schema_version"] == baseline["schema_version"]
        assert depth_only["runtime"] == baseline["runtime"]
        assert depth_only["model"] == baseline["model"]
        assert depth_only["training"] == baseline["training"]
        assert depth_only["data"] == baseline["data"]


def test_depth_only_configs_use_depth_only_objective():
    expected_loss = {
        "depth_weight": 1.0,
        "overflow_weight": 0.0,
        "flood_weight": 1.0,
    }
    for config_path in DEPTH_ONLY_CONFIGS.values():
        config = load_yaml(config_path)
        assert config["loss"] == expected_loss
        assert "baseline_depth_only" in config["output"]["checkpoint_dir"]
