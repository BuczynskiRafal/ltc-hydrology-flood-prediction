"""Benchmark single-sample inference latency using runtime-compatible checkpoints."""

import time
import traceback

import numpy as np
import torch

from experiments.regression_pipeline import (
    get_device,
    load_split_data_for_config,
    load_trained_model,
)

DEFAULT_ITERATIONS = 1000
DEFAULT_WARMUP = 100
MODEL_NAMES = ("lstm", "gru", "mlp", "tcn", "lnn")


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_benchmark_input(runtime_config):
    test_data = load_split_data_for_config(runtime_config, "test")
    benchmark_input = torch.from_numpy(test_data["X"][:1]).float()

    if benchmark_input.shape[-1] != runtime_config["model"]["input_size"]:
        raise ValueError(
            "Configured input_size does not match test data channels: "
            f"{runtime_config['model']['input_size']} != {benchmark_input.shape[-1]}"
        )
    if (
        "seq_len" in runtime_config["model"]
        and benchmark_input.shape[1] != runtime_config["model"]["seq_len"]
    ):
        raise ValueError(
            "Configured seq_len does not match test data length: "
            f"{runtime_config['model']['seq_len']} != {benchmark_input.shape[1]}"
        )

    return benchmark_input, bool(runtime_config["data"].get("use_reduced", True))


def benchmark_model(
    model_name, num_iterations=DEFAULT_ITERATIONS, warmup=DEFAULT_WARMUP
):
    device = get_device()
    artifact = load_trained_model(model_name, device=device)
    benchmark_input, use_reduced = load_benchmark_input(artifact["runtime_config"])
    benchmark_input = benchmark_input.to(device)

    model = artifact["model"]
    checkpoint = artifact["checkpoint"]
    checkpoint_path = artifact["checkpoint_path"]

    print(
        f"[{model_name}] device={device} use_reduced={use_reduced} "
        f"input_shape={tuple(benchmark_input.shape)} checkpoint={checkpoint_path} "
        f"config_source={artifact['config_source']}"
    )

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(benchmark_input)
        synchronize_device(device)

        times = []
        for _ in range(num_iterations):
            synchronize_device(device)
            start = time.perf_counter()
            _ = model(benchmark_input)
            synchronize_device(device)
            times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    num_params = checkpoint.get("n_params")
    if num_params is None:
        num_params = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

    return {
        "model": model_name,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "config_source": artifact["config_source"],
        "use_reduced": use_reduced,
        "seq_len": int(benchmark_input.shape[1]),
        "input_size": int(benchmark_input.shape[2]),
        "batch_size": 1,
        "iterations": int(num_iterations),
        "warmup": int(warmup),
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "num_params": int(num_params),
    }


def main():
    print("=" * 80)
    print("MODEL INFERENCE LATENCY MICROBENCHMARK")
    print("=" * 80)
    print("Measures single-sample forward-pass latency on runtime-compatible inputs.")
    print(f"Iterations: {DEFAULT_ITERATIONS}")
    print(f"Warmup iterations: {DEFAULT_WARMUP}")
    print("=" * 80)

    all_results = []
    failed_models = []

    for model_name in MODEL_NAMES:
        try:
            result = benchmark_model(model_name)
            all_results.append(result)

            print(f"\n{model_name.upper()} Results:")
            print(f"  Device:   {result['device']}")
            print(
                f"  Input:    (1, {result['seq_len']}, {result['input_size']}) "
                f"use_reduced={result['use_reduced']}"
            )
            print(f"  Source:   {result['config_source']}")
            print(f"  Mean:     {result['mean_ms']:.3f} ms")
            print(f"  Std:      {result['std_ms']:.3f} ms")
            print(f"  Median:   {result['median_ms']:.3f} ms")
            print(f"  Min:      {result['min_ms']:.3f} ms")
            print(f"  Max:      {result['max_ms']:.3f} ms")
            print(f"  P95:      {result['p95_ms']:.3f} ms")
            print(f"  P99:      {result['p99_ms']:.3f} ms")
            print(f"  Params:   {result['num_params']:,}")
            print("-" * 80)
        except Exception as exc:
            failed_models.append((model_name, str(exc)))
            print(f"Error benchmarking {model_name}: {exc}")
            traceback.print_exc()

    if all_results:
        all_results.sort(key=lambda item: item["mean_ms"])
        print("\n" + "=" * 80)
        print("SUMMARY (sorted by mean inference time)")
        print("=" * 80)
        print(
            f"{'Model':<8} {'Mean (ms)':<12} {'Median (ms)':<12} "
            f"{'P95 (ms)':<12} {'Params':<12} {'Shape':<14} {'Source':<16}"
        )
        print("-" * 96)
        for result in all_results:
            shape = f"1x{result['seq_len']}x{result['input_size']}"
            print(
                f"{result['model']:<8} "
                f"{result['mean_ms']:<12.3f} "
                f"{result['median_ms']:<12.3f} "
                f"{result['p95_ms']:<12.3f} "
                f"{result['num_params']:<12,} "
                f"{shape:<14} "
                f"{result['config_source']:<16}"
            )

    if failed_models:
        print("\n" + "=" * 80)
        print("FAILED MODELS")
        print("=" * 80)
        for model_name, error in failed_models:
            print(f"{model_name:<8} {error}")


if __name__ == "__main__":
    main()
