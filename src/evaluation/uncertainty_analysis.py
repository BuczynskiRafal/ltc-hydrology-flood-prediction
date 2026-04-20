from dataclasses import dataclass

import numpy as np
import torch

from src.project_config import CI_Z_SCORE


@dataclass(frozen=True)
class UncertaintyPrediction:
    depths_mean: np.ndarray
    depths_std: np.ndarray
    depths_ci_lower: np.ndarray
    depths_ci_upper: np.ndarray
    overflow_mean: np.ndarray
    overflow_std: np.ndarray
    overflow_ci_lower: np.ndarray
    overflow_ci_upper: np.ndarray
    depths_samples: np.ndarray | None = None
    overflow_samples: np.ndarray | None = None


def _normal_confidence_interval(mean_array, std_array, z_score=CI_Z_SCORE):
    return mean_array - (z_score * std_array), mean_array + (z_score * std_array)


class MCDropoutUncertainty:
    def __init__(self, model, n_samples=100, device="cpu"):
        self.model = model
        self.n_samples = n_samples
        self.device = device

    def enable_dropout(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    def predict_with_uncertainty(self, X):
        self.model.eval()
        self.enable_dropout()

        all_depths = []
        all_overflow = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                depths, overflow, intensity = self.model(X)
                all_depths.append(depths.cpu().numpy())
                all_overflow.append(overflow.cpu().numpy())

        all_depths = np.stack(all_depths, axis=0)
        all_overflow = np.stack(all_overflow, axis=0)

        return UncertaintyPrediction(
            depths_mean=np.mean(all_depths, axis=0),
            depths_std=np.std(all_depths, axis=0),
            depths_ci_lower=np.percentile(all_depths, 2.5, axis=0),
            depths_ci_upper=np.percentile(all_depths, 97.5, axis=0),
            overflow_mean=np.mean(all_overflow, axis=0),
            overflow_std=np.std(all_overflow, axis=0),
            overflow_ci_lower=np.percentile(all_overflow, 2.5, axis=0),
            overflow_ci_upper=np.percentile(all_overflow, 97.5, axis=0),
            depths_samples=all_depths,
            overflow_samples=all_overflow,
        )

    def predict_batch_with_uncertainty(self, dataloader):
        all_depths_mean = []
        all_depths_std = []
        all_depths_ci_lower = []
        all_depths_ci_upper = []
        all_overflow_mean = []
        all_overflow_std = []
        all_overflow_ci_lower = []
        all_overflow_ci_upper = []
        all_depths_true = []
        all_overflow_true = []
        all_depths_samples = []
        all_overflow_samples = []

        for batch in dataloader:
            X = batch["X"].to(self.device)
            y_depths = batch["y_depths"].cpu().numpy()
            y_overflow = batch["y_overflow"].cpu().numpy()

            prediction = self.predict_with_uncertainty(X)

            all_depths_mean.append(prediction.depths_mean)
            all_depths_std.append(prediction.depths_std)
            all_depths_ci_lower.append(prediction.depths_ci_lower)
            all_depths_ci_upper.append(prediction.depths_ci_upper)
            all_overflow_mean.append(prediction.overflow_mean)
            all_overflow_std.append(prediction.overflow_std)
            all_overflow_ci_lower.append(prediction.overflow_ci_lower)
            all_overflow_ci_upper.append(prediction.overflow_ci_upper)
            all_depths_true.append(y_depths)
            all_overflow_true.append(y_overflow)
            if prediction.depths_samples is not None:
                all_depths_samples.append(prediction.depths_samples)
            if prediction.overflow_samples is not None:
                all_overflow_samples.append(prediction.overflow_samples)

        results = {
            "depths_mean": np.concatenate(all_depths_mean, axis=0),
            "depths_std": np.concatenate(all_depths_std, axis=0),
            "depths_ci_lower": np.concatenate(all_depths_ci_lower, axis=0),
            "depths_ci_upper": np.concatenate(all_depths_ci_upper, axis=0),
            "overflow_mean": np.concatenate(all_overflow_mean, axis=0),
            "overflow_std": np.concatenate(all_overflow_std, axis=0),
            "overflow_ci_lower": np.concatenate(all_overflow_ci_lower, axis=0),
            "overflow_ci_upper": np.concatenate(all_overflow_ci_upper, axis=0),
            "depths_true": np.concatenate(all_depths_true, axis=0),
            "overflow_true": np.concatenate(all_overflow_true, axis=0),
        }
        if all_depths_samples:
            results["depths_samples"] = np.concatenate(all_depths_samples, axis=1)
        if all_overflow_samples:
            results["overflow_samples"] = np.concatenate(all_overflow_samples, axis=1)
        return results


class DeltaMethodUncertainty:
    def __init__(
        self,
        model,
        relative_input_error=0.05,
        eps=1e-6,
        device="cpu",
        z_score=CI_Z_SCORE,
    ):
        self.model = model
        self.relative_input_error = float(relative_input_error)
        self.eps = float(eps)
        self.device = device
        self.z_score = float(z_score)

    def predict_with_uncertainty(self, x):
        self.model.eval()
        x = x.to(self.device).detach().clone().requires_grad_(True)

        depths, overflow, _ = self.model(x.unsqueeze(0))
        depths = depths.squeeze(0)
        overflow = overflow.squeeze(0)
        combined_outputs = torch.cat([depths, overflow], dim=0)

        sigma_x = self.relative_input_error * torch.maximum(
            x.detach().abs(),
            torch.full_like(x, self.eps),
        )
        flat_input_variance = sigma_x.pow(2).reshape(-1)

        jacobian_rows = []
        for output_index in range(combined_outputs.shape[0]):
            gradient = torch.autograd.grad(
                combined_outputs[output_index],
                x,
                retain_graph=output_index < combined_outputs.shape[0] - 1,
            )[0]
            jacobian_rows.append(gradient.reshape(-1))

        jacobian = torch.stack(jacobian_rows, dim=0)
        covariance = (jacobian * flat_input_variance.unsqueeze(0)) @ jacobian.transpose(
            0, 1
        )
        output_std = torch.sqrt(torch.clamp(torch.diagonal(covariance), min=0.0))

        depths_mean = depths.detach().cpu().numpy()
        depths_std = output_std[: depths_mean.shape[0]].detach().cpu().numpy()
        overflow_mean = overflow.detach().cpu().numpy().reshape(1)
        overflow_std = output_std[depths_mean.shape[0] :].detach().cpu().numpy()

        depths_ci_lower, depths_ci_upper = _normal_confidence_interval(
            depths_mean,
            depths_std,
            z_score=self.z_score,
        )
        overflow_ci_lower, overflow_ci_upper = _normal_confidence_interval(
            overflow_mean,
            overflow_std,
            z_score=self.z_score,
        )
        return UncertaintyPrediction(
            depths_mean=depths_mean,
            depths_std=depths_std,
            depths_ci_lower=depths_ci_lower,
            depths_ci_upper=depths_ci_upper,
            overflow_mean=overflow_mean,
            overflow_std=overflow_std,
            overflow_ci_lower=overflow_ci_lower,
            overflow_ci_upper=overflow_ci_upper,
        )

    def predict_batch_with_uncertainty(self, dataloader):
        all_depths_mean = []
        all_depths_std = []
        all_depths_ci_lower = []
        all_depths_ci_upper = []
        all_overflow_mean = []
        all_overflow_std = []
        all_overflow_ci_lower = []
        all_overflow_ci_upper = []
        all_depths_true = []
        all_overflow_true = []

        for batch in dataloader:
            X = batch["X"].to(self.device)
            y_depths = batch["y_depths"].cpu().numpy()
            y_overflow = batch["y_overflow"].cpu().numpy()

            batch_depths_mean = []
            batch_depths_std = []
            batch_overflow_mean = []
            batch_overflow_std = []
            batch_depths_ci_lower = []
            batch_depths_ci_upper = []
            batch_overflow_ci_lower = []
            batch_overflow_ci_upper = []

            for sample in X:
                prediction = self.predict_with_uncertainty(sample)
                batch_depths_mean.append(prediction.depths_mean)
                batch_depths_std.append(prediction.depths_std)
                batch_overflow_mean.append(prediction.overflow_mean)
                batch_overflow_std.append(prediction.overflow_std)
                batch_depths_ci_lower.append(prediction.depths_ci_lower)
                batch_depths_ci_upper.append(prediction.depths_ci_upper)
                batch_overflow_ci_lower.append(prediction.overflow_ci_lower)
                batch_overflow_ci_upper.append(prediction.overflow_ci_upper)

            all_depths_mean.append(np.stack(batch_depths_mean, axis=0))
            all_depths_std.append(np.stack(batch_depths_std, axis=0))
            all_overflow_mean.append(np.stack(batch_overflow_mean, axis=0))
            all_overflow_std.append(np.stack(batch_overflow_std, axis=0))
            all_depths_ci_lower.append(np.stack(batch_depths_ci_lower, axis=0))
            all_depths_ci_upper.append(np.stack(batch_depths_ci_upper, axis=0))
            all_overflow_ci_lower.append(np.stack(batch_overflow_ci_lower, axis=0))
            all_overflow_ci_upper.append(np.stack(batch_overflow_ci_upper, axis=0))
            all_depths_true.append(y_depths)
            all_overflow_true.append(y_overflow)

        return {
            "depths_mean": np.concatenate(all_depths_mean, axis=0),
            "depths_std": np.concatenate(all_depths_std, axis=0),
            "depths_ci_lower": np.concatenate(all_depths_ci_lower, axis=0),
            "depths_ci_upper": np.concatenate(all_depths_ci_upper, axis=0),
            "overflow_mean": np.concatenate(all_overflow_mean, axis=0),
            "overflow_std": np.concatenate(all_overflow_std, axis=0),
            "overflow_ci_lower": np.concatenate(all_overflow_ci_lower, axis=0),
            "overflow_ci_upper": np.concatenate(all_overflow_ci_upper, axis=0),
            "depths_true": np.concatenate(all_depths_true, axis=0),
            "overflow_true": np.concatenate(all_overflow_true, axis=0),
        }
