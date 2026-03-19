"""DDPM scheduler for MLX — pure math, no framework dependency.

Scheduler step runs in float32 regardless of model precision to avoid
numerical drift compounding through 50+ timesteps (adversarial Finding 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class DDPMSchedulerOutput:
    prev_sample: mx.array
    pred_original_sample: mx.array


class MLXDDPMScheduler:
    """DDPM scheduler operating on MLX arrays."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "scaled_linear",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        prediction_type: str = "epsilon",
        variance_type: str = "fixed_small",
        thresholding: bool = False,
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
        scaling_factor: float = 0.08333,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.variance_type = variance_type
        self.thresholding = thresholding
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset
        self.scaling_factor = scaling_factor
        self.order = 1

        # Compute betas in float64 for precision
        if beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float64)
        elif beta_schedule == "scaled_linear":
            betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float64) ** 2
        else:
            raise NotImplementedError(f"{beta_schedule} not implemented")

        alphas = 1.0 - betas
        self.alphas_cumprod = mx.array(np.cumprod(alphas), dtype=mx.float32)
        self.one = mx.array(1.0, dtype=mx.float32)

        self.num_inference_steps: int | None = None
        self.timesteps: list[int] = []
        self._custom_timesteps = False

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        timesteps: list[int] | None = None,
    ) -> None:
        if timesteps is not None:
            self.timesteps = list(timesteps)
            self._custom_timesteps = True
            self.num_inference_steps = len(timesteps)
            return

        assert num_inference_steps is not None
        self.num_inference_steps = num_inference_steps

        if self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // num_inference_steps
            ts = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            ts += self.steps_offset
        elif self.timestep_spacing == "linspace":
            ts = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps).round()[::-1].copy().astype(np.int64)
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / num_inference_steps
            ts = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio)).astype(np.int64) - 1
        else:
            raise ValueError(f"Unsupported timestep_spacing: {self.timestep_spacing}")

        self.timesteps = ts.tolist()
        self._custom_timesteps = False

    def scale_model_input(self, sample: mx.array, timestep: int) -> mx.array:
        """DDPM doesn't scale model input."""
        return sample

    def step(
        self,
        model_output: mx.array,
        timestep: int,
        sample: mx.array,
        seed: int | None = None,
    ) -> DDPMSchedulerOutput:
        """One denoising step. All math in float32."""
        t = timestep
        prev_t = self._previous_timestep(t)

        # Cast to float32 for precision
        model_output_f32 = model_output.astype(mx.float32)
        sample_f32 = sample.astype(mx.float32)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1.0 - current_alpha_t

        # Predict x0
        if self.prediction_type == "epsilon":
            pred_x0 = (sample_f32 - mx.sqrt(beta_prod_t) * model_output_f32) / mx.sqrt(alpha_prod_t)
        elif self.prediction_type == "v_prediction":
            pred_x0 = mx.sqrt(alpha_prod_t) * sample_f32 - mx.sqrt(beta_prod_t) * model_output_f32
        elif self.prediction_type == "sample":
            pred_x0 = model_output_f32
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # Clip predicted x0
        if self.clip_sample:
            pred_x0 = mx.clip(pred_x0, -self.clip_sample_range, self.clip_sample_range)

        # Compute coefficients (formula 7 from DDPM paper)
        pred_x0_coeff = (mx.sqrt(alpha_prod_t_prev) * current_beta_t) / beta_prod_t
        current_sample_coeff = mx.sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t

        # Predicted previous sample
        pred_prev = pred_x0_coeff * pred_x0 + current_sample_coeff * sample_f32

        # Add noise for t > 0
        if t > 0:
            variance = (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * current_beta_t
            variance = mx.maximum(variance, mx.array(1e-20))

            if seed is not None:
                key = mx.random.key(seed + t)
            else:
                key = None
            noise = mx.random.normal(model_output.shape, key=key)
            pred_prev = pred_prev + mx.sqrt(variance) * noise.astype(mx.float32)

        # Cast back to model dtype
        return DDPMSchedulerOutput(
            prev_sample=pred_prev.astype(model_output.dtype),
            pred_original_sample=pred_x0.astype(model_output.dtype),
        )

    def _previous_timestep(self, timestep: int) -> int:
        if self._custom_timesteps:
            try:
                idx = self.timesteps.index(timestep)
            except ValueError:
                return -1
            if idx == len(self.timesteps) - 1:
                return -1
            return self.timesteps[idx + 1]
        num_steps = self.num_inference_steps or self.num_train_timesteps
        return timestep - self.num_train_timesteps // num_steps
