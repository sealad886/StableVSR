"""Optimization presets and safety guardrails for MLX inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PRESET_NAMES = ("max-quality", "safe", "balanced", "fast")


@dataclass(frozen=True)
class InferencePreset:
    """Immutable configuration for an inference preset."""

    name: str
    compile_models: bool
    ttg_start_step_fraction: float
    chunk_size: int | None
    chunk_overlap: int
    force_tiled_vae: bool | None  # None = auto

    def resolve_ttg_start_step(self, num_inference_steps: int) -> int:
        return int(self.ttg_start_step_fraction * num_inference_steps)


PRESETS: dict[str, InferencePreset] = {
    "max-quality": InferencePreset(
        name="max-quality",
        compile_models=False,
        ttg_start_step_fraction=0.0,
        chunk_size=None,
        chunk_overlap=0,
        force_tiled_vae=None,
    ),
    "safe": InferencePreset(
        name="safe",
        compile_models=True,
        ttg_start_step_fraction=0.0,
        chunk_size=16,
        chunk_overlap=4,
        force_tiled_vae=None,
    ),
    "balanced": InferencePreset(
        name="balanced",
        compile_models=True,
        ttg_start_step_fraction=0.25,
        chunk_size=12,
        chunk_overlap=3,
        force_tiled_vae=None,
    ),
    "fast": InferencePreset(
        name="fast",
        compile_models=True,
        ttg_start_step_fraction=0.5,
        chunk_size=8,
        chunk_overlap=2,
        force_tiled_vae=None,
    ),
}


def get_preset(name: str) -> InferencePreset:
    """Return a preset by name, raising ValueError for unknown names."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {', '.join(PRESETS)}")
    return PRESETS[name]


@dataclass
class GuardrailWarning:
    """A single guardrail warning."""

    code: str
    message: str
    severity: str  # "warn" or "error"


def check_guardrails(
    *,
    num_frames: int,
    height: int,
    width: int,
    num_inference_steps: int,
    ttg_start_step: int,
    chunk_size: int | None,
    chunk_overlap: int,
    compile_models: bool,
    force_tiled_vae: bool | None,
) -> list[GuardrailWarning]:
    """Check inference parameters for risky or unsupported combinations.

    Returns a list of warnings. Empty list means all checks passed.
    All warnings are advisory — the caller decides whether to abort.
    """
    warnings: list[GuardrailWarning] = []

    # TTG aggressiveness
    if num_inference_steps > 0 and ttg_start_step > 0:
        ttg_frac = ttg_start_step / num_inference_steps
        if ttg_frac > 0.5:
            warnings.append(
                GuardrailWarning(
                    code="TTG_AGGRESSIVE",
                    message=(
                        f"ttg_start_step={ttg_start_step} skips {ttg_frac:.0%} of steps. "
                        f"Temporal consistency may degrade significantly. "
                        f"Recommended: ttg_start_step <= {num_inference_steps // 2}"
                    ),
                    severity="warn",
                )
            )
        if ttg_start_step >= num_inference_steps:
            warnings.append(
                GuardrailWarning(
                    code="TTG_DISABLED",
                    message=(
                        f"ttg_start_step={ttg_start_step} >= num_steps={num_inference_steps}. "
                        f"Temporal texture guidance is effectively disabled."
                    ),
                    severity="warn",
                )
            )

    # Memory estimate for frames
    upscale_factor = 4
    out_h, out_w = height * upscale_factor, width * upscale_factor
    bytes_per_frame_f16 = out_h * out_w * 3 * 2
    latent_bytes = (out_h // 4) * (out_w // 4) * 4 * 2
    effective_frames = num_frames if chunk_size is None else min(num_frames, chunk_size)
    estimated_mb = (
        effective_frames * (bytes_per_frame_f16 + latent_bytes) / (1024 * 1024)
    )

    if estimated_mb > 8000:
        warnings.append(
            GuardrailWarning(
                code="MEMORY_HIGH",
                message=(
                    f"Estimated working set ~{estimated_mb:.0f} MB for {effective_frames} "
                    f"frames at {out_h}x{out_w}. Consider using chunked inference "
                    f"(--chunk-size) or reducing resolution."
                ),
                severity="warn",
            )
        )

    if num_frames > 50 and chunk_size is None:
        warnings.append(
            GuardrailWarning(
                code="LONG_VIDEO_NO_CHUNK",
                message=(
                    f"{num_frames} frames without chunking. Memory usage grows linearly "
                    f"with frame count. Consider --chunk-size 16 --chunk-overlap 4."
                ),
                severity="warn",
            )
        )

    # Chunk validation
    if chunk_size is not None:
        if chunk_size < 2:
            warnings.append(
                GuardrailWarning(
                    code="CHUNK_TOO_SMALL",
                    message="chunk_size must be >= 2 for meaningful inference.",
                    severity="error",
                )
            )
        if chunk_overlap >= chunk_size:
            warnings.append(
                GuardrailWarning(
                    code="OVERLAP_GE_CHUNK",
                    message=(
                        f"chunk_overlap={chunk_overlap} >= chunk_size={chunk_size}. "
                        f"No forward progress would be made."
                    ),
                    severity="error",
                )
            )
        if chunk_overlap < 0:
            warnings.append(
                GuardrailWarning(
                    code="OVERLAP_NEGATIVE",
                    message="chunk_overlap cannot be negative.",
                    severity="error",
                )
            )

    # Compile + single step
    if compile_models and num_inference_steps == 1:
        warnings.append(
            GuardrailWarning(
                code="COMPILE_ONE_STEP",
                message=(
                    "compile_models=True with 1 step: JIT compilation overhead "
                    "may exceed the speedup benefit."
                ),
                severity="warn",
            )
        )

    # Tiling forced off on large resolution
    if force_tiled_vae is False and out_h * out_w > 1920 * 1080:
        warnings.append(
            GuardrailWarning(
                code="NO_TILING_LARGE",
                message=(
                    f"Tiled VAE disabled for {out_h}x{out_w} output. "
                    f"This may cause out-of-memory. Consider auto or forced tiling."
                ),
                severity="warn",
            )
        )

    return warnings


def log_guardrails(warnings: list[GuardrailWarning]) -> bool:
    """Log all guardrail warnings. Returns True if any errors were found."""
    has_errors = False
    for w in warnings:
        if w.severity == "error":
            logger.error("[GUARDRAIL %s] %s", w.code, w.message)
            has_errors = True
        else:
            logger.warning("[GUARDRAIL %s] %s", w.code, w.message)
    return has_errors
