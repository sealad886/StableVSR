"""Tests for inference presets and guardrails."""

import pytest

from stablevsr.mlx.presets import (
    PRESETS,
    GuardrailWarning,
    check_guardrails,
    get_preset,
    log_guardrails,
)


class TestPresetRegistry:
    def test_all_presets_exist(self):
        assert "max-quality" in PRESETS
        assert "safe" in PRESETS
        assert "balanced" in PRESETS
        assert "fast" in PRESETS

    def test_get_preset_valid(self):
        p = get_preset("safe")
        assert p.name == "safe"
        assert p.compile_models is True

    def test_get_preset_invalid(self):
        with pytest.raises(ValueError, match="unknown_preset"):
            get_preset("unknown_preset")

    def test_each_preset_has_name(self):
        for name, preset in PRESETS.items():
            assert preset.name == name

    def test_presets_are_frozen(self):
        p = get_preset("safe")
        with pytest.raises(AttributeError):
            p.name = "modified"


class TestResolveTTG:
    def test_max_quality_fraction_zero(self):
        p = get_preset("max-quality")
        assert p.resolve_ttg_start_step(50) == 0

    def test_balanced_dynamic_ttg(self):
        p = get_preset("balanced")
        assert p.resolve_ttg_start_step(40) == 10  # 0.25 * 40

    def test_fast_dynamic_ttg(self):
        p = get_preset("fast")
        assert p.resolve_ttg_start_step(50) == 25  # 0.5 * 50

    def test_safe_ttg_is_zero(self):
        p = get_preset("safe")
        assert p.resolve_ttg_start_step(50) == 0


class TestGuardrails:
    def _base_kwargs(self, **overrides):
        defaults = dict(
            num_frames=5,
            height=270,
            width=480,
            num_inference_steps=50,
            ttg_start_step=0,
            chunk_size=None,
            chunk_overlap=0,
            compile_models=True,
            force_tiled_vae=None,
        )
        defaults.update(overrides)
        return defaults

    def test_no_warnings_for_safe_settings(self):
        warnings = check_guardrails(**self._base_kwargs())
        assert len(warnings) == 0

    def test_aggressive_ttg_warns(self):
        warnings = check_guardrails(**self._base_kwargs(ttg_start_step=30))
        assert any(w.code == "TTG_AGGRESSIVE" for w in warnings)

    def test_ttg_disabled_warns(self):
        warnings = check_guardrails(**self._base_kwargs(ttg_start_step=50))
        assert any(w.code == "TTG_DISABLED" for w in warnings)

    def test_long_video_no_chunk_warns(self):
        warnings = check_guardrails(**self._base_kwargs(num_frames=100))
        assert any(w.code == "LONG_VIDEO_NO_CHUNK" for w in warnings)

    def test_overlap_ge_chunk_errors(self):
        warnings = check_guardrails(**self._base_kwargs(chunk_size=5, chunk_overlap=5))
        assert any(w.code == "OVERLAP_GE_CHUNK" for w in warnings)
        assert any(w.severity == "error" for w in warnings)

    def test_chunk_too_small_errors(self):
        warnings = check_guardrails(**self._base_kwargs(chunk_size=1, chunk_overlap=0))
        assert any(w.code == "CHUNK_TOO_SMALL" for w in warnings)

    def test_compile_one_step_warns(self):
        warnings = check_guardrails(**self._base_kwargs(num_inference_steps=1))
        assert any(w.code == "COMPILE_ONE_STEP" for w in warnings)

    def test_no_tiling_large_warns(self):
        warnings = check_guardrails(
            **self._base_kwargs(
                height=540,
                width=960,
                force_tiled_vae=False,
            )
        )
        assert any(w.code == "NO_TILING_LARGE" for w in warnings)

    def test_high_memory_warns(self):
        """100 frames at 4K without chunking should trigger memory warning."""
        warnings = check_guardrails(
            **self._base_kwargs(
                num_frames=100,
                height=2160,
                width=3840,
            )
        )
        codes = [w.code for w in warnings]
        assert "MEMORY_HIGH" in codes or "LONG_VIDEO_NO_CHUNK" in codes

    def test_memory_estimate_uses_correct_latent_divisor(self):
        """Latent divisor must be 4 (vae_scale_factor=4), not 8.

        For 480x270 input, output is 1920x1080.
        Correct latent: 480x270 (out_h//4 x out_w//4), 4 ch, 2 bytes = 1,036,800 bytes.
        Wrong (//8) would be 240x135 = 259,200 bytes (4x too small).
        With enough frames the correct formula should trigger MEMORY_HIGH.
        """
        # 200 frames at 1080p-input (4320x7680 output) should easily exceed 8 GB
        warnings = check_guardrails(
            **self._base_kwargs(
                num_frames=200,
                height=1080,
                width=1920,
                chunk_size=None,
            )
        )
        codes = [w.code for w in warnings]
        assert "MEMORY_HIGH" in codes


class TestLogGuardrails:
    def test_returns_true_on_error(self):
        warnings = [GuardrailWarning("TEST", "msg", "error")]
        assert log_guardrails(warnings) is True

    def test_returns_false_on_warnings_only(self):
        warnings = [GuardrailWarning("TEST", "msg", "warn")]
        assert log_guardrails(warnings) is False

    def test_empty_returns_false(self):
        assert log_guardrails([]) is False
