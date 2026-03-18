"""Tests for CLI enhancements: dtype, smoke-test, platform detection, backend logging."""

import argparse
import logging

import pytest

from stablevsr.cli import (
    DTYPE_RESTRICTIONS,
    VALID_DTYPES,
    _detect_platform_info,
    _resolve_dtype,
    build_parser,
)


class TestDtypeResolution:
    """Test _resolve_dtype with device restrictions."""

    def test_default_is_float32(self):
        result = _resolve_dtype(None, False, "cpu")
        assert result == "float32"

    def test_fp16_flag_sets_float16(self):
        result = _resolve_dtype(None, True, "cuda")
        assert result == "float16"

    def test_explicit_dtype_overrides_fp16(self, capsys):
        result = _resolve_dtype("float32", True, "cuda")
        assert result == "float32"
        captured = capsys.readouterr().out
        assert "--dtype" in captured

    def test_cpu_rejects_float16(self, capsys):
        result = _resolve_dtype("float16", False, "cpu")
        assert result == "float32"
        captured = capsys.readouterr().out
        assert "not supported" in captured

    def test_cpu_rejects_bfloat16(self, capsys):
        result = _resolve_dtype("bfloat16", False, "cpu")
        assert result == "float32"

    def test_cuda_accepts_bfloat16(self):
        result = _resolve_dtype("bfloat16", False, "cuda")
        assert result == "bfloat16"

    def test_mps_accepts_float16(self):
        result = _resolve_dtype("float16", False, "mps")
        assert result == "float16"

    def test_mps_rejects_bfloat16(self, capsys):
        result = _resolve_dtype("bfloat16", False, "mps")
        assert result == "float32"
        captured = capsys.readouterr().out
        assert "not supported" in captured

    def test_unknown_device_allows_all(self):
        result = _resolve_dtype("bfloat16", False, "xpu")
        assert result == "bfloat16"


class TestDtypeConstants:
    def test_valid_dtypes_complete(self):
        assert VALID_DTYPES == {"float32", "float16", "bfloat16"}

    def test_cpu_restriction_is_float32_only(self):
        assert DTYPE_RESTRICTIONS["cpu"] == {"float32"}

    def test_mps_excludes_bfloat16(self):
        assert "bfloat16" not in DTYPE_RESTRICTIONS["mps"]

    def test_cuda_allows_all(self):
        assert DTYPE_RESTRICTIONS["cuda"] == VALID_DTYPES


class TestParserDtypeAndSmokeTest:
    def test_dtype_flag_parsed(self):
        parser = build_parser()
        args = parser.parse_args(
            ["infer", "--input", "/a", "--output", "/b", "--dtype", "float16"]
        )
        assert args.dtype == "float16"

    def test_smoke_test_flag_parsed(self):
        parser = build_parser()
        args = parser.parse_args(
            ["infer", "--input", "/a", "--output", "/b", "--smoke-test"]
        )
        assert args.smoke_test is True

    def test_verbose_flag_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["infer", "--input", "/a", "--output", "/b", "-v"])
        assert args.verbose is True

    def test_dtype_rejects_invalid(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["infer", "--input", "/a", "--output", "/b", "--dtype", "int8"]
            )


class TestPlatformDetection:
    def test_returns_dict_with_required_keys(self):
        info = _detect_platform_info()
        assert "os" in info
        assert "arch" in info
        assert "python" in info
        assert "pointer_size" in info

    def test_os_is_nonempty(self):
        info = _detect_platform_info()
        assert len(info["os"]) > 0

    def test_pointer_size_is_valid(self):
        info = _detect_platform_info()
        assert info["pointer_size"] in ("32", "64")


class TestDoctorPlatformInfo:
    """Doctor now reports platform info."""

    def test_doctor_shows_platform(self, capsys):
        args = argparse.Namespace()
        from stablevsr.cli import cmd_doctor

        try:
            cmd_doctor(args)
        except SystemExit:
            pass
        captured = capsys.readouterr().out
        assert "[INFO] Platform:" in captured
        assert "[INFO] Python:" in captured


class TestBackendLogging:
    """Backend registry logs selection reasons."""

    def test_auto_detect_logs_reason(self, caplog):
        from stablevsr.backends import get_backend

        with caplog.at_level(logging.INFO, logger="stablevsr.backends.registry"):
            get_backend()
        assert any("Backend selected" in r.message for r in caplog.records)
        assert any("reason:" in r.message for r in caplog.records)

    def test_explicit_backend_logs_reason(self, caplog):
        from stablevsr.backends import get_backend

        with caplog.at_level(logging.INFO, logger="stablevsr.backends.registry"):
            get_backend("torch-cpu")
        assert any("explicit request" in r.message for r in caplog.records)

    def test_auto_detect_logs_rejected(self, caplog):
        from stablevsr.backends import get_backend

        with caplog.at_level(logging.INFO, logger="stablevsr.backends.registry"):
            get_backend()
        log_text = " ".join(r.message for r in caplog.records)
        assert "rejected" in log_text or "MLX" in log_text or "mlx" in log_text
