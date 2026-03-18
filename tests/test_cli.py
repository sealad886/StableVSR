"""Tests for the StableVSR CLI module."""

import argparse

import pytest

from stablevsr.cli import build_parser, cmd_backend_info, cmd_doctor, main


class TestBuildParser:
    def test_returns_argument_parser(self):
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parse_backend_info(self):
        parser = build_parser()
        args = parser.parse_args(["backend-info"])
        assert args.command == "backend-info"

    def test_parse_doctor(self):
        parser = build_parser()
        args = parser.parse_args(["doctor"])
        assert args.command == "doctor"

    def test_parse_infer_sets_args(self):
        parser = build_parser()
        args = parser.parse_args(["infer", "--input", "/tmp/a", "--output", "/tmp/b"])
        assert args.command == "infer"
        assert args.input == "/tmp/a"
        assert args.output == "/tmp/b"


class TestCmdBackendInfo:
    def test_prints_backend_status(self, capsys):
        args = argparse.Namespace()
        cmd_backend_info(args)
        captured = capsys.readouterr().out
        assert "AVAILABLE" in captured or "NOT AVAILABLE" in captured

    def test_prints_device_name(self, capsys):
        args = argparse.Namespace()
        cmd_backend_info(args)
        captured = capsys.readouterr().out
        assert "Device:" in captured

    def test_prints_inference_and_training(self, capsys):
        args = argparse.Namespace()
        cmd_backend_info(args)
        captured = capsys.readouterr().out
        assert "Inference:" in captured
        assert "Training:" in captured


class TestCmdDoctor:
    def test_reports_pytorch_ok(self, capsys):
        args = argparse.Namespace()
        try:
            cmd_doctor(args)
        except SystemExit:
            pass
        captured = capsys.readouterr().out
        assert "[OK] PyTorch" in captured

    def test_reports_gpu_backend_status(self, capsys):
        args = argparse.Namespace()
        try:
            cmd_doctor(args)
        except SystemExit:
            pass
        captured = capsys.readouterr().out
        assert any(
            kw in captured
            for kw in ["[OK] MPS", "[OK] CUDA", "[WARN] No GPU backend"]
        )

    def test_exits_cleanly_when_all_deps_present(self, capsys):
        args = argparse.Namespace()
        try:
            cmd_doctor(args)
        except SystemExit as exc:
            if exc.code == 1:
                pytest.skip("Some optional deps missing in this environment")

    def test_reports_failure_on_missing_import(self, monkeypatch, capsys):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "diffusers":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        args = argparse.Namespace()
        with pytest.raises(SystemExit) as exc_info:
            cmd_doctor(args)
        assert exc_info.value.code == 1

        captured = capsys.readouterr().out
        assert "[FAIL] diffusers not installed" in captured
        assert "issue(s) found" in captured


class TestMain:
    def test_no_args_exits_with_code_1(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["stablevsr"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
