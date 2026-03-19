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
            kw in captured for kw in ["[OK] MPS", "[OK] CUDA", "[WARN] No GPU backend"]
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


class TestCollectSequences:
    """Adversarial tests for _collect_sequences."""

    def test_hidden_dirs_excluded(self, tmp_path):
        from stablevsr.cli import _collect_sequences

        (tmp_path / ".hidden").mkdir()
        (tmp_path / "visible").mkdir()
        seqs = _collect_sequences(tmp_path)
        names = [n for n, _ in seqs]
        assert "visible" in names
        assert ".hidden" not in names

    def test_flat_folder_returns_self(self, tmp_path):
        from stablevsr.cli import _collect_sequences

        (tmp_path / "frame001.png").touch()
        seqs = _collect_sequences(tmp_path)
        assert len(seqs) == 1
        assert seqs[0][1] == tmp_path


class TestLoadFrames:
    """Adversarial tests for _load_frames."""

    def test_filters_non_image_files(self, tmp_path):
        from PIL import Image

        from stablevsr.cli import _load_frames

        img = Image.new("RGB", (4, 4), "red")
        img.save(tmp_path / "frame.png")
        (tmp_path / ".DS_Store").touch()
        (tmp_path / "readme.txt").touch()

        frames, names = _load_frames(tmp_path)
        assert len(frames) == 1
        assert names == ["frame.png"]

    def test_empty_dir_returns_empty(self, tmp_path):
        from stablevsr.cli import _load_frames

        frames, names = _load_frames(tmp_path)
        assert frames == []
        assert names == []


class TestVersionConsistency:
    """Ensure version string comes from package metadata."""

    def test_version_in_parser(self):
        parser = build_parser()
        # The version action should contain the __version__ value
        version_action = None
        for action in parser._actions:
            if isinstance(action, argparse._VersionAction):
                version_action = action
                break
        assert version_action is not None
        assert version_action.version is not None
        from stablevsr import __version__

        assert __version__ in version_action.version


class TestImageExtensionFiltering:
    """Verify test.py IMAGE_EXTENSIONS constant is importable."""

    def test_image_extensions_defined(self):
        """test.py should now be importable (main guard) and expose IMAGE_EXTENSIONS."""
        # test.py has heavy imports (diffusers, torch) at module level that
        # may not be available in CI, so we just verify the constant exists
        # by reading the file as text.
        from pathlib import Path

        test_py = Path(__file__).parent.parent / "test.py"
        content = test_py.read_text()
        assert "IMAGE_EXTENSIONS" in content
        assert 'if __name__ == "__main__":' in content


class TestEvalMainGuard:
    """Verify eval.py has __main__ guard."""

    def test_eval_has_main_guard(self):
        from pathlib import Path

        eval_py = Path(__file__).parent.parent / "eval.py"
        content = eval_py.read_text()
        assert 'if __name__ == "__main__":' in content


class TestBenchmarkParser:
    """Tests for the benchmark subcommand parser."""

    def test_parse_benchmark_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["benchmark"])
        assert args.command == "benchmark"
        assert args.steps == 3
        assert args.frames == 2
        assert args.resolution == "128x128"
        assert args.dtype == "float32"
        assert args.warmup == 0
        assert args.output is None

    def test_parse_benchmark_custom_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "benchmark",
                "--steps",
                "5",
                "--frames",
                "4",
                "--resolution",
                "256x256",
                "--dtype",
                "float16",
                "--warmup",
                "1",
                "--output",
                "/tmp/bench.json",
            ]
        )
        assert args.steps == 5
        assert args.frames == 4
        assert args.resolution == "256x256"
        assert args.dtype == "float16"
        assert args.warmup == 1
        assert args.output == "/tmp/bench.json"


class TestHelperFunctions:
    """Tests for _sync and _peak_rss_mb."""

    def test_peak_rss_returns_positive(self):
        from stablevsr.cli import _peak_rss_mb

        rss = _peak_rss_mb()
        assert rss is None or rss > 0

    def test_sync_noop_on_cpu(self):
        from stablevsr.cli import _sync

        _sync("cpu")  # should not raise


class TestMPSAutoDefaults:
    """Test that MPS auto-defaults are mentioned in infer command code."""

    def test_infer_has_mps_auto_tiling(self):
        import inspect

        from stablevsr.cli import cmd_infer

        src = inspect.getsource(cmd_infer)
        assert "auto-enabled (MPS)" in src
        assert "enable_vae_tiling" in src
        assert "enable_attention_slicing" in src
