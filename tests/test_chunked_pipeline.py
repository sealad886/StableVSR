"""Tests for temporal chunk planning, overlap blending, and assembly."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from stablevsr.mlx.chunked_pipeline import (
    ChunkManifest,
    ChunkSpec,
    _assemble_chunks,
    blend_overlap,
    plan_chunks,
    run_chunked_inference,
)


class TestPlanChunks:
    def test_single_chunk_no_split(self):
        chunks = plan_chunks(num_frames=5, chunk_size=10, chunk_overlap=2)
        assert len(chunks) == 1
        assert chunks[0].start_frame == 0
        assert chunks[0].end_frame == 5
        assert chunks[0].overlap_before == 0
        assert chunks[0].overlap_after == 0

    def test_exact_two_chunks(self):
        chunks = plan_chunks(num_frames=10, chunk_size=7, chunk_overlap=2)
        assert len(chunks) == 2
        assert chunks[0].start_frame == 0
        assert chunks[0].end_frame == 7
        assert chunks[0].overlap_before == 0
        assert chunks[1].start_frame == 5
        assert chunks[1].end_frame == 10
        assert chunks[1].overlap_before == 2

    def test_three_chunks(self):
        chunks = plan_chunks(num_frames=20, chunk_size=10, chunk_overlap=3)
        assert len(chunks) == 3
        # stride = 10 - 3 = 7
        assert chunks[0].start_frame == 0
        assert chunks[0].end_frame == 10
        assert chunks[1].start_frame == 7
        assert chunks[1].end_frame == 17
        assert chunks[2].start_frame == 14
        assert chunks[2].end_frame == 20

    def test_zero_overlap(self):
        chunks = plan_chunks(num_frames=10, chunk_size=5, chunk_overlap=0)
        assert len(chunks) == 2
        assert chunks[0].end_frame == 5
        assert chunks[1].start_frame == 5
        assert chunks[1].end_frame == 10

    def test_coverage_complete(self):
        """All frame indices are covered at least once."""
        for n in [5, 10, 17, 50]:
            chunks = plan_chunks(num_frames=n, chunk_size=8, chunk_overlap=3)
            covered = set()
            for c in chunks:
                covered.update(range(c.start_frame, c.end_frame))
            assert covered == set(range(n)), f"Failed for n={n}"

    def test_last_chunk_has_no_overlap_after(self):
        chunks = plan_chunks(num_frames=30, chunk_size=10, chunk_overlap=3)
        assert chunks[-1].overlap_after == 0

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            plan_chunks(num_frames=0, chunk_size=5, chunk_overlap=2)
        with pytest.raises(ValueError):
            plan_chunks(num_frames=10, chunk_size=1, chunk_overlap=0)
        with pytest.raises(ValueError):
            plan_chunks(num_frames=10, chunk_size=5, chunk_overlap=-1)
        with pytest.raises(ValueError):
            plan_chunks(num_frames=10, chunk_size=5, chunk_overlap=5)

    def test_chunk_size_equals_frames(self):
        chunks = plan_chunks(num_frames=8, chunk_size=8, chunk_overlap=2)
        assert len(chunks) == 1

    def test_small_remainder_absorbed(self):
        """If remainder is 1 frame, it should be absorbed into last chunk."""
        chunks = plan_chunks(num_frames=11, chunk_size=5, chunk_overlap=0)
        assert chunks[-1].end_frame == 11
        for c in chunks:
            assert c.num_frames >= 2

    def test_num_frames_property(self):
        chunks = plan_chunks(num_frames=15, chunk_size=8, chunk_overlap=2)
        for c in chunks:
            assert c.num_frames == c.end_frame - c.start_frame


class TestBlendOverlap:
    def _make_frames(self, n, value):
        return [np.full((4, 4, 3), value, dtype=np.uint8) for _ in range(n)]

    def test_no_overlap(self):
        result = blend_overlap([], [], 0)
        assert result == []

    def test_single_frame_overlap(self):
        a = self._make_frames(1, 200)
        b = self._make_frames(1, 100)
        blended = blend_overlap(a, b, 1)
        assert len(blended) == 1
        # alpha_a = 1.0 - 1/2 = 0.5, alpha_b = 0.5 → 150
        assert blended[0].mean() == pytest.approx(150.0, abs=1)

    def test_two_frame_overlap(self):
        a = self._make_frames(2, 0)
        b = self._make_frames(2, 255)
        blended = blend_overlap(a, b, 2)
        assert len(blended) == 2
        # i=0: alpha_a = 1 - 1/3 = 0.667, alpha_b = 0.333 → ~85
        # i=1: alpha_a = 1 - 2/3 = 0.333, alpha_b = 0.667 → ~170
        assert blended[0].mean() < blended[1].mean()

    def test_mismatched_counts(self):
        a = self._make_frames(2, 100)
        b = self._make_frames(3, 100)
        with pytest.raises(ValueError, match="Expected 2"):
            blend_overlap(a, b, 2)

    def test_output_dtype(self):
        a = self._make_frames(1, 100)
        b = self._make_frames(1, 200)
        blended = blend_overlap(a, b, 1)
        assert blended[0].dtype == np.uint8


class TestAssembleChunks:
    def _make_frames(self, n, value):
        return [np.full((4, 4, 3), value, dtype=np.uint8) for _ in range(n)]

    def test_single_chunk(self):
        chunks = [ChunkSpec(0, 0, 5, 0, 0)]
        outputs = {0: self._make_frames(5, 100)}
        result = _assemble_chunks(chunks, outputs, 0)
        assert len(result) == 5

    def test_two_chunks_no_overlap(self):
        chunks = [
            ChunkSpec(0, 0, 5, 0, 0),
            ChunkSpec(1, 5, 10, 0, 0),
        ]
        outputs = {
            0: self._make_frames(5, 100),
            1: self._make_frames(5, 200),
        }
        result = _assemble_chunks(chunks, outputs, 0)
        assert len(result) == 10

    def test_two_chunks_with_overlap(self):
        chunks = [
            ChunkSpec(0, 0, 7, 0, 2),
            ChunkSpec(1, 5, 10, 2, 0),
        ]
        outputs = {
            0: self._make_frames(7, 100),
            1: self._make_frames(5, 200),
        }
        result = _assemble_chunks(chunks, outputs, 2)
        # 5 non-overlap from first + 2 blended + 3 non-overlap from second = 10
        assert len(result) == 10
        # Blended frames should be between 100 and 200
        for f in result[5:7]:
            assert 100 < f.mean() < 200

    def test_empty_chunks(self):
        result = _assemble_chunks([], {}, 0)
        assert result == []


class TestChunkManifest:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            m = ChunkManifest(total_frames=20, chunk_size=8, chunk_overlap=2)
            m.mark_done(0, path)
            m.mark_done(1, path)

            loaded = ChunkManifest.load(path)
            assert loaded.total_frames == 20
            assert loaded.chunk_size == 8
            assert loaded.chunk_overlap == 2
            assert loaded.is_chunk_done(0)
            assert loaded.is_chunk_done(1)
            assert not loaded.is_chunk_done(2)

    def test_is_chunk_done(self):
        m = ChunkManifest(total_frames=10, chunk_size=5, chunk_overlap=1)
        assert not m.is_chunk_done(0)
        m.completed_chunks.append(0)
        assert m.is_chunk_done(0)

    def test_saved_json_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            m = ChunkManifest(total_frames=10, chunk_size=5, chunk_overlap=1)
            m.save(path)
            data = json.loads(path.read_text())
            assert "total_frames" in data
            assert "chunk_size" in data
            assert "completed_chunks" in data


class TestDryRun:
    def test_dry_run_returns_empty(self):
        mock_pipeline = MagicMock()
        dummy_images = [np.zeros((4, 4, 3), dtype=np.uint8)] * 10

        result = run_chunked_inference(
            pipeline=mock_pipeline,
            images=dummy_images,
            of_model=MagicMock(),
            chunk_size=5,
            chunk_overlap=1,
            dry_run=True,
        )

        assert result == []
        mock_pipeline.assert_not_called()
