"""Tests for flow utility functions — device portability."""

import torch

from util.flow_utils import (
    _get_base_grid,
    _grid_cache,
    compute_flow_gradients,
    compute_flow_magnitude,
    detect_occlusion,
    flow_warp,
)


class TestFlowUtilsDevicePortability:
    def test_compute_flow_gradients_cpu(self):
        flow = torch.randn(1, 64, 64, 2, device="cpu")
        dx_du, dx_dv, dy_du, dy_dv = compute_flow_gradients(flow)
        assert dx_du.device.type == "cpu"
        assert dx_dv.device.type == "cpu"
        assert dy_du.device.type == "cpu"
        assert dy_dv.device.type == "cpu"

    def test_detect_occlusion_cpu(self):
        fw_flow = torch.randn(1, 32, 32, 2, device="cpu")
        bw_flow = torch.randn(1, 32, 32, 2, device="cpu")
        occlusion = detect_occlusion(fw_flow, bw_flow)
        assert occlusion.device.type == "cpu"
        assert occlusion.shape == (1, 32, 32)


class TestFlowWarp:
    def test_identity_flow_preserves_input(self):
        # Zero flow should preserve pixel values in the interior.
        # Border pixels are affected by the align_corners=False / zeros padding
        # interaction, so we compare only the interior.
        x = torch.full((1, 3, 16, 16), 0.42)
        flow = torch.zeros(1, 16, 16, 2)
        out = flow_warp(x, flow)
        assert torch.allclose(out[:, :, 2:-2, 2:-2], x[:, :, 2:-2, 2:-2], atol=1e-5)

    def test_output_shape_matches_input(self):
        x = torch.randn(1, 3, 16, 16)
        flow = torch.randn(1, 16, 16, 2) * 0.1
        out = flow_warp(x, flow)
        assert out.shape == x.shape

    def test_device_portability(self):
        x = torch.randn(1, 3, 16, 16, device="cpu")
        flow = torch.zeros(1, 16, 16, 2, device="cpu")
        out = flow_warp(x, flow)
        assert out.device.type == "cpu"


class TestGridCache:
    def test_cache_returns_same_tensor(self):
        _grid_cache.clear()
        g1 = _get_base_grid(16, 16, torch.device("cpu"), torch.float32)
        g2 = _get_base_grid(16, 16, torch.device("cpu"), torch.float32)
        assert g1 is g2

    def test_cache_keys_differ_by_shape(self):
        _grid_cache.clear()
        g1 = _get_base_grid(16, 16, torch.device("cpu"), torch.float32)
        g2 = _get_base_grid(32, 32, torch.device("cpu"), torch.float32)
        assert g1 is not g2
        assert g1.shape == (16, 16, 2)
        assert g2.shape == (32, 32, 2)

    def test_flow_warp_uses_cache(self):
        _grid_cache.clear()
        x = torch.randn(1, 3, 16, 16)
        flow = torch.zeros(1, 16, 16, 2)
        flow_warp(x, flow)
        assert len(_grid_cache) >= 1
        flow_warp(x, flow)
        assert len(_grid_cache) == 1  # no new entries


class TestComputeFlowMagnitude:
    def test_zero_flow_zero_magnitude(self):
        flow = torch.zeros(1, 16, 16, 2)
        mag = compute_flow_magnitude(flow)
        assert torch.allclose(mag, torch.zeros(1, 16, 16))

    def test_unit_flow_magnitude(self):
        flow = torch.zeros(1, 16, 16, 2)
        flow[:, :, :, 0] = 1.0  # x=1, y=0
        mag = compute_flow_magnitude(flow)
        assert torch.allclose(mag, torch.ones(1, 16, 16))

    def test_output_shape(self):
        flow = torch.randn(2, 8, 8, 2)
        mag = compute_flow_magnitude(flow)
        assert mag.shape == (2, 8, 8)


class TestComputeFlowGradients:
    def test_constant_flow_zero_gradients(self):
        flow = torch.ones(1, 16, 16, 2) * 5.0
        dx_du, dx_dv, dy_du, dy_dv = compute_flow_gradients(flow)
        assert torch.allclose(dx_du, torch.zeros_like(dx_du))
        assert torch.allclose(dx_dv, torch.zeros_like(dx_dv))
        assert torch.allclose(dy_du, torch.zeros_like(dy_du))
        assert torch.allclose(dy_dv, torch.zeros_like(dy_dv))

    def test_output_shapes(self):
        flow = torch.randn(2, 8, 8, 2)
        dx_du, dx_dv, dy_du, dy_dv = compute_flow_gradients(flow)
        for grad in (dx_du, dx_dv, dy_du, dy_dv):
            assert grad.shape == (2, 8, 8)


class TestDetectOcclusion:
    def test_identical_flows_no_occlusion(self):
        flow = torch.zeros(1, 16, 16, 2)
        occlusion = detect_occlusion(flow, flow.clone())
        assert (occlusion == 1).all()

    def test_output_shape(self):
        fw = torch.randn(1, 16, 16, 2)
        bw = torch.randn(1, 16, 16, 2)
        occlusion = detect_occlusion(fw, bw)
        assert occlusion.shape == (1, 16, 16)


class TestGetFlow:
    """Adversarial tests for get_flow — validates P0-001 fix."""

    def test_subpixel_flow_preserved(self):
        """Ensure float division preserves subpixel flow (was floor division bug)."""
        from util.flow_utils import get_flow

        class FakeModel:
            def __call__(self, target, source):
                # Return flow with subpixel values
                flow = torch.full((1, 2, 8, 8), 0.7)
                return [flow]

        target = torch.randn(1, 3, 8, 8)
        source = torch.randn(1, 3, 8, 8)
        model = FakeModel()

        result = get_flow(model, target, source, rescale_factor=1)
        assert result.shape == (1, 8, 8, 2)
        # Values should be 0.7, NOT 0.0 (which floor division would give)
        assert result.abs().mean() > 0.5

    def test_rescale_factor_preserves_subpixel(self):
        """With rescale_factor != 1, subpixel values must survive."""
        from util.flow_utils import get_flow

        class FakeModel:
            def __call__(self, target, source):
                flow = torch.full((1, 2, 8, 8), 1.5)
                return [flow]

        target = torch.randn(1, 3, 8, 8)
        source = torch.randn(1, 3, 8, 8)
        model = FakeModel()

        result = get_flow(model, target, source, rescale_factor=2)
        # After dividing 1.5 / 2 = 0.75 (float) vs 0 (floor)
        assert result.abs().mean() > 0.3

    def test_rescale_factor_one_returns_permuted(self):
        """rescale_factor=1 should skip interpolation, just permute."""
        from util.flow_utils import get_flow

        class FakeModel:
            def __call__(self, target, source):
                return [torch.randn(1, 2, 16, 16)]

        result = get_flow(
            FakeModel(), torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16)
        )
        assert result.shape == (1, 16, 16, 2)


class TestGetFlowForwardBackward:
    """Tests for get_flow_forward_backward."""

    def test_returns_two_flows(self):
        from util.flow_utils import get_flow_forward_backward

        class FakeModel:
            def __call__(self, target, source):
                return [torch.randn(1, 2, 8, 8)]

        fw, bw = get_flow_forward_backward(
            FakeModel(), torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8)
        )
        assert fw.shape == (1, 8, 8, 2)
        assert bw.shape == (1, 8, 8, 2)


class TestWarpError:
    """Tests for warp_error function."""

    def test_identical_frames_low_error(self):
        from util.flow_utils import warp_error

        class FakeModel:
            def __call__(self, target, source):
                return [torch.zeros(1, 2, 16, 16)]

        model = FakeModel()
        frame = torch.full((1, 3, 16, 16), 0.5)
        error = warp_error(model, frame, frame.clone(), frame.clone(), frame.clone())
        assert error.item() < 0.1
