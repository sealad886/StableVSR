"""Tests for flow utility functions — device portability."""

import torch

from util.flow_utils import (
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
