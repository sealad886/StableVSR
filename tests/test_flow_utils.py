"""Tests for flow utility functions — device portability."""

import torch

from util.flow_utils import compute_flow_gradients, detect_occlusion


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
