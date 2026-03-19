"""RAFT bridge: run PyTorch RAFT, convert flows to MLX arrays."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import mlx.core as mx
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    pass


def load_raft_model(device: str | None = None) -> torch.nn.Module:
    """Load RAFT-Large for optical flow computation.

    Returns a frozen, eval-mode RAFT model.  When *device* is ``None``
    (the default), MPS is used when available, otherwise CPU.
    """
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = raft_large(weights=Raft_Large_Weights.DEFAULT)
    model.requires_grad_(False)
    model = model.to(device, dtype=torch.float32)
    model.eval()
    return model


def compute_flows_via_raft(
    of_model: torch.nn.Module,
    images_nchw: list[torch.Tensor],
    rescale_factor: int = 1,
    device: str = "cpu",
) -> tuple[list[mx.array], list[mx.array]]:
    """Compute forward/backward optical flows using PyTorch RAFT, return as MLX arrays.

    Forces RAFT to CPU to avoid Metal contention with MLX.
    Converts one flow at a time then frees the torch tensors.

    Args:
        of_model: RAFT model (will be moved to device).
        images_nchw: List of (1,3,H,W) torch tensors (upscaled images).
        rescale_factor: Optical flow rescale factor.
        device: Device for RAFT ('cpu' to avoid Metal contention).

    Returns:
        (forward_flows, backward_flows) as lists of MLX arrays in (B,H,W,2) NHWC format.
    """
    of_model = of_model.to(device)
    of_model.eval()

    forward_flows: list[mx.array] = []
    backward_flows: list[mx.array] = []

    with torch.no_grad():
        for i in range(len(images_nchw) - 1):
            src = images_nchw[i].to(device)
            tgt = images_nchw[i + 1].to(device)

            # Forward: i → i+1
            fw = _get_flow(of_model, tgt, src, rescale_factor)
            forward_flows.append(mx.array(fw.numpy()))
            del fw

            # Backward: i+1 → i
            bw = _get_flow(of_model, src, tgt, rescale_factor)
            backward_flows.append(mx.array(bw.numpy()))
            del bw

            del src, tgt
            gc.collect()

    return forward_flows, backward_flows


def _get_flow(
    of_model: torch.nn.Module,
    target: torch.Tensor,
    source: torch.Tensor,
    rescale_factor: int = 1,
) -> torch.Tensor:
    """Run RAFT and return flow in (B, H, W, 2) format."""
    flows = of_model(target, source)
    flow = flows[-1]
    if rescale_factor != 1:
        flow = F.interpolate(
            flow / rescale_factor,
            scale_factor=1.0 / rescale_factor,
            mode="bilinear",
            align_corners=False,
        )
    flow = flow.permute(0, 2, 3, 1).cpu()  # to BHWC
    return flow
