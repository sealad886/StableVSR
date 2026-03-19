"""Custom grid_sample and flow utilities for MLX.

MLX lacks F.grid_sample, so we implement bilinear grid sampling from primitives.
Uses align_corners=False convention matching PyTorch default.
"""

from __future__ import annotations

import mlx.core as mx


def grid_sample(
    x: mx.array,
    grid: mx.array,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> mx.array:
    """Sample from x using grid coordinates.

    Args:
        x: Input tensor (B, H, W, C) in NHWC format.
        grid: Sampling grid (B, H_out, W_out, 2) with values in [-1, 1].
              grid[..., 0] = x coords, grid[..., 1] = y coords.
        mode: 'bilinear' or 'nearest'.
        padding_mode: 'zeros' or 'border'.

    Returns:
        Sampled tensor (B, H_out, W_out, C).
    """
    B, H, W, C = x.shape
    _, H_out, W_out, _ = grid.shape

    # Unnormalize grid from [-1, 1] to pixel coords (align_corners=False)
    grid_x = ((grid[..., 0] + 1.0) * W - 1.0) / 2.0
    grid_y = ((grid[..., 1] + 1.0) * H - 1.0) / 2.0

    if mode == "nearest":
        ix = mx.round(grid_x).astype(mx.int32)
        iy = mx.round(grid_y).astype(mx.int32)
        mask = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        ix = mx.clip(ix, 0, W - 1)
        iy = mx.clip(iy, 0, H - 1)
        result = _gather_2d(x, iy, ix)
        if padding_mode == "zeros":
            result = mx.where(mask[..., None], result, mx.zeros_like(result))
        return result

    # Bilinear interpolation
    x0 = mx.floor(grid_x).astype(mx.int32)
    y0 = mx.floor(grid_y).astype(mx.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Fractional parts
    wa = ((x1.astype(mx.float32) - grid_x) * (y1.astype(mx.float32) - grid_y))[..., None]
    wb = ((x1.astype(mx.float32) - grid_x) * (grid_y - y0.astype(mx.float32)))[..., None]
    wc = ((grid_x - x0.astype(mx.float32)) * (y1.astype(mx.float32) - grid_y))[..., None]
    wd = ((grid_x - x0.astype(mx.float32)) * (grid_y - y0.astype(mx.float32)))[..., None]

    if padding_mode == "zeros":
        def _safe_gather(arr: mx.array, iy: mx.array, ix: mx.array) -> mx.array:
            mask = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            ix_c = mx.clip(ix, 0, W - 1)
            iy_c = mx.clip(iy, 0, H - 1)
            vals = _gather_2d(arr, iy_c, ix_c)
            return mx.where(mask[..., None], vals, mx.zeros_like(vals))

        va = _safe_gather(x, y0, x0)
        vb = _safe_gather(x, y1, x0)
        vc = _safe_gather(x, y0, x1)
        vd = _safe_gather(x, y1, x1)
    else:
        # border: clamp to edges
        x0_c = mx.clip(x0, 0, W - 1)
        y0_c = mx.clip(y0, 0, H - 1)
        x1_c = mx.clip(x1, 0, W - 1)
        y1_c = mx.clip(y1, 0, H - 1)
        va = _gather_2d(x, y0_c, x0_c)
        vb = _gather_2d(x, y1_c, x0_c)
        vc = _gather_2d(x, y0_c, x1_c)
        vd = _gather_2d(x, y1_c, x1_c)

    return (wa * va + wb * vb + wc * vc + wd * vd).astype(x.dtype)


def _gather_2d(x: mx.array, iy: mx.array, ix: mx.array) -> mx.array:
    """Gather from x[b, iy, ix, :] for each batch element.

    Args:
        x: (B, H, W, C)
        iy: (B, H_out, W_out) integer y indices
        ix: (B, H_out, W_out) integer x indices

    Returns:
        (B, H_out, W_out, C)
    """
    B, H, W, C = x.shape
    # Flatten spatial dims for take_along_axis
    flat_idx = iy * W + ix  # (B, H_out, W_out)
    x_flat = x.reshape(B, H * W, C)  # (B, H*W, C)
    flat_idx_exp = flat_idx[..., None]  # (B, H_out, W_out, 1)
    flat_idx_exp = mx.broadcast_to(flat_idx_exp, (*flat_idx.shape, C))
    gathered = mx.take_along_axis(x_flat, flat_idx_exp.reshape(B, -1, C), axis=1)
    return gathered.reshape(B, *iy.shape[1:], C)


def flow_warp(
    x: mx.array,
    flow: mx.array,
    interp_mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> mx.array:
    """Warp x using optical flow.

    Args:
        x: (B, H, W, C) NHWC tensor.
        flow: (B, H, W, 2) flow field in pixel units.
        interp_mode: 'bilinear' or 'nearest'.
        padding_mode: 'zeros' or 'border'.

    Returns:
        Warped tensor (B, H, W, C).
    """
    B, H, W, C = x.shape

    # Build base grid
    grid_y = mx.broadcast_to(mx.arange(H)[:, None], (H, W))
    grid_x = mx.broadcast_to(mx.arange(W)[None, :], (H, W))
    base_grid = mx.stack([grid_x, grid_y], axis=-1).astype(mx.float32)  # (H, W, 2)
    base_grid = mx.broadcast_to(base_grid[None], (B, H, W, 2))

    vgrid = base_grid + flow

    # Normalize to [-1, 1] for grid_sample (align_corners=False)
    vgrid_x = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0
    grid_normalized = mx.stack([vgrid_x, vgrid_y], axis=-1)

    return grid_sample(x, grid_normalized, mode=interp_mode, padding_mode=padding_mode)


def bicubic_upsample(x: mx.array, scale_factor: int = 4) -> mx.array:
    """Bicubic upsampling via separable cubic convolution.

    Args:
        x: (B, H, W, C) NHWC input.
        scale_factor: Integer upscale factor.

    Returns:
        (B, H*scale, W*scale, C) upsampled tensor.
    """
    B, H, W, C = x.shape
    H_out, W_out = H * scale_factor, W * scale_factor

    # Output pixel coords in input space
    # align_corners=False style mapping
    out_y = (mx.arange(H_out).astype(mx.float32) + 0.5) / scale_factor - 0.5
    out_x = (mx.arange(W_out).astype(mx.float32) + 0.5) / scale_factor - 0.5

    # Separable: upsample width first, then height
    result = _cubic_interp_1d(x, out_x, axis=2, length=W)
    result = _cubic_interp_1d(result, out_y, axis=1, length=H)
    return result


def _cubic_weight(t: mx.array) -> mx.array:
    """Mitchell-Netravali cubic kernel (a=-0.5, matching PyTorch bicubic)."""
    t_abs = mx.abs(t)
    t2 = t_abs * t_abs
    t3 = t2 * t_abs

    # |t| <= 1
    w1 = 1.5 * t3 - 2.5 * t2 + 1.0
    # 1 < |t| <= 2
    w2 = -0.5 * t3 + 2.5 * t2 - 4.0 * t_abs + 2.0

    return mx.where(t_abs <= 1.0, w1, mx.where(t_abs <= 2.0, w2, mx.zeros_like(t)))


def _cubic_interp_1d(
    x: mx.array,
    coords: mx.array,
    axis: int,
    length: int,
) -> mx.array:
    """1D cubic interpolation along `axis`.

    Args:
        x: Input array, e.g. (B, H, W, C).
        coords: 1D output coordinates in input space.
        axis: Spatial axis to interpolate (1=H, 2=W).
        length: Input size along that axis.

    Returns:
        Interpolated array with the given axis replaced by len(coords).
    """
    # coords shape will be broadcast along the target axis
    floor_c = mx.floor(coords).astype(mx.int32)

    result = mx.zeros_like(
        mx.broadcast_to(
            mx.zeros(1),
            (*x.shape[:axis], len(coords), *x.shape[axis + 1:]),
        )
    ).astype(x.dtype)

    for offset in range(-1, 3):
        idx = mx.clip(floor_c + offset, 0, length - 1)
        w = _cubic_weight(coords - (floor_c + offset).astype(mx.float32))

        # Gather along axis
        sliced = mx.take(x, idx, axis=axis)

        # Shape weight for broadcasting
        shape = [1] * x.ndim
        shape[axis] = len(coords)
        w = w.reshape(shape)

        result = result + sliced * w

    return result
