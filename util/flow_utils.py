import torch
import torch.nn.functional as F

_grid_cache: dict[tuple[int, int, str, str], torch.Tensor] = {}


def _get_base_grid(H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (H, W, str(device), str(dtype))
    cached = _grid_cache.get(key)
    if cached is not None:
        return cached
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H), torch.arange(0, W), indexing="ij"
    )
    grid = torch.stack((grid_x, grid_y), 2).to(device=device, dtype=dtype)
    _grid_cache[key] = grid
    return grid


def flow_warp(x, flow, interp_mode="bilinear", padding_mode="zeros"):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, H, W = x.size()
    grid = _get_base_grid(H, W, x.device, x.dtype)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=False,
    )
    return output


def warp_error(
    of_model, current_frame, prev_frame, current_gt, prev_gt, use_occlusion_mask=True
):
    """Compute mean squared warp error between frames using optical flow."""
    flow_forward, flow_backward = get_flow_forward_backward(
        of_model, current_gt, prev_gt
    )
    prev_warped = flow_warp(prev_frame, flow_forward)
    prev_gt_warped = flow_warp(prev_gt, flow_forward)
    if use_occlusion_mask:
        mask = detect_occlusion(flow_forward, flow_backward)
        valid_pixels = torch.sum(mask == 1)
        mean_error = torch.sum((mask * current_frame - mask * prev_warped) ** 2) / (
            valid_pixels * 3 + 1e-10
        )
    else:
        mean_error = ((current_frame - prev_warped) ** 2).mean()
    return mean_error


def get_flow(of_model, target, source, rescale_factor=1):
    """Compute optical flow from *target* to *source* using the given model."""
    flows = of_model(target, source)
    flow = flows[-1]
    flow = (
        F.interpolate(
            flow / rescale_factor, scale_factor=1 / rescale_factor, mode="bilinear"
        )
        if rescale_factor != 1
        else flow
    )
    flow = flow.permute(0, 2, 3, 1)  # permute to B, H, W, 2
    return flow


def compute_flow_magnitude(flow):
    """Compute per-pixel squared flow magnitude."""
    flow_mag = flow[:, :, :, 0] ** 2 + flow[:, :, :, 1] ** 2
    return flow_mag


def compute_flow_gradients(flow):
    """Compute spatial gradients (du/dv) for both flow components."""

    B = flow.shape[0]
    H = flow.shape[1]
    W = flow.shape[2]

    flow_x_du = torch.zeros((B, H, W)).to(flow.device)
    flow_x_dv = torch.zeros((B, H, W)).to(flow.device)
    flow_y_du = torch.zeros((B, H, W)).to(flow.device)
    flow_y_dv = torch.zeros((B, H, W)).to(flow.device)

    flow_x = flow[:, :, :, 0]
    flow_y = flow[:, :, :, 1]

    flow_x_du[:, :, :-1] = flow_x[:, :, :-1] - flow_x[:, :, 1:]
    flow_x_dv[:, :-1, :] = flow_x[:, :-1, :] - flow_x[:, 1:, :]
    flow_y_du[:, :, :-1] = flow_y[:, :, :-1] - flow_y[:, :, 1:]
    flow_y_dv[:, :-1, :] = flow_y[:, :-1, :] - flow_y[:, 1:, :]

    return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv


def detect_occlusion(fw_flow, bw_flow):
    """Detect occlusion mask via forward-backward flow consistency check."""
    # inputs: flow_forward, flow_backward
    # return: occlusion mask
    ## fw-flow: img1 => img2
    ## bw-flow: img2 => img1

    tmp = bw_flow  # to this for divergence between their and my interpretation of forward and backward of
    bw_flow = fw_flow
    fw_flow = tmp

    fw_flow_w = flow_warp(fw_flow.permute(0, 3, 1, 2), bw_flow).permute(0, 2, 3, 1)

    ## occlusion
    fb_flow_sum = fw_flow_w + bw_flow
    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
    fw_flow_w_mag = compute_flow_magnitude(fw_flow_w)
    bw_flow_mag = compute_flow_magnitude(bw_flow)

    mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5

    ## motion boundary
    fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow)
    fx_mag = fx_du**2 + fx_dv**2
    fy_mag = fy_du**2 + fy_dv**2

    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

    ## combine mask
    mask = torch.logical_or(mask1, mask2)
    occlusion = torch.ones((fw_flow.shape[0], fw_flow.shape[1], fw_flow.shape[2])).to(
        fw_flow.device
    )
    occlusion[mask == 1] = 0

    return occlusion


def get_flow_forward_backward(net, current, prev, rescale_factor=1):
    """Compute forward and backward optical flow between two frames."""
    flow_forward = get_flow(net, current, prev, rescale_factor=rescale_factor)
    flow_backward = get_flow(net, prev, current, rescale_factor=rescale_factor)
    return flow_forward, flow_backward
