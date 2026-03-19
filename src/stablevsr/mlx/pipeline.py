"""MLX StableVSR pipeline — native Apple Silicon inference.

Port of pipeline/stablevsr_pipeline.py from PyTorch to MLX.
RAFT remains in PyTorch (CPU) via the bridge module.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .flow import bicubic_upsample, flow_warp
from .flow.raft_bridge import compute_flows_via_raft
from .models.controlnet import ControlNetModel
from .models.text_encoder import CLIPTextModel
from .models.unet import UNet2DConditionModel
from .models.vae import AutoencoderKL, DEFAULT_TILE_OVERLAP, DEFAULT_TILE_SIZE
from .scheduler import MLXDDPMScheduler
from .weight_utils import load_safetensors_for_mlx

logger = logging.getLogger(__name__)


class MLXStableVSRPipeline:
    """End-to-end video super-resolution pipeline in MLX.

    Usage:
        pipe = MLXStableVSRPipeline.from_pretrained("path/to/model")
        frames = pipe(images, of_model, prompt="...", num_inference_steps=50)
    """

    def __init__(
        self,
        text_encoder: CLIPTextModel,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        controlnet: ControlNetModel,
        scheduler: MLXDDPMScheduler,
        tokenizer: Any,
    ):
        self.text_encoder = text_encoder
        self.vae = vae
        self.unet = unet
        self.controlnet = controlnet
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self._vae_scale_factor = 2 ** (len(vae.decoder.up_blocks) - 1)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        dtype: mx.Dtype = mx.float16,
    ) -> "MLXStableVSRPipeline":
        """Load all components from a HuggingFace-format model directory."""
        model_path = Path(model_path)

        # Load configs
        def _load_config(subdir: str) -> dict:
            with open(model_path / subdir / "config.json") as f:
                return json.load(f)

        te_config = _load_config("text_encoder")
        vae_config = _load_config("vae")
        unet_config = _load_config("unet")
        cn_config = _load_config("controlnet")

        # Build text encoder
        text_encoder = CLIPTextModel(
            vocab_size=te_config["vocab_size"],
            hidden_size=te_config["hidden_size"],
            num_attention_heads=te_config["num_attention_heads"],
            num_hidden_layers=te_config["num_hidden_layers"],
            intermediate_size=te_config["intermediate_size"],
            max_position_embeddings=te_config["max_position_embeddings"],
        )

        # Build VAE
        vae = AutoencoderKL(
            in_channels=vae_config.get("in_channels", 3),
            out_channels=vae_config.get("out_channels", 3),
            block_out_channels=tuple(vae_config["block_out_channels"]),
            layers_per_block=vae_config.get("layers_per_block", 2),
            latent_channels=vae_config.get("latent_channels", 4),
            norm_num_groups=vae_config.get("norm_num_groups", 32),
            scaling_factor=vae_config.get("scaling_factor", 0.08333),
        )

        # Build UNet
        unet = UNet2DConditionModel(
            in_channels=unet_config["in_channels"],
            out_channels=unet_config.get("out_channels", 4),
            block_out_channels=tuple(unet_config["block_out_channels"]),
            layers_per_block=unet_config.get("layers_per_block", 2),
            cross_attention_dim=unet_config.get("cross_attention_dim", 1024),
            attention_head_dim=unet_config.get("attention_head_dim", 8),
            only_cross_attention=tuple(
                unet_config.get("only_cross_attention", [False] * 4)
            ),
            down_block_types=tuple(
                unet_config.get(
                    "down_block_types",
                    [
                        "DownBlock2D",
                        "CrossAttnDownBlock2D",
                        "CrossAttnDownBlock2D",
                        "CrossAttnDownBlock2D",
                    ],
                )
            ),
            up_block_types=tuple(
                unet_config.get(
                    "up_block_types",
                    [
                        "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D",
                        "UpBlock2D",
                    ],
                )
            ),
        )

        # Build ControlNet
        controlnet = ControlNetModel(
            in_channels=cn_config["in_channels"],
            conditioning_channels=cn_config.get("conditioning_channels", 3),
            block_out_channels=tuple(cn_config["block_out_channels"]),
            layers_per_block=cn_config.get("layers_per_block", 2),
            cross_attention_dim=cn_config.get("cross_attention_dim", 1024),
            attention_head_dim=cn_config.get("attention_head_dim", 8),
            only_cross_attention=tuple(
                cn_config.get("only_cross_attention", [True, True, True, False])
            ),
            conditioning_embedding_out_channels=tuple(
                cn_config.get(
                    "conditioning_embedding_out_channels",
                    [64, 128, 256],
                )
            ),
            down_block_types=tuple(
                cn_config.get(
                    "down_block_types",
                    [
                        "DownBlock2D",
                        "CrossAttnDownBlock2D",
                        "CrossAttnDownBlock2D",
                        "CrossAttnDownBlock2D",
                    ],
                )
            ),
        )

        # Build scheduler
        sched_path = model_path / "scheduler" / "scheduler_config.json"
        if sched_path.exists():
            with open(sched_path) as f:
                sched_config = json.load(f)
        else:
            sched_config = {}

        scheduler = MLXDDPMScheduler(
            num_train_timesteps=sched_config.get("num_train_timesteps", 1000),
            beta_start=sched_config.get("beta_start", 0.0001),
            beta_end=sched_config.get("beta_end", 0.02),
            beta_schedule=sched_config.get("beta_schedule", "scaled_linear"),
            clip_sample=sched_config.get("clip_sample", True),
            prediction_type=sched_config.get("prediction_type", "epsilon"),
            variance_type=sched_config.get("variance_type", "fixed_small"),
            timestep_spacing=sched_config.get("timestep_spacing", "leading"),
            steps_offset=sched_config.get("steps_offset", 0),
            scaling_factor=vae.scaling_factor,
        )

        # Load weights
        logger.info("Loading text encoder weights...")
        te_weights = load_safetensors_for_mlx(
            model_path / "text_encoder" / "model.safetensors",
        )
        _load_weights_into_model(
            text_encoder, te_weights, prefix_map={"text_model.": ""}
        )

        logger.info("Loading VAE weights...")
        vae_weights = load_safetensors_for_mlx(
            model_path / "vae" / "diffusion_pytorch_model.safetensors",
            model_path / "vae" / "config.json",
        )
        _load_weights_into_model(vae, vae_weights)

        logger.info("Loading UNet weights...")
        unet_weights = load_safetensors_for_mlx(
            model_path / "unet" / "diffusion_pytorch_model.safetensors",
            model_path / "unet" / "config.json",
        )
        _load_weights_into_model(unet, unet_weights)

        logger.info("Loading ControlNet weights...")
        cn_weights = load_safetensors_for_mlx(
            model_path / "controlnet" / "diffusion_pytorch_model.safetensors",
            model_path / "controlnet" / "config.json",
        )
        _load_weights_into_model(controlnet, cn_weights)

        # Convert to target dtype
        if dtype != mx.float32:
            for model in (text_encoder, vae, unet, controlnet):
                params = {
                    k: v.astype(dtype)
                    for k, v in nn.utils.tree_flatten(model.parameters())
                }
                model.load_weights(list(params.items()))

        mx.eval(
            text_encoder.parameters(),
            vae.parameters(),
            unet.parameters(),
            controlnet.parameters(),
        )

        # Load tokenizer (keep as HuggingFace for compatibility)
        try:
            from transformers import CLIPTokenizer

            tokenizer = CLIPTokenizer.from_pretrained(str(model_path / "tokenizer"))
        except Exception:
            tokenizer = None

        return cls(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            tokenizer=tokenizer,
        )

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        do_classifier_free_guidance: bool = True,
    ) -> mx.array:
        """Encode text prompt → embeddings."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )
        input_ids = mx.array(tokens["input_ids"])
        prompt_embeds = self.text_encoder(input_ids)

        if do_classifier_free_guidance:
            neg_tokens = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="np",
            )
            neg_ids = mx.array(neg_tokens["input_ids"])
            neg_embeds = self.text_encoder(neg_ids)
            prompt_embeds = mx.concatenate([neg_embeds, prompt_embeds], axis=0)

        mx.eval(prompt_embeds)
        return prompt_embeds

    def __call__(
        self,
        images: list[np.ndarray],
        of_model: Any,
        prompt: str = "clean, high-resolution, 8k, sharp, details",
        negative_prompt: str = "blurry, noise, low-resolution, artifacts",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        of_rescale_factor: int = 1,
        seed: int | None = None,
        progress_callback: Any | None = None,
        force_tiled_vae: bool | None = None,
        tile_size: int = DEFAULT_TILE_SIZE,
        tile_overlap: int = DEFAULT_TILE_OVERLAP,
        ttg_start_step: int = 0,
        compile_models: bool = False,
    ) -> list[np.ndarray]:
        """Run video super-resolution.

        Args:
            images: List of LR frames as (H, W, 3) uint8 numpy arrays.
            of_model: PyTorch RAFT model for optical flow.
            prompt: Text prompt for generation.
            negative_prompt: Negative text prompt.
            num_inference_steps: Number of diffusion steps.
            guidance_scale: Classifier-free guidance scale.
            controlnet_conditioning_scale: ControlNet strength.
            of_rescale_factor: Optical flow rescale factor.
            seed: Random seed for reproducibility.
            progress_callback: Called with (step, total_steps) for progress tracking.
            force_tiled_vae: True → always tile; False → never tile; None → auto.
            tile_size: Tile size in latent pixels for tiled VAE decode.
            tile_overlap: Overlap in latent pixels for tiled VAE decode.
            ttg_start_step: Denoising step to begin temporal texture guidance.
                Steps before this skip VAE decode + flow warp (saves time on
                noisy early steps where temporal guidance has minimal benefit).
                0 = all steps (default). Higher values save more time at the
                cost of reduced temporal consistency.
            compile_models: If True, use mx.compile on UNet/ControlNet forward
                passes for fused GPU kernel execution (~30-50% speedup).

        Returns:
            List of SR frames as (H, W, 3) uint8 numpy arrays.
        """
        import torch

        do_cfg = guidance_scale > 1.0
        n_frames = len(images)

        # 1. Encode prompt
        logger.info("Encoding prompt...")
        prompt_embeds = self.encode_prompt(prompt, negative_prompt, do_cfg)

        # 2. Preprocess images: uint8 HWC → float32 NCHW torch tensors (for RAFT)
        #    and NHWC MLX arrays
        images_torch = []
        images_mlx = []
        for img in images:
            # Normalize to [-1, 1]
            img_f = img.astype(np.float32) / 255.0
            img_f = img_f * 2.0 - 1.0

            # NCHW torch tensor
            img_torch = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0)
            images_torch.append(img_torch)

            # NHWC MLX array
            images_mlx.append(mx.array(img_f[None]))  # (1, H, W, 3)

        # 3. Bicubic 4× upscale (conditioning signal for ControlNet)
        logger.info("Upscaling images 4x with bicubic...")
        upscaled_mlx = [bicubic_upsample(img, scale_factor=4) for img in images_mlx]
        for u in upscaled_mlx:
            mx.eval(u)

        # Also upscale torch tensors for RAFT
        import torch.nn.functional as F

        upscaled_torch = [
            F.interpolate(img, scale_factor=4, mode="bicubic", align_corners=False)
            for img in images_torch
        ]

        _, H, W, _ = upscaled_mlx[0].shape

        # 4. Compute optical flows via RAFT bridge
        logger.info("Computing optical flows via RAFT...")
        forward_flows, backward_flows = compute_flows_via_raft(
            of_model, upscaled_torch, rescale_factor=of_rescale_factor, device="cpu"
        )
        # Free torch tensors
        del images_torch, upscaled_torch
        gc.collect()

        # 5. Set up timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latents
        # vae_scale_factor = 2^(num_downsamples) = 2^(len(block_out_channels)-1)
        # For block_out_channels=[128,256,512]: factor=4
        latent_h, latent_w = H // self._vae_scale_factor, W // self._vae_scale_factor
        latents = []
        for i in range(n_frames):
            key = mx.random.key(seed + i if seed is not None else i)
            noise = mx.random.normal((1, latent_h, latent_w, 4), key=key)
            latents.append(noise.astype(prompt_embeds.dtype))

        interp_mode = "bilinear" if of_rescale_factor == 1 else "nearest"

        # 7. Denoising loop with bidirectional temporal sampling
        logger.info(f"Denoising: {len(timesteps)} steps x {n_frames} frames...")
        if ttg_start_step > 0:
            logger.info(
                f"Temporal texture guidance starts at step {ttg_start_step}"
            )
        reversed_order = False
        total_ops = len(timesteps) * n_frames
        op_count = 0

        x0_est: mx.array | None = None
        warped_prev_est: mx.array | None = None

        # Optionally compile model forward passes for fused GPU execution
        if compile_models:
            logger.info("Compiling UNet and ControlNet with mx.compile...")

            @mx.compile
            def _compiled_unet(sample, timestep, enc, *residuals):
                if len(residuals) == 0:
                    return self.unet(sample, timestep, enc)
                down_res_list = list(residuals[:-1])
                mid_res = residuals[-1]
                return self.unet(
                    sample, timestep, enc,
                    down_block_additional_residuals=down_res_list,
                    mid_block_additional_residual=mid_res,
                )

            @mx.compile
            def _compiled_controlnet(sample, timestep, enc, cond, scale):
                return self.controlnet(
                    sample, timestep, enc,
                    controlnet_cond=cond,
                    conditioning_scale=scale,
                )

        # Stage timing accumulators
        _t_vae_decode = 0.0
        _t_flow_warp = 0.0
        _t_controlnet = 0.0
        _t_unet = 0.0
        _t_scheduler = 0.0

        for step_idx, t in enumerate(timesteps):
            flows = forward_flows if not reversed_order else backward_flows

            for frame_idx in range(n_frames):
                # LR image for UNet conditioning (same spatial res as latents)
                lr_image_cond = images_mlx[frame_idx]

                # Temporal Texture Guidance (for non-first frames, after ttg_start_step)
                use_ttg = (
                    frame_idx != 0
                    and x0_est is not None
                    and step_idx >= ttg_start_step
                )
                if use_ttg:
                    # Decode x0_est to pixel space (tiled to avoid OOM)
                    _t0 = time.perf_counter()
                    x0_decoded = self.vae.smart_decode(
                        x0_est / self.vae.scaling_factor,
                        force_tiled=force_tiled_vae,
                        tile_size=tile_size,
                        tile_overlap=tile_overlap,
                    )
                    mx.eval(x0_decoded)
                    _t_vae_decode += time.perf_counter() - _t0

                    # Warp to current frame
                    _t0 = time.perf_counter()
                    warped_prev_est = flow_warp(
                        x0_decoded, flows[frame_idx - 1], interp_mode=interp_mode
                    )
                    mx.eval(warped_prev_est)
                    _t_flow_warp += time.perf_counter() - _t0

                # Prepare UNet input
                latent_input = latents[frame_idx]
                if do_cfg:
                    latent_input = mx.concatenate([latent_input, latent_input], axis=0)

                latent_input = self.scheduler.scale_model_input(latent_input, t)
                # Concat image conditioning (7 channel input)
                # UNet takes LR image at latent resolution, not upscaled image
                if do_cfg:
                    image_for_unet = mx.concatenate(
                        [lr_image_cond, lr_image_cond], axis=0
                    )
                else:
                    image_for_unet = lr_image_cond
                latent_model_input = mx.concatenate(
                    [latent_input, image_for_unet], axis=-1
                )

                timestep_batch = mx.array([t] * latent_model_input.shape[0])

                # ControlNet (only when temporal guidance is active)
                if not use_ttg or warped_prev_est is None:
                    down_residuals = None
                    mid_residual = None
                else:
                    _t0 = time.perf_counter()
                    if do_cfg:
                        cn_cond = mx.concatenate(
                            [warped_prev_est, warped_prev_est], axis=0
                        )
                    else:
                        cn_cond = warped_prev_est

                    if compile_models:
                        down_residuals, mid_residual = _compiled_controlnet(
                            latent_model_input,
                            timestep_batch,
                            prompt_embeds,
                            cn_cond,
                            controlnet_conditioning_scale,
                        )
                    else:
                        down_residuals, mid_residual = self.controlnet(
                            latent_model_input,
                            timestep_batch,
                            prompt_embeds,
                            controlnet_cond=cn_cond,
                            conditioning_scale=controlnet_conditioning_scale,
                        )
                    mx.eval(down_residuals, mid_residual)
                    _t_controlnet += time.perf_counter() - _t0

                # UNet noise prediction
                _t0 = time.perf_counter()
                if compile_models:
                    if down_residuals is not None:
                        noise_pred = _compiled_unet(
                            latent_model_input,
                            timestep_batch,
                            prompt_embeds,
                            *down_residuals,
                            mid_residual,
                        )
                    else:
                        noise_pred = _compiled_unet(
                            latent_model_input,
                            timestep_batch,
                            prompt_embeds,
                        )
                else:
                    noise_pred = self.unet(
                        latent_model_input,
                        timestep_batch,
                        prompt_embeds,
                        down_block_additional_residuals=down_residuals,
                        mid_block_additional_residual=mid_residual,
                    )
                mx.eval(noise_pred)
                _t_unet += time.perf_counter() - _t0

                # Classifier-free guidance
                if do_cfg:
                    uncond, cond = mx.split(noise_pred, 2, axis=0)
                    noise_pred = uncond + guidance_scale * (cond - uncond)

                # Scheduler step
                _t0 = time.perf_counter()
                output = self.scheduler.step(
                    noise_pred,
                    t,
                    latents[frame_idx],
                    seed=seed,
                )
                latents[frame_idx] = output.prev_sample
                x0_est = output.pred_original_sample
                mx.eval(latents[frame_idx], x0_est)
                _t_scheduler += time.perf_counter() - _t0

                op_count += 1
                if progress_callback:
                    progress_callback(op_count, total_ops)

            # Bidirectional: reverse all lists
            images_mlx.reverse()
            upscaled_mlx.reverse()
            latents.reverse()
            forward_flows, backward_flows = backward_flows, forward_flows
            reversed_order = not reversed_order

        # Restore correct order
        if reversed_order:
            latents.reverse()

        logger.info(
            "Denoising stage timing: "
            f"vae_decode={_t_vae_decode:.1f}s, "
            f"flow_warp={_t_flow_warp:.1f}s, "
            f"controlnet={_t_controlnet:.1f}s, "
            f"unet={_t_unet:.1f}s, "
            f"scheduler={_t_scheduler:.1f}s"
        )

        # 8. Decode final latents
        logger.info("Decoding final latents...")
        output_frames = []
        for latent in latents:
            decoded = self.vae.smart_decode(
                latent / self.vae.scaling_factor,
                force_tiled=force_tiled_vae,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )
            mx.eval(decoded)

            # Convert to uint8 numpy
            img = np.array(decoded[0])
            img = ((img + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
            output_frames.append(img)

        return output_frames


def _load_weights_into_model(
    model: nn.Module,
    weights: dict[str, mx.array],
    prefix_map: dict[str, str] | None = None,
) -> None:
    """Load weights dict into an MLX model, applying prefix remapping."""
    if prefix_map:
        remapped = {}
        for k, v in weights.items():
            new_k = k
            for old_prefix, new_prefix in prefix_map.items():
                if new_k.startswith(old_prefix):
                    new_k = new_prefix + new_k[len(old_prefix) :]
            remapped[new_k] = v
        weights = remapped

    model.load_weights(list(weights.items()), strict=False)
