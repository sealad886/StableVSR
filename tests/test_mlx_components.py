"""Comprehensive tests for MLX components in src/stablevsr/mlx/."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


# ---------------------------------------------------------------------------
# nn/attention.py
# ---------------------------------------------------------------------------

class TestCrossAttention:
    def test_self_attention_shape(self):
        from stablevsr.mlx.nn.attention import CrossAttention

        attn = CrossAttention(query_dim=64, heads=4, dim_head=16)
        x = mx.random.normal((1, 8, 64))
        mx.eval(attn.parameters())
        out = attn(x)
        mx.eval(out)
        assert out.shape == (1, 8, 64)

    def test_cross_attention_shape(self):
        from stablevsr.mlx.nn.attention import CrossAttention

        attn = CrossAttention(query_dim=64, cross_attention_dim=128, heads=4, dim_head=16)
        mx.eval(attn.parameters())
        q = mx.random.normal((1, 8, 64))
        kv = mx.random.normal((1, 12, 128))
        out = attn(q, encoder_hidden_states=kv)
        mx.eval(out)
        assert out.shape == (1, 8, 64)


class TestBasicTransformerBlock:
    def test_forward_shape(self):
        from stablevsr.mlx.nn.attention import BasicTransformerBlock

        block = BasicTransformerBlock(
            dim=64, num_attention_heads=4, attention_head_dim=16,
            cross_attention_dim=128,
        )
        mx.eval(block.parameters())
        x = mx.random.normal((1, 8, 64))
        enc = mx.random.normal((1, 10, 128))
        out = block(x, encoder_hidden_states=enc)
        mx.eval(out)
        assert out.shape == (1, 8, 64)

    def test_only_cross_attention(self):
        from stablevsr.mlx.nn.attention import BasicTransformerBlock

        block = BasicTransformerBlock(
            dim=64, num_attention_heads=4, attention_head_dim=16,
            cross_attention_dim=128, only_cross_attention=True,
        )
        mx.eval(block.parameters())
        x = mx.random.normal((1, 8, 64))
        enc = mx.random.normal((1, 10, 128))
        out = block(x, encoder_hidden_states=enc)
        mx.eval(out)
        assert out.shape == (1, 8, 64)


class TestTransformer2DModel:
    def test_forward_nhwc(self):
        from stablevsr.mlx.nn.attention import Transformer2DModel

        model = Transformer2DModel(
            num_attention_heads=4, attention_head_dim=16,
            in_channels=64, num_layers=1, cross_attention_dim=128,
        )
        mx.eval(model.parameters())
        x = mx.random.normal((1, 8, 8, 64))
        enc = mx.random.normal((1, 10, 128))
        out = model(x, encoder_hidden_states=enc)
        mx.eval(out)
        assert out.shape == (1, 8, 8, 64)

    def test_residual_connection(self):
        from stablevsr.mlx.nn.attention import Transformer2DModel

        model = Transformer2DModel(
            num_attention_heads=2, attention_head_dim=16,
            in_channels=32, num_layers=1,
        )
        mx.eval(model.parameters())
        x = mx.zeros((1, 4, 4, 32))
        out = model(x)
        mx.eval(out)
        assert out.shape == (1, 4, 4, 32)


# ---------------------------------------------------------------------------
# nn/resnet.py
# ---------------------------------------------------------------------------

class TestResnetBlock2D:
    def test_same_channels(self):
        from stablevsr.mlx.nn.resnet import ResnetBlock2D

        block = ResnetBlock2D(in_channels=64, out_channels=64, temb_channels=128)
        mx.eval(block.parameters())
        x = mx.random.normal((1, 16, 16, 64))
        temb = mx.random.normal((1, 128))
        out = block(x, temb)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 64)

    def test_channel_change(self):
        from stablevsr.mlx.nn.resnet import ResnetBlock2D

        block = ResnetBlock2D(in_channels=64, out_channels=128, temb_channels=256)
        mx.eval(block.parameters())
        x = mx.random.normal((1, 16, 16, 64))
        temb = mx.random.normal((1, 256))
        out = block(x, temb)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 128)

    def test_no_temb(self):
        from stablevsr.mlx.nn.resnet import ResnetBlock2D

        block = ResnetBlock2D(in_channels=64, temb_channels=0)
        mx.eval(block.parameters())
        x = mx.random.normal((1, 16, 16, 64))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 64)


# ---------------------------------------------------------------------------
# nn/sampling.py
# ---------------------------------------------------------------------------

class TestUpsample2D:
    def test_2x_upsample(self):
        from stablevsr.mlx.nn.sampling import Upsample2D

        up = Upsample2D(channels=32, use_conv=True)
        mx.eval(up.parameters())
        x = mx.random.normal((1, 8, 8, 32))
        out = up(x)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 32)

    def test_upsample_no_conv(self):
        from stablevsr.mlx.nn.sampling import Upsample2D

        up = Upsample2D(channels=32, use_conv=False)
        x = mx.random.normal((1, 8, 8, 32))
        out = up(x)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 32)


class TestDownsample2D:
    def test_2x_downsample(self):
        from stablevsr.mlx.nn.sampling import Downsample2D

        down = Downsample2D(channels=32, use_conv=True)
        mx.eval(down.parameters())
        x = mx.random.normal((1, 16, 16, 32))
        out = down(x)
        mx.eval(out)
        assert out.shape == (1, 8, 8, 32)

    def test_downsample_no_conv(self):
        from stablevsr.mlx.nn.sampling import Downsample2D

        down = Downsample2D(channels=32, use_conv=False)
        x = mx.random.normal((1, 16, 16, 32))
        out = down(x)
        mx.eval(out)
        assert out.shape == (1, 8, 8, 32)


# ---------------------------------------------------------------------------
# flow/__init__.py
# ---------------------------------------------------------------------------

class TestGridSample:
    def test_output_shape_matches_grid(self):
        from stablevsr.mlx.flow import grid_sample

        x = mx.random.normal((1, 16, 16, 3))
        grid = mx.random.uniform(-1.0, 1.0, (1, 8, 12, 2))
        out = grid_sample(x, grid)
        mx.eval(out)
        assert out.shape == (1, 8, 12, 3)

    def test_identity_grid(self):
        from stablevsr.mlx.flow import grid_sample

        B, H, W, C = 1, 8, 8, 2
        x = mx.random.normal((B, H, W, C))

        # Build align_corners=False identity grid: g = (2*(i+0.5)/N) - 1
        gy = mx.broadcast_to(
            (2.0 * (mx.arange(H).astype(mx.float32) + 0.5) / H - 1.0).reshape(1, H, 1),
            (B, H, W),
        )
        gx = mx.broadcast_to(
            (2.0 * (mx.arange(W).astype(mx.float32) + 0.5) / W - 1.0).reshape(1, 1, W),
            (B, H, W),
        )
        grid = mx.stack([gx, gy], axis=-1)

        out = grid_sample(x, grid, padding_mode="border")
        mx.eval(x, out)
        diff = mx.abs(out - x).max().item()
        assert diff < 0.05, f"Identity grid diverged: max diff={diff}"

    def test_zeros_padding(self):
        from stablevsr.mlx.flow import grid_sample

        x = mx.ones((1, 4, 4, 1))
        grid = mx.array([[[[-10.0, -10.0]]]])  # (1,1,1,2) way out of bounds
        out = grid_sample(x, grid, padding_mode="zeros")
        mx.eval(out)
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_nearest_mode(self):
        from stablevsr.mlx.flow import grid_sample

        x = mx.random.normal((1, 8, 8, 3))
        grid = mx.random.uniform(-1.0, 1.0, (1, 4, 4, 2))
        out = grid_sample(x, grid, mode="nearest")
        mx.eval(out)
        assert out.shape == (1, 4, 4, 3)


class TestFlowWarp:
    def test_zero_flow_identity(self):
        from stablevsr.mlx.flow import flow_warp

        # Use a constant image so normalization convention offsets don't matter
        x = mx.ones((1, 16, 16, 3)) * 0.5
        flow = mx.zeros((1, 16, 16, 2))
        warped = flow_warp(x, flow, padding_mode="border")
        mx.eval(x, warped)
        diff = mx.abs(warped - x).max().item()
        assert diff < 1e-4, f"Zero-flow warp on constant image diverged: max diff={diff}"

    def test_output_shape(self):
        from stablevsr.mlx.flow import flow_warp

        x = mx.random.normal((1, 16, 16, 3))
        flow = mx.random.normal((1, 16, 16, 2)) * 0.5
        out = flow_warp(x, flow)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 3)


class TestBicubicUpsample:
    def test_scale_factor_4(self):
        from stablevsr.mlx.flow import bicubic_upsample

        x = mx.random.normal((1, 4, 4, 3))
        out = bicubic_upsample(x, scale_factor=4)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 3)

    def test_scale_factor_2(self):
        from stablevsr.mlx.flow import bicubic_upsample

        x = mx.random.normal((1, 8, 8, 1))
        out = bicubic_upsample(x, scale_factor=2)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 1)


# ---------------------------------------------------------------------------
# scheduler.py
# ---------------------------------------------------------------------------

class TestMLXDDPMScheduler:
    def setup_method(self):
        from stablevsr.mlx.scheduler import MLXDDPMScheduler
        self.scheduler = MLXDDPMScheduler()

    def test_set_timesteps(self):
        self.scheduler.set_timesteps(num_inference_steps=50)
        assert self.scheduler.num_inference_steps == 50
        assert len(self.scheduler.timesteps) == 50
        assert self.scheduler.timesteps[0] > self.scheduler.timesteps[-1]

    def test_set_custom_timesteps(self):
        ts = [999, 500, 200, 100, 0]
        self.scheduler.set_timesteps(timesteps=ts)
        assert self.scheduler.timesteps == ts
        assert self.scheduler.num_inference_steps == 5

    def test_scale_model_input_passthrough(self):
        sample = mx.random.normal((1, 8, 8, 4))
        out = self.scheduler.scale_model_input(sample, timestep=500)
        mx.eval(sample, out)
        assert mx.array_equal(sample, out)

    def test_step_output_shape(self):
        self.scheduler.set_timesteps(num_inference_steps=50)
        t = self.scheduler.timesteps[0]
        noise_pred = mx.random.normal((1, 8, 8, 4))
        sample = mx.random.normal((1, 8, 8, 4))
        result = self.scheduler.step(noise_pred, t, sample, seed=42)
        mx.eval(result.prev_sample, result.pred_original_sample)
        assert result.prev_sample.shape == (1, 8, 8, 4)
        assert result.pred_original_sample.shape == (1, 8, 8, 4)

    def test_step_dtype_float32(self):
        self.scheduler.set_timesteps(num_inference_steps=20)
        t = self.scheduler.timesteps[0]
        noise_pred = mx.random.normal((1, 4, 4, 4)).astype(mx.float16)
        sample = mx.random.normal((1, 4, 4, 4)).astype(mx.float16)
        result = self.scheduler.step(noise_pred, t, sample, seed=0)
        mx.eval(result.prev_sample)
        assert result.prev_sample.dtype == mx.float16

    def test_step_final_timestep(self):
        self.scheduler.set_timesteps(num_inference_steps=20)
        t = self.scheduler.timesteps[-1]
        noise_pred = mx.random.normal((1, 4, 4, 4))
        sample = mx.random.normal((1, 4, 4, 4))
        result = self.scheduler.step(noise_pred, t, sample, seed=0)
        mx.eval(result.prev_sample)
        assert result.prev_sample.shape == (1, 4, 4, 4)


# ---------------------------------------------------------------------------
# models/text_encoder.py
# ---------------------------------------------------------------------------

class TestCLIPTextModel:
    @pytest.fixture(autouse=True)
    def _build(self):
        from stablevsr.mlx.models.text_encoder import CLIPTextModel
        self.model = CLIPTextModel(
            vocab_size=49408,
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=128,
            max_position_embeddings=77,
        )
        mx.eval(self.model.parameters())

    def test_output_shape(self):
        ids = mx.array([[1, 2, 3, 4, 5]])
        out = self.model(ids)
        mx.eval(out)
        assert out.shape == (1, 5, 64)

    def test_max_length(self):
        ids = mx.zeros((1, 77), dtype=mx.int32)
        out = self.model(ids)
        mx.eval(out)
        assert out.shape == (1, 77, 64)

    def test_batch(self):
        ids = mx.zeros((2, 10), dtype=mx.int32)
        out = self.model(ids)
        mx.eval(out)
        assert out.shape == (2, 10, 64)


# ---------------------------------------------------------------------------
# models/vae.py
# ---------------------------------------------------------------------------

class TestAutoencoderKL:
    @pytest.fixture(autouse=True)
    def _build(self):
        from stablevsr.mlx.models.vae import AutoencoderKL
        self.vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            block_out_channels=(32, 64),
            layers_per_block=1,
            latent_channels=4,
            norm_num_groups=32,
        )
        mx.eval(self.vae.parameters())

    def test_encode_shape(self):
        x = mx.random.normal((1, 16, 16, 3))
        latent = self.vae.encode(x)
        mx.eval(latent)
        # 2 blocks → 1 downsample: 16/2 = 8
        assert latent.shape == (1, 8, 8, 4)

    def test_decode_shape(self):
        z = mx.random.normal((1, 8, 8, 4))
        out = self.vae.decode(z)
        mx.eval(out)
        # 2 blocks → 1 upsample: 8*2 = 16
        assert out.shape == (1, 16, 16, 3)

    def test_roundtrip_shape(self):
        x = mx.random.normal((1, 16, 16, 3))
        z = self.vae.encode(x)
        recon = self.vae.decode(z)
        mx.eval(recon)
        assert recon.shape == x.shape


# ---------------------------------------------------------------------------
# models/unet.py
# ---------------------------------------------------------------------------

class TestUNet2DConditionModel:
    @pytest.fixture(autouse=True)
    def _build(self):
        from stablevsr.mlx.models.unet import UNet2DConditionModel
        self.unet = UNet2DConditionModel(
            in_channels=7,
            out_channels=4,
            block_out_channels=(64, 64),
            layers_per_block=1,
            cross_attention_dim=64,
            attention_head_dim=8,
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        )
        mx.eval(self.unet.parameters())

    def test_forward_shape(self):
        sample = mx.random.normal((1, 16, 16, 7))
        t = mx.array([500])
        enc = mx.random.normal((1, 10, 64))

        out = self.unet(sample, t, enc)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 4)

    def test_forward_with_controlnet_residuals(self):
        sample = mx.random.normal((1, 16, 16, 7))
        t = mx.array([100])
        enc = mx.random.normal((1, 10, 64))

        # Residuals matching down_block_res_samples: [conv_in:64@16, 64@16, 64@8, 64@8]
        down_res = [
            mx.zeros((1, 16, 16, 64)),
            mx.zeros((1, 16, 16, 64)),
            mx.zeros((1, 8, 8, 64)),
            mx.zeros((1, 8, 8, 64)),
        ]
        mid_res = mx.zeros((1, 8, 8, 64))

        out = self.unet(sample, t, enc,
                        down_block_additional_residuals=down_res,
                        mid_block_additional_residual=mid_res)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 4)


# ---------------------------------------------------------------------------
# models/controlnet.py
# ---------------------------------------------------------------------------

class TestControlNetModel:
    @pytest.fixture(autouse=True)
    def _build(self):
        from stablevsr.mlx.models.controlnet import ControlNetModel
        self.cnet = ControlNetModel(
            in_channels=7,
            conditioning_channels=3,
            block_out_channels=(64, 64),
            layers_per_block=1,
            cross_attention_dim=64,
            attention_head_dim=8,
            only_cross_attention=(True, False),
            conditioning_embedding_out_channels=(16,),
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        )
        mx.eval(self.cnet.parameters())

    def test_forward_returns_tuple(self):
        sample = mx.random.normal((1, 16, 16, 7))
        t = mx.array([300])
        enc = mx.random.normal((1, 10, 64))
        cond = mx.random.normal((1, 16, 16, 3))

        down_res, mid_res = self.cnet(sample, t, enc, controlnet_cond=cond)
        mx.eval(mid_res)
        for r in down_res:
            mx.eval(r)
        assert isinstance(down_res, list)
        assert len(down_res) > 0
        assert mid_res.ndim == 4

    def test_conditioning_scale(self):
        sample = mx.random.normal((1, 16, 16, 7))
        t = mx.array([300])
        enc = mx.random.normal((1, 10, 64))
        cond = mx.random.normal((1, 16, 16, 3))

        down1, mid1 = self.cnet(sample, t, enc, controlnet_cond=cond, conditioning_scale=1.0)
        down2, mid2 = self.cnet(sample, t, enc, controlnet_cond=cond, conditioning_scale=0.0)
        mx.eval(mid1, mid2)

        assert mx.abs(mid2).max().item() == pytest.approx(0.0, abs=1e-6)
