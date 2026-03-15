"""
RetinexTapetumRGB model.

This variant uses a classical Retinex decomposition instead of a learned DecomNet:
    I = R * L
where
    I : input image,
    R : reflectance,
    L : illumination.

Then a Tapetum-inspired RGB attention block predicts a residual amplification map on
illumination. The final enhanced illumination is:
    L_t = L * (1 + lambda * T_rgb)

The enhanced image becomes:
    I_enh = clamp(R * L_t, 0, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import RETINEX_KERNEL_SIZE, RETINEX_SIGMA


def gaussian_kernel(kernel_size=15, sigma=3.0, channels=3, device="cpu"):
    """Create a depthwise Gaussian kernel used for illumination estimation."""
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel


def gaussian_blur(x, kernel_size=15, sigma=3.0):
    """Apply depthwise Gaussian blur to estimate smooth illumination."""
    c = x.shape[1]
    kernel = gaussian_kernel(kernel_size=kernel_size, sigma=sigma, channels=c, device=x.device)
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    out = F.conv2d(x, kernel, groups=c)
    return out


def retinex_decompose(x):
    """
    Classical Retinex decomposition.

    Illumination is approximated by a Gaussian-smoothed version of the image.
    Reflectance is computed by element-wise division.
    """
    L = gaussian_blur(x, kernel_size=RETINEX_KERNEL_SIZE, sigma=RETINEX_SIGMA)
    R = x / (L + 1e-6)
    R = torch.clamp(R, 0.0, 5.0)
    return R, L


class ConvBlock(nn.Module):
    """Two-layer convolutional block used throughout the U-Net-like attention module."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TapetumAttentionNet(nn.Module):
    """
    RGB Tapetum attention module.

    The network predicts a 3-channel attention map from the illumination image.
    A learnable channel gate allows the model to amplify R/G/B channels differently.
    This is the RGB-specific part of the model.
    """

    def __init__(self, in_ch=3, base=32, channel_gate_scale=0.25):
        super().__init__()
        self.channel_gate_scale = channel_gate_scale

        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out_conv = nn.Conv2d(base, 3, 1)

        # alpha is converted into a bounded per-channel gate with tanh.
        self.alpha = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def forward(self, L):
        e1 = self.enc1(L)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        raw = self.out_conv(d1)

        # Base attention is bounded to [0,1] and suppressed where illumination is already high.
        T_base = torch.sigmoid(raw) * (1.0 - L)

        # Learnable channel gate around 1.0, bounded by tanh.
        channel_gate = 1.0 + self.channel_gate_scale * torch.tanh(self.alpha)
        gate_map = channel_gate.view(1, 3, 1, 1)
        T_rgb = T_base * gate_map

        return T_rgb, channel_gate, T_base


class RetinexTapetumRGBModel(nn.Module):
    """
    Full RetinexTapetumRGB model.

    Differences from the DecomNet-based variants:
    - Reflectance/illumination are produced by fixed Gaussian Retinex decomposition.
    - There is no learned decomposition loss.
    - Enhancement is driven only by Tapetum RGB attention over illumination.
    """

    def __init__(self, base=32, lambda_init=0.0, channel_gate_scale=0.25, lambda_max=1.68):
        super().__init__()
        self.tapetum_net = TapetumAttentionNet(
            in_ch=3,
            base=base,
            channel_gate_scale=channel_gate_scale,
        )

        # Raw parameter. Real lambda is obtained with a bounded sigmoid in forward.
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))
        self.lambda_max = lambda_max

    def forward(self, x):
        # Step 1: classical Retinex decomposition.
        R, L = retinex_decompose(x)

        # Step 2: Tapetum-inspired RGB attention on illumination.
        T_rgb, channel_gate, T_base = self.tapetum_net(L)

        # Step 3: bounded lambda. This is more stable than softplus + manual clamp.
        lam = self.lambda_max * torch.sigmoid(self.lambda_param)

        # Step 4: enhance illumination.
        L_t = L * (1.0 + lam * T_rgb)

        # Step 5: reconstruct enhanced image.
        I_enh = torch.clamp(R * L_t, 0.0, 1.0)

        return {
            "enhanced": I_enh,
            "reflectance": R,
            "illumination": L,
            "tapetum_attention": T_rgb,
            "tapetum_attention_base": T_base,
            "illumination_t": L_t,
            "lambda": lam,
            "channel_gate": channel_gate,
        }
