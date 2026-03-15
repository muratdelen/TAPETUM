"""RetinexTapetum model definition.

This variant uses classical Gaussian Retinex decomposition instead of a learned
DecomNet. The input image is decomposed into:
- reflectance R
- illumination L

A U-Net-like attention network then predicts a tapetum-inspired modulation map
from the illumination branch. The enhanced illumination is:
    L_t = L * (1 + lambda * T)

The final enhanced image is reconstructed as:
    I_enh = clamp(R * L_t, 0, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(kernel_size=15, sigma=3.0, channels=3, device="cpu"):
    """Create a depthwise 2D Gaussian kernel used for illumination smoothing."""
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel


def gaussian_blur(x, kernel_size=15, sigma=3.0):
    """Apply channel-wise Gaussian blur with reflection padding."""
    c = x.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, c, x.device)
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x, kernel, groups=c)


def retinex_decompose(x, kernel_size=21, sigma=5.0):
    """Classical Retinex decomposition.

    Illumination is approximated by a Gaussian-smoothed version of the input.
    Reflectance is computed by division.
    """
    L = gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)
    R = x / (L + 1e-6)
    R = torch.clamp(R, 0.0, 5.0)
    return R, L


class ConvBlock(nn.Module):
    """Simple double-convolution block used in the attention U-Net."""

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
    """U-Net-like module that predicts the tapetum attention map from L."""

    def __init__(self, in_ch=3, base=32):
        super().__init__()
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

    def forward(self, illumination):
        e1 = self.enc1(illumination)
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
        return torch.sigmoid(raw) * (1.0 - illumination)


class RetinexTapetumModel(nn.Module):
    """Classical Retinex + Tapetum enhancement model.

    lambda is bounded with a sigmoid to keep training stable and aligned with
    the standard used in the other standardized variants.
    """

    def __init__(
        self,
        base=32,
        lambda_init=0.0,
        lambda_max=1.68,
        gaussian_kernel_size=21,
        gaussian_sigma=5.0,
    ):
        super().__init__()
        self.tapetum_net = TapetumAttentionNet(3, base)
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))
        self.lambda_max = lambda_max
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma

    def forward(self, x):
        reflectance, illumination = retinex_decompose(
            x,
            kernel_size=self.gaussian_kernel_size,
            sigma=self.gaussian_sigma,
        )
        tapetum_attention = self.tapetum_net(illumination)
        lam = self.lambda_max * torch.sigmoid(self.lambda_param)
        illumination_t = illumination * (1.0 + lam * tapetum_attention)
        enhanced = torch.clamp(reflectance * illumination_t, 0.0, 1.0)

        return {
            "enhanced": enhanced,
            "reflectance_low": reflectance,
            "illumination_low": illumination,
            "tapetum_attention": tapetum_attention,
            "illumination_t": illumination_t,
            "lambda": lam,
        }
