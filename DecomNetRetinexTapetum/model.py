"""
Model definitions for DecomNetRetinexTapetum.

Architecture summary:
    Input image I
        -> DecomNet produces reflectance R and illumination L
        -> TapetumAttentionNet predicts attention T from illumination L
        -> Enhanced illumination is computed as:
               L_t = L * (1 + lambda * T)
        -> Final enhanced image is reconstructed by:
               I_enh = R * L_t

This version uses a single Tapetum attention branch.
Unlike the RGB-gated variant, there is no per-channel gate here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two-layer convolutional block used repeatedly in encoder/decoder stages."""

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


class DecomNet(nn.Module):
    """
    Retinex-style decomposition network.

    Input:
        RGB image x in [0, 1]

    Output:
        R : reflectance map [B, 3, H, W]
        L : illumination map [B, 3, H, W]
    """

    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.head = nn.Conv2d(in_ch, base, 3, 1, 1)
        self.body = nn.Sequential(
            ConvBlock(base, base),
            ConvBlock(base, base),
            ConvBlock(base, base),
        )
        self.r_out = nn.Conv2d(base, 3, 3, 1, 1)
        self.l_out = nn.Conv2d(base, 3, 3, 1, 1)

    def forward(self, x):
        f = F.relu(self.head(x), inplace=True)
        f = self.body(f)
        R = torch.sigmoid(self.r_out(f))
        L = torch.sigmoid(self.l_out(f))
        return R, L


class TapetumAttentionNet(nn.Module):
    """
    U-Net-like attention network operating on illumination.

    The network predicts a Tapetum-inspired enhancement attention map T.
    T is bounded by sigmoid and modulated by (1 - L), meaning stronger
    enhancement is encouraged in darker illumination regions.
    """

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
        T = torch.sigmoid(raw) * (1.0 - L)
        return T


class DecomNetRetinexTapetumModel(nn.Module):
    """
    Full model combining Retinex decomposition and Tapetum enhancement.

    Forward output keys are standardized so logs and test scripts stay aligned
    with the other model variants in the project.
    """

    def __init__(self, base=32, lambda_init=0.0, lambda_max=1.68):
        super().__init__()
        self.decom_net = DecomNet(in_ch=3, base=base)
        self.tapetum_net = TapetumAttentionNet(in_ch=3, base=base)

        # Bounded scalar enhancement strength.
        # Final lambda used in forward pass is:
        #   lambda = lambda_max * sigmoid(lambda_param)
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))
        self.lambda_max = lambda_max

    def forward(self, low, high=None):
        # Decompose low-light input into reflectance and illumination.
        R_low, L_low = self.decom_net(low)

        # Predict Tapetum enhancement attention from low-light illumination.
        T = self.tapetum_net(L_low)

        # Reference-aligned bounded lambda.
        lam = self.lambda_max * torch.sigmoid(self.lambda_param)

        # Enhance illumination and reconstruct the final image.
        L_t = L_low * (1.0 + lam * T)
        I_enh = torch.clamp(R_low * L_t, 0.0, 1.0)

        out = {
            "enhanced": I_enh,
            "reflectance_low": R_low,
            "illumination_low": L_low,
            "tapetum_attention": T,
            "illumination_t": L_t,
            "lambda": lam,
        }

        # If high-light reference is given, decompose it too so decomposition
        # losses can compare low/high reflectance and reconstruction behavior.
        if high is not None:
            R_high, L_high = self.decom_net(high)
            recon_low = torch.clamp(R_low * L_low, 0.0, 1.0)
            recon_high = torch.clamp(R_high * L_high, 0.0, 1.0)

            out.update({
                "reflectance_high": R_high,
                "illumination_high": L_high,
                "recon_low": recon_low,
                "recon_high": recon_high,
            })

        return out
