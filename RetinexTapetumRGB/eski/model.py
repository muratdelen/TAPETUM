import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_kernel(kernel_size=15, sigma=3.0, channels=3, device="cpu"):
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel

def gaussian_blur(x, kernel_size=15, sigma=3.0):
    c = x.shape[1]
    kernel = gaussian_kernel(kernel_size=kernel_size, sigma=sigma, channels=c, device=x.device)
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    out = F.conv2d(x, kernel, groups=c)
    return out

def retinex_decompose(x):
    L = gaussian_blur(x, kernel_size=21, sigma=5.0)
    R = x / (L + 1e-6)
    R = torch.clamp(R, 0.0, 5.0)
    return R, L

class ConvBlock(nn.Module):
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
        T_base = torch.sigmoid(raw) * (1.0 - L)

        channel_gate = 1.0 + self.channel_gate_scale * torch.tanh(self.alpha)
        gate_map = channel_gate.view(1, 3, 1, 1)
        T_rgb = T_base * gate_map

        return T_rgb, channel_gate, T_base

class RetinexTapetumRGBModel(nn.Module):
    def __init__(self, base=32, lambda_init=1.0, channel_gate_scale=0.25):
        super().__init__()
        self.tapetum_net = TapetumAttentionNet(
            in_ch=3,
            base=base,
            channel_gate_scale=channel_gate_scale
        )
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))

    def forward(self, x):
        R, L = retinex_decompose(x)

        T_rgb, channel_gate, T_base = self.tapetum_net(L)
        lam = F.softplus(self.lambda_param)

        L_t = L * (1.0 + lam * T_rgb)

        I_enh = R * L_t
        I_enh = torch.clamp(I_enh, 0.0, 1.0)

        return {
            "enhanced": I_enh,
            "reflectance": R,
            "illumination": L,
            "tapetum_attention": T_rgb,
            "tapetum_attention_base": T_base,
            "illumination_t": L_t,
            "lambda": lam,
            "channel_gate": channel_gate
        }