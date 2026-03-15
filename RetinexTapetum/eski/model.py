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
    kernel = gaussian_kernel(kernel_size, sigma, c, x.device)
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x, kernel, groups=c)

def retinex_decompose(x):
    L = gaussian_blur(x, 21, 5.0)
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
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        raw = self.out_conv(d1)

        T = torch.sigmoid(raw) * (1.0 - L)

        return T

class RetinexTapetumModel(nn.Module):

    def __init__(self, base=32, lambda_init=1.0):
        super().__init__()
        self.tapetum_net = TapetumAttentionNet(3, base)
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))

    def forward(self, x):

        R, L = retinex_decompose(x)

        T = self.tapetum_net(L)

        lam = torch.nn.functional.softplus(self.lambda_param)

        L_t = L * (1.0 + lam * T)

        I_enh = torch.clamp(R * L_t, 0.0, 1.0)

        return {
            "enhanced": I_enh,
            "tapetum_attention": T,
            "lambda": lam
        }