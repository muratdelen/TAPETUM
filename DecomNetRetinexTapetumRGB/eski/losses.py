import torch
import torch.nn.functional as F
from config import (
    W_L1, W_SSIM, W_COLOR, W_ATTN, W_GATE,
    W_RECON_LOW, W_RECON_HIGH, W_REFLECT, W_SMOOTH_LOW, W_SMOOTH_HIGH
)

def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))

def create_gaussian_window(window_size, channel, device):
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()

def ssim_loss(pred, target, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    channel = pred.size(1)
    window = create_gaussian_window(window_size, channel, pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8
    )
    return 1 - ssim_map.mean()

def color_consistency_loss(pred, target):
    pred_mean = pred.mean(dim=[2, 3])
    target_mean = target.mean(dim=[2, 3])
    return F.l1_loss(pred_mean, target_mean)

def attention_regularization(T):
    return torch.mean(torch.abs(T)) + 0.1 * torch.mean(T ** 2)

def gate_regularization(channel_gate):
    target = torch.ones_like(channel_gate)
    return F.l1_loss(channel_gate, target)

def gradient_x(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def illumination_smoothness_loss(L, x):
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    L_gray = 0.299 * L[:, 0:1] + 0.587 * L[:, 1:2] + 0.114 * L[:, 2:3]

    L_dx = gradient_x(L_gray)
    L_dy = gradient_y(L_gray)
    g_dx = gradient_x(gray)
    g_dy = gradient_y(gray)

    weight_x = torch.exp(-10.0 * torch.abs(g_dx))
    weight_y = torch.exp(-10.0 * torch.abs(g_dy))

    loss_x = torch.mean(torch.abs(L_dx) * weight_x)
    loss_y = torch.mean(torch.abs(L_dy) * weight_y)

    return loss_x + loss_y

def decomposition_loss(output, low, high):
    recon_low = output["recon_low"]
    recon_high = output["recon_high"]
    R_low = output["reflectance_low"]
    R_high = output["reflectance_high"]
    L_low = output["illumination_low"]
    L_high = output["illumination_high"]

    loss_recon_low = charbonnier_loss(recon_low, low)
    loss_recon_high = charbonnier_loss(recon_high, high)
    loss_reflect = F.l1_loss(R_low, R_high)
    loss_smooth_low = illumination_smoothness_loss(L_low, low)
    loss_smooth_high = illumination_smoothness_loss(L_high, high)

    total = (
        W_RECON_LOW * loss_recon_low +
        W_RECON_HIGH * loss_recon_high +
        W_REFLECT * loss_reflect +
        W_SMOOTH_LOW * loss_smooth_low +
        W_SMOOTH_HIGH * loss_smooth_high
    )

    logs = {
        "decomp": total.item(),
        "recon_low": loss_recon_low.item(),
        "recon_high": loss_recon_high.item(),
        "reflect": loss_reflect.item(),
        "smooth_low": loss_smooth_low.item(),
        "smooth_high": loss_smooth_high.item(),
    }
    return total, logs

def total_loss_fn(output, low, gt):
    pred = output["enhanced"]
    T = output["tapetum_attention"]
    channel_gate = output["channel_gate"]

    l1 = charbonnier_loss(pred, gt)
    ssim_l = ssim_loss(pred, gt)
    color_l = color_consistency_loss(pred, gt)
    attn_l = attention_regularization(T)
    gate_l = gate_regularization(channel_gate)

    decomp_l, decomp_logs = decomposition_loss(output, low, gt)

    total = (
        W_L1 * l1 +
        W_SSIM * ssim_l +
        W_COLOR * color_l +
        W_ATTN * attn_l +
        W_GATE * gate_l +
        decomp_l
    )

    logs = {
        "total": total.item(),
        "l1": l1.item(),
        "ssim": ssim_l.item(),
        "color": color_l.item(),
        "attn": attn_l.item(),
        "gate": gate_l.item(),
        "decomp": decomp_logs["decomp"],
        "recon_low": decomp_logs["recon_low"],
        "recon_high": decomp_logs["recon_high"],
        "reflect": decomp_logs["reflect"],
        "smooth_low": decomp_logs["smooth_low"],
        "smooth_high": decomp_logs["smooth_high"],
    }
    return total, logs