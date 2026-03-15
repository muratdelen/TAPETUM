"""
Loss functions for RetinexTapetumRGB.

This model does not use learned decomposition, so the loss focuses on:
- image reconstruction quality,
- structural similarity,
- color consistency,
- regularization of the Tapetum attention,
- regularization of RGB channel gates,
- smoothness of the enhanced illumination map.
"""

import torch
import torch.nn.functional as F

from config import W_L1, W_SSIM, W_COLOR, W_ATTN, W_GATE, W_SMOOTH_ENH


def charbonnier_loss(pred, target, eps=1e-3):
    """Robust L1-like reconstruction loss."""
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def create_gaussian_window(window_size, channel, device):
    """Create Gaussian window for SSIM computation."""
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim_loss(pred, target, window_size=11):
    """Differentiable SSIM loss implemented with depthwise convolutions."""
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
    return 1.0 - ssim_map.mean()


def color_consistency_loss(pred, target):
    """Keep average RGB color close to the target image."""
    pred_mean = pred.mean(dim=[2, 3])
    target_mean = target.mean(dim=[2, 3])
    return F.l1_loss(pred_mean, target_mean)


def attention_regularization(T):
    """Keep attention sparse and controlled so enhancement does not explode."""
    return torch.mean(torch.abs(T)) + 0.03 * torch.mean(T ** 2)


def gate_regularization(channel_gate):
    """Encourage RGB gates to stay close to 1.0 unless learning needs otherwise."""
    target = torch.ones_like(channel_gate)
    return F.l1_loss(channel_gate, target)


def gradient_x(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def illumination_smoothness_loss(L, x):
    """
    Edge-aware smoothness for enhanced illumination.

    The idea is simple:
    - smooth the illumination in flat regions,
    - preserve edges where the input image has strong gradients.
    """
    gray_x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    gray_L = 0.299 * L[:, 0:1] + 0.587 * L[:, 1:2] + 0.114 * L[:, 2:3]

    L_dx = gradient_x(gray_L)
    L_dy = gradient_y(gray_L)
    x_dx = gradient_x(gray_x)
    x_dy = gradient_y(gray_x)

    weight_x = torch.exp(-10.0 * torch.abs(x_dx))
    weight_y = torch.exp(-10.0 * torch.abs(x_dy))

    loss_x = torch.mean(torch.abs(L_dx) * weight_x)
    loss_y = torch.mean(torch.abs(L_dy) * weight_y)
    return loss_x + loss_y


def total_loss_fn(output, low, gt):
    """Compute total training loss and return scalar logs for progress reporting."""
    pred = output["enhanced"]
    T = output["tapetum_attention"]
    channel_gate = output["channel_gate"]
    L_t = output["illumination_t"]

    l1 = charbonnier_loss(pred, gt)
    ssim_l = ssim_loss(pred, gt)
    color_l = color_consistency_loss(pred, gt)
    attn_l = attention_regularization(T)
    gate_l = gate_regularization(channel_gate)
    smooth_enh = illumination_smoothness_loss(L_t, low)

    total = (
        W_L1 * l1
        + W_SSIM * ssim_l
        + W_COLOR * color_l
        + W_ATTN * attn_l
        + W_GATE * gate_l
        + W_SMOOTH_ENH * smooth_enh
    )

    logs = {
        "total": total.item(),
        "l1": l1.item(),
        "ssim": ssim_l.item(),
        "color": color_l.item(),
        "attn": attn_l.item(),
        "gate": gate_l.item(),
        "smooth_enh": smooth_enh.item(),
    }
    return total, logs
