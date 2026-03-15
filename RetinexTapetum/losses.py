"""Loss definitions for RetinexTapetum.

This standardized version uses a richer loss than the original minimal code so
that the training behaviour is more comparable to the other TAPETUM variants.
"""

import torch
import torch.nn.functional as F
from config import W_L1, W_SSIM, W_COLOR, W_ATTN, W_SMOOTH_ENH


def charbonnier_loss(pred, target, eps=1e-3):
    """Robust L1-like reconstruction loss."""
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


def create_gaussian_window(window_size, channel, device):
    """Utility window for SSIM computation."""
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim_loss(pred, target, window_size=11):
    """SSIM-based perceptual structure loss."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
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

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    )
    return 1.0 - ssim_map.mean()


def color_consistency_loss(pred, target):
    """Match global RGB channel means between output and target."""
    pred_mean = pred.mean(dim=[2, 3])
    target_mean = target.mean(dim=[2, 3])
    return F.l1_loss(pred_mean, target_mean)


def attention_regularization(attention):
    """Discourage overly strong or noisy attention maps."""
    return torch.mean(torch.abs(attention)) + 0.03 * torch.mean(attention ** 2)


def gradient_x(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def gradient_y(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def illumination_smoothness_loss(illumination, guide_image):
    """Edge-aware smoothing for the enhanced illumination map."""
    gray_x = 0.299 * guide_image[:, 0:1] + 0.587 * guide_image[:, 1:2] + 0.114 * guide_image[:, 2:3]
    gray_l = 0.299 * illumination[:, 0:1] + 0.587 * illumination[:, 1:2] + 0.114 * illumination[:, 2:3]

    l_dx = gradient_x(gray_l)
    l_dy = gradient_y(gray_l)
    x_dx = gradient_x(gray_x)
    x_dy = gradient_y(gray_x)

    weight_x = torch.exp(-10.0 * torch.abs(x_dx))
    weight_y = torch.exp(-10.0 * torch.abs(x_dy))

    loss_x = torch.mean(torch.abs(l_dx) * weight_x)
    loss_y = torch.mean(torch.abs(l_dy) * weight_y)
    return loss_x + loss_y


def total_loss_fn(output, low, gt):
    """Compute total training loss and per-component logs."""
    pred = output["enhanced"]
    attention = output["tapetum_attention"]
    illumination_t = output["illumination_t"]

    l1 = charbonnier_loss(pred, gt)
    ssim_l = ssim_loss(pred, gt)
    color_l = color_consistency_loss(pred, gt)
    attn_l = attention_regularization(attention)
    smooth_enh = illumination_smoothness_loss(illumination_t, low)

    total = (
        W_L1 * l1
        + W_SSIM * ssim_l
        + W_COLOR * color_l
        + W_ATTN * attn_l
        + W_SMOOTH_ENH * smooth_enh
    )

    logs = {
        "total": total.item(),
        "l1": l1.item(),
        "ssim": ssim_l.item(),
        "color": color_l.item(),
        "attn": attn_l.item(),
        "smooth_enh": smooth_enh.item(),
    }
    return total, logs
