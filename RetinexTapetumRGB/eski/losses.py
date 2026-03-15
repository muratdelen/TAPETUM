import torch
import torch.nn.functional as F

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
    return T.mean()

def gate_regularization(channel_gate):
    target = torch.ones_like(channel_gate)
    return F.l1_loss(channel_gate, target)

def total_loss_fn(output, gt, w_l1=1.0, w_ssim=0.5, w_color=0.1, w_attn=0.01, w_gate=0.002):
    pred = output["enhanced"]
    T = output["tapetum_attention"]
    channel_gate = output["channel_gate"]

    l1 = charbonnier_loss(pred, gt)
    ssim_l = ssim_loss(pred, gt)
    color_l = color_consistency_loss(pred, gt)
    attn_l = attention_regularization(T)
    gate_l = gate_regularization(channel_gate)

    total = w_l1 * l1 + w_ssim * ssim_l + w_color * color_l + w_attn * attn_l + w_gate * gate_l

    logs = {
        "total": total.item(),
        "l1": l1.item(),
        "ssim": ssim_l.item(),
        "color": color_l.item(),
        "attn": attn_l.item(),
        "gate": gate_l.item(),
    }
    return total, logs