import torch
import torch.nn.functional as F

def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))

def total_loss_fn(output, gt):

    pred = output["enhanced"]
    T = output["tapetum_attention"]

    l1 = charbonnier_loss(pred, gt)

    attn = T.mean()

    total = l1 + 0.01 * attn

    logs = {
        "total": total.item(),
        "l1": l1.item(),
        "attn": attn.item(),
    }

    return total, logs