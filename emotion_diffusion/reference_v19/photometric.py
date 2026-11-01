"""Bounded pointwise colour/contrast and source-aligned smooth lighting. No warp."""
import torch
from torch import nn
import torch.nn.functional as F
from .appearance import lab


def distribution_stats(image):
    x = lab(image).flatten(2)
    centers = x.new_tensor([[0., .2, .4, .6, .8, 1.],
                            [-.8, -.48, -.16, .16, .48, .8],
                            [-.8, -.48, -.16, .16, .48, .8]])
    hist = torch.exp(-((x[..., None] - centers[None, :, None]) / .16).square()).mean(2)
    hist = hist / hist.sum(-1, keepdim=True).clamp_min(1e-6)
    return torch.cat((x.mean(-1), x.std(-1, unbiased=False), hist.flatten(1)), 1)


class PhotometricBranch(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(dim + 48, 64), nn.SiLU())
        self.global_head = nn.Linear(64, 9)
        self.light_head = nn.Sequential(nn.Conv2d(67, 32, 3, padding=1), nn.SiLU(),
                                        nn.Conv2d(32, 2, 3, padding=1))
        nn.init.zeros_(self.global_head.weight); nn.init.zeros_(self.global_head.bias)
        nn.init.zeros_(self.light_head[-1].weight); nn.init.zeros_(self.light_head[-1].bias)

    def forward(self, image, source, reference, tokens):
        rgb = ((image.float() + 1) / 2).clamp(0, 1)
        features = torch.cat((tokens.float().mean(1), distribution_stats(source),
                              distribution_stats(reference)), 1)
        cond = self.condition(features)
        params = self.global_head(cond).float().tanh()
        log_gain, bias, log_contrast = params.chunk(3, 1)
        log_gain = .45 * log_gain[:, :, None, None]
        bias = .10 * bias[:, :, None, None]
        contrast = (.30 * log_contrast[:, :, None, None]).exp()
        # Reference spatial layout is NOT copied: light field is aligned to SOURCE.
        low = F.adaptive_avg_pool2d((source.float() + 1) / 2, 8)
        field = self.light_head(torch.cat((low, cond[:, :, None, None].expand(-1, -1, 8, 8)), 1)).float().tanh()
        light = F.interpolate(field, rgb.shape[-2:], mode='bilinear', align_corners=False)
        center = rgb.mean((2, 3), keepdim=True)
        result = ((rgb - center) * contrast + center) * (log_gain + .25 * light[:, :1]).exp()
        result = (result + bias + .05 * light[:, 1:]).clamp(0, 1) * 2 - 1
        smooth = (field[:, :, 1:] - field[:, :, :-1]).square().mean() + (field[:, :, :, 1:] - field[:, :, :, :-1]).square().mean()
        regularizer = .01 * params.square().mean() + .01 * field.square().mean() + smooth
        return result, regularizer
