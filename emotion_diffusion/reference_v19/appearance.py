"""Differentiable appearance statistics with independent frozen CNN evaluators."""
import torch
from torch import nn
import torch.nn.functional as F


def lab(image):
    rgb = ((image.float() + 1) / 2).clamp(0, 1)
    linear = torch.where(rgb > .04045, ((rgb + .055) / 1.055).pow(2.4), rgb / 12.92)
    matrix = rgb.new_tensor([[.4124564, .3575761, .1804375],
                             [.2126729, .7151522, .0721750],
                             [.0193339, .1191920, .9503041]])
    xyz = torch.einsum('ij,bjhw->bihw', matrix, linear)
    xyz = xyz / rgb.new_tensor([.95047, 1., 1.08883])[None, :, None, None]
    f = torch.where(xyz > .008856, xyz.clamp_min(1e-6).pow(1 / 3), 7.787 * xyz + 16 / 116)
    return torch.stack(((116 * f[:, 1] - 16) / 100,
                        500 * (f[:, 0] - f[:, 1]) / 128,
                        200 * (f[:, 1] - f[:, 2]) / 128), 1)


def color_stats(image):
    x = lab(image)
    flat = x.flatten(2)
    bins = x.new_tensor([0., .2, .4, .6, .8, 1.])
    hist = torch.exp(-((flat[:, :, :, None] - bins) / .16).square()).mean(2)
    hist = hist / hist.sum(-1, keepdim=True).clamp_min(1e-6)
    return torch.cat((flat.mean(2), flat.std(2, unbiased=False), hist.flatten(1),
                      F.adaptive_avg_pool2d(x, 4).flatten(1)), 1)


def edge_map(image):
    gray = ((image.float() + 1) / 2).mean(1, keepdim=True)
    k = image.new_tensor([[-1., 0, 1], [-2., 0, 2], [-1., 0, 1]])[None, None].float()
    dx = F.conv2d(gray, k, padding=1)
    dy = F.conv2d(gray, k.transpose(2, 3), padding=1)
    edge = (dx.square() + dy.square() + 1e-8).sqrt()
    return edge / edge.amax((2, 3), keepdim=True).clamp_min(.01)


class FrozenAppearance(nn.Module):
    """Pretrained CNN stats; no global object logits or flattened scene layout in texture."""
    feature_dim = 1992  # 2*(64+128+256+512) + 72 Lab stats.

    def __init__(self, backbone='resnet18', weights_path=None):
        super().__init__()
        from torchvision import models
        self.cnn = getattr(models, backbone)(weights=None)
        if not weights_path:
            raise ValueError('Explicit pretrained appearance weights required')
        self.cnn.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        self.cnn.fc = nn.Identity()
        self.requires_grad_(False).eval()
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def maps(self, image, size=128):
        x = F.interpolate(((image.float() + 1) / 2).clamp(0, 1), (size, size), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        m = self.cnn
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        result = []
        for layer in (m.layer1, m.layer2, m.layer3, m.layer4):
            x = layer(x)
            result.append(x)
        return result

    def forward(self, image):
        stats = []
        for x in self.maps(image):
            stats.extend((x.mean((2, 3)), x.flatten(2).std(-1, unbiased=False)))
        return torch.cat((*stats, color_stats(image)), 1)

    def style(self, image, texture_mask=None):
        # Multiscale, position-independent statistics; reference patch positions are not targets.
        texture = []
        for size in (96, 160):
            maps = self.maps(image, size)
            for x in maps[:2]:
                flat = x.flatten(2)
                if texture_mask is None:
                    weight = torch.ones_like(flat[:, :1])
                else:
                    weight = F.interpolate(texture_mask.float(), x.shape[-2:], mode='bilinear', align_corners=False).flatten(2)
                    # Very small/empty background: skip strong spatial selection.
                    weight = torch.where(weight.mean(-1, keepdim=True) > .01, weight, torch.ones_like(weight))
                mass = weight.sum(-1).clamp_min(1e-6)
                mean = (flat * weight).sum(-1) / mass
                variance = ((flat - mean[..., None]).square() * weight).sum(-1) / mass
                texture.extend((mean, (variance + 1e-8).sqrt()))
                # Compressed Gram correlation, retaining only first 16 pretrained channels.
                compact = flat[:, :16]
                gram = (compact * weight) @ compact.transpose(1, 2) / mass[..., None]
                texture.append(gram.flatten(1))
        atmosphere = F.adaptive_avg_pool2d(self.maps(image)[0], 4)
        return torch.cat(texture, 1), color_stats(image), atmosphere


class EmotionHead(nn.Module):
    def __init__(self, dim=1992, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(.2), nn.Linear(hidden, 8))

    def forward(self, features):
        return self.net(features.float())
