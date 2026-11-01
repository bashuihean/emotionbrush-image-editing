"""Shared differentiable frozen DINOv2 encoder."""
import torch
from torch import nn
import torch.nn.functional as F

class DinoEncoder(nn.Module):
    """Frozen DINOv2, identical differentiable preprocessing for cache/inference/teacher."""
    def __init__(self, model_id, size=224):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_id).eval().requires_grad_(False)
        self.size = size
        self.feature_dim = self.encoder.config.hidden_size
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, images):
        pixels = F.interpolate(((images.float()+1)/2).clamp(0, 1), (self.size, self.size),
                               mode='bicubic', align_corners=False)
        result = self.encoder(pixel_values=(pixels-self.mean)/self.std).last_hidden_state
        return result[:, 1:].float()
