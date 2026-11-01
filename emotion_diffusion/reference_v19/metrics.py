"""CLIP-I and legacy ESMI remain separate from the new fixed-B Emo-A."""
import torch
from torch import nn
import torch.nn.functional as F

EDITOR_CLASSES = ('amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness')


class ImageMetrics(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection
        from torchvision.models import resnet18
        self.clip = CLIPVisionModelWithProjection.from_pretrained(config['clip_model']).to(device).eval().requires_grad_(False)
        self.resnet = resnet18(weights=None, num_classes=8).to(device)
        state = torch.load(config['legacy_esmi_weights'], map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        self.resnet.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
        self.resnet.eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor([.48145466, .4578275, .40821073], device=device)[None, :, None, None])
        self.register_buffer('std', torch.tensor([.26862954, .26130258, .27577711], device=device)[None, :, None, None])

    @torch.no_grad()
    def clip_i(self, source, output):
        pixels = F.interpolate(((torch.cat((source, output)) + 1) / 2).clamp(0, 1), (224, 224), mode='bicubic', align_corners=False)
        features = self.clip(pixel_values=(pixels - self.mean) / self.std).image_embeds
        return float(F.cosine_similarity(features[:len(source)], features[len(source):]).mean())

    def esmi(self, source, output, label):
        # Upstream EmoEditor: RGB [0,1], no ImageNet normalization; source top-class CAM.
        source224 = F.interpolate(((source.float() + 1) / 2).clamp(0, 1), (224, 224), mode='bilinear', align_corners=False)
        output224 = F.interpolate(((output.float() + 1) / 2).clamp(0, 1), (224, 224), mode='bilinear', align_corners=False)
        capture = {}
        def hook(module, inputs, activation):
            capture['activation'] = activation
            activation.register_hook(lambda gradient: capture.update(gradient=gradient))
        handle = self.resnet.layer4[-1].register_forward_hook(hook)
        try:
            with torch.enable_grad(), torch.autocast('cuda', enabled=False):
                x = source224.detach().requires_grad_(True)
                logits = self.resnet(x)
                logits[0, logits.argmax(-1).item()].backward()
            weights = capture['gradient'].detach().mean((2, 3), keepdim=True)
            cam = (weights * capture['activation'].detach()).sum(1, keepdim=True).relu()
            cam = (cam - cam.amin()) / (cam.amax() - cam.amin() + 1e-7)
            mask = (F.interpolate(cam, (224, 224), mode='bilinear', align_corners=False) > .5).float()
        finally:
            handle.remove()
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            source_p = logits.float().softmax(-1)[0]
            output_p = self.resnet(output224.float()).softmax(-1)[0]
            index = EDITOR_CLASSES.index(label)
            ks = -source_p[index].clamp_min(1e-10).log()
            ke = -output_p[index].clamp_min(1e-10).log()
            emotional = (ks - ke).clamp_min(0) / (ks + 1e-10)
            diff = (source224 - output224).abs()
            structural = (diff * mask).mean() / (diff.mean() + 1e-10)
        return dict(esmi_legacy=float((emotional + structural) / 2),
                    esmi_emotional=float(emotional), esmi_structural=float(structural))
