"""Inference with source/reference/emotion only; accepts no caption or edited target."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

from .appearance import edge_map
from .data import pixels, EMOTIONS
from .model import Editor
from .train import load_small


def main():
    from transformers import AutoModelForImageSegmentation
    p = argparse.ArgumentParser()
    for key in ('config', 'checkpoint', 'source', 'reference', 'emotion', 'output'):
        p.add_argument('--' + key, required=True, choices=EMOTIONS if key == 'emotion' else None)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    device = torch.device('cuda')
    model = Editor(config, device)
    load_small(model, torch.load(args.checkpoint, map_location='cpu', weights_only=False)['model'])
    model.eval()
    segmentation = AutoModelForImageSegmentation.from_pretrained(config['segmentation_model'], trust_remote_code=True).to(device).eval()
    with torch.no_grad():
        source = pixels(args.source, config['resolution'])[None].to(device)
        reference = pixels(args.reference, config['resolution'])[None].to(device)
        latent = model.vae.encode(source).latent_dist.mode() * model.vae.config.scaling_factor
        x = F.interpolate((source + 1) / 2, (512, 512), mode='bilinear', align_corners=False)
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            mask = F.interpolate(segmentation((x - mean) / std)[-1].sigmoid().float(), (64, 64), mode='bilinear', align_corners=False)
        area = float((mask > .5).float().mean())
        confidence = float((2 * (mask - .5).abs()).mean()) if .01 < area < .9 else 0.
        batch = dict(source=source, reference=reference, latent=latent, dino=model.dino(source),
                     reference_dino=model.dino(reference).mean(1), mask=mask,
                     edge=F.adaptive_avg_pool2d(edge_map(source), 64), confidence=torch.tensor([confidence], device=device),
                     label=torch.tensor([EMOTIONS.index(args.emotion)], device=device))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            image, _, _ = model.render(batch, args.steps, args.seed)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(((image[0].float().cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()).save(destination)


if __name__ == '__main__':
    main()
