"""Two-GPU frozen-feature preparation. Segmentation masks are not paired edit truth."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm.auto import tqdm

from .appearance import FrozenAppearance, edge_map
from .data import audit, cache_path, pixels


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    from transformers import CLIPTokenizer, CLIPTextModel, AutoModelForImageSegmentation
    from diffusers import AutoencoderKL
    from ..spatial_model import DinoEncoder
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl')
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    parts = audit(config)
    from .data import EMOTIONS
    if args.limit:
        for part in ('train', 'val'):
            selected = []
            for emotion in EMOTIONS:
                selected += [r for r in parts[part] if r['target_emotion'] == emotion][:max(1, args.limit // 8)]
            parts[part] = selected
    sources = {r['image'] for part in ('train', 'val') for r in parts[part]}
    paths = sorted(sources | {r['reference_image'] for part in ('train', 'val') for r in parts[part]})
    metadata = dict(schema='reference_v16_unpaired', resolution=config['resolution'],
                    crop='EXIF transpose, PIL center fit bicubic; no random augment',
                    vae=config['base_model'], dino=config['dino_model'],
                    segmentation_revision=config['segmentation_revision'], mask_resolution=64,
                    appearance_a=fingerprint(config['appearance_a_weights']),
                    appearance_b=fingerprint(config['appearance_b_weights']))
    for key, root, subfolder in (('vae_sha', config['base_model'], 'vae'),
                                 ('text_sha', config['base_model'], 'text_encoder'),
                                 ('dino_sha', config['dino_model'], '')):
        directory = Path(root) / subfolder
        files = sorted(directory.glob('*.safetensors')) or sorted(directory.glob('*.bin'))
        if not files:
            raise FileNotFoundError('Cannot fingerprint ' + str(directory))
        metadata[key] = [fingerprint(f) for f in files]
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
    config['cache_signature'] = signature
    if rank == 0:
        config_path.write_text(json.dumps(config, indent=2))
        cache_root = Path(config['cache_dir']) / signature
        cache_root.mkdir(parents=True, exist_ok=True)
        (cache_root / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        text = CLIPTextModel.from_pretrained(config['base_model'], subfolder='text_encoder').to(device).eval()
        tokenizer = CLIPTokenizer.from_pretrained(config['base_model'], subfolder='tokenizer')
        def encode(strings):
            ids = tokenizer(strings, padding='max_length', truncation=True,
                            max_length=tokenizer.model_max_length, return_tensors='pt').input_ids.to(device)
            return text(ids).last_hidden_state.cpu()
        with torch.no_grad():
            torch.save(dict(emotion=encode(list(EMOTIONS)), empty=encode(['']),
                            classes=EMOTIONS), config['text_cache'])
        del text
        print(json.dumps(dict(stage='cache', images=len(paths), source_images=len(sources),
                              split_counts={k: len(v) for k, v in parts.items()}, signature=signature)), flush=True)
    if world > 1:
        dist.barrier()
    dino = DinoEncoder(config['dino_model']).to(device).eval()
    vae = AutoencoderKL.from_pretrained(config['base_model'], subfolder='vae').to(device).requires_grad_(False).eval()
    appearance_a = FrozenAppearance('resnet18', config['appearance_a_weights']).to(device)
    appearance_b = FrozenAppearance('resnet34', config['appearance_b_weights']).to(device)
    # Explicit pretrained automatic foreground model, revision pinned in config.
    segment = AutoModelForImageSegmentation.from_pretrained(config['segmentation_model'], trust_remote_code=True).to(device).requires_grad_(False).eval()
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    owned = paths[rank::world]
    batch_size = config.get('cache_batch_size', 8)
    for start in tqdm(range(0, len(owned), batch_size), desc='NPZ GPU ' + str(local), dynamic_ncols=True):
        todo = [path for path in owned[start:start + batch_size] if not cache_path(config, path).exists()]
        if not todo:
            continue
        images = torch.stack([pixels(path, config['resolution']) for path in todo]).to(device)
        with torch.no_grad():
            dino_features = dino(images)
            features_a, features_b = appearance_a(images), appearance_b(images)
            source_ids = [i for i, path in enumerate(todo) if path in sources]
            source_results = {}
            if source_ids:
                source_images = images[source_ids]
                latent = vae.encode(source_images).latent_dist.mode() * vae.config.scaling_factor
                seg_input = F.interpolate((source_images + 1) / 2, (512, 512), mode='bilinear', align_corners=False)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = segment((seg_input - mean) / std)[-1].sigmoid().float()
                masks = F.interpolate(prediction, (64, 64), mode='bilinear', align_corners=False)
                edges = F.adaptive_avg_pool2d(edge_map(source_images), 64)
                for j, i in enumerate(source_ids):
                    mask = masks[j]
                    area = float((mask > .5).float().mean())
                    confidence = float((2 * (mask - .5).abs()).mean()) if .01 < area < .90 else 0.
                    source_results[i] = dict(latent=latent[j], dino=dino_features[i], mask=mask,
                                             edge=edges[j], confidence=torch.tensor(confidence))
            for i, path in enumerate(todo):
                values = dict(dino_global=dino_features[i].mean(0), appearance_a=features_a[i],
                              appearance_b=features_b[i], **source_results.get(i, {}))
                destination = cache_path(config, path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, **{k: v.detach().cpu().numpy().astype(np.float16) for k, v in values.items()})
                os.replace(temporary, destination)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        print('CACHE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
