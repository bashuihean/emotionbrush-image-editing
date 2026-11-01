"""Pretrain emotion prototypes using training-reference appearance and raw emotion text."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .data import cache_path, EMOTIONS
from .model import Conditions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.manual_seed(42)
    condition = Conditions().cuda()
    condition.requires_grad_(False)
    condition.brush.requires_grad_(True)
    condition.router.requires_grad_(True)
    condition.brush_projection.requires_grad_(True)
    split = json.loads((Path(config['output_dir']) / 'classifier_reference_split.json').read_text())
    sums = torch.zeros(8, 1992)
    counts = torch.zeros(8)
    for path, label in tqdm(split['train'], desc='Emotion brush reference prototypes'):
        with np.load(cache_path(config, path), allow_pickle=False) as cache:
            sums[label] += torch.from_numpy(cache['appearance_a'].astype(np.float32))
        counts[label] += 1
    visual = (sums / counts[:, None].clamp_min(1)).cuda()
    text = torch.load(config['text_cache'], weights_only=True)['emotion'].cuda().mean(1)
    labels = torch.arange(8, device='cuda')
    optimizer = torch.optim.AdamW([p for p in condition.parameters() if p.requires_grad], lr=3e-4)
    for _ in tqdm(range(5 if args.smoke else 300), desc='Pretrain emotion brush'):
        optimizer.zero_grad(set_to_none=True)
        prototypes = condition.brush.mean(1)
        projected = condition.brush_projection(visual)
        similarity = F.normalize(projected, dim=-1) @ F.normalize(prototypes, dim=-1).T / .1
        loss = F.cross_entropy(condition.router(text), labels) + F.cross_entropy(similarity, labels)
        loss += .1 * (1 - F.cosine_similarity(prototypes, text, dim=-1)).mean()
        loss.backward()
        optimizer.step()
    torch.save(condition.state_dict(), config['condition_init'])
    print('BRUSH READY', float(loss.detach()), flush=True)


if __name__ == '__main__':
    main()
