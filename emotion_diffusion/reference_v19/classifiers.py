"""Fit A/B on labelled customer references only, before generator training."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm

from .appearance import EmotionHead
from .data import audit, cache_path, EMOTIONS


def scores(labels, predictions):
    confusion = torch.zeros(8, 8, dtype=torch.long)
    for y, p in zip(labels, predictions):
        confusion[int(y), int(p)] += 1
    counts = confusion.sum(1)
    recall = confusion.diag().float() / counts.clamp_min(1)
    precision = confusion.diag().float() / confusion.sum(0).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return dict(accuracy=float(confusion.diag().sum() / confusion.sum().clamp_min(1)),
                balanced_accuracy=float(recall[counts > 0].mean()), macro_f1=float(f1[counts > 0].mean()),
                per_class_accuracy=recall.tolist(), counts=counts.tolist(), confusion=confusion.tolist())


def pixel_digest(path):
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def reference_sets(config):
    parts = audit(config)
    heldout = {r[k] for p in ('val', 'test') for r in parts[p] for k in ('image', 'reference_image')}
    heldout_hashes = {pixel_digest(path) for path in tqdm(sorted(heldout), desc='Classifier heldout pixel audit')}
    seen, split, conflicts = {}, {'train': [], 'val': []}, 0
    for part in ('train', 'val'):
        paths = {}
        for r in parts[part]:
            path = r['reference_image']
            label = EMOTIONS.index(r['target_emotion'])
            if path in paths and paths[path] != label:
                raise ValueError('Same reference has contradictory labels: ' + path)
            paths[path] = label
        for path, label in tqdm(sorted(paths.items()), desc='Deduplicate references ' + part):
            digest = pixel_digest(path)
            if part == 'train' and digest in heldout_hashes:
                raise ValueError('Train reference duplicates heldout image: ' + path)
            if digest in seen:
                if seen[digest][0] != part:
                    raise ValueError('Reference pixel leakage')
                if seen[digest][1] != label:
                    conflicts += 1
                continue
            seen[digest] = (part, label)
            split[part].append((path, label))
    if conflicts:
        raise ValueError('Contradictory duplicated reference labels: ' + str(conflicts))
    return split


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    rank = int(os.environ.get('LOCAL_RANK', 0))
    name = 'a' if rank == 0 else 'b'
    device = torch.device('cuda', rank)
    torch.cuda.set_device(rank)
    torch.manual_seed(42 + rank)
    split_path = Path(config['output_dir']) / 'classifier_reference_split.json'
    # Each classifier independently verifies identical split; rank zero writes the record.
    if args.smoke:
        parts = audit(config)
        split = {part: [(r['reference_image'], EMOTIONS.index(r['target_emotion']))
                        for emotion in EMOTIONS for r in [x for x in parts[part] if x['target_emotion'] == emotion][:2]]
                 for part in ('train', 'val')}
    else:
        split = reference_sets(config)
    if rank == 0:
        split_path.write_text(json.dumps(split, indent=2))
    feature_sets = {}
    for part in ('train', 'val'):
        features = []
        labels = []
        for path, label in tqdm(split[part], desc='Classifier ' + name.upper() + ' cached features ' + part):
            with np.load(cache_path(config, path), allow_pickle=False) as cache:
                features.append(cache['appearance_' + name].astype(np.float32))
            labels.append(label)
        feature_sets[part] = (torch.from_numpy(np.stack(features)), torch.tensor(labels))
    x, y = feature_sets['train']
    counts = torch.bincount(y, minlength=8)
    if not (counts > 0).all():
        raise ValueError('Missing reference class: ' + str(counts.tolist()))
    weights = 1 / counts[y].float()
    sampler = WeightedRandomSampler(weights, len(y), replacement=True)
    loader = DataLoader(TensorDataset(x, y), batch_size=256, sampler=sampler)
    head = EmotionHead(hidden=256).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=.01)
    best, history = -1., []
    for epoch in tqdm(range(2 if args.smoke else config['classifier_epochs']), desc='Fit classifier ' + name.upper()):
        head.train()
        loss_sum = 0.
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
        head.eval()
        with torch.no_grad():
            vx, vy = feature_sets['val']
            predictions = torch.cat([head(vx[i:i + 256].to(device)).argmax(-1).cpu() for i in range(0, len(vx), 256)])
        metrics = scores(vy, predictions)
        history.append(dict(epoch=epoch + 1, train_ce=loss_sum / len(y), **metrics))
        print(json.dumps(dict(classifier=name, **history[-1])), flush=True)
        if metrics['macro_f1'] > best:
            best = metrics['macro_f1']
            torch.save(dict(state={k: v.cpu() for k, v in head.state_dict().items()}, classes=EMOTIONS,
                            backbone='resnet18' if name == 'a' else 'resnet34',
                            preprocess='v16 FrozenAppearance 128 + Lab stats; center fit 256',
                            metrics=metrics, epoch=epoch + 1, smoke_only=args.smoke), config['classifier_' + name])
    Path(config['classifier_' + name]).with_suffix('.json').write_text(json.dumps(dict(
        purpose='training supervisor' if name == 'a' else 'fixed NEW Emo-A evaluation, no generator gradient',
        classes=EMOTIONS, train_counts=counts.tolist(), history=history,
        sha256=hashlib.sha256(Path(config['classifier_' + name]).read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
