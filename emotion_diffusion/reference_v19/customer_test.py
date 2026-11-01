"""Read-only customer benchmark evaluation, separate from training/validation."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from .appearance import EmotionHead, FrozenAppearance, edge_map
from .classifiers import scores
from .data import EMOTIONS, pixels
from .metrics import EDITOR_CLASSES, ImageMetrics
from .model import Editor
from .train import grid, load_small, ssim


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def summarize(directory):
    directory = Path(directory)
    records = []
    for path in sorted(directory.glob('records-rank*.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    if not records:
        return
    if len({r['index'] for r in records}) != len(records):
        raise ValueError('Duplicate evaluation indices')
    manifest = json.loads((directory / 'manifest.json').read_text())
    labels = [r['label'] for r in records]
    result = dict(samples=len(records), expected_samples=len(manifest),
                  complete=len(records) == len(manifest), split='customer_test',
                  metric_protocol='Fixed reference-trained B; 256 center fit; float output SSIM11box/CLIP224; legacy ESMI',
                  **scores(labels, [r['prediction_b'] for r in records]))
    result['emo_a_new_B'] = 100 * result['accuracy']
    result['emo_a_legacy_resnet'] = 100 * np.mean([r['prediction_legacy'] == r['label'] for r in records])
    result['reference_emo_a_B'] = 100 * np.mean([r['reference_prediction_b'] == r['label'] for r in records])
    result['original_target_match_B'] = 100 * np.mean([r['source_prediction_b'] == r['label'] for r in records])
    for key in ('ssim', 'clip_i', 'dino_similarity', 'esmi_legacy'):
        result[key] = float(np.mean([r[key] for r in records]))
    result['per_emotion'] = {
        emotion: dict(samples=sum(r['label'] == i for r in records),
                      correct=sum(r['label'] == i and r['prediction_b'] == i for r in records))
        for i, emotion in enumerate(EMOTIONS)}
    write_json(directory / 'metrics.json', result)
    write_json(directory / 'per_image.json', sorted(records, key=lambda r: r['index']))
    return result


@torch.no_grad()
def evaluate(args):
    from transformers import AutoModelForImageSegmentation
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    manifest = json.loads((directory / 'manifest.json').read_text())
    actual_hash = hashlib.sha256(Path(config['classifier_b']).read_bytes()).hexdigest()
    if actual_hash != '0a81984c0694cb0b2904a02687d71a77b0a4f9b984a9406b21de26608fb2e206':
        raise ValueError('Fixed classifier B changed')
    torch.set_num_threads(4)
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    model = Editor(config, device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    load_small(model, checkpoint['model'])
    del checkpoint
    model.eval()
    segment = AutoModelForImageSegmentation.from_pretrained(
        config['segmentation_model'], trust_remote_code=True).to(device).eval().requires_grad_(False)
    appearance = FrozenAppearance('resnet34', config['appearance_b_weights']).to(device).eval()
    head = EmotionHead().to(device).eval()
    b = torch.load(config['classifier_b'], map_location='cpu', weights_only=True)
    if tuple(b['classes']) != EMOTIONS:
        raise ValueError('Classifier ordering mismatch')
    head.load_state_dict(b['state'])
    metrics = ImageMetrics(config, device)
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    record_path = directory / ('records-rank%d.jsonl' % args.rank)
    done = {json.loads(line)['index'] for line in record_path.read_text().splitlines()} if record_path.exists() else set()
    owned = list(range(args.rank, len(manifest), args.world))
    source_cache = {}
    start = time.time()
    for index in tqdm(owned, desc='Customer test rank %d' % args.rank, dynamic_ncols=True):
        if index in done:
            continue
        row = manifest[index]
        source_key = row['image']
        source = pixels(source_key, config['resolution'])[None].to(device)
        if source_key not in source_cache:
            latent = model.vae.encode(source).latent_dist.mode() * model.vae.config.scaling_factor
            dino = model.dino(source)
            x = F.interpolate((source + 1) / 2, (512, 512), mode='bilinear', align_corners=False)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                mask = F.interpolate(segment((x - mean) / std)[-1].sigmoid().float(),
                                     (64, 64), mode='bilinear', align_corners=False)
            area = float((mask > .5).float().mean())
            confidence = float((2 * (mask - .5).abs()).mean()) if .01 < area < .9 else 0.
            source_cache[source_key] = {k: v.cpu() for k, v in dict(
                latent=latent, dino=dino, mask=mask,
                edge=F.adaptive_avg_pool2d(edge_map(source), 64),
                confidence=torch.tensor([confidence], device=device)).items()}
        reference = pixels(row['reference_image'], config['resolution'])[None].to(device)
        label = EMOTIONS.index(row['target_emotion'])
        batch = {k: v.to(device) for k, v in source_cache[source_key].items()}
        batch.update(source=source, reference=reference,
                     reference_dino=model.dino(reference).mean(1),
                     label=torch.tensor([label], device=device))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output, _, _ = model.render(batch, args.steps, config['seed'] + index)
            predictions = head(appearance(torch.cat((output, source, reference)))).argmax(-1).tolist()
            dino_similarity = float(F.cosine_similarity(model.dino(output), batch['dino'], dim=-1).mean())
        legacy_pixels = F.interpolate(((output.float() + 1) / 2).clamp(0, 1),
                                      (224, 224), mode='bilinear', align_corners=False)
        legacy_id = int(metrics.resnet(legacy_pixels).argmax(-1))
        record = dict(index=index, source=source_key, reference=row['reference_image'],
                      emotion=row['target_emotion'], label=label, prediction_b=predictions[0],
                      source_prediction_b=predictions[1], reference_prediction_b=predictions[2],
                      prediction_legacy=EMOTIONS.index(EDITOR_CLASSES[legacy_id]),
                      dino_similarity=dino_similarity, ssim=float(ssim(source, output)),
                      clip_i=metrics.clip_i(source, output),
                      **metrics.esmi(source, output, row['target_emotion']))
        stem = '%s_%s' % (Path(source_key).stem, row['target_emotion'])
        generated_dir, grid_dir = directory / 'generated', directory / 'grids'
        generated_dir.mkdir(exist_ok=True)
        grid_dir.mkdir(exist_ok=True)
        Image.fromarray(((output[0].float().cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()).save(generated_dir / (stem + '.png'))
        grid(source[0], reference[0], output[0], grid_dir / (stem + '.png'), label, predictions[0])
        with record_path.open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        done.add(index)
        write_json(directory / ('status-rank%d.json' % args.rank), dict(
            status='running', completed=len(done), assigned=len(owned), last_index=index,
            elapsed_seconds=time.time() - start))
    write_json(directory / ('status-rank%d.json' % args.rank),
               dict(status='complete', completed=len(done), assigned=len(owned)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world', type=int, default=1)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.output)))
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
