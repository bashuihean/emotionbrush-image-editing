"""Isolated inference grid and matched short subject-detail training experiment."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .appearance import FrozenAppearance, EmotionHead
from .classifiers import scores
from .data import ReferenceDataset, EMOTIONS
from .model import Editor
from .train import load_small, move, small_state, ssim


def detail_loss(source, output, mask, confidence):
    """Confidence-weighted foreground luminance structure, allowing global tone changes."""
    def features(x):
        gray = (x.float() * x.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
        gray = (gray - gray.mean((2, 3), keepdim=True)) / gray.std((2, 3), keepdim=True).clamp_min(.1)
        high = gray - F.avg_pool2d(gray, 5, 1, 2)
        return high, gray[:, :, :, 1:] - gray[:, :, :, :-1], gray[:, :, 1:] - gray[:, :, :-1]
    a, b = features(source), features(output)
    weight = F.interpolate(mask.float(), source.shape[-2:], mode='bilinear', align_corners=False)
    weight = weight * confidence.reshape(-1, 1, 1, 1).float()
    # Exclude the frame: average-pool padding can create artificial edges.
    weight = weight.clone()
    weight[:, :, :3] = 0; weight[:, :, -3:] = 0
    weight[:, :, :, :3] = 0; weight[:, :, :, -3:] = 0
    weights = (weight, weight[:, :, :, 1:], weight[:, :, 1:])
    return sum((F.smooth_l1_loss(x, y, reduction='none') * w).sum() / w.sum().clamp_min(1)
               for x, y, w in zip(a, b, weights)) / 3


def png(x):
    return Image.fromarray(((x.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--updates', type=int, default=32)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    config = copy.deepcopy(config)
    seed = config['seed']
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    initial = checkpoint['model']
    device = torch.device('cuda')
    model = Editor(config, device); load_small(model, initial)
    model.set_stage(True); model.eval()
    evaluator = FrozenAppearance('resnet34', config['appearance_b_weights']).to(device).eval().requires_grad_(False)
    head = EmotionHead().to(device).eval().requires_grad_(False)
    head.load_state_dict(torch.load(config['classifier_b'], map_location='cpu', weights_only=True)['state'])
    val = ReferenceDataset(config, 'val', 32)
    batches = [move({k: v[None] if torch.is_tensor(v) else torch.tensor([v]) for k, v in val[i].items()}, device)
               for i in range(len(val))]
    protocol = dict(checkpoint=args.checkpoint, checkpoint_step=checkpoint['step'],
                    checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                    classifier_b_sha256=hashlib.sha256(Path(config['classifier_b']).read_bytes()).hexdigest(),
                    seed=seed, val_indices=list(range(len(val))), train_updates=args.updates,
                    train_accumulation=2, detail_weight=.2,
                    note='Exploratory 32-image validation; reference is NOT paired ground truth. Detail preservation is not perceptual quality.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    records = []
    def status(phase, **kwargs):
        (out / 'status.json').write_text(json.dumps(dict(phase=phase, **kwargs), indent=2))
        print(phase, kwargs, flush=True)

    @torch.no_grad()
    def evaluate(tag, steps, strength):
        model.eval(); config['edit_strength'] = strength
        folder = out / tag; folder.mkdir(exist_ok=True)
        per_image = []; start = time.time()
        for i, batch in enumerate(batches):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                generated, _, _ = model.render(batch, steps, seed+i)
                prediction = int(head(evaluator(generated)).argmax(-1))
                dino = float(F.cosine_similarity(model.dino(generated), batch['dino'], dim=-1).mean())
            per_image.append(dict(index=i, label=int(batch['label']), prediction=prediction,
                                  ssim=float(ssim(batch['source'], generated)), dino=dino,
                                  detail_error=float(detail_loss(batch['source'], generated, batch['mask'], batch['confidence'])),
                                  mask_coverage=float(batch['mask'].mean()), confidence=float(batch['confidence'])))
            png(generated[0]).save(folder / f'{i:05d}.png')
        metrics = dict(tag=tag, scheduled_steps=steps, actual_denoise_steps=steps-int(steps*(1-strength)),
                       strength=strength, seconds=time.time()-start,
                       emo_a=100*scores([x['label'] for x in per_image], [x['prediction'] for x in per_image])['accuracy'],
                       **{k:float(np.mean([r[k] for r in per_image])) for k in ['ssim','dino','detail_error']})
        records.append(metrics)
        (folder / 'per_image.json').write_text(json.dumps(per_image, indent=2))
        (out / 'metrics.json').write_text(json.dumps(records, indent=2))
        status('evaluated', **metrics)

    for steps in (20, 40, 60):
        for strength in (.25, .4, .55):
            evaluate(f'steps{steps}_strength{strength:.2f}', steps, strength)
    # Same checkpoint, samples, noise seeds, optimiser reset and updates for both arms.
    train = ReferenceDataset(config, 'train', args.updates*2)
    config['edit_strength'] = .55
    from .losses import editing_losses
    for arm, weight in [('baseline_short', 0.), ('detail_short', .2)]:
        load_small(model, initial); model.set_stage(True); model.train()
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        optimizer = torch.optim.AdamW([
            dict(params=[p for p in model.condition.parameters() if p.requires_grad], lr=config['condition_learning_rate']),
            dict(params=[p for p in model.unet.parameters() if p.requires_grad], lr=config['learning_rate'])], weight_decay=.01)
        params = [p for p in model.parameters() if p.requires_grad]
        history = []
        for step in range(args.updates):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(2):
                sample = train[step*2+micro]
                batch = move({k:v[None] if torch.is_tensor(v) else torch.tensor([v]) for k,v in sample.items()}, device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    generated, gate, routing = model.render(batch)
                    values = editing_losses(model, batch, generated, gate, routing)
                    extra = detail_loss(batch['source'], generated, batch['mask'], batch['confidence'])
                    loss = sum(config['loss_weights'][k]*v for k,v in values.items()) + weight*extra
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                (loss/2).backward(); losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            history.append(dict(step=step+1, loss=float(np.mean(losses)), grad_norm=float(norm), detail=float(extra.detach())))
            if (step+1)%8 == 0: status(arm, step=step+1, total=args.updates)
        (out / (arm+'_training.json')).write_text(json.dumps(history, indent=2))
        torch.save(dict(model=small_state(model), protocol=protocol, arm=arm), out / (arm+'.pt'))
        evaluate(arm, 20, .55)
        del optimizer
    # Original + reference + nine inference settings, and matched training arms.
    tags = [r['tag'] for r in records]
    for i, batch in enumerate(batches):
        panels = [('source',png(batch['source'][0])), ('reference '+EMOTIONS[int(batch['label'])],png(batch['reference'][0]))]
        panels += [(tag, Image.open(out/tag/f'{i:05d}.png').convert('RGB')) for tag in tags]
        canvas = Image.new('RGB',(256*4,286*4),'white'); draw = ImageDraw.Draw(canvas)
        for j,(title,im) in enumerate(panels):
            x,y=(j%4)*256,(j//4)*286; canvas.paste(im,(x,y+30));draw.text((x+4,y+8),title,fill='black')
        canvas.save(out/f'comparison-{i:05d}.png')
    status('complete', comparisons=len(batches), experiments=len(records))


if __name__ == '__main__':
    main()
