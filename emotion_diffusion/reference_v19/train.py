"""Two-GPU staged unpaired training with fixed independent Emo-A evaluation."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from .appearance import FrozenAppearance, EmotionHead
from .classifiers import scores
from .data import ReferenceDataset, EMOTIONS
from .model import Editor, RoutedAttention


def small_state(model):
    return dict(condition={k: v.detach().cpu() for k, v in model.condition.state_dict().items()},
                processors={name: {k: v.detach().cpu() for k, v in module.state_dict().items()}
                            for name, module in model.unet.attn_processors.items() if isinstance(module, RoutedAttention)})


def load_small(model, state):
    missing, unexpected = model.condition.load_state_dict(state['condition'], strict=False)
    if unexpected or any(not key.startswith('photometric.') for key in missing):
        raise ValueError('Incompatible condition checkpoint: ' + str((missing, unexpected)))
    for name, values in state['processors'].items():
        model.unet.attn_processors[name].load_state_dict(values, strict=True)


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def grid(source, reference, generated, path, label, predicted):
    def pil(tensor):
        array = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
        return Image.fromarray(array)
    images = list(map(pil, (source, reference, generated)))
    size = images[0].width
    canvas = Image.new('RGB', (size * 3, size + 30), 'white')
    draw = ImageDraw.Draw(canvas)
    titles = ['source', 'reference: ' + EMOTIONS[label], 'output: B=' + EMOTIONS[predicted]]
    for i, (image, title) in enumerate(zip(images, titles)):
        canvas.paste(image, (i * size, 30))
        draw.text((i * size + 5, 8), title, fill='black')
    canvas.save(path)


def ssim(source, output):
    x, y = ((source + 1) / 2).float(), ((output + 1) / 2).float()
    ux, uy = F.avg_pool2d(x, 11, 1, 5), F.avg_pool2d(y, 11, 1, 5)
    vx = F.avg_pool2d(x * x, 11, 1, 5) - ux.square()
    vy = F.avg_pool2d(y * y, 11, 1, 5) - uy.square()
    cov = F.avg_pool2d(x * y, 11, 1, 5) - ux * uy
    return (((2 * ux * uy + .01 ** 2) * (2 * cov + .03 ** 2)) /
            ((ux.square() + uy.square() + .01 ** 2) * (vx + vy + .03 ** 2))).mean()


@torch.no_grad()
def validate(model, config, device, rank, world, step, full=False):
    model.eval()
    dataset = ReferenceDataset(config, 'val', None if full else config['validation_limit'])
    if hashlib.sha256(Path(config['classifier_b']).read_bytes()).hexdigest() != config['classifier_b_sha256']:
        raise ValueError('Fixed evaluation B changed during generator training')
    appearance = FrozenAppearance('resnet34', config['appearance_b_weights']).to(device).eval()
    head = EmotionHead().to(device)
    checkpoint = torch.load(config['classifier_b'], map_location='cpu', weights_only=True)
    if tuple(checkpoint['classes']) != EMOTIONS:
        raise ValueError('Evaluation class ordering mismatch')
    head.load_state_dict(checkpoint['state'])
    head.eval().requires_grad_(False)
    from .metrics import ImageMetrics
    image_metrics = ImageMetrics(config, device)
    visual = Path(config['output_dir']) / ('validation_full' if full else 'validation') / ('step-%06d' % step)
    visual.mkdir(parents=True, exist_ok=True)
    records = []
    for index in tqdm(range(rank, len(dataset), world), desc='Validation B rank ' + str(rank), disable=rank != 0):
        sample = dataset[index]
        batch = move({k: v[None] if torch.is_tensor(v) else torch.tensor([v]) for k, v in sample.items()}, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output, gate, routing = model.render(batch, config['validation_sampling_steps'], config['seed'] + index)
            bpred = int(head(appearance(output)).argmax(-1))
            apred = int(model.supervisor(model.appearance(output)).argmax(-1))
            content = float(F.cosine_similarity(model.dino(output), batch['dino'], dim=-1).mean())
            structure = float(ssim(batch['source'], output))
            from .losses import editing_losses
            distances = editing_losses(model, batch, output, gate, routing)
        label = int(batch['label'][0])
        clip_i = image_metrics.clip_i(batch['source'], output)
        esmi = image_metrics.esmi(batch['source'], output, EMOTIONS[label])
        records.append(dict(index=index, label=label, prediction_b=bpred, prediction_a=apred,
                            dino_similarity=content, ssim=structure, clip_i=clip_i, **esmi,
                            **{key: float(value) for key, value in distances.items()}))
        if not full or index < config['validation_limit']:
            grid(batch['source'][0], batch['reference'][0], output[0], visual / ('%05d.png' % index), label, bpred)
            Image.fromarray(((output[0].float().cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()).save(visual / ('output-%05d.png' % index))
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        records = [row for group in gathered for row in group]
    if rank == 0:
        labels = [r['label'] for r in records]
        metrics = scores(labels, [r['prediction_b'] for r in records])
        result = dict(step=step, split='val', scope='full' if full else 'fixed_preview', samples=len(records),
                      emo_a_new_B=100 * metrics['accuracy'],
                      supervisor_a_accuracy=100 * scores(labels, [r['prediction_a'] for r in records])['accuracy'],
                      dino_similarity=float(np.mean([r['dino_similarity'] for r in records])),
                      ssim=float(np.mean([r['ssim'] for r in records])),
                      clip_i=float(np.mean([r['clip_i'] for r in records])),
                      esmi_legacy=float(np.mean([r['esmi_legacy'] for r in records])),
                      metric_protocol='NEW fixed reference-trained B Emo-A; CLIP-I center-fit RGB tensor224; SSIM11box; legacy ESMI EmoEditor CAM tensor224',
                      classifier_b_sha256=hashlib.sha256(Path(config['classifier_b']).read_bytes()).hexdigest(),
                      **metrics)
        (visual / 'metrics.json').write_text(json.dumps(result, indent=2))
        (visual / 'per_image.json').write_text(json.dumps(sorted(records, key=lambda r: r['index']), indent=2))
        with (Path(config['output_dir']) / 'validation_history.jsonl').open('a') as f:
            f.write(json.dumps(result) + '\n')
        print('VALIDATION', json.dumps(result), flush=True)
    del appearance, head, image_metrics
    model.train()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=['content', 'edit'], required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume')
    p.add_argument('--resume-training', help='Restore same-stage optimizer and data position')
    p.add_argument('--max-updates', type=int, help='Stop after this many additional updates for a probe')
    args = p.parse_args()
    if args.resume and args.resume_training:
        p.error('Choose either a warm start or full training resume')
    config = json.loads(Path(args.config).read_text())
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    if world > 1:
        from datetime import timedelta
        dist.init_process_group('nccl', timeout=timedelta(hours=4))
    device = torch.device('cuda', local)
    torch.manual_seed(config['seed'] + rank)
    random.seed(config['seed'] + rank)
    np.random.seed(config['seed'] + rank)
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    for key in ('classifier_a', 'classifier_b'):
        provenance = torch.load(config[key], map_location='cpu', weights_only=True)
        if provenance.get('smoke_only', False) and not args.smoke:
            raise ValueError('Smoke classifiers cannot be used for formal training')
    config['classifier_b_sha256'] = hashlib.sha256(Path(config['classifier_b']).read_bytes()).hexdigest()
    if rank == 0:
        (output / 'active_config.json').write_text(json.dumps(config, indent=2))
    model = Editor(config, device)
    resumed = None
    if args.resume or args.resume_training:
        resumed = torch.load(args.resume_training or args.resume, map_location='cpu', weights_only=False)
        if args.resume_training and not any(k.startswith('photometric.') for k in resumed['model']['condition']):
            raise ValueError('Use warm start for v18 upgrade; optimizer has new parameters')
        load_small(model, resumed['model'])
    else:
        initial = torch.load(config['condition_init'], map_location='cpu', weights_only=True)
        missing, unexpected = model.condition.load_state_dict(initial, strict=False)
        if unexpected or any(not key.startswith('photometric.') for key in missing):
            raise ValueError('Incompatible initialization')
    editing = args.stage == 'edit'
    model.set_stage(editing)
    model.train()
    if editing and any(p.requires_grad for p in model.condition.qformer.parameters()):
        raise RuntimeError('v18 content QFormer must remain frozen during editing')
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.condition.parameters() if p.requires_grad], lr=config['condition_learning_rate']),
        dict(params=[p for p in model.unet.parameters() if p.requires_grad], lr=config['learning_rate'])], weight_decay=.01)
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    if rank == 0:
        (output / (args.stage + '_trainable_parameters.json')).write_text(json.dumps(trainable_names, indent=2))
    ddp = DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True) if world > 1 else model
    dataset = ReferenceDataset(config, 'train', 16 if args.smoke else None)
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=config['seed']) if world > 1 else None
    loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler, shuffle=sampler is None,
                        num_workers=config['num_workers'], pin_memory=True)
    accumulation = config['gradient_accumulation']
    epochs = 1 if args.smoke else config[args.stage + '_epochs']
    planned = epochs * math.ceil(len(loader) / accumulation)
    full_loader_length = len(loader)
    step = 0
    start_epoch, start_batch = 0, 0
    if args.resume_training:
        from .resume import resume_position
        step, start_epoch, start_batch = resume_position(resumed, config, args.stage, len(loader), world)
        optimizer.load_state_dict(resumed['optimizer'])
        if step > planned:
            raise ValueError('Resume progress exceeds requested training schedule')
        print('FULL_RESUME', dict(step=step, epoch=start_epoch, next_batch=start_batch,
                                 optimizer_states=len(optimizer.state),
                                 rng_available='rng_by_rank' in resumed), flush=True)
        if world == 1:
            raise ValueError('Full resume currently requires the deterministic distributed sampler')
        from .resume import SkippedBatchSampler
        resumed_sampler = SkippedBatchSampler(loader.batch_sampler, start_batch)
        loader = DataLoader(dataset, batch_sampler=resumed_sampler,
                            num_workers=config['num_workers'], pin_memory=True)
    initial_step = step
    previous_seconds = float(resumed.get('elapsed_seconds', 0.)) if args.resume_training else 0.
    start = time.time()
    progress = tqdm(total=planned, initial=step, desc=args.stage + ' optimizer updates', disable=rank != 0, dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)
    gradient_evidence = {}
    def checkpoint(tag):
        rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                   python=random.getstate(), numpy=np.random.get_state())
        rng_by_rank = [None] * world
        if world > 1:
            dist.all_gather_object(rng_by_rank, rng)
        else:
            rng_by_rank[0] = rng
        if rank == 0:
            payload = dict(model=small_state(model), stage=args.stage, step=step,
                           optimizer=optimizer.state_dict(), config=config, smoke_only=args.smoke,
                           cursor=dict(epoch=epoch, next_batch=i + 1), world_size=world,
                           rng_by_rank=rng_by_rank, elapsed_seconds=previous_seconds + time.time() - start)
            temporary = output / (tag + '.tmp.pt')
            torch.save(payload, temporary)
            os.replace(temporary, output / (tag + '.pt'))
    totals = {}
    stopped = False
    rng_pending = bool(args.resume_training and 'rng_by_rank' in resumed)
    if args.resume_training and not rng_pending:
        print('Legacy checkpoint: optimizer/progress restored; exact RNG continuation unavailable', flush=True)
    if (args.resume or args.resume_training) and editing and config.get('validate_on_resume', False) and not args.max_updates:
        validate(model, config, device, rank, world, step)
    for epoch in range(start_epoch, epochs):
        if sampler:
            sampler.set_epoch(epoch)
        offset = start_batch if epoch == start_epoch else 0
        if args.resume_training:
            resumed_sampler.skip = offset
        for i, batch in enumerate(loader, start=offset):
            if rng_pending:
                rng = resumed['rng_by_rank'][rank]
                torch.set_rng_state(rng['torch'])
                torch.cuda.set_rng_state(rng['cuda'])
                random.setstate(rng['python']); np.random.set_state(rng['numpy'])
                rng_pending = False
            batch = move(batch, device)
            synchronize = (i + 1) % accumulation == 0 or i + 1 == full_loader_length
            group_start = (i // accumulation) * accumulation
            group_size = min(accumulation, full_loader_length - group_start)
            context = contextlib.nullcontext() if synchronize or world == 1 else ddp.no_sync()
            with context:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, values = ddp(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite ' + args.stage + ' loss')
                (loss / group_size).backward()
            for key, value in dict(loss=loss.detach(), **values).items():
                totals[key] = totals.get(key, 0.) + float(value) / group_size
            if not synchronize:
                continue
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient')
            if step == 0:
                gradient_evidence = {name: float(p.grad.abs().sum()) for name, p in model.named_parameters()
                                     if p.requires_grad and p.grad is not None}
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            record = dict(stage=args.stage, step=step, epoch=epoch + (i + 1) / full_loader_length,
                          planned_updates=planned, elapsed_seconds=previous_seconds + time.time() - start,
                          train_sampling_steps=config['train_sampling_steps'],
                          grad_norm=float(norm), **totals)
            if rank == 0:
                with (output / 'training_history.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                (output / 'status.json').write_text(json.dumps(record, indent=2))
                progress.update()
                progress.set_postfix(loss=record['loss'], grad=record['grad_norm'])
            totals = {}
            if step % config['save_steps'] == 0:
                checkpoint(args.stage + '-%06d' % step)
            if editing and not args.smoke and step % config['val_steps'] == 0:
                validate(model, config, device, rank, world, step)
            if args.smoke and step >= 2:
                break
            if args.max_updates and step - initial_step >= args.max_updates:
                stopped = True
                break
        checkpoint(args.stage + '_latest')
        if editing and not stopped:
            validate(model, config, device, rank, world, step)
        if args.smoke or stopped:
            break
    if editing and stopped and config.get('validate_probe', False):
        validate(model, config, device, rank, world, step)
    if editing and not args.smoke and not stopped:
        validate(model, config, device, rank, world, step, full=True)
    if rank == 0:
        (output / (args.stage + '_completion.json')).write_text(json.dumps(dict(
            status='probe_stopped' if stopped else 'complete', stage=args.stage, smoke_only=args.smoke, updates=step,
            seconds=time.time() - start, first_update_gradients=gradient_evidence,
            peak_memory_mib=torch.cuda.max_memory_allocated(device) / 1024 ** 2), indent=2))
    progress.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
