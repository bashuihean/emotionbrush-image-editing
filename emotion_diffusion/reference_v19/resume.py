"""Checkpoint progress and optimiser compatibility for continuing the same stage."""
import math
import itertools


class SkippedBatchSampler:
    """Skip index batches without opening/reloading already-consumed images."""
    def __init__(self, batch_sampler, skip=0):
        self.batch_sampler, self.skip = batch_sampler, skip

    def __iter__(self):
        return itertools.islice(iter(self.batch_sampler), self.skip, None)

    def __len__(self):
        return max(0, len(self.batch_sampler) - self.skip)


def resume_position(checkpoint, config, stage, batches, world):
    if checkpoint.get('stage') != stage:
        raise ValueError('Full resume requires the same stage; use --resume for a warm start')
    old = checkpoint['config']
    for key in ('batch_size', 'gradient_accumulation', 'resolution', 'train_manifest',
                'cache_signature', 'classifier_a', 'classifier_b', 'seed'):
        if old.get(key) != config.get(key):
            raise ValueError('Full resume incompatible setting: ' + key)
    previous_world = checkpoint.get('world_size', 2)
    if previous_world != world:
        raise ValueError('Keep the same GPU count to retain the dataset position')
    updates_per_epoch = math.ceil(batches / config['gradient_accumulation'])
    step = int(checkpoint['step'])
    epoch, update = divmod(step, updates_per_epoch)
    cursor = checkpoint.get('cursor', dict(epoch=epoch, next_batch=update * config['gradient_accumulation']))
    if int(cursor['next_batch']) >= batches:
        cursor = dict(epoch=int(cursor['epoch']) + 1, next_batch=0)
    if 'optimizer' not in checkpoint:
        raise ValueError('Checkpoint contains no optimizer state')
    return step, int(cursor['epoch']), int(cursor['next_batch'])
