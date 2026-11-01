"""Native customer data and versioned frozen-feature caches."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

EMOTIONS = ('amusement', 'awe', 'contentment', 'excitement', 'anger',
            'disgust', 'fear', 'sadness')


def rows(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    result = json.loads(text) if text.lstrip().startswith('[') else [json.loads(x) for x in text.splitlines() if x.strip()]
    if not result:
        raise ValueError('Empty split')
    for row in result:
        if 'target_image' in row or row.get('paired_target_available', False):
            raise ValueError('v16 accepts unpaired customer references only')
        if row['target_emotion'] not in EMOTIONS:
            raise ValueError(row['target_emotion'])
        for key in ('image', 'reference_image'):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
    return result


def pixels(path, size):
    with Image.open(path) as image:
        image = ImageOps.fit(ImageOps.exif_transpose(image).convert('RGB'),
                             (size, size), method=Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 127.5 - 1


def cache_path(config, path):
    # Cache namespace includes weights/preprocessing fingerprints, populated by prepare.
    key = hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()
    return Path(config['cache_dir']) / config['cache_signature'] / key[:2] / (key + '.npz')


class ReferenceDataset(Dataset):
    def __init__(self, config, part, limit=None):
        self.config = config
        self.rows = rows(config[part + '_manifest'])
        if limit:
            # Round-robin emotions for smoke/visual checks, not a prefix of one class.
            groups = [[r for r in self.rows if r['target_emotion'] == e] for e in EMOTIONS]
            selected = []
            for i in range(max(map(len, groups))):
                selected.extend(g[i] for g in groups if i < len(g))
                if len(selected) >= limit:
                    break
            self.rows = selected[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        source_path = cache_path(self.config, row['image'])
        with np.load(source_path, allow_pickle=False) as source:
            result = {key: torch.from_numpy(source[key].astype(np.float32))
                      for key in ('latent', 'dino', 'mask', 'edge', 'confidence')}
        with np.load(cache_path(self.config, row['reference_image']), allow_pickle=False) as reference:
            result['reference_dino'] = torch.from_numpy(reference['dino_global'].astype(np.float32))
        result.update(source=pixels(row['image'], self.config['resolution']),
                      reference=pixels(row['reference_image'], self.config['resolution']),
                      label=EMOTIONS.index(row['target_emotion']), index=index)
        return result


def audit(config):
    parts = {p: rows(config[p + '_manifest']) for p in ('train', 'val', 'test')}
    sets = {p: {r[k] for r in rr for k in ('image', 'reference_image')} for p, rr in parts.items()}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if sets[a] & sets[b]:
            raise ValueError('Image leakage across ' + a + '/' + b)
    return parts
