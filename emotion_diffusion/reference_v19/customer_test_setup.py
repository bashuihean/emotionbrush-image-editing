"""Build exact source-ID/reference pairs without consuming old generated outputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from PIL import Image

from .data import EMOTIONS
from .customer_test import write_json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('source', 'reference', 'config', 'checkpoint', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Use the existing manifest for resume; do not overwrite a test protocol')
    sources = {x.stem: x for x in Path(args.source).iterdir()
               if x.is_file() and x.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')}
    records, missing, invalid, unmatched = [], [], [], []
    reference_root = Path(args.reference)
    for stem in sorted(sources, key=lambda x: (int(x) if x.isdigit() else 10**9, x)):
        source = sources[stem]
        with Image.open(source) as image:
            image.verify()
        for emotion in EMOTIONS:
            reference = reference_root / emotion / (stem + '_' + emotion + '_target_' + emotion + '.png')
            if not reference.is_file():
                missing.append(dict(source=str(source), emotion=emotion, expected_reference=str(reference)))
                continue
            try:
                with Image.open(reference) as image:
                    image.verify()
            except Exception as exc:
                invalid.append(dict(path=str(reference), error=str(exc)))
                continue
            records.append(dict(image=str(source), reference_image=str(reference),
                                target_emotion=emotion, paired_target_available=False,
                                reference_role='customer_emotion_style_reference'))
    used = {r['reference_image'] for r in records}
    for emotion in EMOTIONS:
        unmatched.extend(str(x) for x in (reference_root / emotion).iterdir()
                         if x.is_file() and str(x) not in used)
    write_json(out / 'manifest.json', records)
    cfg = json.loads(Path(args.config).read_text())
    write_json(out / 'config_snapshot.json', cfg)
    summary = dict(source_count=len(sources), paired_samples=len(records),
                   per_emotion=dict(Counter(r['target_emotion'] for r in records)),
                   missing=missing, invalid=invalid, unmatched_references=unmatched,
                   excluded_directory=str(reference_root / 'generated'),
                   checkpoint=args.checkpoint, checkpoint_sha256=digest(args.checkpoint),
                   classifier_b_sha256=digest(cfg['classifier_b']),
                   sampling_steps=20, edit_strength=cfg['edit_strength'],
                   seed_policy='config.seed + stable manifest index',
                   original_resolution_policy='EXIF transpose, center fit 256; output 256',
                   reference_is_not_target=True,
                   source_directory=args.source, reference_directory=args.reference)
    write_json(out / 'dataset_audit.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('missing', 'invalid', 'unmatched_references')}), flush=True)
    # Exact byte duplicates only; this is explicitly not a near-duplicate/content audit.
    test_paths = {r[k] for r in records for k in ('image', 'reference_image')}
    by_size = {}
    for path in test_paths:
        by_size.setdefault(Path(path).stat().st_size, {}).setdefault(digest(path), []).append(path)
    split_overlaps = {}
    for split in ('train', 'val', 'test'):
        rows = json.loads(Path(cfg[split + '_manifest']).read_text())
        paths = {r[k] for r in rows for k in ('image', 'reference_image')}
        duplicates = []
        for path in sorted(paths):
            size = Path(path).stat().st_size
            if size in by_size:
                key = digest(path)
                if key in by_size[size]:
                    duplicates.append(dict(previous=path, customer_test=by_size[size][key]))
        split_overlaps[split] = dict(exact_path_overlap=sorted(paths & test_paths),
                                     exact_byte_duplicates=duplicates)
    write_json(out / 'overlap_audit.json', dict(method='SHA256 exact file bytes; not near-duplicate detection',
                                               splits=split_overlaps))
    print('SETUP COMPLETE', flush=True)


if __name__ == '__main__':
    main()
