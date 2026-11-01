"""Explicit public training pipeline; no private paths or historical warm starts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--nproc', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.nproc != 2:
        parser.error('Classifier preparation currently requires exactly two GPUs (A/B).')
    config = json.loads(Path(args.config).read_text())
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    common = ['--config', args.config]
    smoke = ['--smoke'] if args.smoke else []
    def run(module, extra=(), distributed=False):
        cmd = [sys.executable, '-u', '-m']
        if distributed:
            cmd += ['torch.distributed.run', '--standalone', '--nproc_per_node=2', '-m']
        cmd += ['emotion_diffusion.reference_v19.' + module, *common, *extra]
        print('RUN', ' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    run('assets')
    run('prepare', ['--limit', '16'] if args.smoke else [], True)
    run('classifiers', smoke, True)
    run('brush', smoke)
    run('train', ['--stage', 'content', *smoke], True)
    run('train', ['--stage', 'edit', '--resume', str(Path(config['output_dir']) / 'content_latest.pt'), *smoke], True)

if __name__ == '__main__':
    main()
