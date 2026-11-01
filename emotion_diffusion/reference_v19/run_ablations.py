"""Equal-budget warm-start ablations; all use one fixed checkpoint and validation set."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path.cwd()
    out = root / 'outputs/v19_ablation'
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads((root / 'v19_warmstart.json').read_text())['checkpoint']
    results = []
    for variant in ('relax_only', 'light_only', 'combined'):
        config = root / ('config_v19_' + variant + '.json')
        run = out / variant
        run.mkdir(exist_ok=True)
        command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node=2', '-m', 'emotion_diffusion.reference_v19.train',
                   '--config', str(config), '--stage', 'edit', '--resume', checkpoint,
                   '--max-updates', '16']
        with (run / 'edit.log').open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            (out / 'status.json').write_text(json.dumps(dict(variant=variant, pid=os.getpid(),
                child_pid=child.pid, status='running', checkpoint=checkpoint, start=time.time()), indent=2))
            code = child.wait()
        if code:
            (out / 'status.json').write_text(json.dumps(dict(variant=variant, status='failed', exit_code=code)))
            raise SystemExit(code)
        completion = json.loads((run / 'edit_completion.json').read_text())
        assert completion['updates'] == 16
        gradients = completion['first_update_gradients']
        assert not any('qformer' in key or 'content_lora' in key for key in gradients)
        if variant != 'relax_only':
            assert any('photometric.' in key and value > 0 for key, value in gradients.items())
        metrics = json.loads((run / 'validation/step-000016/metrics.json').read_text())
        results.append(dict(variant=variant, completion=completion, validation=metrics))
        (out / 'results.json').write_text(json.dumps(results, indent=2))
    (out / 'status.json').write_text(json.dumps(dict(status='complete', pid=os.getpid(), end=time.time())))


if __name__ == '__main__':
    main()
