"""Run only resumed editing; keep existing caches and fixed classifiers untouched."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--warm-start',action='store_true',help='Load adapter weights with a fresh optimizer, including v18 upgrades')
    args=p.parse_args()
    config=json.loads(Path(args.config).read_text())
    out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',
             '-m','emotion_diffusion.reference_v19.train','--config',args.config,'--stage','edit',
             '--resume' if args.warm_start else '--resume-training',args.checkpoint]
    status=dict(stage='edit',revision='v19',status='running',pid=os.getpid(),start=time.time(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),command=command,
                resume_checkpoint=args.checkpoint)
    path=out/'pipeline_status.json'
    path.write_text(json.dumps(status,indent=2))
    with (out/'edit.log').open('a') as f:
        child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
        status['child_pid']=child.pid;path.write_text(json.dumps(status,indent=2))
        code=child.wait()
    status.update(status='complete' if code==0 else 'failed',exit_code=code,end=time.time())
    path.write_text(json.dumps(status,indent=2))
    raise SystemExit(code)


if __name__=='__main__':main()
