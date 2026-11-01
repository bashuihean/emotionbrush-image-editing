"""Rebuild content adapters then editing, reusing fixed classifiers and frozen caches."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    config=json.loads(Path(args.config).read_text());out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=True)
    for stage in ('content','edit'):
        cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',
             '-m','emotion_diffusion.reference_v19.train','--config',args.config,'--stage',stage]
        if stage=='edit':cmd+=['--resume',str(out/'content_latest.pt')]
        status=dict(stage=stage,revision='v19',status='running',pid=os.getpid(),start=time.time(),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),command=cmd)
        with (out/(stage+'.log')).open('a') as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
            status['child_pid']=child.pid;(out/'pipeline_status.json').write_text(json.dumps(status,indent=2))
            code=child.wait()
        status.update(status='complete' if code==0 else 'failed',exit_code=code,end=time.time())
        (out/(stage+'_pipeline.json')).write_text(json.dumps(status,indent=2))
        if code:
            (out/'pipeline_status.json').write_text(json.dumps(status,indent=2));raise SystemExit(code)
    (out/'pipeline_status.json').write_text(json.dumps(dict(status='complete',revision='v19'),indent=2))


if __name__=='__main__':main()
