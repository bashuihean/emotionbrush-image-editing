"""Source reconstruction and condition-path checks before clean editing restart."""
import argparse,json
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from diffusers import DDIMScheduler
from .model import Editor
from .data import ReferenceDataset
from .train import load_small,move,ssim


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();config=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda');model=Editor(config,device)
    load_small(model,torch.load(args.checkpoint,map_location='cpu',weights_only=False)['model'])
    model.set_stage(True);model.eval();dataset=ReferenceDataset(config,'val',16)
    records=[]
    for i in range(len(dataset)):
        sample=dataset[i];batch=move({k:v[None] if torch.is_tensor(v) else torch.tensor([v]) for k,v in sample.items()},device)
        images={'source':batch['source'],'reference':batch['reference']}
        images['vae']=model.vae.decode(batch['latent']/model.vae.config.scaling_factor).sample
        for name,use_content in [('legacy_scaled',False),('base_no_adapters',False),('content_only',True)]:
            scheduler=DDIMScheduler.from_config(model.scheduler_config,clip_sample=False)
            scheduler.set_timesteps(20,device=device);ts=scheduler.timesteps[int(20*(1-.55)):]
            noise=torch.randn(batch['latent'].shape,device=device,generator=torch.Generator(device=device).manual_seed(config['seed']+i))
            latent=scheduler.add_noise(batch['latent'],noise,ts[:1]);text=model.empty_text
            with torch.autocast('cuda',dtype=torch.bfloat16):
                content=model.condition(batch,text,False)[0] if use_content else None
                for t in ts:
                    if name=='legacy_scaled':
                        eps=model.unet(torch.cat((latent,batch['latent']),1).to(torch.bfloat16),t,
                            encoder_hidden_states=text.to(torch.bfloat16)).sample.float()
                    else:
                        eps=model.predict(latent,batch['latent'],t,text,content,None,None)
                    latent=scheduler.step(eps,t,latent).prev_sample
            images[name]=model.vae.decode(latent/model.vae.config.scaling_factor).sample
        with torch.autocast('cuda',dtype=torch.bfloat16):images['new_full']=model.render(batch,20,config['seed']+i)[0]
        canvas=Image.new('RGB',(256*3,286*3),'white');draw=ImageDraw.Draw(canvas)
        metrics={}
        for j,(name,x) in enumerate(images.items()):
            arr=((x[0].float().cpu().clamp(-1,1)+1)*127.5).byte().permute(1,2,0).numpy()
            canvas.paste(Image.fromarray(arr),((j%3)*256,(j//3)*286+30));draw.text(((j%3)*256+4,(j//3)*286+8),name,fill='black')
            if name not in ('source','reference'):metrics[name]=float(ssim(batch['source'],x))
        canvas.save(out/f'{i:05d}.png');records.append(dict(index=i,ssim=metrics))
    (out/'metrics.json').write_text(json.dumps(records,indent=2))
    print(json.dumps({k:sum(r['ssim'][k] for r in records)/len(records) for k in records[0]['ssim']}),flush=True)


if __name__=='__main__':main()
