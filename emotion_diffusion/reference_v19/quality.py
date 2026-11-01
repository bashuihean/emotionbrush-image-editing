"""Dense weak constraints with stronger subject priorities; never a hard pixel lock."""
import torch
import torch.nn.functional as F


def pack_unet_input(noisy_latent, cached_source_latent, scaling_factor):
    # Cache stores scaled diffusion latent. IP2P image-condition channels expect raw VAE latent.
    return torch.cat((noisy_latent, cached_source_latent / scaling_factor), 1)


def protection_map(mask, confidence):
    # Dilate soft foreground to include nearby hands/thin parts; uncertainty retains a floor.
    halo=F.max_pool2d(mask.float(),5,1,2)
    return (.15+.85*halo*confidence.reshape(-1,1,1,1).float()).clamp(0,1)


def dense_noise_loss(prediction, target, protection):
    weight=.2+.8*F.interpolate(protection.float(),prediction.shape[-2:],mode='bilinear',align_corners=False)
    error=(prediction.float()-target.float()).square().mean(1,keepdim=True)
    return (error*weight).sum()/weight.sum().clamp_min(1)


def spatial_structure(source, generated, protection, coarse_weight=1.):
    def gray(x):
        g=(x.float()*x.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        return (g-g.mean((2,3),keepdim=True))/g.std((2,3),keepdim=True,unbiased=False).clamp_min(.1)
    a,b=gray(source),gray(generated)
    weight=.15+.85*F.interpolate(protection.float(),a.shape[-2:],mode='bilinear',align_corners=False)
    # Coarse luminance layout, not RGB matching: allow global color/brightness changes.
    coarse=generated.new_zeros(())
    for size in (16,32):
        x,y=F.adaptive_avg_pool2d(a,size),F.adaptive_avg_pool2d(b,size)
        w=F.adaptive_avg_pool2d(weight,size)
        coarse=coarse+(F.smooth_l1_loss(x,y,reduction='none')*w).sum()/w.sum().clamp_min(1)/2
    kernel=a.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])[None,None]/8
    def gradient(x):
        return torch.cat((F.conv2d(x,kernel,padding=1),F.conv2d(x,kernel.transpose(2,3),padding=1)),1)
    ga,gb=gradient(a),gradient(b)
    # Direction only at existing source edges; no artificial edge creation requirement.
    edge_strength=ga.square().sum(1,keepdim=True).sqrt().detach()
    w=weight*(edge_strength>.04)
    w=w.clone();w[:,:,:2]=0;w[:,:,-2:]=0;w[:,:,:,:2]=0;w[:,:,:,-2:]=0
    direction=(1-F.cosine_similarity(ga,gb,dim=1,eps=1e-4)).clamp_min(0)[:,None]
    direction=(direction*w).sum()/w.sum().clamp_min(1)
    # One-sided anti-blur: retain a fraction of original edge energy, not every texture.
    missing=(.6*edge_strength-gb.square().sum(1,keepdim=True).add(1e-8).sqrt()).relu()
    anti_blur=(missing*w).sum()/w.sum().clamp_min(1)
    return coarse_weight*coarse+.25*direction+.25*anti_blur
