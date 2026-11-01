"""Source/output subject detail correspondence while permitting global tone changes."""
import torch
import torch.nn.functional as F


def subject_structure(source, generated, mask, confidence):
    def luminance(x):
        g=(x.float()*x.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
        return (g-g.mean((2,3),keepdim=True))/g.std((2,3),keepdim=True,unbiased=False).clamp_min(.1)
    a,b=luminance(source),luminance(generated)
    weight=F.interpolate(mask.float(),source.shape[-2:],mode='bilinear',align_corners=False).clone()
    weight[:,:,:3]=0;weight[:,:,-3:]=0;weight[:,:,:,:3]=0;weight[:,:,:,-3:]=0
    pairs=[(a-F.avg_pool2d(a,5,1,2),b-F.avg_pool2d(b,5,1,2),weight),
           (a[:,:,:,1:]-a[:,:,:,:-1],b[:,:,:,1:]-b[:,:,:,:-1],weight[:,:,:,1:]),
           (a[:,:,1:]-a[:,:,:-1],b[:,:,1:]-b[:,:,:-1],weight[:,:,1:])]
    loss=generated.new_zeros(())
    for x,y,w in pairs:
        per_image=(F.smooth_l1_loss(x,y,reduction='none')*w).sum((1,2,3))/w.sum((1,2,3)).clamp_min(1)
        loss=loss+(per_image*confidence.flatten()).mean()/len(pairs)
    return loss
