"""Unpaired appearance objectives; gradients through frozen generated-image encoders."""
import torch
import torch.nn.functional as F


def editing_losses(editor, batch, generated, gate, routing):
    # Frozen encoder parameters do NOT mean no_grad for the generated image.
    generated_dino = editor.dino(generated)
    content_distance = 1 - F.cosine_similarity(generated_dino, batch['dino'], dim=-1)
    side = int(content_distance.shape[1] ** .5)
    subject = F.adaptive_avg_pool2d(batch['mask'], side).flatten(1)
    confidence = batch['confidence'].reshape(-1, 1)
    weights = .25 + .75 * subject * confidence
    content = (content_distance * weights).sum() / weights.sum().clamp_min(1.)
    from .quality import protection_map, spatial_structure
    protection = protection_map(batch['mask'], batch['confidence'])
    generated_texture, generated_color, generated_atmosphere = editor.appearance.style(generated, 1-protection)
    with torch.no_grad():
        reference_texture, reference_color, reference_atmosphere = editor.appearance.style(batch['reference'])
    # Normalize scales per feature to keep large CNN Gram magnitudes bounded.
    texture = F.smooth_l1_loss(generated_texture / (1 + reference_texture.abs()),
                             reference_texture / (1 + reference_texture.abs()))
    from .photometric import distribution_stats
    # Classifiers/encoder retain their legacy dimensions; only editing loss changes.
    color = F.smooth_l1_loss(distribution_stats(generated), distribution_stats(batch["reference"]))
    # Compare coarse distribution, NOT correspondence of unrelated reference objects.
    atmosphere = F.smooth_l1_loss(generated_atmosphere.mean((2, 3)), reference_atmosphere.mean((2, 3)))
    emotion = F.cross_entropy(editor.supervisor(editor.appearance(generated)), batch['label'])
    route = F.nll_loss(routing.clamp_min(1e-8).log(), batch['label'])
    smooth = (gate[:, :, 1:] - gate[:, :, :-1]).abs().mean() + (gate[:, :, :, 1:] - gate[:, :, :, :-1]).abs().mean()
    structure = spatial_structure(batch['source'], generated, protection, editor.config.get('coarse_structure_weight', 1.))
    # Correspondence belongs to original vs generated, NOT unrelated reference coordinates.
    generated_maps = editor.appearance.maps(generated)[:2]
    with torch.no_grad():
        source_maps = editor.appearance.maps(batch['source'])[:2]
    local = generated.new_zeros(())
    for a, b in zip(generated_maps, source_maps):
        def normalize(x):
            return (x-x.mean((2,3),keepdim=True)) / x.std((2,3),keepdim=True,unbiased=False).clamp_min(.1)
        weight = .15 + .85 * F.interpolate(protection.float(), a.shape[-2:], mode='bilinear', align_corners=False)
        error = F.smooth_l1_loss(normalize(a), normalize(b), reduction='none').mean(1,keepdim=True)
        per_image = (error*weight).sum((1,2,3))/weight.sum((1,2,3)).clamp_min(1)
        local = local + per_image.mean()/len(generated_maps)
    return dict(content=content, texture=texture, color=color, atmosphere=atmosphere,
                emotion=emotion, routing=route, gate_smooth=smooth,
                spatial_structure=structure, spatial_local=local)
