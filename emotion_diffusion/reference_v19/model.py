"""QFormer, appearance branches and separately routed content/style low-rank attention."""
import math
import torch
from torch import nn
import torch.nn.functional as F

from .appearance import color_stats


def cnn(channels):
    return nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
                         nn.Conv2d(32, channels, 3, 2, 1), nn.GroupNorm(8, channels), nn.SiLU())


class StyleEncoder(nn.Module):
    """Trainable reference CNNs remain online; frozen features never replace their outputs."""
    def __init__(self, dim=768):
        super().__init__()
        self.texture_cnn = cnn(64)
        self.color_cnn = cnn(32)
        self.atmosphere_cnn = cnn(64)
        self.texture = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 4 * dim))
        self.color = nn.Sequential(nn.LayerNorm(104), nn.Linear(104, 2 * dim))
        self.atmosphere = nn.Linear(64, dim)
        self.reference_dino = nn.Linear(384, dim)
        self.semantic_strength = nn.Parameter(torch.tensor(.02))
        self.norm = nn.LayerNorm(dim)
        self.branch_strength = nn.Parameter(torch.ones(3))
        self.dim = dim

    def forward(self, reference, reference_dino=None):
        b = len(reference)
        textures = []
        for grid in (2, 4):
            # True multiscale patches, order discarded by pooled stats.
            x = F.interpolate(reference.float(), (grid * 32, grid * 32), mode='bilinear', align_corners=False)
            patches = x.unfold(2, 32, 32).unfold(3, 32, 32)
            patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(b * grid * grid, 3, 32, 32)
            f = self.texture_cnn(patches).reshape(b, grid * grid, 64, -1)
            flat = f.permute(0, 2, 1, 3).flatten(2)
            textures.append(torch.cat((flat.mean(-1), flat.std(-1, unbiased=False),
                                        flat.square().mean(-1)), 1))
        tex = self.texture(torch.stack(textures).mean(0)).reshape(b, 4, self.dim)
        low = F.interpolate(reference.float(), (32, 32), mode='bilinear', align_corners=False)
        col = torch.cat((color_stats(reference), self.color_cnn(low).mean((2, 3))), 1)
        col = self.color(col).reshape(b, 2, self.dim)
        atm = self.atmosphere_cnn(F.interpolate(reference.float(), (96, 96), mode='bilinear', align_corners=False))
        atm = self.atmosphere(F.adaptive_avg_pool2d(atm, 2).flatten(2).transpose(1, 2))
        if reference_dino is not None:
            atm = atm + .1 * self.semantic_strength.tanh() * self.reference_dino(reference_dino.float())[:, None]
        strength = self.branch_strength.sigmoid()
        return self.norm(torch.cat((tex * strength[0], col * strength[1], atm * strength[2]), 1))


class Conditions(nn.Module):
    def __init__(self, dino_dim=384, dim=768):
        super().__init__()
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        self.content = nn.Sequential(nn.LayerNorm(dino_dim), nn.Linear(dino_dim, dim))
        self.position = nn.Linear(2, dim, bias=False)
        self.region = nn.Linear(3, dim)
        self.queries = nn.Parameter(torch.randn(1, 32, dim) * .02)
        self.qformer = Blip2QFormerModel(Blip2QFormerConfig(hidden_size=dim,
            encoder_hidden_size=dim, num_hidden_layers=2, num_attention_heads=8,
            intermediate_size=1536, cross_attention_frequency=1))
        self.style = StyleEncoder(dim)
        from .photometric import PhotometricBranch
        self.photometric = PhotometricBranch(dim)
        self.brush = nn.Parameter(torch.randn(8, 4, dim) * .02)
        self.router = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 8))
        self.brush_projection = nn.Linear(1992, dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2 + 3, 64), nn.SiLU(), nn.Linear(64, 1))
        self.norm = nn.LayerNorm(dim)

    def spatial(self, batch, emotion):
        dino = batch['dino'].float()
        side = math.isqrt(dino.shape[1])
        axis = torch.linspace(-1, 1, side, device=dino.device)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        pos = torch.stack((xx, yy), -1).reshape(1, -1, 2)
        mask = F.adaptive_avg_pool2d(batch['mask'].float(), side).flatten(2).transpose(1, 2)
        edge = F.adaptive_avg_pool2d(batch['edge'].float(), side).flatten(2).transpose(1, 2)
        conf = batch['confidence'].reshape(-1, 1, 1).expand(-1, side * side, 1)
        region = torch.cat((mask, edge, conf), -1)
        content = self.content(dino) + self.position(pos) + self.region(region)
        return content, region, side

    def forward(self, batch, emotion, editing=True):
        content, region, side = self.spatial(batch, emotion)
        emo = emotion.float().mean(1)
        # Content queries must be identical across emotion/reference conditions.
        queries = self.queries.expand(len(content), -1, -1)
        fused = self.qformer(query_embeds=queries, encoder_hidden_states=content,
                              encoder_attention_mask=torch.ones(content.shape[:2], device=content.device)).last_hidden_state
        content_tokens = self.norm(torch.cat((content, fused), 1))
        if not editing:
            return content_tokens, None, None, None
        weights = self.router(emo).softmax(-1)
        brush = torch.einsum('be,ekd->bkd', weights, self.brush)
        style = self.style(batch['reference'], batch.get('reference_dino'))
        style_tokens = self.norm(torch.cat((style, brush), 1))
        logits = self.gate(torch.cat((content, emo[:, None].expand_as(content), region), -1))
        # Soft prior, never a hard pixel lock; uncertainty weakens subject protection.
        prior = 1 - .65 * region[..., :1] * region[..., 2:3]
        gate = (.15 + .85 * logits.sigmoid()) * prior
        return content_tokens, style_tokens, gate.transpose(1, 2).reshape(-1, 1, side, side), weights


class LoRA(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8):
        super().__init__()
        self.a = nn.Linear(in_dim, rank, bias=False)
        self.b = nn.Linear(rank, out_dim, bias=False)
        nn.init.normal_(self.a.weight, std=.02)
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.b(self.a(x.float()))


class RoutedAttention(nn.Module):
    """Two genuine low-rank Q/K/V adapters with different condition consumers.

    Both add to frozen projections. Content reads only source/QFormer;
    style reads only reference/brush. Both global and local style residuals receive soft subject protection.
    """
    def __init__(self, hidden, context, rank=8):
        super().__init__()
        from diffusers.models.attention_processor import AttnProcessor2_0
        self.base = AttnProcessor2_0()
        self.content_lora = nn.ModuleDict({'q': LoRA(hidden, hidden, rank),
            'k': LoRA(context, hidden, rank), 'v': LoRA(context, hidden, rank)})
        self.style_lora = nn.ModuleDict({'q': LoRA(hidden, hidden, rank),
            'k': LoRA(context, hidden, rank), 'v': LoRA(context, hidden, rank)})
        self.content_strength = nn.Parameter(torch.tensor(.05))
        self.style_strength = nn.Parameter(torch.tensor(.05))
        self.local_style_strength = nn.Parameter(torch.tensor(.02))

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 temb=None, content_context=None, style_context=None, region_map=None, subject_protection=None):
        base = self.base(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        if content_context is None:
            return base
        b, n, h = hidden_states.shape
        def attend(context, branch):
            q = attn.to_q(hidden_states) + branch['q'](hidden_states).to(hidden_states.dtype)
            k = attn.to_k(context.to(hidden_states.dtype)) + branch['k'](context).to(hidden_states.dtype)
            v = attn.to_v(context.to(hidden_states.dtype)) + branch['v'](context).to(hidden_states.dtype)
            q = q.reshape(b, n, attn.heads, -1).transpose(1, 2)
            k = k.reshape(b, -1, attn.heads, q.shape[-1]).transpose(1, 2)
            v = v.reshape(b, -1, attn.heads, q.shape[-1]).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, h)
            # Cancel the frozen projection bias for a true residual.
            out = attn.to_out[0](out) - attn.to_out[0](torch.zeros_like(out))
            return attn.to_out[1](out) / attn.rescale_output_factor
        extra = self.content_strength.tanh() * attend(content_context, self.content_lora)
        if style_context is not None:
            style = attend(style_context, self.style_lora)
            side = math.isqrt(n)
            gate = F.interpolate(region_map.float(), (side, side), mode='bilinear', align_corners=False)
            gate = gate.flatten(2).transpose(1, 2).to(style.dtype)
            coefficient = self.style_strength.tanh() + self.local_style_strength.tanh() * gate
            if subject_protection is not None:
                subject = F.interpolate(subject_protection.float(), (side, side), mode='bilinear', align_corners=False)
                subject = subject.flatten(2).transpose(1, 2).to(style.dtype)
                coefficient = coefficient * (1 - .8 * subject.clamp(0, 1))
            extra = extra + coefficient * style
        return base + extra.to(base.dtype)


class SelfAttention:
    def __init__(self):
        from diffusers.models.attention_processor import AttnProcessor2_0
        self.base = AttnProcessor2_0()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 temb=None, content_context=None, style_context=None, region_map=None, subject_protection=None):
        return self.base(attn, hidden_states, encoder_hidden_states, attention_mask, temb)


class Editor(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler
        from ..spatial_model import DinoEncoder
        from .appearance import FrozenAppearance, EmotionHead
        self.config = config
        base = config['base_model']
        self.vae = AutoencoderKL.from_pretrained(base, subfolder='vae').requires_grad_(False).eval()
        self.unet = UNet2DConditionModel.from_pretrained(base, subfolder='unet').requires_grad_(False)
        if self.unet.config.in_channels != 8:
            raise ValueError('Expected pretrained InstructPix2Pix 8-channel UNet')
        self.scheduler_config = DDPMScheduler.from_pretrained(base, subfolder='scheduler').config
        self.noise_scheduler = DDPMScheduler.from_config(self.scheduler_config)
        self.condition = Conditions()
        processors = {}
        for name in self.unet.attn_processors:
            if name.endswith('attn1.processor'):
                processors[name] = SelfAttention()
                continue
            if name.startswith('mid_block'):
                hidden = self.unet.config.block_out_channels[-1]
            elif name.startswith('up_blocks'):
                hidden = list(reversed(self.unet.config.block_out_channels))[int(name.split('.')[1])]
            else:
                hidden = self.unet.config.block_out_channels[int(name.split('.')[1])]
            processors[name] = RoutedAttention(hidden, 768, config.get('rank', 8))
        self.unet.set_attn_processor(processors)
        self.dino = DinoEncoder(config['dino_model']).requires_grad_(False).eval()
        self.appearance = FrozenAppearance('resnet18', config['appearance_a_weights'])
        self.supervisor = EmotionHead()
        checkpoint = torch.load(config['classifier_a'], map_location='cpu', weights_only=True)
        self.supervisor.load_state_dict(checkpoint['state'])
        self.supervisor.requires_grad_(False).eval()
        text = torch.load(config['text_cache'], map_location='cpu', weights_only=True)
        self.register_buffer('emotion_text', text['emotion'].float())
        self.register_buffer('empty_text', text['empty'].float())
        self.to(device)
        self.unet.to(dtype=torch.bfloat16)
        # Trainable small adapters stay fp32. VAE/DINO remain fp32 for stable losses.
        self.condition.float()
        for processor in self.unet.attn_processors.values():
            if isinstance(processor, nn.Module):
                processor.float()
        if config.get('gradient_checkpointing', True):
            self.unet.enable_gradient_checkpointing()

    def set_stage(self, editing):
        self.condition.requires_grad_(False)
        if editing:
            for module in (self.condition.style, self.condition.router, self.condition.gate):
                module.requires_grad_(True)
            self.condition.brush.requires_grad_(True)
            self.condition.photometric.requires_grad_(self.config.get("photometric_enabled", True))
        else:
            for module in (self.condition.content, self.condition.position, self.condition.region,
                           self.condition.qformer, self.condition.norm):
                module.requires_grad_(True)
            self.condition.queries.requires_grad_(True)
        for processor in self.unet.attn_processors.values():
            if isinstance(processor, RoutedAttention):
                processor.content_lora.requires_grad_(not editing)
                processor.content_strength.requires_grad_(not editing)
                processor.style_lora.requires_grad_(editing)
                processor.style_strength.requires_grad_(editing)
                processor.local_style_strength.requires_grad_(editing)
        self.editing = editing

    def train(self, mode=True):
        super().train(mode)
        for module in (self.vae, self.dino, self.appearance, self.supervisor):
            module.eval()
        if getattr(self, 'editing', False):
            self.condition.qformer.eval()  # Frozen content must not vary with dropout.
        return self

    def predict(self, latent, source, timestep, text, content, style, gate, subject_protection=None):
        from .quality import pack_unet_input
        return self.unet(pack_unet_input(latent, source, self.vae.config.scaling_factor).to(torch.bfloat16), timestep,
            encoder_hidden_states=text.to(torch.bfloat16),
            cross_attention_kwargs={'content_context': content, 'style_context': style,
                                    'region_map': gate, 'subject_protection': subject_protection}).sample.float()

    def render(self, batch, steps=None, seed=None):
        from diffusers import DDIMScheduler
        scheduler = DDIMScheduler.from_config(self.scheduler_config, clip_sample=False)
        scheduler.set_timesteps(steps or self.config['train_sampling_steps'], device=batch['latent'].device)
        start = int(len(scheduler.timesteps) * (1 - self.config['edit_strength']))
        timesteps = scheduler.timesteps[start:]
        generator = None if seed is None else torch.Generator(device=batch['latent'].device).manual_seed(seed)
        noise = torch.randn(batch['latent'].shape, device=batch['latent'].device, generator=generator)
        latent = scheduler.add_noise(batch['latent'].float(), noise, timesteps[:1])
        text = self.emotion_text[batch['label']]
        content, style, gate, routing = self.condition(batch, text, True)
        for timestep in timesteps:
            from .quality import protection_map
            protection = protection_map(batch['mask'], batch['confidence'])
            epsilon = self.predict(latent, batch['latent'], timestep, text, content, style, gate, protection)
            latent = scheduler.step(epsilon, timestep, latent).prev_sample
        image = self.vae.decode(latent.float() / self.vae.config.scaling_factor).sample
        self.photometric_regularizer = image.new_zeros(())
        if self.config.get("photometric_enabled", True):
            image, self.photometric_regularizer = self.condition.photometric(image, batch["source"], batch["reference"], style)
        return image, gate, routing

    def forward(self, batch):
        if not self.editing:
            text = self.empty_text.expand(len(batch['label']), -1, -1)
            content, _, _, _ = self.condition(batch, text, False)
            t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (len(text),), device=text.device)
            noise = torch.randn_like(batch['latent'])
            noisy = self.noise_scheduler.add_noise(batch['latent'], noise, t)
            pred = self.predict(noisy, batch['latent'], t, text, content, None, None)
            target = noise if self.noise_scheduler.config.prediction_type == 'epsilon' else self.noise_scheduler.get_velocity(batch['latent'], noise, t)
            loss = F.mse_loss(pred, target)
            return loss, {'reconstruction_noise': loss.detach()}
        from .losses import editing_losses
        image, gate, routing = self.render(batch)
        values = editing_losses(self, batch, image, gate, routing)
        values["photometric_regularizer"] = self.photometric_regularizer
        # Source is the anchor only within confident subject regions; reference is NOT a target.
        text = self.emotion_text[batch['label']]
        content, style, gate, _ = self.condition(batch, text, True)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (len(text),), device=text.device)
        noise = torch.randn_like(batch['latent'])
        noisy = self.noise_scheduler.add_noise(batch['latent'], noise, t)
        from .quality import protection_map, dense_noise_loss
        protection = protection_map(batch['mask'], batch['confidence'])
        pred = self.predict(noisy, batch['latent'], t, text, content, style, gate, protection)
        target = noise if self.noise_scheduler.config.prediction_type == 'epsilon' else self.noise_scheduler.get_velocity(batch['latent'], noise, t)
        values['dense_noise'] = dense_noise_loss(pred, target, protection)
        # Frozen source/content teacher, without style residuals, on the SAME noise/time/text.
        # This is a prior regularizer, not a ground-truth edited image or a quality score.
        with torch.no_grad():
            teacher = self.predict(noisy, batch['latent'], t, text, content, None, None)
        values['noise_prior'] = F.mse_loss(pred, teacher)
        loss = sum(self.config['loss_weights'][key] * value for key, value in values.items())
        return loss, {key: value.detach() for key, value in values.items()}
