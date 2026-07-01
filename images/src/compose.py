"""Energy-based / compositional guidance for the class-conditional DiT (E3 main build).

In a class-conditional model there is no text prompt, so the faithful analog of
"compose two guidance directions at sampling time" is ComposableDiffusion
(Liu et al. 2022): treat each class conditional as an expert and form the
conjunction (product of experts) score

    eps = eps_uncond + s * (eps_a - eps_uncond) + s * (eps_b - eps_uncond)

where a is an animal class and b is a nature class. This pushes each sample
toward BOTH concepts at once — the natural "telling for two properties at once"
baseline. We then measure its joint hit-rate against the same bars as telling
and showing. If showing (bin-mixing on both-high classes) still wins, composing
guidance directions does not reach the entangled joint.

Re-implements DiTPipeline's sampling loop (see pipeline_dit.py) with 3-way
conditioning per latent. Only the eps channels are composed; the learned-sigma
'rest' channels are dropped (DPMSolver uses the eps model_output only).
"""
import numpy as np
import torch


@torch.no_grad()
def generate_composed(pipe, classes_a, classes_b, scale=4.0, seed=0, batch_size=16, steps=None):
    """classes_a, classes_b: equal-length lists of class ids to compose pairwise.
    Returns list[PIL.Image], one per pair. scale = per-concept guidance weight."""
    assert len(classes_a) == len(classes_b)
    dev = pipe._execution_device
    C = pipe.transformer.config.in_channels
    S = pipe.transformer.config.sample_size
    steps = steps or getattr(pipe, "_num_steps", 50)
    imgs = []
    for i in range(0, len(classes_a), batch_size):
        ca = list(classes_a[i:i + batch_size]); cb = list(classes_b[i:i + batch_size])
        n = len(ca)
        gen = torch.Generator(device=dev).manual_seed(seed + i)
        latents = torch.randn((n, C, S, S), generator=gen, device=dev, dtype=pipe.transformer.dtype)
        cls = torch.tensor(ca + cb + [1000] * n, device=dev)   # [a, b, null]
        pipe.scheduler.set_timesteps(steps)
        for t in pipe.scheduler.timesteps:
            inp = torch.cat([latents] * 3, dim=0)
            inp = pipe.scheduler.scale_model_input(inp, t)
            ts = t
            if not torch.is_tensor(ts):
                ts = torch.tensor([t], dtype=torch.int64, device=dev)
            elif len(ts.shape) == 0:
                ts = ts[None].to(dev)
            ts = ts.expand(inp.shape[0])
            noise = pipe.transformer(inp, timestep=ts, class_labels=cls).sample
            eps = noise[:, :C]
            eps_a, eps_b, eps_u = torch.split(eps, n, dim=0)
            composed = eps_u + scale * (eps_a - eps_u) + scale * (eps_b - eps_u)
            latents = pipe.scheduler.step(composed, t, latents).prev_sample
        lat = 1 / pipe.vae.config.scaling_factor * latents
        samples = pipe.vae.decode(lat).sample
        samples = (samples / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
        imgs.extend(pipe.numpy_to_pil(samples))
    return imgs


@torch.no_grad()
def generate_composed_multi(pipe, class_lists, scale=4.0, seed=0, batch_size=12, steps=None):
    """Energy-based conjunction of K concepts (K>=2): eps_u + scale*sum_k (eps_k - eps_u).
    class_lists: list of K equal-length lists of class ids (one per concept). Returns one PIL
    image per column. Generalizes generate_composed to arbitrary K (used for E11's 3-way)."""
    K = len(class_lists)
    n_total = len(class_lists[0])
    assert all(len(cl) == n_total for cl in class_lists)
    dev = pipe._execution_device
    C = pipe.transformer.config.in_channels
    S = pipe.transformer.config.sample_size
    steps = steps or getattr(pipe, "_num_steps", 50)
    imgs = []
    for i in range(0, n_total, batch_size):
        cols = [cl[i:i + batch_size] for cl in class_lists]
        n = len(cols[0])
        gen = torch.Generator(device=dev).manual_seed(seed + i)
        latents = torch.randn((n, C, S, S), generator=gen, device=dev, dtype=pipe.transformer.dtype)
        cls = torch.tensor(sum([list(c) for c in cols], []) + [1000] * n, device=dev)  # [c1..cK, null]
        pipe.scheduler.set_timesteps(steps)
        for t in pipe.scheduler.timesteps:
            inp = torch.cat([latents] * (K + 1), dim=0)
            inp = pipe.scheduler.scale_model_input(inp, t)
            ts = t
            if not torch.is_tensor(ts):
                ts = torch.tensor([t], dtype=torch.int64, device=dev)
            elif len(ts.shape) == 0:
                ts = ts[None].to(dev)
            ts = ts.expand(inp.shape[0])
            eps = pipe.transformer(inp, timestep=ts, class_labels=cls).sample[:, :C]
            parts = torch.split(eps, n, dim=0)         # K conditionals then uncond
            eps_u = parts[-1]
            composed = eps_u + scale * sum(parts[k] - eps_u for k in range(K))
            latents = pipe.scheduler.step(composed, t, latents).prev_sample
        lat = 1 / pipe.vae.config.scaling_factor * latents
        samples = pipe.vae.decode(lat).sample
        samples = (samples / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
        imgs.extend(pipe.numpy_to_pil(samples))
    return imgs
