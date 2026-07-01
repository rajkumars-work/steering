"""
DiT-XL/2-256 generation for the ImageNet-domain evidence (reconstruction).

Wraps the diffusers DiTPipeline (facebook/DiT-XL-2-256). Provides:
  - class-conditional sampling at a chosen CFG scale and step count
  - sampling under a recipe mu_P over the 1000 ImageNet classes (telling = point mass;
    showing = any distribution over classes)

Reconstruction choices (recorded for reproducibility): fp16, DPMSolver scheduler,
NUM_STEPS=50 (the original used full DDPM-250; we trade exactness of pixels for
throughput on one A10G — the steering claims are about distributions over many
samples, not single-image fidelity). CFG sweep follows Evidence: {1,3,7,12}.
"""
import numpy as np
import torch

REPO = "facebook/DiT-XL-2-256"
NUM_STEPS = 50


def load_pipe(dtype=torch.float16, steps=NUM_STEPS, device="cuda"):
    from diffusers import DiTPipeline, DPMSolverMultistepScheduler
    pipe = DiTPipeline.from_pretrained(REPO, torch_dtype=dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe._num_steps = steps
    return pipe


@torch.no_grad()
def generate(pipe, class_ids, guidance_scale=4.0, seed=0, batch_size=32):
    """class_ids: list[int] (0..999). Returns list[PIL.Image], one per id."""
    imgs = []
    for i in range(0, len(class_ids), batch_size):
        chunk = list(class_ids[i:i + batch_size])
        g = torch.Generator(device=pipe.device).manual_seed(seed + i)
        out = pipe(class_labels=chunk, guidance_scale=guidance_scale,
                   num_inference_steps=pipe._num_steps, generator=g)
        imgs.extend(out.images)
    return imgs


def sample_mu(mu, n, seed=0):
    """mu: array[1000] of probabilities (need not be normalized), OR dict{class:prob}.
    Returns n class ids drawn ~ mu (the showing recipe)."""
    p = np.zeros(1000, np.float64)
    if isinstance(mu, dict):
        for k, v in mu.items():
            p[int(k)] = v
    else:
        p[:] = np.asarray(mu, np.float64)
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(1000, size=n, p=p).tolist()


def uniform_pool(class_ids):
    """A uniform mu_P over a set of class ids (e.g., a top-k pool or an intersection)."""
    mu = np.zeros(1000, np.float64)
    for c in class_ids:
        mu[int(c)] = 1.0
    return mu


if __name__ == "__main__":
    # Smoke test: download pipeline, generate a few known classes, label them.
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from labelers import Labeler
    pipe = load_pipe()
    # 207 golden retriever, 388 giant panda, 933 cheeseburger, 817 sports car
    ids = [207, 388, 933, 817]
    imgs = generate(pipe, ids, guidance_scale=4.0, seed=0)
    L = Labeler()
    lab = L.labels(imgs)
    keys = L.key(imgs)
    print("generated", len(imgs), "images; sizes", [im.size for im in imgs])
    for j, c in enumerate(ids):
        print(f"  class {c}: resnet_key={keys[j]}  aesthetic={lab['aesthetic'][j]:.2f} "
              f"bright={lab['brightness'][j]:.2f} sim_animal={lab['sim_animal'][j]:.3f} "
              f"sim_vehicle={lab['sim_vehicle'][j]:.3f} sim_food={lab['sim_food'][j]:.3f}")
    print("GEN-SMOKE-OK")
