"""Soft-prompt (textual-inversion-style) conditioning for DiT — the *learned knob* (round-6 E14).

DiT-XL re-embeds the class label independently in each of its 28 blocks (each block has its own
`class_embedder.embedding_table`, verified: distinct tensors). A faithful continuous conditioning
("soft prompt") is therefore a *per-block* vector set {e_b}, not a single vector.

Mechanism (verified pixel-identical to a real class when e_b := table_b[class]):
  - append ONE scratch row (index SOFT_ID = 1001) to every block's table (null stays 1000);
  - write the per-block soft vectors into that row;
  - generate with `class_labels = [SOFT_ID]*n` (the pipeline's CFG null token 1000 is untouched).

Parameterization of the learned knob (audit-calibrated, low-dim so a gradient-free ES converges):
  e_b(theta) = E_b[best] + sum_k theta_k * (E_b[anchor_k] - E_b[best]),  theta in R^K unconstrained.
theta=0 reproduces the single best (audit) class; theta free lets the optimum leave the convex hull
of the anchor class tokens (genuinely learned, strictly more expressive than any single class). The
anchors are the top-K audit classes for the target. This is optimized to the target scorer, so it is
the *strongest* per-prompt (single-conditioning = telling) knob we can give the reviewer.
"""
import numpy as np
import torch

SOFT_ID = 1001


def install_soft_slot(pipe):
    """Append a writable scratch row (index SOFT_ID) to every block's class-embedding table.
    Returns the list of 28 embedding_table modules (the soft-slot handles)."""
    from diffusers.models.embeddings import LabelEmbedding
    tables = [m.embedding_table for _, m in pipe.transformer.named_modules()
              if isinstance(m, LabelEmbedding)]
    for t in tables:
        w = t.weight.data
        if w.shape[0] > SOFT_ID:            # already installed
            continue
        new = torch.zeros(w.shape[0] + 1, w.shape[1], dtype=w.dtype, device=w.device)
        new[:w.shape[0]] = w
        new[SOFT_ID] = w[0].clone()         # placeholder; set_soft overwrites before use
        t.weight = torch.nn.Parameter(new, requires_grad=False)
        t.num_embeddings = new.shape[0]
    return tables


def class_rows(tables, cls):
    """Per-block embedding rows for a real class: list of 28 tensors [D]."""
    return [t.weight.data[cls].clone() for t in tables]


def set_soft_from_theta(tables, best_rows, anchor_rows, theta):
    """Write e_b(theta) = best_b + sum_k theta_k (anchor_k_b - best_b) into every soft slot.
    best_rows: list[28] of [D]; anchor_rows: list[K] of list[28] of [D]; theta: [K]."""
    for b, t in enumerate(tables):
        e = best_rows[b].clone().float()
        for k in range(len(theta)):
            e = e + float(theta[k]) * (anchor_rows[k][b].float() - best_rows[b].float())
        t.weight.data[SOFT_ID] = e.to(t.weight.dtype)


def optimize_soft(pipe, labeler, target, best_class, anchor_classes, tables,
                  n_es=32, iters=25, sigma0=0.35, seed=0, log=print):
    """(1+1)-ES on theta to MAXIMISE mean(target) of a soft-prompt batch. Returns (theta*, mean*).
    Audit-calibrated: best_class + anchors come from the shadow audit. Warm start theta=0 (=best)."""
    import generate as G
    best_rows = class_rows(tables, best_class)
    anchor_rows = [class_rows(tables, c) for c in anchor_classes]
    K = len(anchor_classes)
    rng = np.random.default_rng(seed)

    def score(theta, es):
        set_soft_from_theta(tables, best_rows, anchor_rows, theta)
        imgs = G.generate(pipe, [SOFT_ID] * n_es, guidance_scale=4.0, seed=es)
        return float(np.asarray(labeler.labels(imgs)[target], float).mean())

    theta = np.zeros(K)
    fbest = score(theta, 7000)               # theta=0 == best audit class, fixed eval seed
    sigma = sigma0
    succ = 0
    for it in range(iters):
        cand = theta + sigma * rng.standard_normal(K)
        fc = score(cand, 7000)               # same eval seed -> low-variance comparison
        if fc > fbest:
            theta, fbest, succ = cand, fc, succ + 1
        # 1/5-success rule step-size control
        if (it + 1) % 5 == 0:
            rate = succ / 5.0
            sigma *= 1.5 if rate > 0.2 else 0.7
            succ = 0
        log(f"    ES it{it+1:2d} sigma={sigma:.3f} f*={fbest:.4f}", flush=True)
    return theta, fbest
