"""End-to-end test: download Samanantar Hindi-English, train, evaluate with BLEU."""

import argparse
import collections
import csv
import math
import os
import tempfile
from pathlib import Path

import torch

from configs import Config, get_device
from ed_data import build_tokenizer, get_loader, load_sp
from ed_model import EdGPT, get_lr
from ed_train import generate


def parse_args():
    p = argparse.ArgumentParser(description="Test encoder-decoder with Samanantar Hindi-English")
    p.add_argument("--lang", default="hi", help="Samanantar language code")
    p.add_argument("--max_pairs", type=int, default=10000, help="Max training pairs to use")
    p.add_argument("--test_pairs", type=int, default=200, help="Number of test pairs")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs")
    p.add_argument("--vocab_size", type=int, default=16000, help="Vocabulary size")
    p.add_argument("--edim", type=int, default=256, help="Embedding dimension")
    p.add_argument("--layers", type=int, default=4, help="Number of layers")
    p.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size")
    p.add_argument("--max_seq_len", type=int, default=128, help="Max sequence length")
    p.add_argument("--work_dir", default=None, help="Working directory (default: temp)")
    return p.parse_args()


def download_samanantar(lang, max_pairs, test_pairs, work_dir):
    """Download Samanantar and create train/test CSV files."""
    from datasets import load_dataset

    train_csv = os.path.join(work_dir, "train.csv")
    test_csv = os.path.join(work_dir, "test.csv")

    if Path(train_csv).exists() and Path(test_csv).exists():
        print("Using existing CSV files")
        return train_csv, test_csv

    total = max_pairs + test_pairs
    print(f"Downloading Samanantar {lang} ({total} pairs)...")
    ds = load_dataset("ai4bharat/samanantar", lang, split=f"train[:{total}]")

    with open(train_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target"])
        writer.writeheader()
        for i, item in enumerate(ds):
            if i >= max_pairs:
                break
            writer.writerow({"source": item["src"], "target": item["tgt"]})

    with open(test_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target"])
        writer.writeheader()
        for i, item in enumerate(ds):
            if i < max_pairs:
                continue
            writer.writerow({"source": item["src"], "target": item["tgt"]})

    print(f"Created {train_csv} ({max_pairs} pairs) and {test_csv} ({test_pairs} pairs)")
    return train_csv, test_csv


def compute_bleu(references, hypotheses, max_n=4):
    """Compute corpus BLEU score (simplified implementation)."""

    def ngrams(tokens, n):
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    clipped_counts = collections.Counter()
    total_counts = collections.Counter()
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        ref_len += len(ref_tokens)
        hyp_len += len(hyp_tokens)

        for n in range(1, max_n + 1):
            ref_ngrams = collections.Counter(ngrams(ref_tokens, n))
            hyp_ngrams = collections.Counter(ngrams(hyp_tokens, n))
            for ng, count in hyp_ngrams.items():
                clipped_counts[n] += min(count, ref_ngrams.get(ng, 0))
                total_counts[n] += count

    precisions = []
    for n in range(1, max_n + 1):
        if total_counts[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_counts[n] / total_counts[n])

    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n

    # Brevity penalty
    if hyp_len == 0:
        return 0.0
    bp = min(1.0, math.exp(1 - ref_len / hyp_len))
    return bp * math.exp(log_avg) * 100  # return as percentage


def train_model(args, train_csv, work_dir):
    """Train model and return path to checkpoint."""
    sp_prefix = os.path.join(work_dir, "test_sp")
    sp_model_path = sp_prefix + ".model"
    ckpt_path = os.path.join(work_dir, "test_ckpt.pt")

    if Path(ckpt_path).exists():
        print("Using existing checkpoint")
        return ckpt_path, sp_model_path

    # Build tokenizer
    if not Path(sp_model_path).exists():
        print("Training tokenizer...")
        build_tokenizer(train_csv, sp_prefix, args.vocab_size)

    sp = load_sp(sp_model_path)
    vocab_size = sp.get_piece_size()

    c = Config(
        name="test",
        edim=args.edim,
        layers=args.layers,
        heads=args.heads,
        tdim=args.max_seq_len,
        dropout=0.1,
        vocab_size=vocab_size,
        B=args.batch_size,
        lr=6e-4 * 3,
    )

    device = torch.device(get_device())
    model = EdGPT(c).to(device)
    model.train()
    device_type = device.type

    optimizer = model.configure_optimizers(0.1, c.lr, device_type)
    loader = get_loader(train_csv, sp_model_path, args.batch_size, args.max_seq_len)

    from contextlib import nullcontext
    import itertools

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    steps_per_epoch = len(loader)
    max_steps = steps_per_epoch * args.epochs
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params, {steps_per_epoch} steps/epoch, {max_steps} total steps")

    loader_iter = itertools.cycle(iter(loader))
    for step in range(1, max_steps + 1):
        batch = next(loader_iter)
        src, dec_in, labels, src_mask, tgt_mask = batch
        src, dec_in, labels = src.to(device), dec_in.to(device), labels.to(device)
        src_mask, tgt_mask = src_mask.to(device).bool(), tgt_mask.to(device).bool()

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            _, loss = model(dec_in, src, targets=labels, src_mask=src_mask, tgt_mask=tgt_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step, max_steps=max_steps, max_lr=c.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.step()

        if step % 100 == 0:
            print(f"  step {step}/{max_steps} loss {loss.item():.3f}")

    torch.save({"model": model.state_dict(), "config": c}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")
    return ckpt_path, sp_model_path


def evaluate(ckpt_path, sp_model_path, test_csv, max_seq_len):
    """Evaluate model on test set, compute BLEU, print samples."""
    device = torch.device(get_device())
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = EdGPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sp = load_sp(sp_model_path)

    sources, references = [], []
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sources.append(row["source"])
            references.append(row["target"])

    print(f"\nEvaluating on {len(sources)} test pairs...")
    hypotheses = []
    for i, src in enumerate(sources):
        hyp = generate(model, sp, src, device, max_length=max_seq_len)
        hypotheses.append(hyp)
        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{len(sources)}")

    bleu = compute_bleu(references, hypotheses)

    print(f"\n{'=' * 60}")
    print(f"  BLEU Score: {bleu:.2f}")
    print(f"{'=' * 60}")

    print("\nSample translations:")
    for i in range(min(5, len(sources))):
        print(f"\n  Source:     {sources[i][:100]}")
        print(f"  Reference:  {references[i][:100]}")
        print(f"  Generated:  {hypotheses[i][:100]}")

    return bleu


def main():
    args = parse_args()

    if args.work_dir:
        work_dir = args.work_dir
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = tempfile.mkdtemp(prefix="ed_test_")

    print(f"Working directory: {work_dir}")

    train_csv, test_csv = download_samanantar(
        args.lang, args.max_pairs, args.test_pairs, work_dir
    )
    ckpt_path, sp_model_path = train_model(args, train_csv, work_dir)
    bleu = evaluate(ckpt_path, sp_model_path, test_csv, args.max_seq_len)
    return bleu


if __name__ == "__main__":
    main()
