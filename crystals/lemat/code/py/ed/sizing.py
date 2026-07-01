"""Suggest model hyperparameters based on dataset statistics."""

import argparse
import csv
import math

import sentencepiece as spm


def parse_args():
    p = argparse.ArgumentParser(description="Suggest model parameters from dataset")
    p.add_argument("--data", required=True, help="Path to CSV with 'source' and 'target' columns")
    p.add_argument("--sp_model", default=None, help="Path to SentencePiece model (for token-level stats)")
    return p.parse_args()


def analyze_dataset(csv_path, sp=None):
    src_lens = []
    tgt_lens = []
    src_char_lens = []
    tgt_char_lens = []
    chars = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src, tgt = row["source"], row["target"]
            src_char_lens.append(len(src))
            tgt_char_lens.append(len(tgt))
            chars.update(src)
            chars.update(tgt)
            if sp:
                src_lens.append(len(sp.encode(src, out_type=int)))
                tgt_lens.append(len(sp.encode(tgt, out_type=int)))

    num_rows = len(src_char_lens)
    stats = {
        "num_rows": num_rows,
        "unique_chars": len(chars),
        "avg_src_chars": sum(src_char_lens) / max(num_rows, 1),
        "avg_tgt_chars": sum(tgt_char_lens) / max(num_rows, 1),
        "max_src_chars": max(src_char_lens) if src_char_lens else 0,
        "max_tgt_chars": max(tgt_char_lens) if tgt_char_lens else 0,
    }

    if sp and src_lens:
        all_lens = src_lens + tgt_lens
        all_lens.sort()
        p95_idx = int(0.95 * len(all_lens))
        stats.update({
            "avg_src_tokens": sum(src_lens) / max(len(src_lens), 1),
            "avg_tgt_tokens": sum(tgt_lens) / max(len(tgt_lens), 1),
            "max_src_tokens": max(src_lens),
            "max_tgt_tokens": max(tgt_lens),
            "p95_tokens": all_lens[p95_idx],
        })

    return stats


def suggest_params(stats):
    n = stats["num_rows"]

    # Vocab size: scale with corpus size
    if n < 5_000:
        vocab_size = 8_000
    elif n < 50_000:
        vocab_size = 16_000
    else:
        vocab_size = 32_000

    # Model dimensions based on dataset size
    if n < 10_000:
        edim, layers = 256, 4
    elif n < 100_000:
        edim, layers = 512, 6
    else:
        edim, layers = 768, 12

    heads = edim // 64

    # Max sequence length from token stats or char heuristic
    if "p95_tokens" in stats:
        max_seq_len = min(2048, int(math.ceil(stats["p95_tokens"] / 64) * 64))
        max_seq_len = max(128, max_seq_len)
    else:
        avg_chars = max(stats["avg_src_chars"], stats["avg_tgt_chars"])
        max_seq_len = min(2048, max(128, int(math.ceil(avg_chars * 0.5 / 64) * 64)))

    # Batch size heuristic (smaller for longer sequences)
    if max_seq_len <= 128:
        batch_size = 32
    elif max_seq_len <= 512:
        batch_size = 16
    elif max_seq_len <= 1024:
        batch_size = 4
    else:
        batch_size = 2

    # Epochs: more data = fewer epochs
    if n < 5_000:
        epochs = 50
    elif n < 50_000:
        epochs = 20
    elif n < 500_000:
        epochs = 10
    else:
        epochs = 3

    # Estimate parameter count
    # Rough: 2 * layers * (4*edim^2 + 2*edim^2) + 2*vocab_size*edim
    params = 2 * layers * (6 * edim * edim) + 2 * vocab_size * edim
    params_m = params / 1e6

    return {
        "vocab_size": vocab_size,
        "edim": edim,
        "layers": layers,
        "heads": heads,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "epochs": epochs,
        "est_params_M": round(params_m, 1),
    }


def print_table(title, data):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    max_key = max(len(str(k)) for k in data)
    for k, v in data.items():
        if isinstance(v, float):
            print(f"  {k:<{max_key}}  {v:>12.1f}")
        else:
            print(f"  {k:<{max_key}}  {v:>12}")
    print()


def main():
    args = parse_args()

    sp = None
    if args.sp_model:
        sp = spm.SentencePieceProcessor()
        sp.load(args.sp_model)

    stats = analyze_dataset(args.data, sp)
    suggestions = suggest_params(stats)

    print_table("Dataset Statistics", stats)
    print_table("Suggested Parameters", suggestions)

    print("Example command:")
    print(f"  python ed_train.py --data {args.data} "
          f"--vocab_size {suggestions['vocab_size']} "
          f"--edim {suggestions['edim']} "
          f"--layers {suggestions['layers']} "
          f"--heads {suggestions['heads']} "
          f"--max_seq_len {suggestions['max_seq_len']} "
          f"--batch_size {suggestions['batch_size']} "
          f"--epochs {suggestions['epochs']}")


if __name__ == "__main__":
    main()
