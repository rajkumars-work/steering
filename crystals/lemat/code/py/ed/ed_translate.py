"""Inference script for encoder-decoder transformer translation."""

import argparse
import csv
import warnings
import os

import numpy as np
import torch
import torch.nn.functional as F
from ase import Atoms
from ase.io import write as ase_write

from configs import get_device
from ed_data import load_sp
from ed_model import EdGPT
from expert_xattn_model import ExpertXAttnEdGPT
from d_model import DecoderOnlyGPT
from dielectric_data.reader import get_dataset_info, parse_target as dielectric_parse_target


def parse_args():
    p = argparse.ArgumentParser(description="Translate using trained encoder-decoder model")
    p.add_argument("--model", default="checkpoints/ed_ckpt_final.pt", help="Path to model checkpoint (.pt)")
    p.add_argument("--sp_model", default="checkpoints/model_sp.model", help="Path to SentencePiece model (.model)")
    p.add_argument("--input", required=True, help="Input CSV with 'source' column, or text file (one line per source)")
    p.add_argument("--output", help="Output path (default: translations.extxyz or translations.csv)")
    p.add_argument("--csv", action="store_true", dest="use_csv", help="Output CSV instead of extxyz")
    p.add_argument("--n", type=int, default=None, help="Translate only the first N entries (default: all)")
    p.add_argument("--max_length", type=int, default=512, help="Maximum generation length")
    p.add_argument("--top_k", type=int, default=10, help="Top-k sampling (0 for greedy)")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    p.add_argument("--max_tokens", type=int, default=1024, help="Max source token length; entries exceeding this are skipped")
    p.add_argument("--split", default="eval",
                   help="Only use rows with this split value (default: 'eval'). "
                        "Pass --split '' to use all rows.")
    p.add_argument("--version", default=None, help="Manually specify data version (e.g. d6). If None, detected from input.")
    args = p.parse_args()
    if args.output is None:
        args.output = "translations.csv" if args.use_csv else "translations.extxyz"
    return args


def translate_batch(model, sp, src_texts, device, max_length=512, top_k=10):
    """Translate a batch of source strings.

    Dispatches on model type: encoder-decoder (EdGPT, ExpertXAttnEdGPT) uses
    the cached-encoder + decoder-loop path; decoder-only (DecoderOnlyGPT)
    uses the concatenated-prefix path in `translate_batch_d`.
    """
    if isinstance(model, DecoderOnlyGPT) or not hasattr(model, "encode"):
        return translate_batch_d(model, sp, src_texts, device,
                                 max_length=max_length, top_k=top_k)

    model.eval()
    batch_size = len(src_texts)

    # Tokenize all
    src_ids_list = [sp.encode(t, out_type=int) for t in src_texts]
    max_src_len = max(len(ids) for ids in src_ids_list)

    src_tensor = torch.zeros((batch_size, max_src_len), dtype=torch.long, device=device)
    src_mask = torch.zeros((batch_size, max_src_len), dtype=torch.bool, device=device)
    for i, ids in enumerate(src_ids_list):
        src_tensor[i, :len(ids)] = torch.tensor(ids)
        src_mask[i, :len(ids)] = True

    pad_id = sp.pad_id()
    eos_id = sp.eos_id()
    bos_id = sp.bos_id()

    # Decoder input: [BOS, ...]
    tokens = torch.full((batch_size, max_length), pad_id, dtype=torch.long, device=device)
    tokens[:, 0] = bos_id

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    cur_len = 1

    with torch.no_grad():
        encoder_out = model.encode(src_tensor, src_mask).contiguous()

        while cur_len < max_length and not finished.all():
            tgt_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
            tgt_mask[:, :cur_len] = True

            logits, _ = model(tokens, targets=None, tgt_mask=tgt_mask, encoder_out=encoder_out)
            # Only need logits for the last generated token
            logits = logits[:, cur_len - 1, :]

            if top_k == 0:
                # Greedy
                next_tokens = torch.argmax(logits, dim=-1)
            else:
                # Top-k sampling
                probs = F.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
                next_indices = torch.multinomial(top_probs, 1).squeeze(1)
                next_tokens = top_indices[torch.arange(batch_size), next_indices]

            # Update tokens for unfinished sequences
            tokens[~finished, cur_len] = next_tokens[~finished]

            # Check for EOS
            finished |= (next_tokens == eos_id)
            cur_len += 1

    results = []
    for i in range(batch_size):
        row_ids = tokens[i, 1:cur_len].tolist()
        # Trim at first EOS if present
        if eos_id in row_ids:
            row_ids = row_ids[:row_ids.index(eos_id)]
        results.append(sp.decode(row_ids))

    return results


def translate_batch_d(model, sp, src_texts, device, max_length=512, top_k=10):
    """Decoder-only batched generation: concat each source as a prefix.

    No KV cache yet — re-feeds the full prefix at each step. Inference
    cost is O(n²) in total length but acceptable at our target lengths
    (source <20 tokens, target <800 tokens). At benchmark scale of n=1000
    this adds ~15% to wall time vs. the cached-encoder path; flag for
    future optimization (KV cache via `transformers`-style layer-wise cache).
    """
    model.eval()
    B = len(src_texts)

    src_ids_list = [sp.encode(t, out_type=int) for t in src_texts]
    src_lens = [len(ids) for ids in src_ids_list]
    max_src_len = max(src_lens)

    pad_id = sp.pad_id()
    eos_id = sp.eos_id()
    bos_id = sp.bos_id()

    # Cap context at model's tdim.
    tdim = getattr(model, "config", None).tdim if hasattr(model, "config") else 1024
    total_max = min(tdim, max_src_len + 1 + max_length)  # +1 for BOS separator

    # Pre-allocate. Layout per row: [src ; BOS ; tgt_so_far ; PAD...].
    tokens = torch.full((B, total_max), pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(src_ids_list):
        tokens[i, :src_lens[i]] = torch.tensor(ids, device=device)
        tokens[i, src_lens[i]] = bos_id

    # Per-row write pointer (next position to fill). Starts at src_len + 1
    # (after BOS). Per-row attention mask grows as we fill.
    write_ptr = torch.tensor([s + 1 for s in src_lens], dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    src_lens_t = torch.tensor(src_lens, dtype=torch.long, device=device)

    with torch.no_grad():
        while not finished.all() and (write_ptr < total_max).any():
            # padding mask: True at positions [0 .. write_ptr-1] inclusive of source.
            cur_T = int(write_ptr.max().item())
            pos = torch.arange(cur_T, device=device)
            mask = pos.unsqueeze(0) < write_ptr.unsqueeze(1)  # (B, cur_T)

            x = tokens[:, :cur_T]
            logits, _ = model(x, targets=None, key_padding_mask=mask)

            # Per-row, the position to read logits at = write_ptr - 1
            # (predictions for the next-token at position write_ptr).
            row_idx = torch.arange(B, device=device)
            last_logits = logits[row_idx, write_ptr - 1, :]

            if top_k == 0:
                next_tokens = torch.argmax(last_logits, dim=-1)
            else:
                probs = F.softmax(last_logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
                next_idx = torch.multinomial(top_probs, 1).squeeze(1)
                next_tokens = top_indices[row_idx, next_idx]

            # Write next token at write_ptr (only for unfinished + non-overflow rows).
            can_advance = (~finished) & (write_ptr < total_max)
            advance_idx = can_advance.nonzero(as_tuple=True)[0]
            if advance_idx.numel() > 0:
                tokens[advance_idx, write_ptr[advance_idx]] = next_tokens[advance_idx]
                write_ptr[advance_idx] += 1
                finished[advance_idx] |= (next_tokens[advance_idx] == eos_id)

    results = []
    for i in range(B):
        # Generated tokens are at [src_lens[i]+1 .. write_ptr[i]).
        row_ids = tokens[i, src_lens[i] + 1:int(write_ptr[i].item())].tolist()
        if eos_id in row_ids:
            row_ids = row_ids[:row_ids.index(eos_id)]
        results.append(sp.decode(row_ids))

    return results


def main():
    args = parse_args()
    device = get_device()
    
    print(f"Loading SentencePiece model from {args.sp_model}...")
    sp = load_sp(args.sp_model)
    
    print(f"Loading model checkpoint from {args.model}...")
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    
    # Handle DDP prefix if present
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    
    # Check if xattn
    is_xattn = any("attn_x.kv_projs" in k for k in state_dict)
    if is_xattn:
        print("Detected Expert Cross-Attention model architecture")
        model = ExpertXAttnEdGPT(config).to(device)
    else:
        model = EdGPT(config).to(device)
        
    model.load_state_dict(state_dict)
    model.eval()

    # Load inputs
    sources = []
    metadata = [] # List of (origin, id)
    
    version_id = args.version
    if args.input.endswith(".csv"):
        if not version_id:
            ds_info = get_dataset_info(args.input)
            version_id = ds_info.get("version_id", "d6")
            print(f"Detected input dataset version: {version_id}")
            
        with open(args.input, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if args.split:
                    label = row.get("label") or row.get("split", "")
                    if label != args.split:
                        continue
                sources.append(row["source"])
                metadata.append({
                    "origin": row.get("origin", "mp"),
                    "id": row.get("id", "")
                })
    else:
        if not version_id: version_id = "d6"
        with open(args.input, "r") as f:
            for line in f:
                sources.append(line.strip())
                metadata.append({"origin": "mp", "id": ""})

    if args.n:
        sources = sources[:args.n]
        metadata = metadata[:args.n]

    print(f"Translating {len(sources)} entries...")
    
    all_translations = []
    for i in range(0, len(sources), args.batch_size):
        batch_src = sources[i:i+args.batch_size]
        batch_res = translate_batch(model, sp, batch_src, device, args.max_length, args.top_k)
        all_translations.extend(batch_res)
        if (i // args.batch_size) % 10 == 0:
            print(f"  Processed {i + len(batch_src)}/{len(sources)}")

    if args.use_csv:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "target", "origin", "id"])
            for src, tgt, meta in zip(sources, all_translations, metadata):
                writer.writerow([src, tgt, meta["origin"], meta["id"]])
    else:
        all_atoms = []
        parsed_count = 0
        for tgt, meta in zip(all_translations, metadata):
            atoms = dielectric_parse_target(tgt, version_id, meta["origin"])
            if atoms is not None:
                for k, v in meta.items():
                    atoms.info[k] = v
                all_atoms.append(atoms)
                parsed_count += 1
        
        ase_write(args.output, all_atoms)
        print(f"Successfully parsed and wrote {parsed_count}/{len(all_translations)} structures to {args.output}")


if __name__ == "__main__":
    main()
