"""General-purpose data loading for encoder-decoder translation from CSV files."""

import csv
import os
import random
import tempfile
from pathlib import Path
from collections import Counter

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from dielectric_data.reader import get_dataset_info
from dielectric_data.versions import (
    get_version,
    S_ANON_FORMULA_ELEMENTS,
    S_ELEMENTS_ONLY,
    S_EREF,
    S_PROP_LABELS,
    S_PROP_LABELS_IC,
)

# Set CSV field size limit globally to handle large atomic coordinate strings (e.g., 4MB+ rows)
csv.field_size_limit(10 * 10**6)  # 10MB limit

# ---------------------------------------------------------------------------
# Source-field dropout
# ---------------------------------------------------------------------------

def _dropout_source(
    src_text: str,
    p: float = 0.2,
    version_id: str = "d6",
    origin: str = "mp",
    p_label: float = 0.0,
) -> str:
    """Drop pipe-separated source segments (and, optionally, individual
    property labels inside the kept props segment).

    Outer loop: every segment is dropped independently with probability `p`,
    except the composition (elements) and Eref segments which are dropped
    together as a unit.

    Inner loop (new): when the property-labels segment survives the outer
    drop and `p_label > 0`, each individual label inside that segment is
    independently dropped with probability `p_label`. This exposes the model
    to partial-label prompts (e.g., "wide-gap stable" with no eps_0 label)
    without going fully unconditional.

    If the inner drop empties the props segment, the segment is omitted —
    equivalent to the outer segment-level drop.
    """
    if p <= 0 and p_label <= 0:
        return src_text
    segments = [s.strip() for s in src_text.split("|")]
    if len(segments) <= 1:
        return src_text

    try:
        spec = get_version(version_id)
        source_fields = spec.source_segments.get(origin, spec.source_segments["mp"])
    except Exception:
        # Fallback to legacy heuristic if version lookup fails
        source_fields = []

    # Identify segment types
    elem_idx = None
    eref_idx = None
    prop_idx = None

    if source_fields:
        for i, field in enumerate(source_fields):
            if field in (S_ANON_FORMULA_ELEMENTS, S_ELEMENTS_ONLY):
                elem_idx = i
            elif field == S_EREF:
                eref_idx = i
            elif field in (S_PROP_LABELS, S_PROP_LABELS_IC):
                prop_idx = i
    else:
        # Legacy heuristic
        for i, seg in enumerate(segments):
            if seg.startswith("Eref "):
                eref_idx = i
            elif i == 1:
                elem_idx = i
        # Legacy: assume props segment is the last one
        if len(segments) >= 4:
            prop_idx = len(segments) - 1

    # Outer: decide segment-level drops
    drop_composition = random.random() < p
    kept = []
    for i, seg in enumerate(segments):
        if i == elem_idx or i == eref_idx:
            if not drop_composition:
                kept.append(seg)
            continue

        # Outer segment drop
        if p > 0 and random.random() < p:
            continue

        # Inner per-label drop, only on the props segment
        if i == prop_idx and p_label > 0 and seg:
            labels = seg.split()
            kept_labels = [lbl for lbl in labels if random.random() >= p_label]
            if kept_labels:
                kept.append(" ".join(kept_labels))
            # if all labels dropped, omit segment entirely
        else:
            kept.append(seg)

    # If everything was dropped, keep at least the first segment
    if not kept:
        kept.append(segments[0])

    return " | ".join(kept)


class TranslationDataset(Dataset):
    """Reads a CSV with "source"/"target" columns and tokenizes with SentencePiece.

    Automatically detects dataset version and origin for robust processing.
    """

    def __init__(self, csv_path: str, sp_model_path: str, max_seq_len: int = 1024,
                 prop_dropout: float = 0.0, emit_wp_class: bool = False,
                 split: str = None, label_dropout: float = 0.0):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        self.max_seq_len = max_seq_len
        self.prop_dropout = prop_dropout
        self.label_dropout = label_dropout
        self.emit_wp_class = emit_wp_class

        # Detect version info
        self.ds_info = get_dataset_info(csv_path)
        self.version_id = self.ds_info.get("version_id", "d6")

        self.pairs = [] # List of (source, target, origin)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split:
                    row_label = row.get("label") or row.get("split", "")
                    if row_label != split:
                        continue
                origin = row.get("origin", "mp")
                self.pairs.append((row["source"], row["target"], origin))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text, origin = self.pairs[idx]
        if self.prop_dropout > 0 or self.label_dropout > 0:
            src_text = _dropout_source(
                src_text, self.prop_dropout, self.version_id, origin,
                p_label=self.label_dropout,
            )
            
        src_ids = self.sp.encode(src_text, out_type=int)[: self.max_seq_len]
        tgt_ids = self.sp.encode(tgt_text, out_type=int)[: self.max_seq_len - 1]
        dec_in_ids = [self.sp.bos_id()] + tgt_ids
        label_ids = tgt_ids + [self.sp.eos_id()]

        if not self.emit_wp_class:
            return src_ids, dec_in_ids, label_ids

        from wp_class_builder import build_wp_class_sequence
        pieces = [self.sp.id_to_piece(tid) for tid in tgt_ids]
        tgt_classes = build_wp_class_sequence(pieces)
        wp_class = [0] + tgt_classes
        return src_ids, dec_in_ids, label_ids, wp_class


def collate_fn(batch, max_seq_len=1024, emit_wp_class=False):
    """Pads sequences in a batch to max_seq_len and emits masks expected by training."""
    # Find actual max lengths in this batch
    max_src = min(max_seq_len, max(len(x[0]) for x in batch))
    max_tgt = min(max_seq_len, max(len(x[1]) for x in batch))

    batch_size = len(batch)
    src_ids = torch.zeros((batch_size, max_src), dtype=torch.long)
    dec_in_ids = torch.zeros((batch_size, max_tgt), dtype=torch.long)
    label_ids = torch.full((batch_size, max_tgt), -100, dtype=torch.long)
    src_mask = torch.zeros((batch_size, max_src), dtype=torch.bool)
    tgt_mask = torch.zeros((batch_size, max_tgt), dtype=torch.bool)

    if emit_wp_class:
        wp_classes = torch.zeros((batch_size, max_tgt), dtype=torch.long)

    for i, item in enumerate(batch):
        s_ids, d_ids, l_ids = item[:3]

        s_len = min(len(s_ids), max_src)
        src_ids[i, :s_len] = torch.tensor(s_ids[:s_len])
        src_mask[i, :s_len] = True

        t_len = min(len(d_ids), max_tgt)
        dec_in_ids[i, :t_len] = torch.tensor(d_ids[:t_len])
        label_ids[i, :t_len] = torch.tensor(l_ids[:t_len])
        tgt_mask[i, :t_len] = True

        if emit_wp_class:
            wp_ids = item[3]
            wp_classes[i, :t_len] = torch.tensor(wp_ids[:t_len])

    if emit_wp_class:
        return src_ids, dec_in_ids, label_ids, src_mask, tgt_mask, wp_classes
    return src_ids, dec_in_ids, label_ids, src_mask, tgt_mask


def concat_collate_fn(batch, max_seq_len=1024):
    """Decoder-only collate: concatenate `[src; dec_in]` and `[-100*src_len; label_ids]`.

    Produced for `--model_type d` (DecoderOnlyGPT). Source positions in
    the labels tensor are filled with -100 so cross-entropy ignores them
    (prefix-LM training).

    Returns:
        input_ids: (B, T) — concatenated [src; BOS; tgt] token ids, padded.
        target_ids: (B, T) — labels (-100 on source + pad, real ids on target output positions).
        padding_mask: (B, T) bool — True at valid positions.
        src_lens: (B,) long — per-row source length (for diagnostics; loss
                              does not need it because of -100 masking).
    """
    B = len(batch)
    # Per-row total length = src_len + dec_in_len; cap at max_seq_len.
    per_row_lens = [min(max_seq_len, len(x[0]) + len(x[1])) for x in batch]
    T = max(per_row_lens)

    input_ids = torch.zeros((B, T), dtype=torch.long)
    target_ids = torch.full((B, T), -100, dtype=torch.long)
    padding_mask = torch.zeros((B, T), dtype=torch.bool)
    src_lens = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        s_ids, d_ids, l_ids = item[:3]
        s_len = len(s_ids)
        d_len = len(d_ids)

        # Truncate target if needed to fit max_seq_len; keep all of source.
        if s_len + d_len > max_seq_len:
            d_len = max(0, max_seq_len - s_len)
            d_ids = d_ids[:d_len]
            l_ids = l_ids[:d_len]

        if s_len > 0:
            input_ids[i, :s_len] = torch.tensor(s_ids)
        if d_len > 0:
            input_ids[i, s_len:s_len + d_len] = torch.tensor(d_ids)
            # Target positions in labels — source/pad positions stay at -100.
            target_ids[i, s_len:s_len + d_len] = torch.tensor(l_ids)
        padding_mask[i, :s_len + d_len] = True
        src_lens[i] = s_len

    return input_ids, target_ids, padding_mask, src_lens


def get_loader(csv_path, sp_model_path, batch_size, max_seq_len=1024,
               prop_dropout=0.0, emit_wp_class=False, split=None,
               shuffle=True, num_workers=4, pin_memory=True,
               distributed_rank=None, distributed_world_size=None,
               label_dropout=0.0, model_type="ed"):
    """Builds a DataLoader for the translation dataset.

    `model_type='ed'` (default) uses the encoder-decoder collate
    (separate src and dec_in tensors). `model_type='d'` uses the
    decoder-only concat collate (single concatenated input + ignore-mask
    labels).
    """
    dataset = TranslationDataset(csv_path, sp_model_path, max_seq_len,
                                 prop_dropout, emit_wp_class, split,
                                 label_dropout=label_dropout)

    sampler = None
    if distributed_rank is not None and distributed_world_size is not None:
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed_world_size,
            rank=distributed_rank,
            shuffle=shuffle,
        )

    from functools import partial
    if model_type == "d":
        if emit_wp_class:
            raise ValueError(
                "emit_wp_class is incompatible with model_type='d' "
                "(decoder-only path doesn't pass wp_class through)."
            )
        collate = partial(concat_collate_fn, max_seq_len=max_seq_len)
    else:
        collate = partial(collate_fn, max_seq_len=max_seq_len, emit_wp_class=emit_wp_class)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(shuffle and sampler is None),
                        sampler=sampler, num_workers=num_workers, collate_fn=collate,
                        pin_memory=pin_memory, persistent_workers=num_workers > 0)
    return loader


def build_tokenizer(csv_paths, sp_prefix="model_sp", vocab_size=16000):
    """Trains a SentencePiece tokenizer on source + target text from CSVs."""
    import sentencepiece as spm

    if isinstance(csv_paths, (str, Path)):
        csv_paths = [str(csv_paths)]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        for path in csv_paths:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tmp.write(row["source"] + "\n")
                    tmp.write(row["target"] + "\n")
        tmp_path = tmp.name

    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=sp_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )
    os.remove(tmp_path)


def load_sp(model_path):
    """Loads a SentencePiece model."""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp
