"""DPO dataset and collation for preference-pair training.

Loads dpo_pairs.csv (source, chosen, rejected, score) and returns tokenized
triples for the DPO training loop.
"""

import csv

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset


class DPODataset(Dataset):
    """Reads a DPO preference-pairs CSV and tokenizes with SentencePiece.

    Each item returns 5 tensors:
        (src_ids, chosen_dec_in, chosen_labels,
         rejected_dec_in, rejected_labels)
    """

    def __init__(self, csv_path, sp_model_path, max_seq_len=1024):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)
        self.max_seq_len = max_seq_len
        self.triples = []  # (source, chosen, rejected)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip truncation-caused parse failures — they are just valid
                # prefixes cut off at max_length and poison DPO training by
                # penalising token patterns shared with correct outputs.
                if "parse_failure" in row.get("score", ""):
                    continue
                self.triples.append((row["source"], row["chosen"], row["rejected"]))

    def __len__(self):
        return len(self.triples)

    def _tokenize_target(self, text):
        """Tokenize a target string into (dec_in_ids, label_ids)."""
        tgt_ids = self.sp.encode(text, out_type=int)[:self.max_seq_len - 1]
        dec_in_ids = [self.sp.bos_id()] + tgt_ids
        label_ids = tgt_ids + [self.sp.eos_id()]
        return dec_in_ids, label_ids

    def __getitem__(self, idx):
        src_text, chosen_text, rejected_text = self.triples[idx]

        src_ids = self.sp.encode(src_text, out_type=int)[:self.max_seq_len]
        chosen_dec_in, chosen_labels = self._tokenize_target(chosen_text)
        rejected_dec_in, rejected_labels = self._tokenize_target(rejected_text)

        return (src_ids, chosen_dec_in, chosen_labels,
                rejected_dec_in, rejected_labels)


def dpo_collate_fn(batch, max_seq_len=1024):
    """Pad all 5 tensors and create masks.

    Returns:
        src_padded:          (B, S)  int64
        chosen_dec_in:       (B, Tc) int64
        chosen_labels:       (B, Tc) int64, padded with -100
        rejected_dec_in:     (B, Tr) int64
        rejected_labels:     (B, Tr) int64, padded with -100
        src_mask:            (B, S)  bool
        chosen_tgt_mask:     (B, Tc) bool
        rejected_tgt_mask:   (B, Tr) bool
    """
    (src_batch, chosen_di_batch, chosen_lb_batch,
     rejected_di_batch, rejected_lb_batch) = zip(*batch)

    # Truncate to max_seq_len
    src_batch = [s[:max_seq_len] for s in src_batch]
    chosen_di_batch = [s[:max_seq_len] for s in chosen_di_batch]
    chosen_lb_batch = [s[:max_seq_len] for s in chosen_lb_batch]
    rejected_di_batch = [s[:max_seq_len] for s in rejected_di_batch]
    rejected_lb_batch = [s[:max_seq_len] for s in rejected_lb_batch]

    def pad_seqs(seqs, pad_value=0):
        return torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(s, dtype=torch.long) for s in seqs],
            batch_first=True, padding_value=pad_value,
        )

    src_padded = pad_seqs(src_batch, pad_value=0)
    chosen_dec_in = pad_seqs(chosen_di_batch, pad_value=0)
    chosen_labels = pad_seqs(chosen_lb_batch, pad_value=-100)
    rejected_dec_in = pad_seqs(rejected_di_batch, pad_value=0)
    rejected_labels = pad_seqs(rejected_lb_batch, pad_value=-100)

    src_mask = (src_padded != 0).bool()
    chosen_tgt_mask = (chosen_dec_in != 0).bool()
    rejected_tgt_mask = (rejected_dec_in != 0).bool()

    return (src_padded, chosen_dec_in, chosen_labels,
            rejected_dec_in, rejected_labels,
            src_mask, chosen_tgt_mask, rejected_tgt_mask)


def get_dpo_loader(csv_path, sp_model_path, batch_size, max_seq_len=1024,
                   shuffle=True):
    """Convenience function to create a DPO DataLoader."""
    dataset = DPODataset(csv_path, sp_model_path, max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: dpo_collate_fn(batch, max_seq_len=max_seq_len),
        shuffle=shuffle,
    )
