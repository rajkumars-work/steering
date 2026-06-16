# wp_class_builder.py
"""
Utilities to build the wp_class (B, T) tensor from decoded token sequences.

Two modes:
  1. Batch mode (dataset pre-processing): build_wp_class_sequence()
  2. Incremental mode (autoregressive inference): IncrementalWPClassTracker
"""

import re

import torch
from wp_moe import wp_letter_to_dof, WPDOFClass

# Known Wyckoff letters (a-z lowercase single characters used as site labels)
_WP_LETTERS = set("abcdefghijklmnopqrstuvwxyz")

# Regex for coordinate-like tokens (digits, dots, minus signs)
_COORD_RE = re.compile(r"^-?[\d.]+$")

# SentencePiece word-boundary marker
_SP_PREFIX = "▁"


def _clean(tok: str) -> str:
    """Strip SentencePiece prefix and whitespace from a piece."""
    return tok.strip().lstrip(_SP_PREFIX).strip()


def _is_coord_like(tok: str) -> bool:
    """Check if a token looks like a fractional coordinate."""
    return bool(_COORD_RE.match(_clean(tok)))


def _is_element_like(tok: str) -> bool:
    """Check if a token looks like an element symbol (1-2 alpha chars, first upper)."""
    t = _clean(tok)
    if not t or not t[0].isupper():
        return False
    return len(t) <= 2 and t.isalpha()


def build_wp_class_sequence(
    token_strs: list[str],
    atom_block_start_token: str = "|",
) -> list[int]:
    """
    Given decoded token strings for one target sequence, return a parallel list
    of WP DOF class indices.

    Uses a state machine to parse the atom section:
      State 0 (pre-atom): all tokens -> FIXED. Count pipe delimiters to find
                           the atom section (after the last pipe).
      State 1 (expect element): next element-like token is the element symbol.
      State 2 (expect WP letter): next single lowercase alpha is the WP letter.
      State 3 (expect coords): next 3 coord-like tokens get the current DOF class.

    The token stream format is:
      Formula | SG ... | CN:... NN:... WP:... | a b c alpha beta gamma | Elem wl x y z ...

    Args:
        token_strs: list of token strings (SentencePiece pieces or split words)
        atom_block_start_token: delimiter between sections

    Returns:
        list of int, same length as token_strs, values in {0, 1, 2, 3}
    """
    classes = [int(WPDOFClass.FIXED)] * len(token_strs)

    # Find the start of the atom section: after the last pipe delimiter.
    # The format has 4 pipe-separated sections before atoms.
    # SentencePiece may encode "|" as "▁|", so clean before comparing.
    last_pipe = -1
    for i, tok in enumerate(token_strs):
        if _clean(tok) == atom_block_start_token:
            last_pipe = i

    if last_pipe < 0:
        return classes  # no atom section found

    # State machine for atom section
    # States: 1=expect_element, 2=expect_wp_letter, 3=expect_coords
    state = 1
    current_dof = int(WPDOFClass.FIXED)
    coord_count = 0

    for i in range(last_pipe + 1, len(token_strs)):
        cleaned = _clean(token_strs[i])
        if not cleaned:
            continue

        if state == 1:  # expect element symbol
            if _is_element_like(token_strs[i]):
                state = 2
            # else: skip (sub-word continuation, whitespace piece, etc.)

        elif state == 2:  # expect WP letter
            if len(cleaned) == 1 and cleaned.lower() in _WP_LETTERS:
                current_dof = wp_letter_to_dof(cleaned.lower())
                classes[i] = current_dof
                coord_count = 0
                state = 3
            elif _is_element_like(token_strs[i]):
                # Two element symbols in a row — previous atom had no WP letter?
                # Stay in state 2, treat this as the element of next atom.
                pass
            # else: skip sub-word piece

        elif state == 3:  # expect coordinate tokens
            if _is_coord_like(token_strs[i]):
                classes[i] = current_dof
                coord_count += 1
                if coord_count >= 3:
                    state = 1  # done with this atom, expect next element
            elif _is_element_like(token_strs[i]):
                # Coordinate section cut short — start of next atom
                state = 2
                current_dof = int(WPDOFClass.FIXED)
                coord_count = 0

    return classes


class IncrementalWPClassTracker:
    """
    Tracks wp_class assignments during autoregressive generation.

    After each token is generated, call update() to rebuild wp_class for
    the full sequence so far. Uses build_wp_class_sequence() internally.

    This is O(T^2) over the full sequence but T <= 512 and string processing
    is negligible compared to the transformer forward pass.
    """
    def __init__(self, sp, batch_size: int, max_length: int, device):
        self.sp = sp
        self.batch_size = batch_size
        self.wp_class = torch.zeros(batch_size, max_length, dtype=torch.long,
                                    device=device)
        # Cache decoded pieces per batch element
        self._pieces: list[list[str]] = [[] for _ in range(batch_size)]

    def update(self, token_ids: torch.Tensor, pos: int, finished: torch.Tensor):
        """
        Called after generating token at position `pos` for all batch elements.

        Args:
            token_ids: (B,) the token id generated at position pos
            pos:       int, the position in the sequence (0 = BOS)
            finished:  (B,) bool tensor, True if sequence is done
        """
        for b in range(self.batch_size):
            if finished[b]:
                continue
            piece = self.sp.id_to_piece(token_ids[b].item())
            self._pieces[b].append(piece)
            # Rebuild wp_class from full decoded sequence
            classes = build_wp_class_sequence(self._pieces[b])
            # Position 0 is BOS (class 0), pieces start at position 1
            for j, c in enumerate(classes):
                self.wp_class[b, j + 1] = c

    def reset(self):
        """Reset state for a new batch."""
        self.wp_class.zero_()
        self._pieces = [[] for _ in range(self.batch_size)]


def collate_wp_class(batch_classes: list[list[int]], pad_to: int, device) -> torch.Tensor:
    """
    Collate a list of wp_class sequences into a (B, T) tensor, right-padded with 0.
    """
    B = len(batch_classes)
    out = torch.zeros(B, pad_to, dtype=torch.long, device=device)
    for i, seq in enumerate(batch_classes):
        L = min(len(seq), pad_to)
        out[i, :L] = torch.tensor(seq[:L], dtype=torch.long)
    return out
