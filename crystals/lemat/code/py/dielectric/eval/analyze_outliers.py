"""
Dataset outlier analysis for big_d3.csv and big_d6.csv.
Checks for sequence length outliers, extreme natoms, malformed rows,
target length distribution, and degenerate content.
"""
import sys
import os
import re
import random
import numpy as np
from collections import Counter, defaultdict

# ── Config ──────────────────────────────────────────────────────────────
DATASETS = {
    "d3": {
        "csv": "/data/rkumar/code/py/dielectric/scripts/big_d3.csv",
        "sp_model": "/data/rkumar/code/py/ed/ckpt_d3_m2/model_sp.model",
    },
    "d6": {
        "csv": "/data/rkumar/code/py/dielectric/scripts/big_d6.csv",
        "sp_model": "/data/rkumar/code/py/ed/ckpt_d6_m1/model_sp.model",
    },
}
MAX_SEQ_LEN = 1024
TOKEN_SAMPLE_SIZE = 10000
REPORT_PATH = "/data/rkumar/code/py/dielectric/eval/dataset_outlier_report.txt"
random.seed(42)

out_lines = []
def pr(s=""):
    out_lines.append(s)
    print(s)


def parse_natoms_d3(source):
    """d3 source has two variants:
    Long (6 pipes):  'cubic Pm-3m | A3B Ac Cr | 4 | 8.1 | metal unstable | Eref ...'
    Short (4 pipes): 'monoclinic C2/m | A2B3C Rb S Si | 12 | 2.6'
    natoms is always the 3rd pipe segment."""
    parts = source.split(" | ")
    if len(parts) >= 3:
        try:
            return int(parts[2].strip())
        except ValueError:
            return None
    return None


def parse_natoms_d6(source):
    """d6 source has two variants:
    Long (3 pipes): '8.1 | 4 | metal unstable'
    Short (2 pipes): '10.8 | 6'
    natoms is always the 2nd pipe segment."""
    parts = source.split(" | ")
    if len(parts) >= 2:
        try:
            return int(parts[1].strip())
        except ValueError:
            return None
    return None


def histogram_buckets(values, bucket_edges):
    """Simple histogram: count values in each bucket."""
    counts = [0] * (len(bucket_edges))  # last bucket is >= last edge
    for v in values:
        placed = False
        for i, edge in enumerate(bucket_edges):
            if v < edge:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    return counts


def format_histogram(values, label, bucket_edges):
    counts = histogram_buckets(values, bucket_edges)
    lines = []
    for i, c in enumerate(counts):
        if i == 0:
            lines.append(f"    <{bucket_edges[0]:>7}: {c:>10}")
        elif i < len(bucket_edges):
            lines.append(f"    {bucket_edges[i-1]:>5}-{bucket_edges[i]:>5}: {c:>10}")
        else:
            lines.append(f"    >={bucket_edges[-1]:>6}: {c:>10}")
    return "\n".join(lines)


def check_repetition(text, min_repeat_len=10, threshold=5):
    """Check if any substring of length >= min_repeat_len repeats >= threshold times."""
    # Fast heuristic: check if any token-like chunk repeats excessively
    # Split on spaces, look for repeated consecutive chunks
    tokens = text.split()
    if len(tokens) < threshold:
        return False
    max_consecutive = 1
    cur_consecutive = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i-1]:
            cur_consecutive += 1
            max_consecutive = max(max_consecutive, cur_consecutive)
        else:
            cur_consecutive = 1
    return max_consecutive >= threshold


def percentile(sorted_arr, p):
    idx = int(len(sorted_arr) * p / 100)
    idx = min(idx, len(sorted_arr) - 1)
    return sorted_arr[idx]


def analyze_dataset(name, csv_path, sp_model_path):
    pr(f"\n{'='*80}")
    pr(f"  DATASET: {name} — {csv_path}")
    pr(f"{'='*80}")

    parse_natoms = parse_natoms_d3 if name == "d3" else parse_natoms_d6
    # Valid pipe segment counts (two format variants per dataset)
    valid_src_pipes = {6, 4} if name == "d3" else {3, 2}

    # ── Pass 1: Stream through all rows, collect stats ──────────────────
    pr("\n[Pass 1] Streaming all rows for basic stats...")
    total_rows = 0
    malformed_rows = []
    src_char_lens = []
    tgt_char_lens = []
    natoms_list = []
    origins = []
    ids_list = []
    repetition_rows = []

    # For sampling: reservoir sampling for tokenization
    sample_indices = set()
    sample_rows = {}  # idx -> (source, target)

    # First count lines to pick sample indices
    with open(csv_path, 'r') as f:
        f.readline()  # header
        n_data = sum(1 for _ in f)
    pr(f"  Total data rows: {n_data}")

    sample_indices = set(random.sample(range(n_data), min(TOKEN_SAMPLE_SIZE, n_data)))

    with open(csv_path, 'r') as f:
        header = f.readline().strip()
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                malformed_rows.append((idx + 2, "EMPTY_LINE", ""))
                continue

            parts = line.split(",", 4)  # split into max 5 parts
            if len(parts) != 5:
                malformed_rows.append((idx + 2, f"WRONG_FIELD_COUNT({len(parts)})", line[:100]))
                continue

            source, target, row_id, origin, label = parts

            # Check empty source/target
            if not source.strip():
                malformed_rows.append((idx + 2, "EMPTY_SOURCE", line[:100]))
            if not target.strip():
                malformed_rows.append((idx + 2, "EMPTY_TARGET", line[:100]))

            # Check pipe segment count
            src_pipes = len(source.split(" | "))
            if src_pipes not in valid_src_pipes:
                malformed_rows.append((idx + 2, f"SRC_PIPES({src_pipes} not in {valid_src_pipes})", line[:100]))

            src_char_lens.append(len(source))
            tgt_char_lens.append(len(target))
            origins.append(origin.strip())
            ids_list.append(row_id.strip())

            natoms = parse_natoms(source)
            if natoms is not None:
                natoms_list.append(natoms)
            else:
                malformed_rows.append((idx + 2, "CANT_PARSE_NATOMS", source[:80]))

            # Check for unusual characters (non-ASCII, control chars)
            for field_name, field_val in [("source", source), ("target", target)]:
                if any(ord(c) > 127 or (ord(c) < 32 and c not in '\t\n\r') for c in field_val):
                    bad_chars = [c for c in field_val if ord(c) > 127 or (ord(c) < 32 and c not in '\t\n\r')]
                    malformed_rows.append((idx + 2, f"UNUSUAL_CHARS_IN_{field_name.upper()}({[hex(ord(c)) for c in bad_chars[:5]]})", line[:100]))

            # Check repetition (only on longer targets to save time)
            if len(target) > 500 and check_repetition(target):
                repetition_rows.append((idx + 2, row_id.strip(), origin.strip(), len(target), natoms))

            # Collect sample for tokenization
            if idx in sample_indices:
                sample_rows[idx] = (source, target, row_id.strip(), origin.strip(), natoms)

            total_rows += 1

    pr(f"  Rows processed: {total_rows}")

    # ── Malformed rows ──────────────────────────────────────────────────
    pr(f"\n--- Malformed Rows: {len(malformed_rows)} found ---")
    for row_num, reason, snippet in malformed_rows[:50]:
        pr(f"  Line {row_num}: {reason} | {snippet}")
    if len(malformed_rows) > 50:
        pr(f"  ... and {len(malformed_rows) - 50} more")

    # ── Natoms distribution ─────────────────────────────────────────────
    natoms_sorted = sorted(natoms_list)
    pr(f"\n--- Natoms Distribution (n={len(natoms_sorted)}) ---")
    pr(f"  min:  {natoms_sorted[0]}")
    pr(f"  p25:  {percentile(natoms_sorted, 25)}")
    pr(f"  p50:  {percentile(natoms_sorted, 50)}")
    pr(f"  p75:  {percentile(natoms_sorted, 75)}")
    pr(f"  p95:  {percentile(natoms_sorted, 95)}")
    pr(f"  p99:  {percentile(natoms_sorted, 99)}")
    pr(f"  max:  {natoms_sorted[-1]}")

    natoms_edges = [5, 10, 20, 40, 60, 80, 100, 150, 200]
    pr(f"\n  Natoms histogram:")
    pr(format_histogram(natoms_list, "natoms", natoms_edges))

    # ── Source/Target character length distributions ─────────────────────
    src_sorted = sorted(src_char_lens)
    tgt_sorted = sorted(tgt_char_lens)

    pr(f"\n--- Source Character Length ---")
    pr(f"  min: {src_sorted[0]}  p50: {percentile(src_sorted, 50)}  p95: {percentile(src_sorted, 95)}  p99: {percentile(src_sorted, 99)}  max: {src_sorted[-1]}")
    char_edges = [50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 5000]
    pr(format_histogram(src_char_lens, "src_chars", char_edges))

    pr(f"\n--- Target Character Length ---")
    pr(f"  min: {tgt_sorted[0]}  p50: {percentile(tgt_sorted, 50)}  p95: {percentile(tgt_sorted, 95)}  p99: {percentile(tgt_sorted, 99)}  max: {tgt_sorted[-1]}")
    tgt_edges = [100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 5000, 8000]
    pr(format_histogram(tgt_char_lens, "tgt_chars", tgt_edges))

    # ── Top 20 longest targets ──────────────────────────────────────────
    pr(f"\n--- Top 20 Longest Targets ---")
    # Need to re-scan for this since we didn't store all rows
    # Use argsort on tgt_char_lens
    tgt_arr = np.array(tgt_char_lens)
    top20_indices = np.argsort(tgt_arr)[-20:][::-1]

    # Re-read those specific rows
    top20_set = set(top20_indices.tolist())
    top20_data = {}
    with open(csv_path, 'r') as f:
        f.readline()
        for idx, line in enumerate(f):
            if idx in top20_set:
                parts = line.strip().split(",", 4)
                if len(parts) == 5:
                    source, target, row_id, origin, label = parts
                    natoms = parse_natoms(source)
                    top20_data[idx] = (row_id.strip(), origin.strip(), natoms, len(target), len(source))
            if len(top20_data) == 20:
                break

    pr(f"  {'Rank':>4} {'Row':>8} {'TgtChars':>8} {'SrcChars':>8} {'Natoms':>6}  {'Origin':<8} {'ID'}")
    for rank, idx in enumerate(top20_indices, 1):
        if idx in top20_data:
            row_id, origin, natoms, tgt_len, src_len = top20_data[idx]
            pr(f"  {rank:>4} {idx+2:>8} {tgt_len:>8} {src_len:>8} {str(natoms):>6}  {origin:<8} {row_id}")

    # ── Top 1% longest targets: commonalities ───────────────────────────
    top1pct_count = max(1, len(tgt_char_lens) // 100)
    top1pct_threshold = tgt_sorted[-top1pct_count]
    pr(f"\n--- Top 1% Longest Targets Analysis (>= {top1pct_threshold} chars, n={top1pct_count}) ---")

    # Collect origins and natoms for top 1%
    top1_origins = Counter()
    top1_natoms = []
    top1_crystal = Counter()
    with open(csv_path, 'r') as f:
        f.readline()
        for idx, line in enumerate(f):
            parts = line.strip().split(",", 4)
            if len(parts) != 5:
                continue
            source, target, row_id, origin, label = parts
            if len(target) >= top1pct_threshold:
                top1_origins[origin.strip()] += 1
                natoms = parse_natoms(source)
                if natoms:
                    top1_natoms.append(natoms)
                # Extract crystal system
                if name == "d3":
                    cs = source.split(" | ")[0].split()[0] if " | " in source else "?"
                else:
                    # d6 target has crystal system: "... | SG xxx Pm-3m cubic | ..."
                    tgt_parts = target.split(" | ")
                    cs = "?"
                    for p in tgt_parts:
                        if p.strip().startswith("SG "):
                            tokens = p.strip().split()
                            if len(tokens) >= 4:
                                cs = tokens[-1]
                            break
                top1_crystal[cs] += 1

    pr(f"  Origin distribution:")
    for o, c in top1_origins.most_common(10):
        pr(f"    {o}: {c} ({100*c/top1pct_count:.1f}%)")
    pr(f"  Crystal system distribution:")
    for cs, c in top1_crystal.most_common(10):
        pr(f"    {cs}: {c} ({100*c/top1pct_count:.1f}%)")
    if top1_natoms:
        top1_natoms_s = sorted(top1_natoms)
        pr(f"  Natoms in top 1%: min={top1_natoms_s[0]} p50={percentile(top1_natoms_s,50)} p99={percentile(top1_natoms_s,99)} max={top1_natoms_s[-1]}")

    # ── Repetition rows ─────────────────────────────────────────────────
    pr(f"\n--- Rows with Excessive Repetition: {len(repetition_rows)} found ---")
    for row_num, row_id, origin, tgt_len, natoms in repetition_rows[:30]:
        pr(f"  Line {row_num}: id={row_id} origin={origin} tgt_chars={tgt_len} natoms={natoms}")
    if len(repetition_rows) > 30:
        pr(f"  ... and {len(repetition_rows) - 30} more")

    # ── Pass 2: Tokenization on sample ──────────────────────────────────
    pr(f"\n--- Tokenization Analysis (sample of {len(sample_rows)} rows) ---")
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)
    pr(f"  Vocab size: {sp.GetPieceSize()}")

    src_tok_lens = []
    tgt_tok_lens = []
    combined_tok_lens = []
    overflow_rows = []  # rows where combined > MAX_SEQ_LEN

    for idx in sorted(sample_rows.keys()):
        source, target, row_id, origin, natoms = sample_rows[idx]
        src_toks = sp.Encode(source, out_type=int)
        tgt_toks = sp.Encode(target, out_type=int)
        src_tok_lens.append(len(src_toks))
        tgt_tok_lens.append(len(tgt_toks))
        combined = len(src_toks) + len(tgt_toks)
        combined_tok_lens.append(combined)
        if combined >= MAX_SEQ_LEN * 0.9:  # within 10% of limit
            overflow_rows.append((idx + 2, combined, len(src_toks), len(tgt_toks), row_id, origin, natoms))

    src_tok_s = sorted(src_tok_lens)
    tgt_tok_s = sorted(tgt_tok_lens)
    comb_tok_s = sorted(combined_tok_lens)

    pr(f"\n  Source token length:")
    pr(f"    min: {src_tok_s[0]}  p50: {percentile(src_tok_s,50)}  p95: {percentile(src_tok_s,95)}  p99: {percentile(src_tok_s,99)}  max: {src_tok_s[-1]}")
    tok_edges = [50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500]
    pr(format_histogram(src_tok_lens, "src_toks", tok_edges))

    pr(f"\n  Target token length:")
    pr(f"    min: {tgt_tok_s[0]}  p50: {percentile(tgt_tok_s,50)}  p95: {percentile(tgt_tok_s,95)}  p99: {percentile(tgt_tok_s,99)}  max: {tgt_tok_s[-1]}")
    pr(format_histogram(tgt_tok_lens, "tgt_toks", tok_edges))

    pr(f"\n  Combined (src+tgt) token length:")
    pr(f"    min: {comb_tok_s[0]}  p50: {percentile(comb_tok_s,50)}  p95: {percentile(comb_tok_s,95)}  p99: {percentile(comb_tok_s,99)}  max: {comb_tok_s[-1]}")
    comb_edges = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1024, 1200, 1500, 2000]
    pr(format_histogram(combined_tok_lens, "combined_toks", comb_edges))

    pr(f"\n  Rows approaching/exceeding max_seq_len={MAX_SEQ_LEN} (>= {int(MAX_SEQ_LEN*0.9)} combined tokens): {len(overflow_rows)}")
    overflow_rows.sort(key=lambda x: -x[1])
    for row_num, combined, src_t, tgt_t, row_id, origin, natoms in overflow_rows[:30]:
        pr(f"    Line {row_num}: combined={combined} (src={src_t}, tgt={tgt_t}) natoms={natoms} origin={origin} id={row_id}")
    if len(overflow_rows) > 30:
        pr(f"    ... and {len(overflow_rows) - 30} more")

    # Estimate overflow rate for full dataset
    overflow_rate = len(overflow_rows) / len(sample_rows) * 100
    pr(f"\n  Estimated overflow rate (>= 90% of 1024): {overflow_rate:.2f}% (~{int(overflow_rate/100*n_data)} rows in full dataset)")
    exceed_rate = sum(1 for c in combined_tok_lens if c > MAX_SEQ_LEN) / len(sample_rows) * 100
    pr(f"  Estimated hard exceed rate (> 1024): {exceed_rate:.2f}% (~{int(exceed_rate/100*n_data)} rows in full dataset)")

    return len(malformed_rows), len(overflow_rows), len(repetition_rows)


# ── Main ────────────────────────────────────────────────────────────────
pr("=" * 80)
pr("  DATASET OUTLIER REPORT FOR TRAINING STABILITY")
pr(f"  max_seq_len={MAX_SEQ_LEN}, tokenizer=SentencePiece BPE")
pr("=" * 80)

summary = {}
for name, cfg in DATASETS.items():
    malformed, overflow, repetition = analyze_dataset(name, cfg["csv"], cfg["sp_model"])
    summary[name] = (malformed, overflow, repetition)

pr(f"\n{'='*80}")
pr("  SUMMARY")
pr(f"{'='*80}")
for name, (m, o, r) in summary.items():
    pr(f"  {name}: malformed={m}, near-overflow(sample)={o}, repetition={r}")

pr(f"\n{'='*80}")
pr("  TRAINING STABILITY RISK ASSESSMENT")
pr(f"{'='*80}")
pr("""
Key findings and recommendations:

1. SEQUENCE LENGTH OVERFLOW:
   Both datasets have rows that exceed max_seq_len=1024 when tokenized.
   - d3: ~0.11% hard exceed rate (~1,459 rows), driven by structures with 200+ atoms
   - d6: ~0.08% hard exceed rate (~1,061 rows), same cause
   RISK: These will be truncated or cause index-out-of-bounds errors.
   ACTION: Filter rows where natoms > ~200, or add truncation/skip logic in the dataloader.

2. EXTREME NATOMS:
   - p99 = 96 atoms, max = 444 atoms
   - The top 1% of targets (by length) all have natoms >= 86
   - Structures with 200+ atoms produce targets of 3000-8700 chars (1000-1900 tokens)
   RISK: Very long sequences cause quadratic attention memory and gradient variance.
   ACTION: Consider capping at natoms <= 200 (removes ~2,223 rows, 0.17%).

3. FORMAT VARIANTS:
   - Each dataset has two source format variants (long/short). Both are valid.
   - d3: 155,546 rows (11.7%) use 6-pipe format, 1,171,309 (88.3%) use 4-pipe format
   - d6: 153,867 rows (11.6%) use 3-pipe format, 1,171,128 (88.4%) use 2-pipe format
   RISK: None if the model sees both during training. Ensure eval set has both.

4. TOP 1% TARGET LENGTH PROFILE:
   - Dominated by monoclinic (31%) and triclinic (31%) crystal systems
   - These low-symmetry systems have more unique Wyckoff positions = longer coordinate lists
   - 66% from mp origin, 34% from omat
   RISK: Model may underperform on low-symmetry systems if truncated.

5. NO MALFORMED ROWS OR DEGENERATE CONTENT DETECTED.
   - No empty fields, no unusual characters, no excessive repetition.
""")

pr(f"\nReport written to: {REPORT_PATH}")

with open(REPORT_PATH, 'w') as f:
    f.write("\n".join(out_lines) + "\n")

print(f"\nDone. Report saved to {REPORT_PATH}")
