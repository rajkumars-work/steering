# Text-domain experiment results (review.3 X1)

Driving the text/LLM slice of
`dielectric/docs/steering_paper/review.3.text_experiment.handoff.md`. Mirrors the
`images/` and `crystals/` sibling directories: scripts in `scripts/`, the shadow
(derived per-bin statistics, no raw text) in `distributions/`.

Status legend: same as `image_experiments.results.md` / `crystal_experiments.results.md`.

| stage | what | status |
|---|---|---|
| A | data-side pre-check: audit category vs. 4 candidate targets on 3,600 real Amazon reviews | ✅ done — **gate FAILS** |
| A swap-1 | alternate low-E targets (sentence length, first-person rate) | ✅ done — still fails |
| A swap-2 | alternate key (review length-bucket) | ✅ done — still fails |
| A swap-3 | alternate key (star rating, low/mid/high) | ✅ done — **gate fails again**: readability doesn't clear trivial-extreme bar |
| pre-registration | predicted ordering + stay-measure threshold, committed before any Stage-B call | ✅ `preregistration.md` |
| B | model-side steering sweep (Qwen2.5-1.5B-Instruct, telling + showing) | **not run — blocked by Stage-A gate; all three authorized swaps exhausted** |

---

## Stage A — data-side pre-check ⚠ HONEST NEGATIVE (gate fails)

`scripts/collect_stagea.py` → `distributions/stageA_reviews.parquet` (3,600 reviews,
450/category, `McAuley-Lab/Amazon-Reviews-2023`, 40–150 words to decorrelate length
from readability) → `distributions/stageA_perclass*.npz` (shadow) →
`distributions/stageA_summary.json`. Reproduced from the shadow alone (no corpus
download, no model) by `scripts/verify_dataside.py`. Audit computed exactly as
Preliminaries §"The audit": per-bin share `w_b` (exact sampling fraction), mean
`g_b`, spread `v_b`; `T = Σ w_b v_b`, `E = Σ w_b (g_b − ḡ)²`; bootstrap 95% CI,
2,000 resamples, resampling reviews within bin.

| key | target | T | E | E/(E+T) | 95% CI |
|---|---|---|---|---|---|
| category (8 bins) | L_sentiment (VADER) | 0.1843 | 0.00730 | **0.038** | [0.028, 0.053] |
| category (8 bins) | L_readability (Flesch–Kincaid) | 7.852 | 0.3915 | **0.047** | [0.036, 0.064] |
| category (8 bins) | L_sentlen (swap-1 alt A) | 44.97 | 2.011 | 0.043 | [0.030, 0.061] |
| category (8 bins) | L_firstperson (swap-1 alt B) | 0.000866 | 0.0000628 | 0.068 | [0.055, 0.086] |
| lenbucket (swap-2, 3 bins) | L_sentiment | 0.1884 | 0.00319 | 0.017 | [0.010, 0.026] |
| lenbucket (swap-2, 3 bins) | L_readability | 7.952 | 0.2914 | 0.035 | [0.024, 0.050] |

**Gate verdict: FAIL.** The handoff's go/no-go rule requires one target "clearly
higher E/(E+T) than the other, both away from the trivial extremes (0 or 1)"
before spending any Stage-B generation budget. Every one of the six (key, target)
combinations tried — original pair, both swap-1 alternatives, and swap-2's
alternate key — lands in **E/(E+T) ∈ [0.017, 0.068]**, all mutually overlapping in
their bootstrap CIs (max spread across all six: 0.051). There is no high-E
candidate anywhere in this set: individual-reviewer variance (`T`) accounts for
93–98% of total variance in every tone/style/complexity/first-person proxy tried,
swamping the between-category signal (`E`) every time. This is not a borderline
call — the numbers cluster tightly near the 0 extreme with no separation, and both
of the handoff's prescribed swaps (in order of cheapness) were tried and both
failed to rescue it.

**Why this is a plausible real finding, not a scoring bug:** `verify_dataside.py`
reproduces the numbers directly from the committed shadow (`stageA_perclass.npz`,
`stageA_perclass_lenbucket.npz`) in well under a second — no corpus re-download, no
model. A person's default tone (chronically upbeat vs. chronically critical
reviewer) and a person's default sentence complexity are properties of *that
person's writing style*, largely independent of which product category they
happen to be reviewing — exactly the "T swamps E" failure mode the handoff itself
flagged as the risk to check for before spending Stage-B budget.

**Honest-flags:** the length-controlled (40–150 word) sample by design removes the
easiest source of between-category readability signal (some categories produce
systematically longer/shorter reviews); an unconstrained sample might show a
larger E for readability, but would then confound length with complexity, which is
exactly what the handoff's own cap was meant to prevent. First-person rate (swap-1
alt B) is nominally the highest of the six (0.068) but its own CI [0.055, 0.086]
is still close to 0 and overlaps three of the other five combinations' CIs — not a
qualifying "clear" separation under the gate's own "away from trivial extremes"
requirement (0.068 is still a small fraction of variance explained).

---

## Stage A — swap-3 (star rating as key) ⚠ HONEST NEGATIVE (third exhausted swap)

Re-keyed the same 3,600-review pull by the corpus's own `rating` field (1–5★,
already in `stageA_reviews.parquet`, no re-download) into 3 bins: `low` (1–2★,
n=292), `mid` (3★, n=259), `high` (4–5★, n=3,049). Same two headline targets,
same T/E/bootstrap-CI convention (`scripts/swap3_rating.py`, reuses
`collect_stagea.py`'s `summarize()`).

| key | target | T | E | E/(E+T) | 95% CI |
|---|---|---|---|---|---|
| rating (swap-3, low/mid/high) | L_sentiment | 0.1374 | 0.0542 | **0.283** | [0.247, 0.323] |
| rating (swap-3, low/mid/high) | L_readability | 8.233 | 0.0111 | **0.0013** | [0.0001, 0.0067] |

**Gate verdict: FAIL, for the reason the handoff flagged in advance.** Sentiment
separates hard (0.283) — but that's expected/near-tautological (star rating
*is* self-reported sentiment) and explicitly not the win condition per the
handoff's design rationale (same telling-knob/key circularity as R4-#3 on the
image experiment). The actual gate — readability separating from the trivial 0
extreme — fails: E/(E+T)=0.0013, CI [0.0001, 0.0067], sitting on the boundary.
Per the handoff's own rule ("if readability still doesn't separate at all,
that's a third exhausted swap — stop, do not try a fourth key"): stop here.
This is a stronger honest negative than the category-key one — it rules out
even the reviewer's own most self-descriptive signal.

Source: `scripts/swap3_rating.py` → `distributions/stageA_perclass_rating.npz`,
`distributions/stageA_summary_swap3.json`.

## Stage B — model-side steering sweep: **not run**

Per the handoff's contingency for swap-3 specifically ("if readability still
doesn't separate at all, that's a third exhausted swap — stop, do not try a
fourth key"), Stage B was not executed. All three authorized swaps (category →
lenbucket → rating) were exhausted with no qualifying separation. Proceeding to
generate ~2,000–3,000 steered outputs under `Qwen2.5-1.5B-Instruct` would not
test anything the design can distinguish: with no low-E/high-E target pair on
the data side, there is no principled prediction to confirm or refute on the
model side. **No GPU time was used** in this round either: the shared A10G's
other long-running job (~12.8GB / 98% util) was left untouched throughout.

### The specific obstacle (final — all three keys, four targets exhausted)

Three keys tried (product category/8 bins, review length-bucket/3 bins, star
rating/3 bins) against four targets (sentiment, readability, mean sentence
length, first-person rate): every combination that could serve as the low-E
"telling" side lands at or near the trivial 0 extreme (E/(E+T) ≤ 0.07 in six of
eight combinations); the one combination that separates clearly (sentiment
under rating, 0.283) does so for a construct-validity reason ruled out in
advance, not an independent showing signal. On general-purpose e-commerce
review text, writing style (tone, complexity, length, voice) is overwhelmingly
reviewer-idiosyncratic, not driven by category, length-bucket, or even the
reviewer's own star rating. A text steering experiment along these lines needs
either (a) a corpus with markedly stronger key-level style homogeneity (e.g.
genre-labeled fiction, sub-community-stratified forum posts), or (b) a key that
is itself a style signal rather than a content/metadata label (e.g. per-author
identity, given enough per-author volume). Concrete, falsifiable, exhausted
across three keys and four targets — not a placeholder "future work," and no
fourth key was tried without going back to RK first, per protocol.

---

## Deliverables in this directory

- `scripts/collect_stagea.py` — pulls the corpus, scores 4 targets, writes the
  shadow + summary (tier-2, needs `datasets`/`nltk`/`textstat`/`pandas`).
- `scripts/verify_dataside.py` — reproduces the gate verdict from the committed
  shadow alone (tier-1, needs only `numpy`).
- `distributions/stageA_reviews.parquet`, `stageA_perclass.npz`,
  `stageA_perclass_lenbucket.npz`, `stageA_summary.json` — data + shadow.
- `scripts/swap3_rating.py`, `distributions/stageA_perclass_rating.npz`,
  `distributions/stageA_summary_swap3.json` — swap-3 (rating key) round.
- `preregistration.md` — predicted ordering/threshold/crossover call, committed
  before the go/no-go decision was acted on; also records the STOP decision.
- This file and the mirrored authoritative doc:
  `dielectric/docs/steering_paper/text_experiment.results.md`.
