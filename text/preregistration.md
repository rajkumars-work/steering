# Pre-registration — text/LLM steering experiment (review.3 X1)

Written before any Stage-B generation call. Per the handoff
(`dielectric/docs/steering_paper/review.3.text_experiment.handoff.md`), Stage A's
go/no-go gate must pass *before* any steered generation is run, and this document
must be committed before that gate is evaluated for a Stage-B decision. Below:
the pre-registered predictions the design rationale implies, immediately followed
by the Stage-A gate evaluation and the resulting go/no-go call.

## 1. Predicted E/(E+T) ordering (headline targets)

Per the handoff's own design rationale (§Stage A step 2):

- **L_sentiment (VADER compound)** — predicted **high E/(E+T)**: categories
  plausibly differ in default tone (Beauty/Toys warmer, Electronics carrying more
  defect/complaint reviews).
- **L_readability (Flesch–Kincaid grade)** — predicted **low E/(E+T)**: writing
  complexity is plausibly a property of the individual reviewer, not the product
  category.

Predicted ordering: **E/(E+T)[sentiment] > E/(E+T)[readability]**, with sentiment
landing clearly away from the 0 extreme and readability landing close to it.

## 2. Stay-measure threshold

Per the handoff (non-negotiable, stated before any telling sweep): an operation is
classified **telling** iff, across its sweep, the generated output's classified
category matches the requested category at a rate **≥ 90%**; below that rate the
operation is classified as having left its bin. This threshold is fixed now and
will not be adjusted after seeing which method (telling vs. showing) wins.

## 3. Predicted crossover call (Stage B, contingent on gate passing)

If the gate passes: telling's working reach is ≈√T, showing's is ≈√(χ²·E).
Predicted call — **sentiment (predicted high-E) is showing-favored** (telling
alone can't reach the needed shift because T is large relative to E only in the
sense that the ceiling √T is small vs. what showing's √(χ²E) can realize once E is
non-trivial); **readability (predicted low-E) is telling-favored** (E is small, so
showing has little between-bin signal to exploit, while a direct instruction
easily moves grade level within a single bin).

---

## Stage-A gate evaluation (done BEFORE this document's predictions are acted on)

Source: `distributions/stageA_summary.json`, `distributions/stageA_perclass*.npz`,
reproduced standalone by `scripts/verify_dataside.py` (3,600 Amazon reviews, 450 x 8
categories, `McAuley-Lab/Amazon-Reviews-2023`, 40-150 words/review, bootstrap
95% CIs, 2000 resamples).

| key | target | E/(E+T) | 95% CI |
|---|---|---|---|
| category | L_sentiment | 0.038 | [0.028, 0.053] |
| category | L_readability | 0.047 | [0.036, 0.064] |
| category | L_sentlen (swap-1 alt) | 0.043 | [0.030, 0.061] |
| category | L_firstperson (swap-1 alt) | 0.068 | [0.055, 0.086] |
| lenbucket (swap-2 key) | L_sentiment | 0.017 | [0.010, 0.026] |
| lenbucket (swap-2 key) | L_readability | 0.035 | [0.024, 0.050] |

**Observed ordering is inverted from the prediction above** (readability 0.047 >
sentiment 0.038), though the two headline-target CIs overlap almost entirely
([0.028,0.053] vs [0.036,0.064]) — there is no clear winner either way. Both
prescribed swaps (alternate low-E targets sentlen/firstperson; alternate key
lenbucket) were already run by the prior session and *also* fail to separate:
every one of the six (key, target) combinations tried lands in E/(E+T) ∈
[0.017, 0.068], all mutually overlapping in their bootstrap CIs, and — critically
— **none is "clearly higher... away from the trivial extremes"**: all six sit
close to the 0 extreme. There is no high-E candidate at all in this data, so the
required *separation* (one high, one low, both non-trivial) does not exist,
independent of which of the six pairings is nominated "headline."

**Gate verdict: FAIL.** Per the handoff's own protocol ("if nothing separates
after two swaps, stop and report that as the Stage-A finding"), both prescribed
swaps have now been exhausted with no separation found.

## Go/no-go decision

**STOP. Stage B is not run.** Running the steered-generation sweep would
contradict the handoff's explicit gate ("Proceed to Stage B only if the two
targets separate... If they don't separate on this key/target pair, try one swap
before giving up entirely... if nothing separates after two swaps, stop and
report that as the Stage-A finding"). Both swaps are exhausted and none
separates. This document therefore also serves as the terminal pre-registration
record: the Stage-B predictions in §1/§3 above are recorded for completeness and
audit-trail (so a reader can see what would have been tested), but they are
**not evaluated against any Stage-B run**, because per protocol none was
performed. See `RESULTS.md` and
`dielectric/docs/steering_paper/text_experiment.results.md` for the "why not
text (yet)" write-up naming the specific obstacle
(individual-reviewer variance dominates category-level variance for every
tone/style/complexity proxy tried on real review text).

No GPU work follows from this decision — the shared A10G (running the other
long-running experiment at ~12.8GB/98% util at time of writing) is untouched.
