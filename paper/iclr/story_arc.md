# Story arc — plan before prose

Purpose: lay out the logical sequence the *whole story* (Intro → Abstract →
Method) must follow *before* writing sentences, so any section's prose can
be checked against one shared arc instead of drifting section by section.
Each beat has a **claim**, the **why**, and the **paper section** it's
grounded in.

Supersedes the old `intro_arc.md`. `abstract_arc.md` should be read as this
arc's abstract-length compression — if the two disagree, this file wins and
`abstract_arc.md` gets fixed to match.

Update this file whenever the story's structure changes; treat drift
between this file and the paper's actual prose as a bug to fix.

## Terminology fix

"Tagging" is retired — it doesn't name what's being done, and it invited
confusion with the *output*-side "bin" during editing. Replacement:

- **Slicing**: specifying an input property — an image's class, a
  crystal's element-set — that narrows the pool of outputs drawn from. A
  slicing parameter *picks a subset*. (Left unsliced, the "subset" is the
  whole space.)
- **Turning a knob**: varying some *other* input, continuous or discrete,
  hoping to bias the property we care about, *without itself picking a
  subset*.
- **Telling**: the umbrella — slicing, knob-turning, or both.
- **Binning**: partitioning *outputs* by an easy-to-calculate output
  property, purely to audit. Output-side, not input-side — a bin doesn't
  change what gets generated, it's a lens on what already was. This is
  what keeps binning distinct from slicing, which is an input-side move.
- **Showing**: choosing a recipe — proportions across several audited bins
  combined into one batch — to reach a target value the audit revealed.

"Slicing" was chosen so it echoes forward cleanly: telling only ever
reaches "that slice's share of the budget" — the word already carries the
right image.

## The five-act arc

### Act I — The limits: budget × slice
1. **Hook.** A knob saturates: push it far enough and it stops moving the
   property you care about. *Why:* concrete, familiar entry point. (§1)
2. **Reframe.** Two things cap it, not one: (a) a **budget** — the actual
   range the property can take across *all* outputs, fixed by the training
   data before the model exists; (b) whatever **slice** telling has
   confined outputs to (or the whole space, if unsliced).
   *Why it's two things, not one — the two degenerate cases:* a knob with
   no slice is still drawing from the whole, heterogeneous population, so
   you can't tell how much of any shift is the knob and how much is just
   which part of that population you happened to draw from — you don't
   know how the knob is influencing the property at all. A slice with no
   knob fixes *which* population you draw from, but you're then sampling
   it arbitrarily: whether you reach that slice's own range of the
   property, and what that range even is, is an accident of how the slice
   was chosen, not something you targeted. Neither move alone tells you
   where you stand inside the budget — that not-knowing is exactly what
   Act II's audit answers. (§1, §3.2)

### Act II — Understanding the full scope: why binning, not just auditing
3. **Motivating example** (candidate for the Introduction, not just this
   doc). Some desired properties have no input you can slice on directly —
   there is no "nature" class to specify, only "forest" as a proxy. Binning
   outputs by color-composition and auditing each bin's actual
   *natureness* is how you find out what a proxy slice actually got you,
   when telling can't name the property directly.
   *Why:* motivates binning as a response to a real gap (imperfect
   proxies), not an arbitrary methodological choice. (§3.1, "key π chosen
   from a data-side score")
4. **Binning's two jobs.** (a) *Audit* — reveal the budget and any slice's
   share of it, via each bin's share, mean, and variance; (b) *unit of
   granularity* — bins are also what showing draws its recipe from, so bin
   size trades targeting precision against estimate quality.
   *Why:* binning isn't only a measuring device; it's simultaneously
   building the vocabulary showing will use in Act III. Keeping both jobs
   explicit stops "we group outputs into bins" from reading as one
   unmotivated step. (§3.1, key π)
5. **The two extremes make the tradeoff concrete.** One giant bin (the
   whole dataset): the mean is knowable, but nothing is *targetable* —
   $\psi$ is expensive, so any draw just returns the population average.
   One bin per output: every value is technically present, but no bin's
   mean is *estimable* (too few samples each) and nothing generalizes past
   the training set. The right bin count sits between: coarse enough to
   estimate reliably, fine enough that different bins actually differ.
   *Why:* answers "why not just make one huge bin" and "why not just bin
   everything separately" before a reader thinks to ask either. (§3.1)

### Act III — Targeting any point in the full scope
6. **Showing.** Given audited bins, *showing* composes a recipe —
   proportions across bins — that reaches any value the audit says is
   achievable, not only one slice's average.
   *Why:* this is the actual payoff Act II was building toward — not the
   audit itself, but what the audit lets you *do*. If a reader comes away
   thinking "the contribution is binning," this beat was under-stated
   relative to it. (§3.1, $\mu_P$)

### Act IV — The formalism
7. **The split (result 1).** A target property's variance divides exactly
   into a within-bin part $T$ (telling's reach) and a between-bin part $E$
   (showing's reach) — law of total variance, turning Acts I–III into one
   computable quantity. *Why:* the qualitative telling-vs-showing
   distinction becomes exact and checkable. (§3.2)

### Act V — Grounded in data, before a model exists
8. **Practicality + predictability (result 2).** Both $T$ and $E$, and
   whether they transfer to a trained model, are computable from data
   alone — no generative model, no GPU. Demonstrated: a carry-over order
   forecast on crystals *before* auditing the model, confirmed after.
   *Why:* answers "is this just theory" with a falsifiable, pre-registered
   prediction. (Contribution 2; do not let this quietly disappear in
   trims.)
9. **Empirical verification (result 3).** Two unrelated domains (image,
   crystal-structure generation); showing beats the strongest knob
   baselines built by 4.8–26× (crystal properties), 3× (image targets).
   *Why:* generality + magnitude. (§4.2)
10. **Practicality of showing itself (result 4).** Hitting a chosen point
    via showing needs no fine-tuning — just a different choice of
    examples. *Why:* distinct from beat 8's practicality claim (that one
    is about the audit/prediction step, not about running showing).
11. **When telling still wins (result 5, honesty).** On occasion telling
    is the better choice, and the same $T$, $E$ numbers say when, in
    advance. *Why:* prevents overclaiming "showing always wins." (§4.2,
    brightness/aesthetic crossover)
12. **Curvature (result 6).** Given showing is the right call, a
    property's curvature (average vs. coverage goal) sets whether to
    concentrate on one bin or spread across several. *Why:* a second,
    distinct decision the framework resolves — do not let this stand in
    for beat 11's job (that was an actual bug once; see git history around
    commit 38213ac).
13. **Expressiveness (bonus, separate axis).** Showing also lets you steer
    toward a target you can recognize but never name — no scorer needed.
    *Why:* a genuinely separate benefit from the reach/budget story above
    it — mark it "also," don't fold it into the quantitative claims.
    (§1 intro, Polanyi 1966; §3.4, lift)

## Contributions, reframed

Binning and auditing are Act II's *device* for finding the budget's
boundaries — they exist to serve contribution 3, not to stand in for it.
The actual claims, in order of weight:

1. Steering has a budget, fixed by the training data, independent of any
   knob.
2. That budget divides exactly between what a knob can reach ($T$) and
   what only showing can reach ($E$).
3. A method — showing, via bin recipes — that hits any point the budget
   allows, not just an average within one slice.
4. Evidence that auditing training data alone predicts what carries over
   to the trained model, before the model is ever audited.
5. Showing is strictly more expressive than a knob: it reaches further
   (1–3 above) and can target what has no name, only examples.

If a draft's first paragraph reads as "we group outputs and audit them,"
that's beat 6 and this section failing to land — the sentence should read
as "we find out how far steering can go, and how to get anywhere in that
range," with binning named as the tool, not the headline.

**Where this must show up.** A reviewer reads three places closely:
Abstract, Introduction, Conclusion. This contributions list, appropriately
compressed for each, must appear recognizably in all three — check each
one after any wording change here, not just the section being actively
edited.

## Known-missing (flagged, judged non-essential so far)

- The key $\pi$ itself can be *chosen* via a data-side score, not just
  assumed given (Act II beat 3 gestures at this; Contribution 4's closing
  clause, §4.2).
- The scorer-free "lift" tool for tacit steering (rank correlation 0.94,
  §3.4) — related to but distinct from beat 13's expressiveness framing.
- ~~Bin-granularity / choice-of-$\pi$ discussion.~~ **Done.** Added
  Appendix F: the two-extremes argument (Act II beat 5), plus the
  already-verified in-paper evidence that a key's *structure*, not its
  bin count, sets the split (§4.2's structured-vs-random 50-bin ablation)
  and that key choice is itself a data-side audit (§4.2's rank-correlation
  0.88 result). Cross-referenced from §3.2 and Introduction Contribution
  3. No new experiments — built entirely from results already in §4.2.
