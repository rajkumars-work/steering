# Intro arc — plan before prose

Purpose: lay out the logical sequence the intro must follow *before*
writing sentences, so edits (especially trims) can be checked against it.
Each beat has a **claim**, the **why** (what it follows from / why the
reader needs it before the next beat), and the **paper section** it's
grounded in. If a future edit drops a "why," that's a sign of drift, not
just tightening — check here first before cutting.

Update this file whenever the abstract's structure changes; treat drift
between this file and `paper.md`'s actual intro as a bug to fix, not
something to silently let happen.

## The beats

1. **Hook.** Knobs (prompts, guidance scales, tags) hit a ceiling — push
   harder and the property stops moving.
   *Why:* concrete, familiar entry point before any formalism. (§1 intro)

2. **Reframe.** The ceiling isn't a flaw in the model — it's a budget fixed
   by the training data before the model exists.
   *Why:* turns an empirical annoyance into something knowable in advance;
   this is what makes it a framework, not a trick. (§1, §3.1)

3. **Why telling is restricted.** A knob works by naming one value of some
   attribute and pushing on it — it can't simultaneously draw on a
   *different* named value. So whatever it reaches is confined to outputs
   sharing that one value.
   *Why:* the mechanistic reason the ceiling must exist, not just an
   assertion that it does. (§3.2, "stays inside the named bin")

4. **Why binning is the right lens** — *the connective tissue most often
   dropped; check this beat explicitly on every edit.* Because a knob
   already restricts you to outputs sharing one named value, sorting
   outputs by that same *kind* of attribute — into bins — is what makes
   the size of that restriction computable: compare one bin against all
   of them. Binning isn't an arbitrary analysis choice; it's forced by how
   knobs already work.
   *Why:* without this, "we group outputs into bins" reads as an
   unmotivated methodological step instead of a consequence of beat 3.
   (§3.1, key π)

5. **The split (result 1).** A target property's variance splits exactly
   into a within-bin part (telling's reach) and a between-bin part
   (reachable only by showing — a batch drawn from several bins at once).
   *Why:* turns the qualitative knob-vs-showing distinction into an exact,
   computable quantity. (§3.2, law of total variance)

6. **Practicality + predictability (result 2).** Both parts, and whether
   they transfer to a trained model, are knowable before the model exists
   — no generative model, no GPU — demonstrated concretely: a carry-over
   ordering forecast on crystals before auditing, confirmed after.
   *Why:* answers "is this just theory" — shows falsifiable, pre-registered
   prediction, not post-hoc explanation. This is Contribution 2 in the
   Introduction; do not let it quietly disappear during trims.

7. **Empirical verification (result 3).** Two unrelated domains (image,
   crystal-structure generation); showing beats the strongest knob
   baselines by a large, quantified margin (4.8–26×, 3×).
   *Why:* generality (not a one-domain artifact) + magnitude (large enough
   to matter). (§4.2)

8. **Practicality of showing itself (result 4).** Hitting a chosen point
   via showing needs no fine-tuning — just a different choice of examples.
   *Why:* answers "is showing hard to adopt" — no, it's inference-time
   only. Distinct from beat 6's practicality claim (that one is about the
   audit/prediction step, not about running showing itself) — don't
   collapse the two into one dangling "no X" clause.

9. **When knobs still win (result 5, honesty).** On occasion telling is
   the better choice, and the same two budget numbers say when, in
   advance.
   *Why:* prevents overclaiming ("showing always wins"); shows the
   framework cuts both ways. (§4.2, brightness/aesthetic crossover)

10. **Curvature (result 6).** Given showing is the right call, a
    property's curvature (average vs. coverage goal) sets whether to
    concentrate on one bin or spread across several.
    *Why:* a second, distinct decision the framework resolves — do not
    let this stand in as the reason knobs vs. showing is decided (that's
    beat 9's job, not this one — this was an actual bug once, see git
    history around commit 38213ac).

11. **Expressiveness (bonus benefit, separate axis).** Showing also lets
    you steer toward targets you can only recognize, not name — works
    without ever having a scorer for the property.
    *Why:* a genuinely separate benefit, not part of the reach/budget
    story — keep it clearly marked as "also," not folded into the
    quantitative claims above it. (§1 intro, Polanyi 1966; §3.4, lift)

## Known-missing (flagged, not yet in the abstract, judged non-essential
so far — revisit if space allows or reviewers ask)

- The key π itself can be *chosen* via a data-side score, not just
  assumed given (Contribution 3's closing clause, §4.2).
- The scorer-free "lift" tool for tacit steering (rank correlation 0.94,
  §3.4) — related to but distinct from beat 11's expressiveness framing.
