# Abstract arc — plan before prose

Purpose: lay out the logical sequence the abstract must follow *before*
writing sentences, so edits (especially trims) can be checked against it.
Each beat has a **claim**, the **why** (what it follows from / why the
reader needs it before the next beat), and the **paper section** it's
grounded in. If a future edit drops a "why," that's a sign of drift, not
just tightening — check here first before cutting.

Update this file whenever the abstract's structure changes; treat drift
between this file and `paper.md`'s actual abstract as a bug to fix, not
something to silently let happen.

## The beats

1. **Hook.** Knobs (prompts, guidance scales, tags) hit a ceiling — push
   harder and the property stops moving.
   *Why:* concrete, familiar entry point before any formalism. (§1 intro)

2. **Reframe.** The ceiling isn't a flaw in the model — it's a budget fixed
   by the training data before the model exists.
   *Why:* turns an empirical annoyance into something knowable in advance;
   this is what makes it a framework, not a trick. (§1, §3.1)

3. **Two ways to tell.** *Tagging* names a value of an attribute — a class,
   a chemistry — and restricts the output space to that subset. *Turning a
   knob* adjusts a continuous or discrete input, hoping to bias the output
   toward a property, without picking any subset itself.
   *Why:* the paper uses both under one posture, "telling"; defining each
   plainly, before saying anything about a shared limit, stops the reader
   from assuming one is being reduced to the other. (§1, "a prompt, a
   guidance scale, a property tag")

4. **Telling's limit.** Telling — tag or knob — only nudges the average
   property value within whatever subset it's confined to. Without an
   audit, you don't know what that average is or what else is reachable —
   telling operates in the dark.
   *Why:* the ceiling isn't just a restriction, it's a blind one; that
   blindness, not the restriction alone, is what beat 6's audit fixes.
   (§3.2, "shifts weight among the outputs ... but cannot change which
   bin")

5. **The budget.** The actual range of property values achievable across
   *all* outputs, not just one subset, is the **budget**. Telling only
   ever moves the average within its own subset's slice of that budget, so
   it cannot deliver a target average outside what that slice contains —
   whether the slice came from a tag or was left as the whole space.
   *Why:* names the resource being rationed, restates beat 4's per-subset
   ceiling as a comparison against the whole, and closes the "leave it
   untagged" escape explicitly. (§3.2, $T=\sum_b w_b v_b$)

6. **Bins and showing.** Binning — partitioning outputs by an
   easy-to-calculate property, then auditing each partition's average —
   is what turns the dark of beat 4 into a map of what's achievable where.
   *Showing* then targets any value that map covers, by combining bins in
   a chosen recipe rather than nudging just one.
   *Why:* binning's payoff is the audit itself; showing is only possible
   because binning made the budget legible. (§3.1, key $\pi$; §3.2,
   $E=\sum_b w_b(g_b-\bar g)^2$)

7. **The split (result 1).** A target property's variance splits exactly
   into a within-bin part $T$ (telling's reach) and a between-bin part $E$
   (showing's reach) — the exact version of beats 5–6.
   *Why:* turns the qualitative knob-vs-showing distinction into an exact,
   computable quantity. (§3.2, law of total variance)

8. **Practicality + predictability (result 2).** Both parts, and whether
   they transfer to a trained model, are knowable before the model exists
   — no generative model, no GPU — demonstrated concretely: a carry-over
   ordering forecast on crystals before auditing, confirmed after.
   *Why:* answers "is this just theory" — shows falsifiable, pre-registered
   prediction, not post-hoc explanation. This is Contribution 2 in the
   Introduction; do not let it quietly disappear during trims.

9. **Empirical verification (result 3).** Two unrelated domains (image,
   crystal-structure generation); showing beats the strongest knob
   baselines by a large, quantified margin (4.8–26×, 3×).
   *Why:* generality (not a one-domain artifact) + magnitude (large enough
   to matter). (§4.2)

10. **Practicality of showing itself (result 4).** Hitting a chosen point
    via showing needs no fine-tuning — just a different choice of examples.
    *Why:* answers "is showing hard to adopt" — no, it's inference-time
    only. Distinct from beat 8's practicality claim (that one is about the
    audit/prediction step, not about running showing itself) — don't
    collapse the two into one dangling "no X" clause.

11. **When knobs still win (result 5, honesty).** On occasion telling is
    the better choice, and the same two budget numbers say when, in
    advance.
    *Why:* prevents overclaiming ("showing always wins"); shows the
    framework cuts both ways. (§4.2, brightness/aesthetic crossover)

12. **Curvature (result 6).** Given showing is the right call, a
    property's curvature (average vs. coverage goal) sets whether to
    concentrate on one bin or spread across several.
    *Why:* a second, distinct decision the framework resolves — do not
    let this stand in as the reason knobs vs. showing is decided (that's
    beat 11's job, not this one — this was an actual bug once, see git
    history around commit 38213ac).

13. **Expressiveness (bonus benefit, separate axis).** Showing also lets
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
  §3.4) — related to but distinct from beat 13's expressiveness framing.
