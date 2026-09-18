<!-- ICLR-style condensation of the arXiv long-form (../arxiv/*.v2.md). Fresh
     rewrite 2026-07-01 for the examples-beat-knobs reframe + Option B (telling =
     within-bin sqrt(T); the single-best-bin move is *concentrated showing*, not
     telling; curvature sets the shape of the showing recipe). Regenerated the
     framing to track the finalized arXiv: abstract/intro lead with the *felt
     ceiling* (a knob saturates), not "a knob needs the target named"; telling is
     "turn a knob", never "describe the target"; §3.3 curvature stated as F(mu)
     maximized over the simplex (convex->vertex/one bin, concave->interior/spread),
     no Jensen name, no averaged-sqrt(T) parenthetical. Evidence numbers/tables
     carried verbatim from the reconciled arXiv sections (95% CIs). ~9pp + appendix. -->

# The Steering Budget: Examples beat Knobs

## Abstract

Generative models are steered by turning knobs — prompts, guidance scales,
tags — but every knob saturates: push it far enough and it stops moving
the property you care about. This ceiling has two sources: a budget fixed
by the training data before the model exists, and the subset of outputs a
knob is confined to — the class, chemistry, or other value an input
property names. Turning a knob — changing some other input — only ever
moves the target property within that subset's slice of the budget, never past
it; we call this *telling*. We introduce *showing*, which exposes
the property's whole budget and reaches any value within it, not just
one subset's share, by binning outputs on an easy-to-calculate property,
auditing each bin, and choosing which bins to draw from.

This audit makes the split exact: a target property's variance divides
into a within-bin part (telling's reach) and a between-bin part (showing's
reach), both computable from data alone, before a model exists. These two
numbers predict, in advance, whether they transfer to a trained model: on
crystals, they correctly forecast the carry-over order of three properties
before the model was ever audited. Across two unrelated domains — image
and crystal-structure generation — showing out-reaches the strongest knob
baselines we could build by 4.8–26× on crystal properties and 3× on image
targets, with no fine-tuning: only a different choice of examples. The
same two numbers also say, in advance, the occasions when telling remains
the better choice. Given showing wins, a property's curvature determines
the shape of the fix: concentrate on one bin for an average goal, spread
across bins for coverage. Showing further buys expressiveness a knob
cannot: because the specification lives in the examples, it can steer
toward properties hard to quantify but easy to *show* such as a favorite
collection of images.

## 1 Introduction

You have a generative model and want it to lean a certain way — images with
the feel of nature photography, crystals with a wider band gap. You reach for
the controls: a prompt, a guidance scale, a property tag. Past a point they
stop moving the property — you push harder and the output barely changes. We
show this ceiling is not a shortcoming of the model but a **budget**, fixed by
the training data before the model is trained, and that a different move —
*showing* the model examples instead of turning a knob — reaches a part of the
budget the knob structurally cannot.

*Tagging* specifies an input property — an image's class, a crystal's
element-set — and thereby fixes the subset of outputs you draw from.
*Turning a knob* changes some other input, continuous or discrete, hoping
to bias the property, without itself picking a subset. Both are *telling*:
whichever subset a tag has fixed, or the whole space if untagged, a knob
can only reweight the outputs *within* it, never past that subset's own
share of the budget. *Showing* — handing over a set of examples and asking
for more like them — reweights the mix *across* bins instead, reaching
values no single subset contains. Our central result makes this
quantitative. A generative model produces outputs $D$ (an image, a crystal
structure). A cheap key $P=\pi(D)$ sorts each output into a bin — an image's
class, say; for a target property $L=\psi(D)$ — an image's aesthetic score, a
crystal's band gap — the law of total variance splits $\mathrm{Var}(L)$ into a
within-bin budget $T$ and a between-bin budget $E$. Telling reweights the
outputs *within* one named bin and reaches $\sqrt{T}$; showing hands over a
set of examples, which reweights the mix *across* bins, and reaches
$\sqrt{\chi^2 E}$. Both budgets are fixed by the data before the model exists,
so one can predict, in advance, which targets a knob cannot reach — and,
where a knob falls short, which *shape* of exemplar recipe closes the gap.

Showing additionally adds something a knob cannot: **expressiveness**. Because the
specification lives in the examples, you can steer toward a target you can only
point to — the curator who keeps ten images of a hundred, the materials
scientist who flags a promising structure, for reasons they cannot fully state
(Polanyi 1966). A knob demands the target be named first; showing does not.

**Contributions.**

1. **A computable steering budget.** The steerable mean-shift of a labeled
   property decomposes into a *telling* reach $\sqrt{\chi^2_{\text{tell}}T}$
   (about $\sqrt{T}$ in practice) and a *showing* reach $\sqrt{\chi^2 E}$,
   both read from training data before the model exists — law of total
   variance for the split, Cauchy–Schwarz for the reaches. §3.
2. **A model-free audit with a pre-registrable carry-over test.** An *audit*
   reads three numbers per bin — its share, mean, and variance — straight
   from data, no model needed; whether those numbers transfer to a trained
   model is decided by two diagnostics — bin-mean drift $\delta$ and
   within-bin spread ratio $\rho$. On crystals we *predicted the carry-over
   ordering of three targets before auditing*, from these signals alone, and
   the audit confirmed it. §3, §4.
3. **Examples beat knobs, measured — and the budget says when.** Pitted head
   to head against the strongest knob each domain allows (best-of-twelve; a
   learned soft-prompt optimised on the scorer), choosing bins moves the batch
   mean 4.8–26× further on the crystal properties and ~3× on high-share image
   targets; where the between-bin share is small (brightness, aesthetic) a
   strong knob wins, a crossover $E/(E{+}T)$ predicts in advance — and the same
   score, computed data-side, picks the key $\pi$. §4.
4. **Curvature sets the recipe.** A goal that is an *average* (a rate, a
   conjunction) is reached by *concentrating* the batch on one audit-chosen
   bin; a *spread* goal (coverage of opposed properties) only by a *mix* of
   bins. Both are showing; naming the goal (telling) reaches neither. We
   verify both shapes, and give closed-form recipes plus a scorer-free *lift*
   that forecasts which bins pay off (rank correlation 0.94). §3, §4.

The same predictions hold in domains that share no model, no data, and no
code, which argues they are properties of the framework rather than of one
setup. The two one-line proofs, the carry-over diagnostics, the lift, and the
crystal-model spec are in the Appendix, which carries the full derivations
and additional results.

## 2 Related Work

**Controllable generation** builds *knobs* for steering — plug-and-play and
attribute-guided text decoding (Dathathri et al. 2020; Krause et al. 2021; Li
& Liang 2021; Lester et al. 2021) and latent-direction image editing
(Jahanian et al. 2020; Shen et al. 2020). These ask which interventions
exist; we ask the prior, quantitative question of how far *any* such
intervention can move a target, read off the data. Most are *telling* in our
terms.

**Guidance and alignment.** Classifier and classifier-free guidance (Dhariwal
& Nichol 2021; Ho & Salimans 2022) and fine-tuning toward a reward — RLHF
(Ziegler et al. 2019; Ouyang et al. 2022), its KL-constrained Boltzmann
optimum (Korbak et al. 2022), DPO (Rafailov et al. 2023) — reshape the output
distribution. We absorb these as closed-form rewrites of the data
distribution and recompute the budget on the reshaped data.

**Exemplar- and reference-conditioned generation** (Gal et al. 2023; Ruiz et
al. 2023; Ye et al. 2023; Blattmann et al. 2022) conditions on example sets —
the *showing* posture. This is established; what is new here is not the
posture but the *budget* that bounds what it can reach.

**Retrieval-augmented and in-context generation.** Retrieval-augmented
generation (Lewis et al. 2020) and in-context / few-shot prompting (Brown et al.
2020) are *showing* in our exact sense — a set of retrieved or in-context
examples reshapes the output distribution without a target being named — in a
domain (text) we do not test here. As above, our contribution is not this
posture but the budget that predicts, before generation, how far it will
reach: the split applies unchanged (any cheap key $\pi$ bins the outputs;
$L$ splits into $T+E$ as usual), and predicts that such examples move a
target only to the extent its variance is *between*-bin. We verify this
prediction head-to-head against the strongest knob available, in two domains
neither RAG nor in-context prompting addresses — image and crystal
generation, where "showing" is realized as choosing the conditioning
distribution a batch is drawn from rather than literal in-context exemplars
— and leave the text-domain measurement to future work.

**Diversity and mode collapse** (Goodfellow et al. 2014; Salimans et al.
2016) and fidelity/diversity metrics (Sajjadi et al. 2018; Kynkäänniemi et
al. 2019; Naeem et al. 2020) document that optimizing for quality costs
variety. We give a structural account: a per-output objective reaches only
the within-bin budget; between-bin variety is reachable only by acting on the
bin mixture. The showing bound is an importance-sampling result (Owen 2013)
applied to the bin distribution; the split is the law of total variance
(Casella & Berger 2002).

## 3 Method

### 3.1 Setup and assumptions

A generator $\mathrm{G}$ produces outputs $D$. A **cheap key** $\pi$ assigns each a
label $P=\pi(D)$ (an image's class, a crystal's chemistry), which sorts
outputs into $B$ **bins**. An **expensive target** $L=\psi(D)$ — we call
$\psi$ the **scorer** — is the property to steer (aesthetic, band gap). An
**audit** — run on any distribution over outputs, e.g. the training
distribution $\mathcal{J}$, or a model's own samples once one exists —
records, per bin $b$: its share $w_b$, mean $g_b=\mathbb{E}[L\mid P_b]$, and
within-bin variance $v_b$. ($\pi$ is cheap, so $w_b$ is exact over all
outputs; $\psi$ is expensive, so $g_b, v_b$ are estimated from a modest scored
sample per bin.) Unless noted otherwise, "the audit" means the data-side
audit, on $\mathcal{J}$.

| symbol | meaning |
|----------------|----------------------------------------|
| $\pi,\ P=\pi(D)$ | cheap **key**; the label it assigns (sorts outputs into **bins**) |
| $\psi,\ L=\psi(D)$ | expensive scorer; the **target** property to steer |
| $b,\ B$ | a bin; the number of bins |
| $w_b,\ g_b,\ v_b$ | bin $b$'s **share**, **mean** of $L$, within-bin **variance** of $L$ |
| $\bar g$ | overall mean $\sum_b w_b g_b$ |
| $T=\sum_b w_b v_b$ | **within-bin budget** — telling's room |
| $E=\sum_b w_b(g_b-\bar g)^2$ | **between-bin budget** — showing's room |
| $\mathcal{J},\ \mathcal{J}_P$ | training distribution; its bin distribution |
| $\mu_P$ | the **recipe** — a designed distribution over bins (showing sets it) |
| $\chi^2(\mu_P\|\mathcal{J}_P)$ | **effort** — how far the recipe departs from the data's mix |
| reach $=\sqrt{\chi^2\cdot\text{budget}}$ | the most a reweighting can shift the mean |
| $\mathrm{lift}=\mu_P(b)/w_b$ | scorer-free estimate of a bin's target-clearing rate |
| $\delta,\ \rho$ | carry-over diagnostics: bin-mean drift; within-bin spread ratio |

: Notation.

**Assumption (carry-over).** For the target $L$, the trained model's per-bin
statistics match the data's: $g_b(\mathrm{G})\approx g_b$ and $v_b(\mathrm{G})\approx v_b$
on the bins in play. This holds to the extent $L$ is faithfully reproduced by
the model (it is trained to match its data, not to optimize $L$); we do not
assume it, we **test** it per target with the drift $\delta$ and spread ratio
$\rho$ (§4, Claim 1).

### 3.2 The budget and the two reaches

**The split.** By the law of total variance (Casella & Berger 2002),
$$\mathrm{Var}_{\mathcal{J}}(L) \;=\; \underbrace{\textstyle\sum_b w_b v_b}_{T\ \text{(within-bin)}} \;+\; \underbrace{\textstyle\sum_b w_b (g_b-\bar g)^2}_{E\ \text{(between-bin)}},\qquad \bar g=\textstyle\sum_b w_b g_b.$$
$T$ is the room *telling* works in (within one bin); $E$ the room *showing*
works in (across bins). The split is exact for any $(\pi,L,\mathcal{J})$ and is set,
before any model exists, by the choice of key $\pi$ — finer bins move variance
into $E$, coarser into $T$.

![The budget: $\mathrm{Var}(L)$ splits into within-bin $T$ (telling) and between-bin $E$ (showing). Illustrative numbers, not measured data.](figures/fig_lotv_bandgap.png){width=70%}

*Worked example (Fig. 1's numbers).* Four bins, equal shares, with means
$g=(1.2,3.0,4.2,6.8)$ and within-bin variances $v=(0.16,1.21,0.25,0.18)$: then
$\bar g=3.8$, $T=\sum_b w_b v_b=0.45$, $E=\sum_b w_b(g_b-\bar g)^2=4.14$, so
$\mathrm{Var}(L)=4.59$ and $E/(E+T)=0.90$. Telling's unit-effort reach is
$\sqrt{T}=0.67$, showing's is $\sqrt{E}=2.03$: the property lives mostly *between*
bins, so a knob confined to one column can move its mean only about a third as
far as choosing columns.

**The two reaches.** Telling and showing are two ways to *reweight* the same
distribution — to change which outputs the batch draws more of — and they
differ only in *which* set each is allowed to reweight; both obey one law:
departing from a set's own proportions shifts its mean by at most
$\sqrt{\chi^2\cdot\sigma^2}$, where $\sigma^2$ is the variance of the set
reweighted and the Pearson $\chi^2$-divergence measures the size of the
departure (Cauchy–Schwarz).

*Telling* stays inside the named bin: a prompt or guidance knob shifts weight
among the outputs *within* a bin (a "golden retriever" prompt still gives you
retrievers, just brighter or more textbook ones) but cannot change which bin,
so it draws on $T$,
$$|\Delta\mathbb{E}[L]|\le\sqrt{\chi^2_{\text{tell}}\,T}.$$
In practice the controls push gently — even maximum guidance realizes
$\chi^2_{\text{tell}}\approx 1$ (§4) — so telling's working reach is about
$\sqrt{T}$.

*Showing* changes the bin mix instead: handing over an exemplar set is
steered not by any one example but by the **recipe** $\mu_P$ it implies — a
designed distribution over bins (e.g. "70% high-band-gap chemistries, 30%
low"; Notation, §3.1). The shift is $\Delta\mathbb{E}[L]=\sum_b(\mu_P(b)-\mathcal{J}_P(b))\,g_b$, and by
Cauchy–Schwarz
$$|\Delta\mathbb{E}[L]| \;\le\; \sqrt{\chi^2(\mu_P\|\mathcal{J}_P)\cdot E},\qquad \chi^2(\mu_P\|\mathcal{J}_P)=\sum_b\frac{(\mu_P(b)-\mathcal{J}_P(b))^2}{\mathcal{J}_P(b)}.$$
The reaches have the same form; the asymmetry is (i) **structural** — telling
is locked to $T$ and cannot reach $E$ *at all*, since it cannot cross bins —
and (ii) **empirical** — a recipe departs far harder from the data mix than a
prompt does (a recipe rewrites the whole bin mixture; a prompt only nudges
within one bin), so showing realizes $\chi^2$ of order 100 against telling's
≈1 (§4). Both are **ceilings**: Cauchy–Schwarz is tight only when weight
concentrates on the extreme-mean bins, so realized shifts land below and the
gap grows with $\chi^2$; the min-$\chi^2$ recipe (§3.4) attains the ceiling.

**Equal effort, structural gap.** At *matched* effort (equal $\chi^2$) the two
reaches are $\sqrt{\chi^2 T}$ and $\sqrt{\chi^2 E}$, so showing out-shifts
telling by $\sqrt{E/T}$ — a data-fixed ratio **no knob strength can change**
(Fig. 1's numbers: $\sqrt{E/T}=3.0\times$; crystal band gap $7\times$, density
$4.9\times$). The realized head-to-head gaps of §4.2 are *larger* still, because
real knobs spend far less effort than recipes ($\chi^2\approx1$ vs $\approx100$,
§4.2) — but the knob-agnostic floor is $\sqrt{E/T}$: even a knob driven to
showing's effort is out-shifted by that much. Where the between-bin share is
small ($E<T$), $\sqrt{E/T}<1$ and the knob wins — the brightness case, called in
advance.

Telling draws only on $T$, showing only on $E$: whichever the data makes
larger, wins.

### 3.3 Which operation, and which shape of recipe

Two questions, in order. **Telling or showing?** — a shift you can find
*inside* your bin ($\Delta<\sqrt{T}$) needs only telling; a larger one is out
of any bin's reach, so you must show. **If showing, concentrate or spread?**
— set by the goal's **curvature**. A writable goal is a function $F(\mu_P)$ of
the recipe, maximized over the simplex of bin distributions. If $F$ is convex
(linear included), its maximum sits at a vertex: *concentrate* $\mu_P$ on one
bin. Batch averages, rates, and conjunctions ("$A$ and $B$",
$F=\sum_b\mu_P(b)\,c_b$) are the type case — and the winning bin is the one
the *audit* finds, not the bin telling starts in. If $F$ is strictly concave
its maximum is interior — *spread* $\mu_P$ across bins: coverage of two
opposed properties is
the type case, since no single bin holds both corners. Both shapes are
showing; *naming* the goal (telling) reaches neither.

### 3.4 Designing recipes

To hit a target mean $g^*$: **pool-mixing** blends a low and a high pool with
a dial $\alpha$; **min-$\chi^2$** is the gentlest recipe reaching $g^*$, in
closed form
$\mu_P(P_b)=\mathcal{J}_P(P_b)\,[1+(g^*-\bar g)(g_b-\bar g)/\mathrm{Var}_{\mathcal{J}}(g_b)]$,
which is the Cauchy–Schwarz-tight direction. When no scorer $\psi$ for $L$ is
available (*tacit* steering), $\mu_P$ is the bin histogram of an exemplar set,
and each bin's **lift** $=\mu_P(b)/\mathcal{J}_P(b)$ estimates the fraction of that
bin clearing the target — with no scorer (Appendix).

### 3.5 Predictions

Four falsifiable claims, tested in §4:

1. **Split is exact and carries.** The split is exact, moves with the key,
   and *carries to the model* for faithfully reproduced targets.
2. **Reaches are orthogonal.** Telling and showing reach orthogonal parts of
   the budget; we measure how much further showing reaches than a knob.
3. **Curvature sets the shape.** A goal's curvature sets the shape of the
   showing recipe — concentrate for an average, spread for coverage — while
   naming (telling) reaches neither.
4. **Fine-tuning reshapes the budget.** Standard fine-tuning changes the
   budget predictably (recompute on the reshaped data).

## 4 Experiments

### 4.1 Setup

Two domains that share no model, data, or code: **images**, where you can
inspect a bin's contents directly by eye, and **crystals**, where the targets
are real materials-discovery quantities (would this structure be worth
synthesizing?) that no eyeballing can check.

| | **Images** (inspect by eye) | **Crystals** (real materials-discovery stakes) |
|--------|----------------------------------|----------------------------------|
| generator | pretrained DiT-XL/2-256, used as-is | pretrained encoder–decoder (Appendix: full spec) |
| training data | ImageNet | Alexandria-derived data |
| key $\pi$ | ResNet-50 class (1000 bins) | chemistry: elements + atom count |
| targets $L$ | brightness; a learned aesthetic score; file-size/MP; four CLIP content scores (animal/vehicle/food/nature) | band gap; energy-above-hull; density; dielectric constant; *Combined* — the fraction of generated structures that are novel, unique, and at least metastable, a proxy for viable new-material yield |

**Uncertainty:** every headline number carries a 95% CI from bootstrap over
bins/structures and/or ≥3 generation seeds. **Cost:** the data-side claims
need only the audit *shadow* (per-bin share, mean, variance — 132 KB) and
reproduce with `numpy` in under a second, no model and no GPU.

### 4.2 Results

**Claim 1 — split exact, moves with the key, carries to the model.** The
split is an identity (a measurement check, not evidence); coarsening the key
shifts it toward within-bin as predicted (images: between $0.0030\!\to\!0.0002$
merging 1000 classes into a structured 50; a *random* 50-bin key instead
collapses $E$ ≈18× — so the budget reads the key's structure, not its
resolution). The substantive test is carry-over: per-bin agreement between
model and data, by target, 95% CIs from bootstrap over chemistries × 3 seeds.

| target (crystal) | carry-over (0–1) | 95% CI |
|---|---:|---|
| density | 0.985 | [0.91, 1.00] |
| bonding (BVS-GII) | 0.879 | [0.04, 0.97] |
| stability (e-hull) | 0.488 | [0.17, 0.71] |

Geometry-like targets carry tightly; stability carries weakly and uncertainly
(a small structural error flips it off the hull), exactly where the carry-over
assumption (§3.1) is expected to fail. **This ordering was pre-registered:**
from an independent pilot using only support overlap and the $\delta,\rho$
diagnostics, we committed in advance (timestamped) to density > bonding >
stability, density $\gtrsim0.9$ and stability $\lesssim0.4$. The converged audit
confirmed the **ordering** exactly (density 0.99, bonding 0.88, stability 0.49);
the magnitude calls were close, with stability landing at 0.49 — just above its
pre-registered $\lesssim0.4$ band, the one number that overshot. So carry-over
is *forecast* — from cheap pilot signals,
committed before the full audit — then confirmed on the model, not a pattern
read off afterward. (Note the split $T/E$ is pure data, but the diagnostics
$\delta,\rho$ are model-vs-data, Appendix B — so carry-over, unlike the split,
is a claim *about* the model, not something knowable before it exists.) On images, the model
reproduces $0.96$–$1.14\times$ the data's within-bin spread ($\rho\approx1$),
and class-aligned per-bin means track the data at $r^2\,0.85$–$0.94$.

![Carry-over by target (crystal), 95% CIs: density 0.99, bonding 0.88, stability 0.49.](figures/fig_claim1_survival_ladder.png){width=60%}

**Claim 2 — telling and showing reach orthogonal budgets, and how much
further.** Telling stays within one bin; showing ranges across them. We first
measure the *effort* each spends: telling swept to maximum classifier-free
guidance (CFG = 15) realizes $\chi^2\approx1$–$1.6$, while exemplar recipes
reach $\chi^2$ of order 100. At every guidance scale the realized shift stayed
under telling's own within-bin ceiling $\sqrt{\chi^2 T}$ — so $\sqrt{T}$ is
not a wall telling hits, just the $\chi^2\approx1$ scale it can afford. As the
recipe concentrates (top 400→100→25→8 of 1000 classes) the realized showing
shift grows and stays under $\sqrt{\chi^2 E}$ (animal $0.031\!\to\!0.051$;
brightness $0.036\!\to\!0.118$). Under the **min-$\chi^2$** recipe the ceiling
becomes a *forecast*: at a small target shift the realized move ($0.0162$)
lands on the predicted ceiling ($0.0195$, inside CI $[0.0114, 0.0209]$), while
blunt top-$k$ at the same $\chi^2$ reaches only a fraction of it.

![Showing's realized shift vs the ceiling $\sqrt{\chi^2E}$ as the recipe concentrates.](figures/fig_chi2_reach.png){width=60%}

*How much further: bins vs knobs, property by property.* Pitting the strongest
available knob against bin selection on the *same* property:

| domain / target | $E/(E{+}T)$ | $\sqrt{E/T}$ (equal-effort) | bin / knob (realized) |
|------------------------------------------|:------------:|:------------:|:----------------:|
| crystal band gap | 0.98 | 7.0 | **≈800×** |
| crystal density | 0.96 | 4.9 | **≈91×** |
| crystal stability (metastable rate) | 0.57 | 1.15 | **≈41×** |
| image vehicle content | 0.57 | 1.15 | 8.0 |
| image food content | 0.63 | 1.31 | 7.2 |
| image file size / MP | 0.30 | 0.65 | 6.0 |
| image aesthetic | 0.27 | 0.61 | 4.6 |
| image animal content | 0.72 | 1.60 | 3.0 |
| image nature content | 0.53 | 1.06 | 2.8 |
| image brightness | 0.17 | 0.45 | **0.5** |

The $\sqrt{E/T}$ column is the **equal-effort** gap — what showing wins over a
knob driven to the *same* $\chi^2$. It is knob-agnostic ($7\times$ band gap,
$4.9\times$ density), exceeds $1$ wherever the between-bin share is large, and
falls below $1$ for brightness (where the knob wins). The realized bin/knob
column is *larger* because real knobs realize $\chi^2\approx1$ while recipes
reach $\approx100$ (Claim 2) — an **effort** gap, not a definitional one — but a
stronger, audit-calibrated knob is still out-shifted by at least the
$\sqrt{E/T}$ floor.
We also pushed the knob to the strongest each domain allows. On crystals,
**best-of-twelve** within a fixed chemistry — the largest within-bin reweighting
a per-output selector can do, at matched effort $\chi^2\approx5.3$ — is still
out-shifted 26× / 14× / 4.8× on gap / density / stability. On images, a
**learned soft-prompt** optimised directly on each scorer makes the bin/knob
ratio *monotone* in $E/(E{+}T)$ (0.29, 0.41, 2.97, 2.93) — and, being that
strong, **flips the two lowest-share targets**, aesthetic and brightness, to
knob-wins, while bins still win ~3× on the semantic targets. But the knob wins
those two only by *leaving its bin* — the winning images land outside the class
they started from (measured *stay* of zero) — so the experiment lets us
*construct* a knob-win, but one that breaks the bin barrier defining telling. In
one line: **$\sqrt{E/T}$ is the fair comparison; best-of-twelve shows no
within-bin knob escapes it; the soft-prompt shows the only escape is to stop
being a knob — and $E/(E{+}T)$ tells you, in advance, the one regime (low
$E/(E{+}T)$) where that escape is worth making.** The same $E/(E{+}T)$, scored
data-side across candidate keys, also **picks the key $\pi$** (rank correlation
0.88 to the realized ratio; a random partition floors it).

On all three crystal targets the knob's 95% CI includes zero (fix the chemistry
and the property tag does essentially nothing) while bin selection's excludes it.
Because that denominator straddles zero, the realized *ratio* (≈41–800×) is
statistically unstable — we report it only to convey that the tag moves
essentially nothing, and treat the knob-free $\sqrt{E/T}$ column (which needs no
knob estimate) as the trustworthy equal-effort comparison. The honest statement
for these targets is "knob shift indistinguishable from zero; showing shift
$\Delta$ = 2.71 / 8.40 / 0.385 [CI], a $\sqrt{E/T}$ = 7.0 / 4.9 / 1.15× floor."

**Claim 3 — curvature sets the recipe.** Curvature sets the shape:
*concentrate* for an averaging goal, *spread* for a coverage goal; naming
(telling) reaches neither.

*An averaging goal (concentrate).*
Ask for images that are *both* animal and natural — a fraction of the batch
clearing both bars, an average, so the theory says *concentrate*. No class is
"animal in nature," so naming fails; the audit points to bins already both
(otters, flamingos) and the best move is to draw the batch from the single
richest (128 images/seed × 10 seeds):

| how we steered | strongly both | 95% CI |
|------------------------------------------------|:------------:|:----------:|
| ask for animals (naming) | 14.5% | [12, 17] |
| ask for nature (naming) | 12.8% | [11, 15] |
| compose both — Composable Diffusion | 13.5% | [12, 15] |
| mix 30 audit-chosen both-high bins | 45.0% | [42, 48] |
| mix the top 5 | 61.0% | [58, 65] |
| **single richest bin (showing, concentrated)** | **66.5%** | [64, 69] |

Naming stays near one in seven (Composable Diffusion, the strongest way to
compose two conditions, included); showing works, and the *more concentrated*
the recipe the better — the averaging prediction exactly. This is the extreme
of showing, not telling: the bin is the one the audit *chose*, whereas a knob
held inside any one bin reaches only $\sqrt{T}$. Which bins pay off is itself
forecast by the scorer-free **lift** (rank correlation 0.94 to realized rates).

![Averaging goal (images): naming the property stays near one in seven; concentrating the batch on audit-chosen bins climbs, peaking at the single richest bin.](figures/fig_joint_target_bars.png){width=60%}

The *same averaging goal on crystals* — wide-gap *and* stable — adds a
wrinkle. Mixing chemistries (13.2%, [10.5, 16.5]) beats a *naive* request
five-fold (2.7%, [1.7, 4.3]) but ties the *strong* single request —
conditioning on the model's own wide-gap prior (15.5%, [12.5, 19.1]).
Concentrating on the right region is enough; spreading adds nothing, as for an
average. The framework says why the tie is *low*: showing produces the most
wide-gap structures of any recipe, but stability sits near one in two for
every recipe, and stability is the property Claim 1 found does not carry over
— so it caps the joint, and the two recipes converge on it.

![Crystals (averaging goal): showing beats a naive request ~4× but ties a strong one — stability, which does not carry over (Claim 1), caps both.](figures/fig_crystal_joint_bars.png){width=60%}

*A spread goal (spread).* Now a batch that must **cover both corners** of two
properties that pull apart; score coverage as the smaller corner-fraction, so
a recipe scores only if the batch holds *both*. Predictions committed before
running.

| coverage (min corner-fraction) | images | crystals |
|------------------------------------------------|:------------:|:------------:|
| ask for one (telling, one bin) | 0.00 | 0.00 |
| compose / condition on both | 0.00 | 0.02 |
| **mix a corner-$A$ and a corner-$B$ bin (showing)** | **0.30** | **0.40** |

Telling sits in one corner (coverage zero by construction); composing the two
conditions collapses to the muddy middle (car–food chimeras, 92% of the batch
— *the measured finding* that the go-to "both at once" tool returns the
average, not the spread); only showing puts genuine examples of both corners
in the batch, the only recipe above zero by a CI-separated margin. **The gap
widens with the number of corners:** counting corners covered with real,
CI-separated mass, telling covers exactly one, composing one, and showing all
$N$ — two of two at $N=2$, three of three at $N=3$ (clean $N=3$ on images) —
so the gap is $N-1$ — showing's coverage advantage over telling and
composing, made quantitative.
Curvature called both shapes correctly in both domains.

**Claim 4 — training changes the budget predictably.** Standard training
moves reshape $\mathcal{J}$ into a new $\mathcal{J}'$ in closed form
(Appendix B), so the new budget is computable without retraining — up to a
point. Under plain pretraining, model per-bin means match the data's (on the
diagonal: 0.99 on crystal density, 0.85–0.94 on image content targets,
tightening with sample size) — training reproduces $\mathcal{J}$ unchanged, so
the budget is the data's budget. **Filtering** — keeping only the
reward-best fraction of the training set, the reweighting
$\mathcal{J}'(D)\propto\mathcal{J}(D)e^{R(D)/\beta}$ of §2 and Appendix B —
moves the per-bin means measurably, and the moved values match what that
closed-form rewrite predicts to 0.03 (crystal) / 0.002 (image) of a bin
width. Guidance sweep (1,3,7,12): monotone drift (animal 0.18→0.20,
brightness 0.52→0.57), the same rewrite applied at increasing strength. The
boundary: a fine-tune aggressive enough to leave the training distribution
(held-out structures scored −15.1/token, CI [−15.2, −15.0], vs −9.7 random —
every structure below) cannot be predicted from the data — past that point,
sample and audit the model directly.

### 4.3 One mechanism behind familiar failures (interpretation)

Four familiar training failures — mode collapse, the alignment tax, reward
hacking, and compositional-generalization failure — read, in this framework,
as one cause wearing four names. Each is produced by a per-output objective
(a likelihood, a per-sample reward): such an objective can only push each
output it sees toward "better," so it can only ever draw on the within-bin
budget $T$. But what these failures actually cost is collection-level spread
— variety, coverage, a joint of two properties — and that spread lives in
$E$, reachable only by a move across bins, which is what showing does. So a
per-output objective structurally cannot recover the spread it destroys; only
acting on the mixture (showing) can. This is an interpretation the framework
invites, not one of the tested claims.

## 5 Limitations

The bounds in §3 apply to one specific move — shifting a batch's *mean* by
*reweighting* which existing outputs it draws from (selection), not by
inventing new ones — and only where the model matches the data on the bins
in play; a model pushed off that support ($\chi^2\!\to\!\infty$) is flagged,
not bounded. They are second-moment ceilings: heavy-tailed
bin means make them loose. Set-level targets (coverage, diversity) are a
different object than a mean — the framework says *which* operation reaches
them (a mixture, showing) but not how large the metric gets. The carry-over
assumption is the one empirical input and fails for targets the model
reproduces poorly (e.g. stability) — which, as Claim 3 shows, can cap what
*any* steering method reaches for a joint that leans on that target; we
therefore report it as a measured, pre-registered diagnostic rather than
assuming it. The crystal-joint band gap uses a composition surrogate (stricter
at the threshold than the full structural calculation); the direction and the
stability ceiling are robust to this, the absolute rate is not. Finally, each
domain uses one generator: but the $T/E$ budget is a property of the data and
the key, not the model, so it is invariant to model scale and architecture by
construction — only *carry-over* is model-specific, with a per-model test
(Appendix B). A second generator per domain would probe carry-over robustness,
not the budget; we leave it to future work.

## 6 Conclusion

How far a generative model can be steered toward a property is fixed, before
training, by a budget read off the data and split into a part a knob reaches
and a part only examples reach. The split is computable, testable, and —
verified in two unrelated domains — predicts which targets a knob cannot
reach and, where it falls short, which *shape* of exemplar recipe closes the
gap: concentrate the batch on one audit-chosen bin for an average, spread it
across bins for coverage. The practical rule: audit, see which half of the
budget a goal lives in, and use the operation that covers it.

## Ethics Statement

This work uses only public, non-personal data — image-generation benchmarks
and materials-generation datasets — and pretrained/openly released
generators. It does not involve human subjects, personal data, or crowd
labor, and we do not foresee dual-use or societal-harm concerns beyond those
already associated with the underlying generative models.

## Reproducibility Statement

All experiments use public datasets and openly released or reproducible
generators; architectures, training data, and checkpoint hashes for the
crystal generator are given in Appendix D, and scorers/diagnostics in
Appendices B–C. A complete code repository that reproduces every reported
number exists; to preserve double-blind anonymity it is not linked here, but
will be provided to reviewers on request and made public upon acceptance.

## AI Use Statement

In this work, we used generative AI tools to assist with
executing experiments — writing and running the code that produced the
results reported here — and with formatting the manuscript (the pandoc →
LaTeX build pipeline and document preparation). We did not use generative AI
for the theory, claims, proofs, hypotheses, methodology, data cleaning,
qualitative analysis, or interpretation of results. All AI-assisted output
was reviewed and verified by the authors.

## References

- Batatia, Kovács, Simm, Ortner, Csányi. "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields." NeurIPS 2022.
- Blattmann, Rombach, Oktay, Müller, Ommer. "Retrieval-Augmented Diffusion Models." NeurIPS 2022.
- Brown, Mann, Ryder, Subbiah, Kaplan, Dhariwal, et al. "Language Models are Few-Shot Learners." NeurIPS 2020.
- Casella, Berger. *Statistical Inference*, 2nd ed. Duxbury, 2002.
- Dathathri, Madotto, Lan, Hung, Frank, Molino, Yosinski, Liu. "Plug and Play Language Models: A Simple Approach to Controlled Text Generation." ICLR 2020.
- Deng, Dong, Socher, Li, Li, Fei-Fei. "ImageNet: A Large-Scale Hierarchical Image Database." CVPR 2009.
- Dhariwal, Nichol. "Diffusion Models Beat GANs on Image Synthesis." NeurIPS 2021.
- Emmerich, Deutz, Klinkenberg. "The Computation of the Expected Improvement in Dominated Hypervolume of Pareto Front Approximations." Technical Report, Leiden University, 2006.
- Gal, Alaluf, Atzmon, Patashnik, Bermano, Chechik, Cohen-Or. "An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion." ICLR 2023.
- Goodfellow, Pouget-Abadie, Mirza, Xu, Warde-Farley, Ozair, Courville, Bengio. "Generative Adversarial Nets." NeurIPS 2014.
- He, Zhang, Ren, Sun. "Deep Residual Learning for Image Recognition." CVPR 2016.
- Ho, Salimans. "Classifier-Free Diffusion Guidance." arXiv:2207.12598, 2022.
- Jahanian, Chai, Isola. "On the 'Steerability' of Generative Adversarial Networks." ICLR 2020.
- Korbak, Perez, Buckley. "RL with KL Penalties is Better Viewed as Bayesian Inference." Findings of EMNLP 2022.
- Krause, Gotmare, McCann, Keskar, Joty, Socher, Rajani. "GeDi: Generative Discriminator Guided Sequence Generation." Findings of EMNLP 2021.
- Kynkäänniemi, Karras, Laine, Lehtinen, Aila. "Improved Precision and Recall Metric for Assessing Generative Models." NeurIPS 2019.
- Lester, Al-Rfou, Constant. "The Power of Scale for Parameter-Efficient Prompt Tuning." EMNLP 2021.
- Lewis, Perez, Piktus, Petroni, Karpukhin, Goyal, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.
- Li, Liang. "Prefix-Tuning: Optimizing Continuous Prompts for Generation." ACL-IJCNLP 2021.
- Naeem, Oh, Uh, Choi, Yoo. "Reliable Fidelity and Diversity Metrics for Generative Models." ICML 2020.
- Ouyang, Wu, Jiang, Almeida, Wainwright, et al. "Training Language Models to Follow Instructions with Human Feedback." NeurIPS 2022.
- Owen. *Monte Carlo Theory, Methods and Examples.* 2013.
- Peebles, Xie. "Scalable Diffusion Models with Transformers." ICCV 2023.
- Polanyi. *The Tacit Dimension.* Doubleday, 1966.
- Radford, Kim, Hallacy, Ramesh, Goh, Agarwal, et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.
- Rafailov, Sharma, Mitchell, Ermon, Manning, Finn. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023.
- Ruiz, Li, Jampani, Pritch, Rubinstein, Aberman. "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation." CVPR 2023.
- Sajjadi, Bachem, Lucic, Bousquet, Gelly. "Assessing Generative Models via Precision and Recall." NeurIPS 2018.
- Salimans, Goodfellow, Zaremba, Cheung, Radford, Chen. "Improved Techniques for Training GANs." NeurIPS 2016.
- Schmidt, Hoffmann, Wang, Borlido, Carriço, Cerqueira, Botti, Marques. "Machine-Learning-Assisted Determination of the Global Zero-Temperature Phase Diagram of Materials." *Advanced Materials* 35(22):2210788, 2023.
- Schuhmann, Beaumont, Vencu, Gordon, Wightman, et al. "LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models." NeurIPS 2022 Datasets and Benchmarks Track.
- Shen, Gu, Tang, Zhou. "Interpreting the Latent Space of GANs for Semantic Face Editing." CVPR 2020.
- Xie, Fu, Ganea, Barzilay, Jaakkola. "Crystal Diffusion Variational Autoencoder for Periodic Material Generation." ICLR 2022.
- Ye, Zhang, Liu, Han, Yang. "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models." arXiv:2308.06721, 2023.
- Zeni, Pinsler, Zügner, Fowler, Horton, Fu, et al. "A Generative Model for Inorganic Materials Design" (MatterGen). *Nature* 639:624–632, 2025.
- Ziegler, Stiennon, Wu, Brown, Radford, Amodei, Christiano, Irving. "Fine-Tuning Language Models from Human Preferences." arXiv:1909.08593, 2019.

## Appendix

### A. Proofs of the two reaches (one line each)

**The split (law of total variance).** For any $(\pi, L, \mathcal{J})$,
conditioning $L$ on its bin label $P$ gives
$\mathrm{Var}(L)=\mathbb{E}[\mathrm{Var}(L\mid P)]+\mathrm{Var}(\mathbb{E}[L\mid
P])=\sum_b w_b v_b+\sum_b w_b(g_b-\bar g)^2=T+E.$ ∎

**The reaches (Cauchy–Schwarz).** Reweighting a set of variance $\sigma^2$ by a
distribution $\mu$ (from its own proportions $\mathcal{J}$) shifts its mean by
$\Delta=\sum_b(\mu_b-\mathcal{J}_b)g_b=\langle \mu/\mathcal{J}-1,\ g-\bar g\rangle_{\mathcal{J}}$.
By Cauchy–Schwarz in the $\mathcal{J}$-inner product, $|\Delta|\le
\|\mu/\mathcal{J}-1\|_{\mathcal{J}}\,\|g-\bar g\|_{\mathcal{J}}=\sqrt{\chi^2(\mu\|\mathcal{J})}\,\sqrt{\sigma^2}$,
tight when $\mu-\mathcal{J}\propto(g-\bar g)$ (the min-$\chi^2$ recipe of §3.4).
Applied within a bin ($\sigma^2=T$, telling) or across bins ($\sigma^2=E$,
showing) gives the two reaches. ∎

### B. Carry-over diagnostics ($\delta,\rho$) and the data→model bridge

Carry-over is **not** knowable before the model exists: $\delta$ and $\rho$
compare the *trained model's* per-bin audit to the data's, so they need model
samples. What is pre-model is the *prediction* of carry-over from cheap pilot
signals (support overlap + a loss-sensitivity proxy) — which is what we
pre-registered (§4.2, Claim 1). The two checks:
- **Bin-mean drift** $\delta(\mu_P)=\sqrt{\mathbb{E}_{\mu_P}[(g_{\mathrm{G}}(P)-g_{\mathcal{J}}(P))^2/v_P]}$ — did the bin means move?
- **Within-bin spread ratio** $\rho(b)=v_{\mathrm{G}}(b)/v_{\mathcal{J}}(b)$ — did the within-bin variety carry?

A model can pass one and fail the other (right bins, wrong labels within them).
For a *known* training intervention the model approximates a reshaped $\mathcal{J}'$ in
closed form — MLE: $\mathcal{J}'=\mathcal{J}$; RLHF/KL-PPO: $\mathcal{J}'(D)\propto\mathcal{J}(D)e^{R(D)/\beta}$;
filtering on $S$: $\mathcal{J}'=\mathcal{J}\,\mathbf{1}[D\in S]/Z$; DPO: Boltzmann with implicit
reward — so recompute the per-bin $w_b,g_b,v_b$ on $\mathcal{J}'$ and every result
carries over at no new audit cost. When the intervention is aggressive enough to
leave $\mathcal{J}$'s support, audit the model's own samples instead.

### C. The lift — scorer-free bin selection

With no scorer for $L$, the recipe is the bin histogram of an exemplar set, and
each bin's **lift** $=\mu_P(b)/w_b$ estimates the fraction $q_b=s\cdot\mathrm{lift}_b$
of that bin clearing the target ($s$ = the exemplar set's overall hit rate).
A small bin that soaks up exemplars has large lift; a bin the exemplars ignore
has lift $\approx1$. The global $\chi^2=\sum_b w_b(\mathrm{lift}_b-1)^2$ is just the
$w$-weighted spread of the lifts (rank correlation 0.94 to realized rates, §4.2).

### D. The crystal model

Encoder–decoder transformer (6 encoder / 18 decoder layers, model dim 768,
**≈238 M** params; 16k SentencePiece vocab), trained by supervised next-token
cross-entropy on an Alexandria-derived set (**≈225k** structures, <20 atoms, low
energy-above-hull, LeMat-Bulk overlap excluded). A prompt is
`<elements> | <natoms> | <property-bin tags>`; the decoder emits a structure
encoding parsed to an ASE `Atoms`. Distribution-steering uses naked
(`elements | natoms |`) prompts. Stability = single-MACE energy vs an MP hull;
*Combined* = SUN + MSUN (stable/metastable ∧ novel ∧ unique); also BVS-GII,
SMACT/validity. The checkpoint (hash-pinned) and all scorers are released.

### E. Scope and the curvature table

The reaches bound the **mean shift by selection** — not literal extremes — and
hold where the model matches the data on the bins in play (off-support,
$\chi^2\to\infty$, flags failure); they are second-moment ceilings, so
heavy-tailed bin means make them loose. Set-level targets (coverage, diversity)
are a *different object* than a mean: the framework says **which** operation
reaches them (a mixture — showing) but not how large the metric gets.
**Curvature table:** convex/linear goals (mean, rate, precision, conjunction) →
*concentrate* on one bin; concave goals (coverage, diversity, spread, match) →
*spread* across bins; both are showing. For $n$ properties the reachable means
fill the convex hull of the bin points, and covering a Pareto front needs
specialist bins unless one bin's within-bin spread already reaches the corners.
(The variance-vs-range argument and the full pitfalls list are in the extended
version.)
