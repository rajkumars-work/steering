"""Generate synthetic source prompts from saved statistics.

Produces realistic 4-segment source strings (elements | natoms | density | tags)
without needing the original training data — only a JSON statistics file.

Three enhanced generation modes (beyond baseline independent sampling):

  1. stability_conditioned: All parameters drawn from the distribution of
     structures with a specific stability class. If you ask for 'stable',
     you get elements/natoms/density characteristic of stable structures.

  2. cooccurrence: Element sets sampled from observed pair/triple frequencies
     instead of independently. Produces realistic element combinations.

  3. natoms_correlated: Natoms sampled conditioned on the number of elements,
     capturing the element-count → cell-size correlation.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

from chem.source_gen.statistics import SourceStatistics

STABILITY_TAGS = {"stable", "metastable", "unstable"}
BANDGAP_TAGS = {"metal", "narrow-gap", "small-gap", "semiconductor", "wide-gap", "very-wide-gap"}


class SourceGenerator:
    """Generate synthetic source prompts from saved statistics.

    Modes:
        - default: Independent sampling (baseline)
        - stability_conditioned: Draw from per-stability-class distributions
        - cooccurrence: Use element pair/triple frequencies
        - natoms_correlated: Condition natoms on n_elements
    """

    def __init__(self, stats: SourceStatistics, seed: int = 42):
        self.stats = stats
        self.rng = random.Random(seed)
        self._init_global_samplers()

    def _init_global_samplers(self):
        """Precompute weighted sampling arrays from global distributions."""
        s = self.stats
        self._elements = list(s.element_freq.keys())
        self._element_weights = [s.element_freq[e] for e in self._elements]

        self._n_elem_vals = list(s.n_elements_dist.keys())
        self._n_elem_weights = [s.n_elements_dist[n] for n in self._n_elem_vals]

        self._natoms_vals = list(s.natoms_hist.keys())
        self._natoms_weights = [s.natoms_hist[n] for n in self._natoms_vals]

        self._tag_combos = list(s.tag_combos.keys())
        self._tag_combo_weights = [s.tag_combos[c] for c in self._tag_combos]

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _weighted_sample_no_replace(self, pool, weights, n):
        """Sample n items from pool weighted without replacement."""
        pool = list(pool)
        weights = list(weights)
        chosen = []
        for _ in range(n):
            if not pool:
                break
            idx = self.rng.choices(range(len(pool)), weights, k=1)[0]
            chosen.append(pool[idx])
            pool.pop(idx)
            weights.pop(idx)
        return chosen

    def _sample_density(self, elements=None, natoms=None,
                        element_density=None, global_density=None):
        """Sample density conditioned on elements and/or natoms.

        Priority: elements > natoms > global fallback.
        When both are available, uses element-based conditioning (more precise).
        When only natoms is available (no-elements mode), uses the joint
        natoms-density distribution.
        """
        gd = global_density or self.stats.global_density

        if elements:
            # Condition on elements (original behavior)
            ed = element_density or self.stats.element_density
            means, variances = [], []
            for el in elements:
                if el in ed:
                    means.append(ed[el]["mean"])
                    variances.append(ed[el]["std"] ** 2)
                else:
                    means.append(gd["mean"])
                    variances.append(gd["std"] ** 2)
            mean = sum(means) / len(means)
            std = math.sqrt(sum(variances) / len(variances)) / max(math.sqrt(len(elements)), 1)
        elif natoms is not None and str(natoms) in self.stats.density_by_natoms:
            # Condition on natoms (joint distribution)
            dbn = self.stats.density_by_natoms[str(natoms)]
            mean = dbn["mean"]
            std = dbn["std"]
        else:
            # Global fallback
            mean = gd["mean"]
            std = gd["std"]

        density = self.rng.gauss(mean, std)
        density = max(gd["min"], min(density, gd["max"]))
        return round(density, 1)

    @staticmethod
    def _format_source(elements, natoms, density, tags):
        """Format source string. Always rounds density to 1 decimal place."""
        density_str = f"{density:.1f}"
        if elements:
            return f"{' '.join(sorted(elements))} | {natoms} | {density_str} | {tags}"
        else:
            return f"| {natoms} | {density_str} | {tags}"

    # ------------------------------------------------------------------
    # Mode 0: Baseline (independent sampling)
    # ------------------------------------------------------------------

    def generate_baseline(self, n: int, force_tag: str | None = None) -> list[str]:
        """Baseline: independent sampling of all parameters."""
        results = []
        for _ in range(n):
            n_elem = self.rng.choices(self._n_elem_vals, self._n_elem_weights, k=1)[0]
            n_elem = max(1, min(n_elem, len(self._elements)))
            elements = self._weighted_sample_no_replace(self._elements, self._element_weights, n_elem)
            natoms = self.rng.choices(self._natoms_vals, self._natoms_weights, k=1)[0]
            density = self._sample_density(elements)

            if force_tag:
                tags = force_tag
            elif self._tag_combos:
                combo = self.rng.choices(self._tag_combos, self._tag_combo_weights, k=1)[0]
                tags = " ".join(sorted(combo.split()))
            else:
                tags = ""

            results.append(self._format_source(elements, natoms, density, tags))
        return results

    # ------------------------------------------------------------------
    # Mode 1: Stability-conditioned
    # ------------------------------------------------------------------

    def generate_stability_conditioned(self, n: int, stability: str = "stable") -> list[str]:
        """All parameters drawn from the per-stability-class distribution.

        Elements, natoms, density are all from the subset of rows that had
        the specified stability tag. Produces compositions characteristic
        of stable/metastable/unstable structures.
        """
        ps = self.stats.per_stability.get(stability)
        if not ps:
            # Fallback to baseline with forced tag
            return self.generate_baseline(n, force_tag=stability)

        # Build per-stability samplers
        elem_list = list(ps["element_freq"].keys())
        elem_weights = [ps["element_freq"][e] for e in elem_list]

        ne_dist = {int(k): v for k, v in ps["n_elements_dist"].items()}
        ne_vals = list(ne_dist.keys())
        ne_weights = [ne_dist[k] for k in ne_vals]

        na_dist = {int(k): v for k, v in ps["natoms_hist"].items()}
        na_vals = list(na_dist.keys())
        na_weights = [na_dist[k] for k in na_vals]

        elem_density = ps["element_density"]
        global_density = ps["global_density"]

        results = []
        for _ in range(n):
            n_elem = self.rng.choices(ne_vals, ne_weights, k=1)[0]
            n_elem = max(1, min(n_elem, len(elem_list)))
            elements = self._weighted_sample_no_replace(elem_list, elem_weights, n_elem)
            natoms = self.rng.choices(na_vals, na_weights, k=1)[0]
            density = self._sample_density(elements=elements, element_density=elem_density, global_density=global_density)
            results.append(self._format_source(elements, natoms, density, stability))
        return results

    # ------------------------------------------------------------------
    # Mode 2: Element co-occurrence
    # ------------------------------------------------------------------

    def generate_cooccurrence(self, n: int, force_tag: str | None = None) -> list[str]:
        """Element sets sampled from observed pair/triple frequencies.

        For 2-element sets: sample directly from pair frequencies.
        For 3-element sets: sample from triple frequencies.
        For 4+ elements: sample a triple, then extend with weighted independent elements.
        """
        pairs = self.stats.element_pairs
        triples = self.stats.element_triples

        pair_keys = list(pairs.keys())
        pair_weights = [pairs[k] for k in pair_keys]
        triple_keys = list(triples.keys())
        triple_weights = [triples[k] for k in triple_keys]

        results = []
        for _ in range(n):
            n_elem = self.rng.choices(self._n_elem_vals, self._n_elem_weights, k=1)[0]
            n_elem = max(2, min(n_elem, len(self._elements)))

            if n_elem == 2 and pair_keys:
                pair = self.rng.choices(pair_keys, pair_weights, k=1)[0]
                elements = pair.split("-")
            elif n_elem == 3 and triple_keys:
                triple = self.rng.choices(triple_keys, triple_weights, k=1)[0]
                elements = triple.split("-")
            elif n_elem >= 3 and triple_keys:
                # Start from a triple, extend
                triple = self.rng.choices(triple_keys, triple_weights, k=1)[0]
                elements = set(triple.split("-"))
                remaining = n_elem - len(elements)
                if remaining > 0:
                    pool = [e for e in self._elements if e not in elements]
                    weights = [self.stats.element_freq.get(e, 1) for e in pool]
                    if pool:
                        extra = self._weighted_sample_no_replace(pool, weights, remaining)
                        elements.update(extra)
                elements = list(elements)
            else:
                elements = self._weighted_sample_no_replace(
                    self._elements, self._element_weights, n_elem)

            natoms = self.rng.choices(self._natoms_vals, self._natoms_weights, k=1)[0]
            density = self._sample_density(elements)

            if force_tag:
                tags = force_tag
            elif self._tag_combos:
                combo = self.rng.choices(self._tag_combos, self._tag_combo_weights, k=1)[0]
                tags = " ".join(sorted(combo.split()))
            else:
                tags = ""

            results.append(self._format_source(elements, natoms, density, tags))
        return results

    # ------------------------------------------------------------------
    # Mode 3: Element-natoms correlation
    # ------------------------------------------------------------------

    def generate_natoms_correlated(self, n: int, force_tag: str | None = None) -> list[str]:
        """Natoms sampled conditioned on the number of elements.

        Captures the correlation between composition complexity and cell size
        (e.g., binary compounds tend to have smaller cells than quaternary ones).
        """
        na_by_ne = self.stats.natoms_by_nelements

        results = []
        for _ in range(n):
            n_elem = self.rng.choices(self._n_elem_vals, self._n_elem_weights, k=1)[0]
            n_elem = max(1, min(n_elem, len(self._elements)))
            elements = self._weighted_sample_no_replace(
                self._elements, self._element_weights, n_elem)

            # Natoms conditioned on n_elements
            if n_elem in na_by_ne and na_by_ne[n_elem]:
                na_dist = na_by_ne[n_elem]
                na_vals = list(na_dist.keys())
                na_weights = [na_dist[k] for k in na_vals]
                natoms = self.rng.choices(na_vals, na_weights, k=1)[0]
            else:
                natoms = self.rng.choices(self._natoms_vals, self._natoms_weights, k=1)[0]

            density = self._sample_density(elements)

            if force_tag:
                tags = force_tag
            elif self._tag_combos:
                combo = self.rng.choices(self._tag_combos, self._tag_combo_weights, k=1)[0]
                tags = " ".join(sorted(combo.split()))
            else:
                tags = ""

            results.append(self._format_source(elements, natoms, density, tags))
        return results

    # ------------------------------------------------------------------
    # Mode 4: Joint natoms-density (no elements)
    # ------------------------------------------------------------------

    def generate_noelements(self, n: int, force_tag: str | None = None) -> list[str]:
        """Generate sources with natoms + density (no elements).

        Natoms sampled from empirical distribution, density sampled
        conditioned on natoms from the joint distribution. Elements
        are omitted — the model infers them from physical constraints.

        This mode produced the best synthetic MSUN (0.148) in ablation
        experiments because the model picks chemically reasonable
        compositions when given only physical constraints.
        """
        results = []
        for _ in range(n):
            natoms = self.rng.choices(self._natoms_vals, self._natoms_weights, k=1)[0]
            density = self._sample_density(natoms=natoms)

            if force_tag:
                tags = force_tag
            elif self._tag_combos:
                combo = self.rng.choices(self._tag_combos, self._tag_combo_weights, k=1)[0]
                tags = " ".join(sorted(combo.split()))
            else:
                tags = ""

            results.append(self._format_source(None, natoms, density, tags))
        return results

    # ------------------------------------------------------------------
    # Convenience: original API preserved
    # ------------------------------------------------------------------

    def generate(self, n: int) -> list[str]:
        """Generate n sources using baseline mode (backward compatible)."""
        return self.generate_baseline(n)

    def generate_one(self) -> str:
        """Generate a single source using baseline mode."""
        return self.generate_baseline(1)[0]

    def generate_source_file(self, path: str, n: int,
                             mode: str = "baseline",
                             force_tag: str | None = None,
                             stability: str = "stable") -> None:
        """Write n sources to a text file.

        Args:
            mode: 'baseline', 'stability_conditioned', 'cooccurrence',
                  'natoms_correlated', 'noelements'
            force_tag: Force this tag (e.g., 'stable') for non-conditioned modes
            stability: Stability class for 'stability_conditioned' mode
        """
        if mode == "stability_conditioned":
            sources = self.generate_stability_conditioned(n, stability=stability)
        elif mode == "cooccurrence":
            sources = self.generate_cooccurrence(n, force_tag=force_tag)
        elif mode == "natoms_correlated":
            sources = self.generate_natoms_correlated(n, force_tag=force_tag)
        elif mode == "noelements":
            sources = self.generate_noelements(n, force_tag=force_tag)
        else:
            sources = self.generate_baseline(n, force_tag=force_tag)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for src in sources:
                f.write(src + "\n")
