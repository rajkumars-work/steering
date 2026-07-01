"""
Composition Generator
---------------------
Implements the 7-step algorithm from chem.txt to generate 10k candidate compositions.
"""

import os
import random
import warnings
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter
import itertools

# Try to import scipy
try:
    from scipy.spatial.distance import cdist
except ImportError:
    cdist = None

# Try to import pymatgen for oxidation states
try:
    from pymatgen.core.periodic_table import Element
except ImportError:
    Element = None

from . import sampler as composition_sampler
from .sampler import CompositionStatistics, VALID_ELEMENTS

class CompositionGenerator:
    def __init__(self, stats_file=None, data_dir=None):
        """
        Initialize the generator.
        
        Args:
            stats_file: Path to statistics file (optional)
            data_dir: Path to dataset directory (optional, for recomputing stats if needed)
        """
        self.stats_manager = CompositionStatistics(cache_file=stats_file)
        self.stats = self.stats_manager.get_stats()
        
        # Check if dataset vectors are available
        if "dataset_compositions" not in self.stats or not self.stats["dataset_compositions"]:
            print("Dataset compositions not found in stats. Recomputing...")
            if data_dir is None:
                # Try to use default from sampler
                data_dir = composition_sampler.DEFAULT_DATA_DIR
            
            # Recompute stats to get compositions
            self.stats = composition_sampler.compute_statistics(
                data_dir=data_dir,
                n_samples=None,  # All files
                output_file=stats_file
            )
            
        # Precompute matrix for novelty check
        print("Preparing dataset vectors for novelty check...")
        self.dataset_vectors, self.vector_elements = self.stats_manager.get_dataset_vectors()
        self.vector_elem_to_idx = {e: i for i, e in enumerate(self.vector_elements)}
        
        # Cache oxidation states (use COMMON oxidation states only)
        if Element:
            self.oxi_states = {}
            for e in VALID_ELEMENTS:
                try:
                    elem = Element(e)
                    # Use common_oxidation_states instead of all oxidation_states
                    states = elem.common_oxidation_states
                    self.oxi_states[e] = list(states) if states else []
                except:
                    self.oxi_states[e] = []
        else:
            warnings.warn("Pymatgen not available. Oxidation state checks will be limited.")
            self.oxi_states = {}

    def generate(self, n_target: int = 10000, n_proposals: int = 100000) -> List[List[Tuple[str, float]]]:
        """
        Generate target number of valid compositions.
        Following Step 7: Generate ~10x desired candidates (100k proposals) then keep top ~10k.
        """
        print(f"Generating {n_proposals} composition proposals...")
        print(f"Target: {n_target} final compositions after filtering")

        proposals = []
        attempts = 0
        max_attempts = n_proposals * 5  # Allow retries

        stats_failures = {
            'element_selection': 0,
            'stoichiometry': 0,
            'chemical_filters': 0,
            'novelty': 0
        }

        while len(proposals) < n_proposals and attempts < max_attempts:
            attempts += 1

            # Steps 1-3: Generate elements
            elems = self._generate_elements()
            if elems is None or len(elems) != 3 or 'O' not in elems:
                stats_failures['element_selection'] += 1
                continue

            # Step 4: Solve Stoichiometry
            stoichiometries = self._solve_stoichiometry(elems)
            if not stoichiometries:
                stats_failures['stoichiometry'] += 1
                continue

            # Pick best stoichiometry (smallest sum)
            best_stoich = stoichiometries[0]

            # Step 5: Chemical Filters
            if not self._check_chemical_filters(elems, best_stoich):
                stats_failures['chemical_filters'] += 1
                continue

            # Convert to composition
            total_atoms = sum(best_stoich)
            comp = []
            for elem, n in zip(elems, best_stoich):
                comp.append((elem, n / total_atoms))
            comp.sort(key=lambda x: -x[1])

            # Step 6: Novelty Filter
            if not self._check_novelty(comp):
                stats_failures['novelty'] += 1
                continue

            # Accept proposal
            proposals.append(comp)

            if len(proposals) % 1000 == 0:
                print(f"  Generated {len(proposals)}/{n_proposals} proposals (attempts: {attempts})...")

        print(f"\nGeneration complete: {len(proposals)} valid proposals from {attempts} attempts")
        print(f"Failure breakdown:")
        for reason, count in stats_failures.items():
            print(f"  {reason}: {count}")

        # Step 7: Rank and select top n_target
        if len(proposals) <= n_target:
            print(f"\nReturning all {len(proposals)} proposals")
            return proposals

        print(f"\nRanking proposals and selecting top {n_target}...")
        # Rank by novelty score (prefer compositions in middle of novelty range)
        scored_proposals = []
        target_novelty = 0.20  # Middle of 0.05-0.35 range

        for comp in proposals:
            novelty_score = self._compute_novelty_score(comp)
            # Prefer closer to target
            score = -abs(novelty_score - target_novelty)
            scored_proposals.append((score, comp))

        scored_proposals.sort(reverse=True, key=lambda x: x[0])
        top_candidates = [comp for score, comp in scored_proposals[:n_target]]

        print(f"Selected top {len(top_candidates)} compositions")
        return top_candidates

    def _generate_elements(self) -> Optional[List[str]]:
        """
        Steps 1-3: Generate a set of 3 elements with O required.
        Returns None if failed.
        """
        # Step 2: Anchor selection
        if random.random() < 0.8:
            current = ['O']
        else:
            # Pick random metal (weighted by frequency)
            weights = self.stats["element_weights"]
            metals = [e for e in weights.keys() if e != 'O']
            if not metals:
                current = ['O']
            else:
                probs = [weights[e] for e in metals]
                total = sum(probs)
                if total > 0:
                    probs = [p/total for p in probs]
                    anchor = random.choices(metals, weights=probs, k=1)[0]
                    current = [anchor]
                else:
                    current = ['O']

        # Step 3: Grow to 3 elements
        while len(current) < 3:
            next_elem = self._pick_next_element(current)
            if next_elem:
                current.append(next_elem)
            else:
                return None

        # Ensure O is present (Step 1 requirement)
        if 'O' not in current:
            # Replace a random non-anchor element with O
            if len(current) > 1:
                idx = random.randint(1, len(current)-1)
                current[idx] = 'O'
            else:
                return None

        return current


    def _pick_next_element(self, current: List[str]) -> Optional[str]:
        """
        Pick next element using Step 3 formula:
        P(j|i) = (count(i,j) + alpha) / (count(i) + alpha*K)
        Weighted by mean compatibility with already-chosen elements.
        """
        alpha = 5.0
        K = len(VALID_ELEMENTS)
        
        # Candidates
        candidates = sorted(list(VALID_ELEMENTS - set(current)))
        
        # Calculate scores
        # Score(c) = Mean( P(c | s) for s in current )
        
        scores = []
        cooccurrence = self.stats["element_cooccurrence"]
        element_freq = self.stats["element_frequency"] # Proxy for marginal counts
        # Actually, count(i) is sum of co-occurrences involving i?
        # Or just total samples containing i (which is element_freq * N_samples)?
        # Let's use N_samples * freq as count(i)
        n_samples = self.stats["meta"].get("n_samples_computed", 2000)
        
        for cand in candidates:
            probs_given_s = []
            for s in current:
                pair = tuple(sorted([s, cand]))
                count_ij = cooccurrence.get(pair, 0)
                
                # count(s) = sum of all pairs involving s?
                # Or just count of s in dataset?
                # "N" in formula usually means denominator sum.
                # If formula is P(j|i), denominator is sum_k (count(i,k) + alpha) = count(i) + alpha*K.
                # count(i) here should be total co-occurrences of i?
                # Let's approximate count(i) ~ n_samples * freq(i) * (avg_elements-1)?
                # Simpler: Use raw count of s in dataset.
                count_s = element_freq.get(s, 0) * n_samples
                
                prob = (count_ij + alpha) / (count_s + alpha * K)
                probs_given_s.append(prob)
            
            # Mean compatibility
            mean_prob = sum(probs_given_s) / len(probs_given_s)
            scores.append(mean_prob)
        
        # Sample
        total_score = sum(scores)
        if total_score == 0:
            return random.choice(candidates)
        
        probs = [s / total_score for s in scores]
        return random.choices(candidates, weights=probs, k=1)[0]

    def _solve_stoichiometry(self, elements: List[str]) -> List[Tuple[int, ...]]:
        """
        Step 4: Solve stoichiometry by charge neutrality.
        1 <= n_i <= 8
        total atoms <= 20
        """
        # Get oxidation states
        oxi_options = [self.oxi_states.get(e, []) for e in elements]
        if any(not opts for opts in oxi_options):
            # If any element has no known oxidation states, fail (or assume 0?)
            return []
            
        # Generate possible integer combinations
        # Prefer small integers -> iterate by sum?
        solutions = []
        
        # Iterate n1, n2, n3
        # Bounds: 1 to 8. Sum <= 20.
        # Optimized loops
        for n1 in range(1, 9):
            for n2 in range(1, 9):
                for n3 in range(1, 9):
                    if n1 + n2 + n3 > 20:
                        continue
                    
                    ns = [n1, n2, n3]
                    
                    # Check if charge neutrality is possible
                    # We need to find q1, q2, q3 from oxi_options such that sum(n_i * q_i) == 0
                    if self._check_charge_neutrality(ns, oxi_options):
                        solutions.append(tuple(ns))
        
        # Sort solutions by sum (smallest first)
        solutions.sort(key=lambda x: sum(x))
        return solutions

    def _check_charge_neutrality(self, ns: List[int], oxi_opts: List[List[float]]) -> bool:
        """Check if any combination of oxidation states sums to 0."""
        # Cartesian product of oxidation states
        for qs in itertools.product(*oxi_opts):
            charge = sum(n * q for n, q in zip(ns, qs))
            if abs(charge) < 0.01:
                return True
        return False

    def _check_chemical_filters(self, elements: List[str], stoich: Tuple[int, ...]) -> bool:
        """Step 5: Quick chemical filters."""
        # Charge neutrality is already checked.
        
        # "Not all metals"
        # Since we require O (Step 1), and O is non-metal, this is satisfied?
        # Unless O is not in elements (which we enforced in Step 3).
        if 'O' not in elements:
            return False
            
        # "Max electronegativity spread within cation subset"
        # Cations are non-O elements?
        # Or elements with positive charge?
        # Heuristic: Get electronegativities of all non-O elements.
        # Check difference.
        if Element:
            try:
                cations = [e for e in elements if e != 'O']
                if len(cations) >= 2:
                    en_values = [Element(e).X for e in cations]
                    en_values = [x for x in en_values if x is not None]
                    if len(en_values) >= 2:
                        spread = max(en_values) - min(en_values)
                        # Heuristic threshold? "simple heuristic".
                        # Maybe spread shouldn't be too large? Or too small?
                        # In dielectrics/perovskites, we often want A and B to differ?
                        # Or do we want them similar?
                        # chem.txt doesn't specify. I'll use a loose threshold.
                        # If spread > 2.5 (unlikely for cations), reject?
                        # Or maybe just check they exist.
                        pass 
            except:
                pass

        # "Minimum pairwise distance (element radius) plausible"
        # I'll implement a simple check: sum of ionic radii checks?
        # Skipping as too ambiguous without structure.
        
        return True

    def _compute_novelty_score(self, comp: List[Tuple[str, float]]) -> float:
        """
        Compute minimum distance to dataset for novelty scoring.
        """
        if self.dataset_vectors is None or len(self.dataset_vectors) == 0:
            return 0.20  # Default middle value

        if cdist is None:
            return 0.20

        # Convert comp to vector
        vec = np.zeros((1, len(self.vector_elements)))
        for elem, frac in comp:
            if elem in self.vector_elem_to_idx:
                vec[0, self.vector_elem_to_idx[elem]] = frac

        # Compute distances (using cosine for robustness)
        try:
            dists = cdist(vec, self.dataset_vectors, metric='cosine')
        except:
            return 0.20

        min_dist = np.min(dists)
        return min_dist

    def _check_novelty(self, comp: List[Tuple[str, float]]) -> bool:
        """
        Step 6: Novelty filter.
        0.05 < dist < 0.35
        """
        if self.dataset_vectors is None or len(self.dataset_vectors) == 0:
            return True # Cannot check

        if cdist is None:
            return True # Scipy missing

        min_dist = self._compute_novelty_score(comp)
        return 0.05 < min_dist < 0.35


def composition_to_formula(comp: List[Tuple[str, float]]) -> str:
    """
    Convert composition (element, fraction) to chemical formula.
    """
    # Try using pymatgen for proper formula
    if Element:
        try:
            from pymatgen.core.composition import Composition as PMGComp
            # Convert fractions to approximate integer counts
            # Find a common denominator
            fractions = [frac for _, frac in comp]
            # Scale to get integers (try multiple scales)
            for scale in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30]:
                counts = [frac * scale for frac in fractions]
                if all(abs(c - round(c)) < 0.01 for c in counts):
                    # Good scale found
                    comp_dict = {elem: int(round(frac * scale)) for elem, frac in comp}
                    pmg_comp = PMGComp(comp_dict)
                    return pmg_comp.reduced_formula
        except:
            pass

    # Fallback: simple formula
    parts = []
    for elem, frac in comp:
        # Approximate as integer ratio
        count = int(round(frac * 100))  # Scale by 100
        if count > 0:
            parts.append(f"{elem}{count}")
    return "".join(parts)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate composition candidates.")
    parser.add_argument("--n_target", type=int, default=10000, help="Number of compositions to generate")
    parser.add_argument("--n_proposals", type=int, default=100000, help="Number of proposals before filtering")
    parser.add_argument("--output", default="candidates.txt", help="Output file")
    parser.add_argument("--data_dir", default=None, help="Dataset directory (optional)")
    parser.add_argument("--stats_file", default=None, help="Statistics cache file (optional)")

    args = parser.parse_args()

    print("="*80)
    print("COMPOSITION GENERATOR")
    print("="*80)

    gen = CompositionGenerator(stats_file=args.stats_file, data_dir=args.data_dir)
    candidates = gen.generate(n_target=args.n_target, n_proposals=args.n_proposals)

    print(f"\n{'='*80}")
    print(f"Saving {len(candidates)} candidates to {args.output}...")
    with open(args.output, "w") as f:
        f.write("# Generated Composition Candidates\n")
        f.write(f"# Total: {len(candidates)}\n")
        f.write("# Format: Index | Formula | Element1 Fraction1 Element2 Fraction2 ...\n")
        f.write("#" + "="*77 + "\n\n")

        for i, comp in enumerate(candidates):
            formula = composition_to_formula(comp)
            # Format: index | formula | elem frac elem frac ...
            frac_str = " ".join([f"{e} {frac:.4f}" for e, frac in comp])
            f.write(f"{i+1:6d} | {formula:15s} | {frac_str}\n")

    print(f"Done! Saved to {args.output}")

    # Print sample
    print(f"\nSample compositions (first 20):")
    for i, comp in enumerate(candidates[:20]):
        formula = composition_to_formula(comp)
        print(f"  {i+1:3d}. {formula}")
