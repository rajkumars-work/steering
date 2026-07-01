"""
Composition Sampler Module

Samples realistic material compositions based on training dataset statistics.
Ensures generated structures match the chemical space of real materials.

Usage:
    from composition_sampler import sample_composition, sample_compositions

    # Sample a single ternary composition
    comp = sample_composition(n_elements=3)
    # Returns: [('O', 0.625), ('Li', 0.250), ('F', 0.125)]

    # Batch sample with constraints
    comps = sample_compositions(n_samples=10, include=['O'])
"""

import os
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from ase.io import read as ase_read
except ImportError:
    ase_read = None

# Import from voxel.py with fallback
try:
    from voxel import VALID_ELEMENTS, elem_to_idx, ELEMENTS, COMPOSITION_KEYS
except ImportError:
    warnings.warn("Could not import from voxel.py. Using fallback definitions.")
    # Fallback definitions for standalone use
    VALID_ELEMENTS = {
        'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Cs', 'Ba', 'Hf', 'Ta', 'W', 'Re', 'Os',
        'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi'
    }
    ELEMENTS = sorted(VALID_ELEMENTS)
    elem_to_idx = {e: i for i, e in enumerate(ELEMENTS)}
    COMPOSITION_KEYS = [
        "n_unique_elements",
        "top1_elem_idx", "top1_elem_frac",
        "top2_elem_idx", "top2_elem_frac",
        "top3_elem_idx", "top3_elem_frac",
        "top4_elem_idx", "top4_elem_frac",
        "top5_elem_idx", "top5_elem_frac",
    ]

# Default cache location
DEFAULT_CACHE_FILE = os.path.join(
    os.path.dirname(__file__),
    "composition_sampler_stats.pt"
)

# Default training data location
DEFAULT_DATA_DIR = "/data/assets/atlas/dielectrics/DATABASE/db-mp_updated/"


# ============================================================================
# Statistics Management
# ============================================================================

class CompositionStatistics:
    """Manages lazy loading and caching of composition statistics."""

    _instance = None
    _stats = None

    @classmethod
    def get_instance(cls, cache_file=None):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(cache_file)
        return cls._instance

    def __init__(self, cache_file=None):
        self.cache_file = cache_file or DEFAULT_CACHE_FILE
        self._load_stats()

    def _load_stats(self):
        """Load statistics from cache file."""
        if os.path.exists(self.cache_file):
            if torch is not None:
                self._stats = torch.load(self.cache_file, weights_only=False)
                print(f"Loaded composition statistics from {self.cache_file}")
            else:
                import pickle
                with open(self.cache_file, 'rb') as f:
                    self._stats = pickle.load(f)
                print(f"Loaded composition statistics from {self.cache_file}")
        else:
            warnings.warn(
                f"Statistics cache not found at {self.cache_file}. "
                "Using default uniform statistics. "
                "Run compute_statistics() for better results."
            )
            self._stats = self._get_default_stats()

    def get_stats(self):
        """Get statistics dictionary."""
        return self._stats

    def get_dataset_vectors(self, elements: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Get dataset compositions as a dense matrix.
        
        Args:
            elements: List of elements to include in vector. If None, uses all VALID_ELEMENTS.
            
        Returns:
            (vectors, elements_list)
            vectors: (N, n_elements) float array
            elements_list: List of element symbols corresponding to columns
        """
        if elements is None:
            elements = sorted(VALID_ELEMENTS)
        
        elem_to_idx = {e: i for i, e in enumerate(elements)}
        n_dims = len(elements)
        
        comps = self._stats.get("dataset_compositions", [])
        if not comps:
            warnings.warn("No dataset compositions found in stats.")
            return np.zeros((0, n_dims)), elements
            
        vectors = np.zeros((len(comps), n_dims))
        
        for i, comp in enumerate(comps):
            for elem, frac in comp.items():
                if elem in elem_to_idx:
                    vectors[i, elem_to_idx[elem]] = frac
                    
        return vectors, elements

    @staticmethod
    def _get_default_stats():
        """Return minimal statistics for fallback."""
        # Uniform distribution over valid elements
        n_elements = len(VALID_ELEMENTS)
        uniform_freq = 1.0 / n_elements

        return {
            "meta": {
                "dataset_path": "default",
                "n_samples_computed": 0,
                "version": "1.0.0-default"
            },
            "element_frequency": {elem: uniform_freq for elem in VALID_ELEMENTS},
            "element_weights": {elem: uniform_freq for elem in VALID_ELEMENTS},
            "element_cooccurrence": {},
            "n_elements_dist": {2: 0.1, 3: 0.6, 4: 0.3},  # Typical distribution
            "element_fraction_stats": {elem: {"mean": 0.5, "std": 0.2} for elem in VALID_ELEMENTS},
        }


def compute_statistics(data_dir: str, n_samples: Optional[int] = 2000,
                      output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute composition statistics from training dataset.

    Args:
        data_dir: Path to directory containing .extxyz files
        n_samples: Number of structures to sample (None = all)
        output_file: Path to save statistics (None = don't save)

    Returns:
        Dictionary containing statistics
    """
    if ase_read is None:
        raise ImportError("ASE is required for statistics computation. Install with: pip install ase")

    print(f"Computing composition statistics from {data_dir}...")

    # Discover all .extxyz files
    data_path = Path(data_dir)
    files = list(data_path.glob("*.extxyz"))
    n_total = len(files)
    print(f"Found {n_total} .extxyz files")

    if n_total == 0:
        raise ValueError(f"No .extxyz files found in {data_dir}")

    # Sample files
    if n_samples is not None and n_samples < n_total:
        files = random.sample(files, n_samples)
        print(f"Sampling {n_samples} files")
    else:
        n_samples = n_total

    # Initialize counters
    element_counts = Counter()  # How many structures contain each element
    element_atom_counts = Counter()  # Total atoms of each element across all structures
    element_pairs = Counter()  # Co-occurrence of element pairs
    n_elements_counts = Counter()  # Distribution of number of elements
    element_fractions = {elem: [] for elem in VALID_ELEMENTS}  # Fraction when element is present
    dataset_compositions = []  # List of {elem: fraction} for each structure

    # Process files
    n_processed = 0
    n_errors = 0

    for file_path in files:
        try:
            atoms = ase_read(str(file_path))
            symbols = atoms.get_chemical_symbols()

            # Skip if any invalid elements
            unique_elements = set(symbols)
            if not unique_elements.issubset(VALID_ELEMENTS):
                n_errors += 1
                continue

            # Count elements
            element_count_dict = Counter(symbols)
            n_unique = len(element_count_dict)
            n_total_atoms = len(symbols)

            # Update statistics
            n_elements_counts[n_unique] += 1

            for elem, count in element_count_dict.items():
                element_counts[elem] += 1
                element_atom_counts[elem] += count
                frac = count / n_total_atoms
                element_fractions[elem].append(frac)

            # Store composition for novelty checks
            comp_dict = {elem: count / n_total_atoms for elem, count in element_count_dict.items()}
            dataset_compositions.append(comp_dict)

            # Co-occurrence (pairs)
            elements_list = sorted(unique_elements)
            for i in range(len(elements_list)):
                for j in range(i + 1, len(elements_list)):
                    pair = tuple(sorted([elements_list[i], elements_list[j]]))
                    element_pairs[pair] += 1

            n_processed += 1
            if n_processed % 500 == 0:
                print(f"Processed {n_processed}/{n_samples} files...")

        except Exception as e:
            n_errors += 1
            if n_errors < 10:  # Only show first few errors
                print(f"Error reading {file_path}: {e}")

    print(f"Successfully processed {n_processed} files ({n_errors} errors)")

    if n_processed == 0:
        raise ValueError("No files could be processed successfully")

    # Compute derived statistics

    # Element frequencies (probability of appearing)
    element_frequency = {elem: count / n_processed for elem, count in element_counts.items()}
    # Fill in zeros for elements that never appeared
    for elem in VALID_ELEMENTS:
        if elem not in element_frequency:
            element_frequency[elem] = 0.0

    # Element weights (for weighted sampling)
    element_weights = element_frequency.copy()

    # N elements distribution
    n_elements_dist = {n: count / n_processed for n, count in n_elements_counts.items()}

    # Element fraction statistics
    element_fraction_stats = {}
    for elem, fracs in element_fractions.items():
        if fracs:
            element_fraction_stats[elem] = {
                "mean": float(np.mean(fracs)),
                "std": float(np.std(fracs)),
                "min": float(np.min(fracs)),
                "max": float(np.max(fracs)),
            }
        else:
            element_fraction_stats[elem] = {
                "mean": 0.5,
                "std": 0.2,
                "min": 0.0,
                "max": 1.0,
            }

    # Package statistics
    stats = {
        "meta": {
            "dataset_path": str(data_dir),
            "n_samples_computed": n_processed,
            "n_total_files": n_total,
            "version": "1.0.0"
        },
        "element_frequency": element_frequency,
        "element_weights": element_weights,
        "element_cooccurrence": dict(element_pairs),
        "dataset_compositions": dataset_compositions,
        "n_elements_dist": n_elements_dist,
        "element_fraction_stats": element_fraction_stats,
    }

    # Save if requested
    if output_file:
        output_path = Path(output_file)
        if torch is not None:
            torch.save(stats, output_path)
            print(f"Saved statistics to {output_path}")
        else:
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(stats, f)
            print(f"Saved statistics to {output_path}")

    # Print summary
    print("\n=== Statistics Summary ===")
    print(f"Number of elements distribution:")
    for n in sorted(n_elements_dist.keys()):
        print(f"  {n} elements: {n_elements_dist[n]*100:.1f}%")
    print(f"\nTop 10 most common elements:")
    for elem, freq in sorted(element_frequency.items(), key=lambda x: -x[1])[:10]:
        print(f"  {elem}: {freq*100:.1f}%")

    return stats


# ============================================================================
# Core Sampling Logic
# ============================================================================

class CompositionSampler:
    """Core sampling logic for realistic compositions."""

    def __init__(self, stats: Dict[str, Any]):
        self.stats = stats

    def sample_n_elements(self) -> int:
        """Sample number of elements from empirical distribution."""
        dist = self.stats["n_elements_dist"]
        elements_options = list(dist.keys())
        probabilities = [dist[n] for n in elements_options]

        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        return random.choices(elements_options, weights=probabilities, k=1)[0]

    def sample_elements(self, n_elements: int, include: Optional[List[str]] = None,
                       exclude: Optional[List[str]] = None) -> List[str]:
        """
        Sample elements using conditional probabilities based on co-occurrence.

        Args:
            n_elements: Number of elements to sample
            include: Elements that must be included
            exclude: Elements that must not be included

        Returns:
            List of element symbols
        """
        include = include or []
        exclude = exclude or []

        # Validate constraints
        if len(include) > n_elements:
            raise ValueError(
                f"Cannot satisfy constraint: {len(include)} required elements "
                f"but n_elements={n_elements}"
            )

        for elem in include:
            if elem not in VALID_ELEMENTS:
                raise ValueError(f"Invalid element in include: {elem}")

        for elem in exclude:
            if elem not in VALID_ELEMENTS:
                raise ValueError(f"Invalid element in exclude: {elem}")

        elements = list(include)

        # Sample remaining elements
        while len(elements) < n_elements:
            if len(elements) == 0:
                # First element: sample from frequency distribution
                probs = self._get_element_probabilities([], exclude)
            else:
                # Subsequent elements: use conditional probabilities
                probs = self._get_conditional_probabilities(elements, exclude)

            # Sample element
            elem_options = list(probs.keys())
            probabilities = list(probs.values())

            if not elem_options:
                warnings.warn("No valid elements available. Using fallback.")
                # Fallback: use any valid element not in exclude or already selected
                available = VALID_ELEMENTS - set(elements) - set(exclude)
                if available:
                    elem = random.choice(list(available))
                else:
                    break
            else:
                elem = random.choices(elem_options, weights=probabilities, k=1)[0]

            elements.append(elem)

        return elements

    def _get_element_probabilities(self, selected: List[str], exclude: List[str]) -> Dict[str, float]:
        """Get element probabilities (frequency-based)."""
        probs = {}
        for elem, weight in self.stats["element_weights"].items():
            if elem not in selected and elem not in exclude:
                probs[elem] = weight

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs

    def _get_conditional_probabilities(self, selected: List[str],
                                      exclude: List[str]) -> Dict[str, float]:
        """Get conditional probabilities based on co-occurrence."""
        cooccurrence = self.stats["element_cooccurrence"]
        element_freq = self.stats["element_frequency"]
        n_samples = self.stats["meta"]["n_samples_computed"]

        if n_samples == 0:
            # Fallback to frequency-based
            return self._get_element_probabilities(selected, exclude)

        probs = {}

        for elem in VALID_ELEMENTS:
            if elem in selected or elem in exclude:
                continue

            # Compute conditional probability using co-occurrence
            # P(elem | selected) = geometric mean of P(elem | s) for s in selected
            cond_probs = []
            for s in selected:
                pair = tuple(sorted([elem, s]))
                pair_count = cooccurrence.get(pair, 0)
                selected_count = element_freq.get(s, 0) * n_samples

                if selected_count > 0:
                    cond_prob = pair_count / selected_count
                else:
                    cond_prob = element_freq.get(elem, 0)

                cond_probs.append(cond_prob)

            if cond_probs:
                # Geometric mean
                prob = np.prod(cond_probs) ** (1.0 / len(cond_probs))
            else:
                # Fallback to frequency
                prob = element_freq.get(elem, 0)

            probs[elem] = prob

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs

    def sample_fractions(self, elements: List[str]) -> List[float]:
        """
        Sample composition fractions for given elements.

        Args:
            elements: List of element symbols

        Returns:
            List of fractions (same order as elements), sum to 1.0
        """
        n = len(elements)
        if n == 0:
            return []

        # Sample fractions using element-specific statistics
        fractions = []
        for elem in elements:
            stats = self.stats["element_fraction_stats"].get(elem, {"mean": 0.5, "std": 0.2})
            mean = stats["mean"]
            std = stats["std"]

            # Sample from truncated normal
            frac = np.random.normal(mean, std)
            frac = np.clip(frac, 0.01, 1.0)  # Keep positive and <= 1
            fractions.append(frac)

        # Normalize to sum to 1.0
        total = sum(fractions)
        fractions = [f / total for f in fractions]

        return fractions


# ============================================================================
# Public API
# ============================================================================

def sample_composition(n_elements: Optional[int] = None,
                      include: Optional[List[str]] = None,
                      exclude: Optional[List[str]] = None,
                      random_state: Optional[int] = None,
                      statistics: Optional[Dict] = None) -> List[Tuple[str, float]]:
    """
    Sample a realistic material composition.

    Args:
        n_elements: Number of unique elements. If None, samples from distribution (mode=3).
        include: Elements that must be present.
        exclude: Elements that cannot be present.
        random_state: Random seed for reproducibility.
        statistics: Preloaded statistics. If None, loads from cache.

    Returns:
        Composition as [(element, fraction), ...] sorted by decreasing fraction.

    Examples:
        >>> sample_composition()
        [('O', 0.625), ('Li', 0.250), ('F', 0.125)]

        >>> sample_composition(n_elements=2, include=['O'])
        [('O', 0.667), ('Li', 0.333)]
    """
    # Set random seed
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # Load statistics
    if statistics is None:
        stats_manager = CompositionStatistics.get_instance()
        statistics = stats_manager.get_stats()

    # Create sampler
    sampler = CompositionSampler(statistics)

    # Sample number of elements
    if n_elements is None:
        n_elements = sampler.sample_n_elements()

    # Sample elements
    elements = sampler.sample_elements(n_elements, include=include, exclude=exclude)

    # Sample fractions
    fractions = sampler.sample_fractions(elements)

    # Create composition (sorted by decreasing fraction)
    composition = list(zip(elements, fractions))
    composition.sort(key=lambda x: -x[1])

    return composition


def sample_compositions(n_samples: int = 10,
                       n_elements: Optional[int] = None,
                       include: Optional[List[str]] = None,
                       exclude: Optional[List[str]] = None,
                       random_state: Optional[int] = None,
                       statistics: Optional[Dict] = None) -> List[List[Tuple[str, float]]]:
    """
    Sample multiple compositions in batch.

    Args:
        n_samples: Number of compositions to generate.
        (other args same as sample_composition)

    Returns:
        List of compositions.
    """
    # Load statistics once
    if statistics is None:
        stats_manager = CompositionStatistics.get_instance()
        statistics = stats_manager.get_stats()

    compositions = []
    for i in range(n_samples):
        seed = None if random_state is None else random_state + i
        comp = sample_composition(
            n_elements=n_elements,
            include=include,
            exclude=exclude,
            random_state=seed,
            statistics=statistics
        )
        compositions.append(comp)

    return compositions


def validate_composition(composition: List[Tuple[str, float]],
                        check_realism: bool = False,
                        statistics: Optional[Dict] = None) -> Tuple[bool, List[str]]:
    """
    Validate a composition for correctness and optionally realism.

    Args:
        composition: List of (element, fraction) tuples
        check_realism: Whether to check against training data
        statistics: Statistics dict (required if check_realism=True)

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    if not composition:
        issues.append("Empty composition")
        return False, issues

    # Check elements
    for elem, frac in composition:
        if elem not in VALID_ELEMENTS:
            issues.append(f"Invalid element: {elem}")

    # Check fractions
    fractions = [frac for _, frac in composition]

    for i, (elem, frac) in enumerate(composition):
        if frac <= 0:
            issues.append(f"Non-positive fraction for {elem}: {frac}")
        if frac > 1:
            issues.append(f"Fraction > 1 for {elem}: {frac}")

    # Check sum
    total = sum(fractions)
    if abs(total - 1.0) > 1e-6:
        issues.append(f"Fractions sum to {total:.6f}, not 1.0")

    # Check number of elements
    n_elem = len(composition)
    if n_elem < 2 or n_elem > 6:
        issues.append(f"Unusual number of elements: {n_elem} (expected 2-6)")

    # Optional realism check
    if check_realism and statistics:
        # Check if this element combination appears in training data
        elements = {elem for elem, _ in composition}
        cooccurrence = statistics.get("element_cooccurrence", {})

        # For now, just warn if combination is very rare
        # (more sophisticated checks could be added)
        pass

    is_valid = len(issues) == 0
    return is_valid, issues


def composition_to_features(composition: List[Tuple[str, float]],
                           keys: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Convert composition to model feature dict.

    Args:
        composition: List of (element, fraction) tuples (sorted by decreasing fraction)
        keys: Feature keys to extract (default: COMPOSITION_KEYS)

    Returns:
        Feature dict compatible with voxel.extract_composition() output.
    """
    if keys is None:
        keys = COMPOSITION_KEYS

    n_unique = len(composition)
    features = {"n_unique_elements": float(n_unique)}

    # Extract top-k elements and fractions
    for i in range(5):  # top5
        if i < len(composition):
            elem, frac = composition[i]
            elem_idx = elem_to_idx.get(elem, 0)
        else:
            elem_idx = 0
            frac = 0.0

        features[f"top{i+1}_elem_idx"] = float(elem_idx)
        features[f"top{i+1}_elem_frac"] = float(frac)

    return features


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Composition Sampler")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Compute statistics
    compute_parser = subparsers.add_parser("compute", help="Compute statistics from dataset")
    compute_parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Dataset directory")
    compute_parser.add_argument("--n_samples", type=int, default=8000, help="Number of samples")
    compute_parser.add_argument("--output", default=DEFAULT_CACHE_FILE, help="Output file")

    # Sample compositions
    sample_parser = subparsers.add_parser("sample", help="Sample compositions")
    sample_parser.add_argument("--n_samples", type=int, default=10, help="Number of compositions")
    sample_parser.add_argument("--n_elements", type=int, default=None, help="Number of elements")
    sample_parser.add_argument("--include", nargs="+", help="Elements to include")
    sample_parser.add_argument("--exclude", nargs="+", help="Elements to exclude")
    sample_parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Show statistics
    stats_parser = subparsers.add_parser("stats", help="Show statistics in human-friendly format")
    stats_parser.add_argument("--cache_file", default=DEFAULT_CACHE_FILE, help="Statistics cache file")
    stats_parser.add_argument("--top_n", type=int, default=20, help="Number of top elements to show")

    args = parser.parse_args()

    if args.command == "compute":
        stats = compute_statistics(
            data_dir=args.data_dir,
            n_samples=args.n_samples,
            output_file=args.output
        )

    elif args.command == "sample":
        comps = sample_compositions(
            n_samples=args.n_samples,
            n_elements=args.n_elements,
            include=args.include,
            exclude=args.exclude,
            random_state=args.seed
        )

        print(f"\nSampled {len(comps)} compositions:")
        for i, comp in enumerate(comps):
            elem_str = ", ".join([f"{elem}({frac:.3f})" for elem, frac in comp])
            print(f"{i+1}. {elem_str}")

    elif args.command == "stats":
        # Load statistics
        stats_manager = CompositionStatistics(cache_file=args.cache_file)
        stats = stats_manager.get_stats()

        print("\n" + "=" * 80)
        print("COMPOSITION STATISTICS - HUMAN-FRIENDLY REPORT")
        print("=" * 80)

        # Meta information
        meta = stats.get("meta", {})
        print(f"\nDataset: {meta.get('dataset_path', 'unknown')}")
        print(f"Samples analyzed: {meta.get('n_samples_computed', 0):,}")
        print(f"Version: {meta.get('version', 'unknown')}")

        # Number of elements distribution
        print("\n" + "-" * 80)
        print("NUMBER OF ELEMENTS DISTRIBUTION")
        print("-" * 80)
        n_elements_dist = stats.get("n_elements_dist", {})
        total_prob = sum(n_elements_dist.values())
        for n in sorted(n_elements_dist.keys()):
            prob = n_elements_dist[n]
            percentage = prob * 100
            bar_length = int(percentage / 2)  # Scale to fit in terminal
            bar = "█" * bar_length
            print(f"{n} elements: {percentage:5.1f}% {bar}")

        # Element frequency (most common elements)
        print("\n" + "-" * 80)
        print(f"TOP {args.top_n} MOST COMMON ELEMENTS")
        print("-" * 80)
        print(f"{'Element':<10} {'Frequency':<12} {'Appears in':<20} {'Distribution'}")
        print("-" * 80)
        element_frequency = stats.get("element_frequency", {})
        sorted_elements = sorted(element_frequency.items(), key=lambda x: -x[1])
        n_samples = meta.get('n_samples_computed', 1)

        for elem, freq in sorted_elements[:args.top_n]:
            percentage = freq * 100
            count = int(freq * n_samples)
            bar_length = int(percentage / 2)  # Scale to fit
            bar = "█" * bar_length
            print(f"{elem:<10} {freq:>6.4f} ({percentage:>5.1f}%)  {count:>6,} samples    {bar}")

        # Element fraction statistics (for common elements)
        print("\n" + "-" * 80)
        print(f"ELEMENT FRACTION STATISTICS (Top {min(10, args.top_n)} elements)")
        print("-" * 80)
        print(f"{'Element':<10} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max'}")
        print("-" * 80)
        element_fraction_stats = stats.get("element_fraction_stats", {})

        for elem, freq in sorted_elements[:min(10, args.top_n)]:
            frac_stats = element_fraction_stats.get(elem, {})
            mean = frac_stats.get("mean", 0)
            std = frac_stats.get("std", 0)
            min_val = frac_stats.get("min", 0)
            max_val = frac_stats.get("max", 0)
            print(f"{elem:<10} {mean:>6.4f} ({mean*100:>5.1f}%)  "
                  f"{std:>6.4f}       {min_val:>6.4f}       {max_val:>6.4f}")

        # Co-occurrence statistics (most common pairs)
        print("\n" + "-" * 80)
        print("TOP 20 ELEMENT PAIR CO-OCCURRENCES")
        print("-" * 80)
        print(f"{'Pair':<15} {'Count':<10} {'Frequency':<12} {'Distribution'}")
        print("-" * 80)
        element_cooccurrence = stats.get("element_cooccurrence", {})
        sorted_pairs = sorted(element_cooccurrence.items(), key=lambda x: -x[1])

        for (elem1, elem2), count in sorted_pairs[:20]:
            pair_str = f"{elem1}-{elem2}"
            freq = count / n_samples if n_samples > 0 else 0
            percentage = freq * 100
            bar_length = int(percentage / 2)
            bar = "█" * bar_length
            print(f"{pair_str:<15} {count:>6,}     {freq:>6.4f} ({percentage:>5.1f}%)  {bar}")

        # Summary statistics
        print("\n" + "-" * 80)
        print("SUMMARY STATISTICS")
        print("-" * 80)
        n_elements_with_data = sum(1 for freq in element_frequency.values() if freq > 0)
        print(f"Total unique elements in dataset: {n_elements_with_data}")
        print(f"Total valid elements in vocabulary: {len(VALID_ELEMENTS)}")

        avg_n_elements = sum(n * prob for n, prob in n_elements_dist.items())
        print(f"Average number of elements per composition: {avg_n_elements:.2f}")

        most_common_elem = sorted_elements[0] if sorted_elements else ("N/A", 0)
        print(f"Most common element: {most_common_elem[0]} ({most_common_elem[1]*100:.1f}%)")

        print("\n" + "=" * 80)

    else:
        parser.print_help()
