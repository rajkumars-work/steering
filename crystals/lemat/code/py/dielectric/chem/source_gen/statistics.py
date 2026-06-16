"""Extract and store distributional statistics from a source CSV.

Captures the joint structure of source prompts (elements, natoms, density,
property tags) in a lightweight, JSON-serializable format. The saved statistics
are sufficient to generate realistic synthetic sources without the original data.

Supports three levels of correlation:
  1. Per-stability-class distributions (elements, natoms, density conditional on stable/metastable/unstable)
  2. Element co-occurrence (pair frequencies for realistic element combinations)
  3. Element-natoms correlation (natoms distribution per n_elements bucket)
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

STABILITY_CLASSES = ("stable", "metastable", "unstable")


class SourceStatistics:
    """Distributional statistics of source prompts from a dataset CSV.

    Captures global distributions plus conditional distributions keyed by
    stability class, element pair co-occurrence, and element-natoms correlation.
    """

    def __init__(
        self,
        element_freq: dict[str, int],
        n_elements_dist: dict[int, int],
        natoms_hist: dict[int, int],
        element_density: dict[str, dict],  # {elem: {mean, std, count}}
        global_density: dict,  # {mean, std, min, max}
        tag_freq: dict[str, int],
        tag_combos: dict[str, int],  # "tag1 tag2" -> count
        n_tags_dist: dict[int, int],
        total_rows: int,
        # --- Extended statistics ---
        per_stability: dict | None = None,  # {class: {element_freq, n_elements_dist, natoms_hist, element_density, global_density}}
        element_pairs: dict | None = None,  # {"El1-El2": count} sorted pairs
        element_triples: dict | None = None,  # {"El1-El2-El3": count}
        natoms_by_nelements: dict | None = None,  # {n_elements: {natoms: count}}
        density_by_natoms: dict | None = None,  # {natoms: {mean, std}}
        metadata: dict | None = None,
    ):
        self.element_freq = element_freq
        self.n_elements_dist = {int(k): v for k, v in n_elements_dist.items()}
        self.natoms_hist = {int(k): v for k, v in natoms_hist.items()}
        self.element_density = element_density
        self.global_density = global_density
        self.tag_freq = tag_freq
        self.tag_combos = tag_combos
        self.n_tags_dist = {int(k): v for k, v in n_tags_dist.items()}
        self.total_rows = total_rows
        self.per_stability = per_stability or {}
        self.element_pairs = element_pairs or {}
        self.element_triples = element_triples or {}
        self.natoms_by_nelements = {int(k): {int(k2): v2 for k2, v2 in v.items()}
                                    for k, v in (natoms_by_nelements or {}).items()}
        # density_by_natoms: {natoms: {mean, std}} for joint sampling
        self.density_by_natoms = density_by_natoms or {}
        self.metadata = metadata or {}

    @classmethod
    def from_csv(cls, csv_path: str, split: str | None = None) -> "SourceStatistics":
        """Extract statistics from a dataset CSV.

        Args:
            csv_path: Path to CSV with 'source' column (pipe-delimited, 4 segments)
            split: If set, only include rows with this label ('train', 'eval', etc.)
                   If None, include all rows.
        """
        # Global accumulators
        element_freq: Counter = Counter()
        n_elements_dist: Counter = Counter()
        natoms_hist: Counter = Counter()
        density_per_element: dict[str, list[float]] = defaultdict(list)
        all_densities: list[float] = []
        tag_freq: Counter = Counter()
        tag_combos: Counter = Counter()
        n_tags_dist: Counter = Counter()
        total = 0

        # Per-stability accumulators
        stab_element_freq: dict[str, Counter] = {c: Counter() for c in STABILITY_CLASSES}
        stab_n_elements_dist: dict[str, Counter] = {c: Counter() for c in STABILITY_CLASSES}
        stab_natoms_hist: dict[str, Counter] = {c: Counter() for c in STABILITY_CLASSES}
        stab_density_per_element: dict[str, dict[str, list]] = {c: defaultdict(list) for c in STABILITY_CLASSES}
        stab_all_densities: dict[str, list] = {c: [] for c in STABILITY_CLASSES}
        stab_count: dict[str, int] = {c: 0 for c in STABILITY_CLASSES}

        # Element co-occurrence
        pair_freq: Counter = Counter()
        triple_freq: Counter = Counter()

        # Natoms by n_elements
        natoms_by_nelements: dict[int, Counter] = defaultdict(Counter)

        # Density by natoms (for joint natoms-density sampling)
        density_by_natoms_raw: dict[int, list] = defaultdict(list)

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split and row.get("label", "") != split:
                    continue

                src = row.get("source", "")
                parts = src.split("|")
                if len(parts) < 3:
                    continue

                total += 1

                # Segment 1: elements
                elements = parts[0].strip().split()
                for el in elements:
                    element_freq[el] += 1
                n_elements_dist[len(elements)] += 1

                # Element pairs and triples
                sorted_elems = sorted(elements)
                for i in range(len(sorted_elems)):
                    for j in range(i + 1, len(sorted_elems)):
                        pair_freq[f"{sorted_elems[i]}-{sorted_elems[j]}"] += 1
                        for k in range(j + 1, len(sorted_elems)):
                            triple_freq[f"{sorted_elems[i]}-{sorted_elems[j]}-{sorted_elems[k]}"] += 1

                # Segment 2: natoms
                natoms = None
                try:
                    natoms = int(parts[1].strip())
                    natoms_hist[natoms] += 1
                    natoms_by_nelements[len(elements)][natoms] += 1
                except ValueError:
                    pass

                # Segment 3: density
                density = None
                try:
                    density = float(parts[2].strip())
                    all_densities.append(density)
                    for el in elements:
                        density_per_element[el].append(density)
                except ValueError:
                    pass

                # Joint natoms-density
                if natoms is not None and density is not None:
                    density_by_natoms_raw[natoms].append(density)

                # Segment 4: tags
                tags_str = parts[3].strip() if len(parts) >= 4 else ""
                tags = tags_str.split() if tags_str else []
                for t in tags:
                    tag_freq[t] += 1
                n_tags_dist[len(tags)] += 1
                if tags:
                    combo_key = " ".join(sorted(tags))
                    tag_combos[combo_key] += 1

                # Per-stability accumulation
                stab_class = None
                for t in tags:
                    if t in STABILITY_CLASSES:
                        stab_class = t
                        break
                if stab_class:
                    stab_count[stab_class] += 1
                    for el in elements:
                        stab_element_freq[stab_class][el] += 1
                    stab_n_elements_dist[stab_class][len(elements)] += 1
                    if natoms is not None:
                        stab_natoms_hist[stab_class][natoms] += 1
                    if density is not None:
                        stab_all_densities[stab_class].append(density)
                        for el in elements:
                            stab_density_per_element[stab_class][el].append(density)

        # Compute per-element density stats (global)
        element_density = _compute_element_density(density_per_element)
        global_density = _compute_global_density(all_densities)

        # Compute per-stability stats
        per_stability = {}
        for c in STABILITY_CLASSES:
            if stab_count[c] == 0:
                continue
            per_stability[c] = {
                "count": stab_count[c],
                "element_freq": dict(stab_element_freq[c]),
                "n_elements_dist": {str(k): v for k, v in stab_n_elements_dist[c].items()},
                "natoms_hist": {str(k): v for k, v in stab_natoms_hist[c].items()},
                "element_density": _compute_element_density(stab_density_per_element[c]),
                "global_density": _compute_global_density(stab_all_densities[c]),
            }

        # Keep only top-N pairs/triples to keep JSON manageable
        top_pairs = dict(pair_freq.most_common(5000))
        top_triples = dict(triple_freq.most_common(5000))

        # Per-natoms density stats (for joint sampling)
        density_by_natoms = {}
        for natoms_val, densities in density_by_natoms_raw.items():
            if len(densities) >= 3:  # need enough data for stats
                n = len(densities)
                mean = sum(densities) / n
                variance = sum((d - mean) ** 2 for d in densities) / max(n - 1, 1)
                density_by_natoms[natoms_val] = {
                    "mean": round(mean, 4),
                    "std": round(math.sqrt(variance), 4),
                    "count": n,
                }

        metadata = {"csv_path": str(csv_path), "split": split}

        return cls(
            element_freq=dict(element_freq),
            n_elements_dist=dict(n_elements_dist),
            natoms_hist=dict(natoms_hist),
            element_density=element_density,
            global_density=global_density,
            tag_freq=dict(tag_freq),
            tag_combos=dict(tag_combos),
            n_tags_dist=dict(n_tags_dist),
            total_rows=total,
            per_stability=per_stability,
            element_pairs=top_pairs,
            element_triples=top_triples,
            natoms_by_nelements={str(k): dict(v) for k, v in natoms_by_nelements.items()},
            density_by_natoms={str(k): v for k, v in density_by_natoms.items()},
            metadata=metadata,
        )

    def to_json(self, path: str) -> None:
        """Save statistics to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "element_freq": self.element_freq,
            "n_elements_dist": {str(k): v for k, v in self.n_elements_dist.items()},
            "natoms_hist": {str(k): v for k, v in self.natoms_hist.items()},
            "element_density": self.element_density,
            "global_density": self.global_density,
            "tag_freq": self.tag_freq,
            "tag_combos": self.tag_combos,
            "n_tags_dist": {str(k): v for k, v in self.n_tags_dist.items()},
            "total_rows": self.total_rows,
            "per_stability": self.per_stability,
            "element_pairs": self.element_pairs,
            "element_triples": self.element_triples,
            "natoms_by_nelements": {str(k): {str(k2): v2 for k2, v2 in v.items()}
                                    for k, v in self.natoms_by_nelements.items()},
            "density_by_natoms": {str(k): v for k, v in self.density_by_natoms.items()},
            "metadata": self.metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "SourceStatistics":
        """Load statistics from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"SourceStatistics: {self.total_rows:,} rows",
            f"  Source: {self.metadata.get('csv_path', '?')} (split={self.metadata.get('split', 'all')})",
            f"  Elements: {len(self.element_freq)} unique",
            f"  Top 10: {', '.join(k for k, _ in Counter(self.element_freq).most_common(10))}",
            f"  N-elements: {dict(sorted(self.n_elements_dist.items()))}",
            f"  Natoms: min={min(self.natoms_hist)}, max={max(self.natoms_hist)}, "
            f"mode={max(self.natoms_hist, key=self.natoms_hist.get)}",
            f"  Density: mean={self.global_density['mean']}, std={self.global_density['std']}, "
            f"range=[{self.global_density['min']}, {self.global_density['max']}]",
            f"  Tags: {len(self.tag_freq)} unique",
            f"  Top tags: {', '.join(f'{k}({v})' for k, v in Counter(self.tag_freq).most_common(5))}",
            f"  Tag combos: {len(self.tag_combos)} unique",
            f"  Element pairs: {len(self.element_pairs)} (top 5000)",
            f"  Element triples: {len(self.element_triples)} (top 5000)",
        ]
        for c in STABILITY_CLASSES:
            if c in self.per_stability:
                s = self.per_stability[c]
                lines.append(f"  [{c}] {s['count']:,} rows, "
                             f"{len(s['element_freq'])} elements, "
                             f"density mean={s['global_density']['mean']}")
        return "\n".join(lines)


def _compute_element_density(density_per_element: dict) -> dict:
    """Compute per-element density {mean, std, count} from raw lists."""
    result = {}
    for el, densities in density_per_element.items():
        if isinstance(densities, list):
            n = len(densities)
            mean = sum(densities) / n
            variance = sum((d - mean) ** 2 for d in densities) / max(n - 1, 1)
            result[el] = {"mean": round(mean, 4), "std": round(math.sqrt(variance), 4), "count": n}
        else:
            result[el] = densities  # already computed (from JSON)
    return result


def _compute_global_density(all_densities: list) -> dict:
    """Compute global density stats from raw list."""
    if not all_densities:
        return {"mean": 6.0, "std": 3.0, "min": 0.3, "max": 22.0}
    n = len(all_densities)
    mean = sum(all_densities) / n
    variance = sum((d - mean) ** 2 for d in all_densities) / max(n - 1, 1)
    return {
        "mean": round(mean, 4),
        "std": round(math.sqrt(variance), 4),
        "min": round(min(all_densities), 1),
        "max": round(max(all_densities), 1),
    }
