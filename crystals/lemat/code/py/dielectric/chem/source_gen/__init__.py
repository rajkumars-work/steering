"""Synthetic source prompt generation for crystal structure models.

Extract statistics from a dataset CSV, serialize to JSON, then generate
realistic source prompts without needing the original data.

Usage:
    # Extract and save statistics
    from chem.source_gen import SourceStatistics, SourceGenerator

    stats = SourceStatistics.from_csv("data/d13_mixed_lemat_ehull.csv", split="train")
    stats.to_json("data/stats/inset_stats.json")

    # Generate synthetic sources (only needs the JSON)
    stats = SourceStatistics.from_json("data/stats/inset_stats.json")
    gen = SourceGenerator(stats, seed=42)
    sources = gen.generate(250)
    gen.generate_source_file("experiments/synth_sources.txt", 250)
"""

from chem.source_gen.statistics import SourceStatistics
from chem.source_gen.generator import SourceGenerator

__all__ = ["SourceStatistics", "SourceGenerator"]
