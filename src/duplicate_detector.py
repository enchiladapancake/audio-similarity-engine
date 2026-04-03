"""Near-duplicate detection by pairwise cosine similarity with union-find grouping."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


def find_duplicates(
    features:  dict[Path, np.ndarray],
    threshold: float = 0.97,
) -> list[list[Path]]:
    """Group files whose pairwise cosine similarity meets or exceeds *threshold*.

    Args:
        features:  Dict mapping Path → 1-D feature vector (same shape as produced
                   by feature_extractor.extract_features_batch).
        threshold: Cosine similarity cut-off in [0.0, 1.0].  Pairs that score
                   at or above this value are considered near-duplicates.

    Returns:
        A list of groups.  Each group is a list of two or more Paths that are
        near-duplicates of each other (transitive closure via union-find).
        Files that have no near-duplicate partner are excluded entirely.
        Each Path appears in at most one group.
    """
    if len(features) < 2:
        return []

    paths  = list(features.keys())
    n      = len(paths)
    matrix = np.stack([features[p] for p in paths])          # (N, F)

    # Normalise rows to unit length (zero vectors become zero rows — safe)
    norms        = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms        = np.where(norms == 0.0, 1.0, norms)
    matrix_unit  = matrix / norms                             # (N, F)

    # Full pairwise cosine-similarity matrix
    sim = matrix_unit @ matrix_unit.T                         # (N, N)

    # ── Union-Find ───────────────────────────────────────────────────────────
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)

    # ── Collect groups ───────────────────────────────────────────────────────
    buckets: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        buckets[find(i)].append(i)

    return [
        [paths[i] for i in indices]
        for indices in buckets.values()
        if len(indices) >= 2
    ]
