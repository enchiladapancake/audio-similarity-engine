"""Reduces high-dimensional feature vectors to 2D for visualisation.

Strategy:
  ≥ 6 files  →  UMAP  (n_neighbors=5, min_dist=0.3, random_state=42)
  < 6 files  →  PCA   (sklearn, n_components=2)

Both paths return the same dict[Path, tuple[float, float]] contract so
the caller never needs to know which reducer was used.
"""
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_UMAP_MIN_FILES = 6
_UMAP_PARAMS = dict(n_components=2, n_neighbors=5, min_dist=0.3, random_state=42)


def reduce(features: dict[Path, np.ndarray]) -> dict[Path, tuple[float, float]]:
    """Project feature vectors into 2D coordinates.

    Args:
        features: Dict mapping Path → 1-D feature vector (all same length).

    Returns:
        Dict mapping each Path to an (x, y) float tuple.
        Returns an empty dict if *features* is empty.

    Raises:
        ValueError: if feature vectors have inconsistent lengths.
    """
    if not features:
        return {}

    paths = list(features.keys())
    matrix = np.stack([features[p] for p in paths]).astype(np.float32)  # (N, F)

    n_files = len(paths)

    if n_files < 2:
        # Single file: place it at the origin
        logger.warning("Only 1 file — returning origin coordinates.")
        return {paths[0]: (0.0, 0.0)}

    if n_files >= _UMAP_MIN_FILES:
        coords = _reduce_umap(matrix)
        logger.debug("Used UMAP for %d files.", n_files)
    else:
        coords = _reduce_pca(matrix)
        logger.debug(
            "Used PCA fallback for %d files (< %d required for UMAP).",
            n_files,
            _UMAP_MIN_FILES,
        )

    return {path: (float(coords[i, 0]), float(coords[i, 1])) for i, path in enumerate(paths)}


# ── Private reducers ─────────────────────────────────────────────────────────

def _reduce_umap(matrix: np.ndarray) -> np.ndarray:
    """Return (N, 2) embedding via UMAP."""
    import umap  # imported here so missing install gives a clear error at call-time

    reducer = umap.UMAP(**_UMAP_PARAMS)
    return reducer.fit_transform(matrix)


def _reduce_pca(matrix: np.ndarray) -> np.ndarray:
    """Return (N, 2) embedding via PCA."""
    from sklearn.decomposition import PCA

    n_components = min(2, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    result = pca.fit_transform(matrix)

    # If only 1 component was possible, pad with zeros
    if result.shape[1] < 2:
        result = np.hstack([result, np.zeros((result.shape[0], 1))])

    return result
