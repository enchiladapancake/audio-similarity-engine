"""Reduces high-dimensional feature vectors to 2D using UMAP."""
import numpy as np


def reduce_to_2d(feature_matrix: np.ndarray) -> np.ndarray:
    """Return a (n_files × 2) array of 2D UMAP coordinates.

    Args:
        feature_matrix: 2-D array (n_files × n_features).

    Returns:
        2-D array of shape (n_files, 2) ready for plotting.
    """
    # TODO: fit UMAP, return embedding
    raise NotImplementedError
