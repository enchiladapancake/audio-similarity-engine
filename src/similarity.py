"""Cosine similarity scoring between a query file and a library of files."""
import numpy as np


def score_similarity(query_vector: np.ndarray, library_vectors: np.ndarray) -> np.ndarray:
    """Return cosine similarity scores of query against every library vector.

    Args:
        query_vector: 1-D feature vector for the selected file.
        library_vectors: 2-D array (n_files × n_features).

    Returns:
        1-D array of similarity scores in [−1, 1], one per library file.
    """
    # TODO: implement cosine similarity (sklearn or manual)
    raise NotImplementedError
