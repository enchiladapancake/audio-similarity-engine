"""Cosine similarity scoring between a target vector and a candidate dict."""
from pathlib import Path

import numpy as np


def score_similarity(
    target: np.ndarray,
    candidates: dict[Path, np.ndarray],
) -> list[tuple[Path, float]]:
    """Score every candidate vector against *target* using cosine similarity.

    Args:
        target:     1-D feature vector for the query file, shape (F,).
        candidates: Dict mapping Path → 1-D feature vector, shape (F,).
                    May include the target file itself; it will score 1.0.

    Returns:
        List of (Path, score) tuples sorted descending by score.
        Scores are in [0.0, 1.0] — cosine similarity clipped to non-negative
        so anti-correlated audio reads as 0 rather than a negative number.
        Returns an empty list if *candidates* is empty.
    """
    if not candidates:
        return []

    paths = list(candidates.keys())
    matrix = np.stack([candidates[p] for p in paths])   # (N, F)

    # ── Normalise both sides to unit length ─────────────────────
    target_norm = np.linalg.norm(target)
    if target_norm == 0.0:
        # Zero vector — all scores undefined; return 0 for every candidate
        return [(p, 0.0) for p in paths]

    target_unit = target / target_norm

    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)   # (N, 1)
    # Avoid division by zero for any all-silent candidate files
    row_norms = np.where(row_norms == 0.0, 1.0, row_norms)
    matrix_unit = matrix / row_norms                             # (N, F)

    raw_scores = matrix_unit @ target_unit                       # (N,)

    # Clip to [0, 1]: negative cosine similarity has no meaningful "less
    # similar than opposite" interpretation for audio feature vectors.
    scores = np.clip(raw_scores, 0.0, 1.0).tolist()

    results = sorted(zip(paths, scores), key=lambda t: t[1], reverse=True)
    return results


# ── Standalone smoke-test ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s  %(message)s")

    TEST_FOLDER = r"C:\Users\mylog\OneDrive\Documents\audio\ASE_library"

    from pathlib import Path as _Path
    folder = _Path(TEST_FOLDER)

    if not folder.exists():
        print(f"Test folder not found: {folder}")
        print("Create the folder, drop some .wav files in it, and re-run.")
        sys.exit(1)

    from src.audio_loader import load_wav_files, AudioLoaderError
    from src.feature_extractor import extract_features_batch

    try:
        paths = load_wav_files(folder)
    except AudioLoaderError as e:
        print(f"Loader error: {e}")
        sys.exit(1)

    print(f"Extracting features for {len(paths)} file(s)…")
    features = extract_features_batch(paths)

    if len(features) < 2:
        print("Need at least 2 readable WAV files to test similarity.")
        sys.exit(1)

    # Use the first file as the query target
    target_path, target_vec = next(iter(features.items()))
    print(f"\nQuery file: {target_path.name}\n")

    ranked = score_similarity(target_vec, features)

    print(f"{'File':<40}  {'Score':>6}")
    print("-" * 50)
    for path, score in ranked:
        marker = " ← query" if path == target_path else ""
        print(f"{path.name:<40}  {score:.4f}{marker}")
