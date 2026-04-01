"""Extracts audio features (MFCCs, etc.) from WAV files using librosa."""
from pathlib import Path
import numpy as np


def extract_features(wav_path: Path) -> np.ndarray:
    """Return a 1-D feature vector (mean MFCCs) for a single WAV file."""
    # TODO: load audio with librosa, compute MFCCs, return mean across frames
    raise NotImplementedError


def extract_features_batch(wav_paths: list[Path]) -> np.ndarray:
    """Return a 2-D array (n_files × n_features) for a list of WAV files."""
    # TODO: call extract_features for each path, stack results
    raise NotImplementedError
