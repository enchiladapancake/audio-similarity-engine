"""Extracts audio features from WAV files using librosa.

Feature vector layout (17 values, all float32):
  [0:13]  MFCC coefficients 1-13   (mean across time)
  [13]    Spectral centroid         (mean across time)
  [14]    Spectral bandwidth        (mean across time)
  [15]    RMS energy                (mean across time)
  [16]    Zero-crossing rate        (mean across time)
"""
import logging
import warnings
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)

N_MFCC = 13
FEATURE_SIZE = 17  # 13 MFCCs + 4 extra features


def extract_features(wav_path: Path) -> np.ndarray:
    """Return a 1-D float32 feature vector of length 17 for one WAV file.

    Args:
        wav_path: Path to a readable .wav file.

    Returns:
        numpy array of shape (17,), dtype float32.

    Raises:
        Exception: re-raised after logging if librosa cannot load the file.
    """
    # librosa may emit UserWarnings for unusual sample rates etc. — capture them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)

    # ── MFCCs ────────────────────────────────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)   # (13, T)
    mfcc_means = np.mean(mfccs, axis=1)                          # (13,)

    # ── Spectral centroid ────────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)    # (1, T)
    centroid_mean = float(np.mean(centroid))

    # ── Spectral bandwidth ───────────────────────────────────────
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)  # (1, T)
    bandwidth_mean = float(np.mean(bandwidth))

    # ── RMS energy ───────────────────────────────────────────────
    rms = librosa.feature.rms(y=y)                              # (1, T)
    rms_mean = float(np.mean(rms))

    # ── Zero-crossing rate ───────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y)                 # (1, T)
    zcr_mean = float(np.mean(zcr))

    vector = np.array(
        [*mfcc_means, centroid_mean, bandwidth_mean, rms_mean, zcr_mean],
        dtype=np.float32,
    )
    assert vector.shape == (FEATURE_SIZE,), f"Unexpected shape: {vector.shape}"
    return vector


def extract_features_batch(
    wav_paths: list[Path],
) -> dict[Path, np.ndarray]:
    """Extract features for every path in *wav_paths*.

    Skips files that cannot be loaded — logs a warning and continues.

    Args:
        wav_paths: List of paths, typically from audio_loader.load_wav_files.

    Returns:
        Dict mapping each successfully processed Path to its (17,) feature
        vector.  Unreadable files are omitted (not raised).
    """
    results: dict[Path, np.ndarray] = {}

    for path in wav_paths:
        try:
            results[path] = extract_features(path)
            logger.debug("Extracted features: %s", path.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s — could not extract features: %s", path.name, exc)

    if not results:
        logger.warning("No features extracted — all files failed or list was empty.")

    return results


# ── Standalone smoke-test ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    # ── Change this path to a folder containing .wav files ──────
    TEST_FOLDER = Path(r"C:\Users\mylog\OneDrive\Documents\audio\ASE_library")
    # ────────────────────────────────────────────────────────────

    if not TEST_FOLDER.exists():
        print(f"Test folder not found: {TEST_FOLDER}")
        print("Create the folder, drop some .wav files in it, and re-run.")
        sys.exit(1)

    # Use audio_loader so the full pipeline is exercised
    from src.audio_loader import load_wav_files, AudioLoaderError

    try:
        paths = load_wav_files(TEST_FOLDER)
    except AudioLoaderError as e:
        print(f"Loader error: {e}")
        sys.exit(1)

    print(f"Found {len(paths)} WAV file(s). Extracting features…\n")
    features = extract_features_batch(paths)

    for path, vec in features.items():
        assert vec.shape == (FEATURE_SIZE,), f"Bad shape for {path.name}: {vec.shape}"
        print(f"  {path.name}")
        print(f"    MFCCs (1-13):  {vec[:13].round(2)}")
        print(f"    Centroid:      {vec[13]:.2f} Hz")
        print(f"    Bandwidth:     {vec[14]:.2f} Hz")
        print(f"    RMS energy:    {vec[15]:.6f}")
        print(f"    Zero-cross:    {vec[16]:.4f}")
        print()

    skipped = len(paths) - len(features)
    print(f"Done. {len(features)} succeeded, {skipped} skipped.")
