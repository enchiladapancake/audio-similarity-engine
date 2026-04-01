"""Loads WAV files from a directory — pure backend, no UI."""
from pathlib import Path


class AudioLoaderError(Exception):
    """Raised for recoverable load problems the UI can display."""


def load_wav_files(directory: str | Path) -> list[Path]:
    """Scan *directory* for WAV files and return their paths.

    Performs a non-recursive, case-insensitive glob for *.wav files.

    Args:
        directory: Path to the folder to scan. Accepts str or Path.

    Returns:
        Sorted list of Path objects for every readable .wav file found.

    Raises:
        AudioLoaderError: if the folder doesn't exist, isn't a directory,
                          contains no WAV files, or no files are readable.
    """
    folder = Path(directory)

    if not folder.exists():
        raise AudioLoaderError(f"Folder not found: {folder}")

    if not folder.is_dir():
        raise AudioLoaderError(f"Path is not a directory: {folder}")

    # Case-insensitive glob: collect both *.wav and *.WAV (and mixed case)
    candidates = [
        p for p in folder.iterdir()
        if p.suffix.lower() == ".wav" and p.is_file()
    ]

    if not candidates:
        raise AudioLoaderError(f"No WAV files found in: {folder}")

    readable: list[Path] = []
    unreadable: list[Path] = []

    for path in candidates:
        try:
            # Cheapest check: open for reading without loading audio data
            with path.open("rb"):
                pass
            readable.append(path)
        except OSError:
            unreadable.append(path)

    if not readable:
        names = ", ".join(p.name for p in unreadable)
        raise AudioLoaderError(
            f"Found {len(unreadable)} WAV file(s) but none could be read: {names}"
        )

    if unreadable:
        # Partial success — caller can log or surface the skipped files
        import warnings
        skipped = ", ".join(p.name for p in unreadable)
        warnings.warn(
            f"Skipped {len(unreadable)} unreadable WAV file(s): {skipped}",
            UserWarning,
            stacklevel=2,
        )

    return sorted(readable)
