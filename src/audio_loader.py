"""Loads audio files from a directory — pure backend, no UI."""
from pathlib import Path

_AUDIO_EXTENSIONS = {".wav", ".mp3"}


class AudioLoaderError(Exception):
    """Raised for recoverable load problems the UI can display."""


def load_wav_files(directory: str | Path) -> list[Path]:
    """Scan *directory* for WAV and MP3 files and return their paths.

    Performs a non-recursive, case-insensitive scan for .wav and .mp3 files.

    Args:
        directory: Path to the folder to scan. Accepts str or Path.

    Returns:
        Sorted list of Path objects for every readable audio file found.

    Raises:
        AudioLoaderError: if the folder doesn't exist, isn't a directory,
                          contains no audio files, or no files are readable.
    """
    folder = Path(directory)

    if not folder.exists():
        raise AudioLoaderError(f"Folder not found: {folder}")

    if not folder.is_dir():
        raise AudioLoaderError(f"Path is not a directory: {folder}")

    candidates = [
        p for p in folder.iterdir()
        if p.suffix.lower() in _AUDIO_EXTENSIONS and p.is_file()
    ]

    if not candidates:
        raise AudioLoaderError(f"No audio files (.wav, .mp3) found in: {folder}")

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
            f"Found {len(unreadable)} audio file(s) but none could be read: {names}"
        )

    if unreadable:
        import warnings
        skipped = ", ".join(p.name for p in unreadable)
        warnings.warn(
            f"Skipped {len(unreadable)} unreadable audio file(s): {skipped}",
            UserWarning,
            stacklevel=2,
        )

    return sorted(readable)
