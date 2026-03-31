from pathlib import Path
from typing import List


def get_audio_files(folder: str) -> List[Path]:
    """Return all .wav files in the given folder."""
    folder_path = Path(folder)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    wav_files = sorted(folder_path.glob("*.wav"))
    return wav_files


if __name__ == "__main__":
    files = get_audio_files("data/library")
    print("Found WAV files:")
    for file in files:
        print(file.name)