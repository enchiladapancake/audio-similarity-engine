import sys
import threading
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow


def _warmup_librosa() -> None:
    """Import librosa and run a tiny extraction to warm up Numba JIT."""
    try:
        import warnings
        import numpy as np
        import librosa
        dummy = np.zeros(22050, dtype=np.float32)  # 1 second of silence
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            librosa.feature.mfcc(y=dummy, sr=22050, n_mfcc=13)
    except Exception:
        pass


def main():
    threading.Thread(target=_warmup_librosa, daemon=True).start()
    app = QApplication(sys.argv)
    app.setApplicationName("Audio Similarity Engine")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
