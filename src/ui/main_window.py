"""Main application window.

Layout
------
- Top-left: app title + loaded directory path
- Top-left (below path): short description label
- Center-left: Load Folder, Sort & Filter, and Check & Score buttons
- Right: live UMAP cluster map (always visible)
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt

from src.audio_loader import load_wav_files, AudioLoaderError
from src.ui.sort_filter_page import SortFilterPage
from src.ui.check_score_page import CheckScorePage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Similarity Engine")
        self.resize(1200, 700)

        self._wav_files: list = []   # populated after folder load

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        # ── Top-level split: left panel | right vector diagram ──
        h_layout = QHBoxLayout(root)

        # ── Left panel ──────────────────────────────────────────
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.title_label = QLabel("Audio Similarity Engine")
        self.dir_label = QLabel("No directory loaded")
        self.dir_label.setWordWrap(True)
        self.file_count_label = QLabel("")
        self.desc_label = QLabel(
            "Load a folder of WAV files to explore their\n"
            "audio features and similarity relationships."
        )

        self.btn_load_folder = QPushButton("Load Folder…")
        self.btn_sort_filter = QPushButton("Sort & Filter")
        self.btn_check_score = QPushButton("Check & Score")

        self.btn_sort_filter.setEnabled(False)
        self.btn_check_score.setEnabled(False)

        self.btn_load_folder.clicked.connect(self._on_load_folder)
        self.btn_sort_filter.clicked.connect(self._open_sort_filter)
        self.btn_check_score.clicked.connect(self._open_check_score)

        left_layout.addWidget(self.title_label)
        left_layout.addWidget(self.dir_label)
        left_layout.addWidget(self.file_count_label)
        left_layout.addWidget(self.desc_label)
        left_layout.addSpacing(24)
        left_layout.addWidget(self.btn_load_folder)
        left_layout.addWidget(self.btn_sort_filter)
        left_layout.addWidget(self.btn_check_score)
        left_layout.addStretch()

        # ── Right panel: UMAP cluster map ───────────────────────
        self.vector_diagram = _VectorDiagram()

        h_layout.addWidget(left_panel, stretch=1)
        h_layout.addWidget(self.vector_diagram, stretch=2)

    # ── Folder loading ───────────────────────────────────────────

    def _on_load_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select WAV folder",
            "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if not folder:
            return  # user cancelled

        try:
            wav_files = load_wav_files(folder)
        except AudioLoaderError as exc:
            QMessageBox.warning(self, "Load Error", str(exc))
            return

        self._wav_files = wav_files
        self.dir_label.setText(folder)
        self.file_count_label.setText(f"{len(wav_files)} WAV file(s) loaded")
        self.btn_sort_filter.setEnabled(True)
        self.btn_check_score.setEnabled(True)

    # ── Navigation stubs ────────────────────────────────────────

    def _open_sort_filter(self):
        # TODO: navigate to SortFilterPage
        pass

    def _open_check_score(self):
        # TODO: navigate to CheckScorePage
        pass


class _VectorDiagram(QWidget):
    """Placeholder for the embedded matplotlib UMAP scatter plot."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        placeholder = QLabel("UMAP cluster map will appear here")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder)
        # TODO: replace placeholder with matplotlib FigureCanvasQTAgg
