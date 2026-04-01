"""Main application window.

Layout
------
- Top-left: app title + loaded directory path
- Top-left (below path): short description label
- Center-left: Sort & Filter and Check & Score buttons
- Right: live UMAP cluster map (always visible)
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QStackedWidget,
)
from PyQt6.QtCore import Qt

from src.ui.sort_filter_page import SortFilterPage
from src.ui.check_score_page import CheckScorePage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Similarity Engine")
        self.resize(1200, 700)

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
        self.desc_label = QLabel(
            "Load a folder of WAV files to explore their\n"
            "audio features and similarity relationships."
        )

        self.btn_sort_filter = QPushButton("Sort & Filter")
        self.btn_check_score = QPushButton("Check & Score")

        # TODO: wire navigation to stacked widget / separate pages
        self.btn_sort_filter.clicked.connect(self._open_sort_filter)
        self.btn_check_score.clicked.connect(self._open_check_score)

        left_layout.addWidget(self.title_label)
        left_layout.addWidget(self.dir_label)
        left_layout.addWidget(self.desc_label)
        left_layout.addSpacing(24)
        left_layout.addWidget(self.btn_sort_filter)
        left_layout.addWidget(self.btn_check_score)
        left_layout.addStretch()

        # ── Right panel: UMAP cluster map ───────────────────────
        self.vector_diagram = _VectorDiagram()

        h_layout.addWidget(left_panel, stretch=1)
        h_layout.addWidget(self.vector_diagram, stretch=2)

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
