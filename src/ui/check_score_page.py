"""Check & Score page.

User selects one WAV file; the app scores all other files by cosine
similarity to it and updates the UMAP diagram to highlight the query
file and its nearest neighbors.
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt


class CheckScorePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout.addWidget(QLabel("Check & Score"))
        # TODO: add file picker for query WAV
        # TODO: add ranked results list with similarity scores
        # TODO: emit signal to update VectorDiagram highlights
