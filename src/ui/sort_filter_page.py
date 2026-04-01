"""Sort & Filter page.

Lets the user sort/filter loaded WAV files by extracted audio properties
(e.g. tempo, loudness, spectral centroid, MFCC values).
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt


class SortFilterPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout.addWidget(QLabel("Sort & Filter"))
        # TODO: add property selector (combo box), sort direction, filter range sliders
        # TODO: add results table / list view
