"""Sort & Filter dialog — sortable/filterable table of loaded audio files."""

from pathlib import Path

from PyQt6.QtCore import Qt, QItemSelection, QItemSelectionModel, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QDialog, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QPushButton, QSlider,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

# Feature vector column indices (must match feature_extractor.py layout)
_IDX_CENTROID = 13
_IDX_RMS      = 15
_IDX_ZCR      = 16

_BG   = "#1e1e1e"
_TEXT = "#dddddd"
_SUB  = "#888888"
_SPIN = "#444444"
_BLUE = "#5b9bd5"


class SortFilterDialog(QDialog):
    """Sortable, filterable table of every loaded audio file.

    Signals
    -------
    apply_selection(object)  — list[Path] of selected/visible files emitted
                               when the user clicks "Apply to Main View"
    """

    apply_selection = pyqtSignal(object)

    def __init__(self, features: dict, durations: dict, parent=None):
        super().__init__(parent)
        self._features  = features   # dict[Path, np.ndarray]
        self._durations = durations  # dict[Path, float]
        self._all_paths = list(features.keys())

        self.setWindowTitle("Sort & Filter")
        self.setMinimumSize(920, 580)

        self._build_data_ranges()
        self._build_ui()
        self._populate_table()
        self._apply_filters()

    # ── Data range pre-computation ────────────────────────────────

    def _build_data_ranges(self):
        rms_vals  = [float(self._features[p][_IDX_RMS])      for p in self._all_paths]
        cent_vals = [float(self._features[p][_IDX_CENTROID]) for p in self._all_paths]

        self._rms_min  = min(rms_vals,  default=0.0)
        self._rms_max  = max(rms_vals,  default=1.0)
        self._cent_min = min(cent_vals, default=0.0)
        self._cent_max = max(cent_vals, default=20000.0)

        # Guard against all-identical values (would cause div-by-zero in scaling)
        if self._rms_max == self._rms_min:
            self._rms_max = self._rms_min + 1e-6
        if self._cent_max == self._cent_min:
            self._cent_max = self._cent_min + 1.0

    # ── UI construction ───────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        # ── Filter panel ──────────────────────────────────────────
        panel = QWidget()
        panel.setObjectName("filtersPanel")
        panel_row = QHBoxLayout(panel)
        panel_row.setSpacing(20)
        panel_row.setContentsMargins(12, 10, 12, 10)

        # Filename search
        search_col = QVBoxLayout()
        search_col.setSpacing(4)
        search_col.addWidget(_hdr("Filename Search"))
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Filter by filename…")
        self._search_box.textChanged.connect(self._apply_filters)
        search_col.addWidget(self._search_box)
        panel_row.addLayout(search_col, stretch=2)

        # RMS Energy range
        self._rms_lo  = _slider()
        self._rms_hi  = _slider()
        self._rms_hi.setValue(1000)
        self._rms_lo_lbl = QLabel(f"{self._rms_min:.5f}")
        self._rms_hi_lbl = QLabel(f"{self._rms_max:.5f}")
        for lbl in (self._rms_lo_lbl, self._rms_hi_lbl):
            lbl.setMinimumWidth(60)
        self._rms_lo.valueChanged.connect(
            lambda _: self._on_range(
                self._rms_lo, self._rms_hi,
                self._rms_lo_lbl, self._rms_hi_lbl,
                self._rms_min, self._rms_max, ".5f", lo=True,
            )
        )
        self._rms_hi.valueChanged.connect(
            lambda _: self._on_range(
                self._rms_lo, self._rms_hi,
                self._rms_lo_lbl, self._rms_hi_lbl,
                self._rms_min, self._rms_max, ".5f", lo=False,
            )
        )
        panel_row.addLayout(
            _range_col("RMS Energy", self._rms_lo, self._rms_hi,
                       self._rms_lo_lbl, self._rms_hi_lbl),
            stretch=3,
        )

        # Spectral Centroid range
        self._cent_lo  = _slider()
        self._cent_hi  = _slider()
        self._cent_hi.setValue(1000)
        self._cent_lo_lbl = QLabel(f"{self._cent_min:.0f}")
        self._cent_hi_lbl = QLabel(f"{self._cent_max:.0f}")
        for lbl in (self._cent_lo_lbl, self._cent_hi_lbl):
            lbl.setMinimumWidth(52)
        self._cent_lo.valueChanged.connect(
            lambda _: self._on_range(
                self._cent_lo, self._cent_hi,
                self._cent_lo_lbl, self._cent_hi_lbl,
                self._cent_min, self._cent_max, ".0f", lo=True,
            )
        )
        self._cent_hi.valueChanged.connect(
            lambda _: self._on_range(
                self._cent_lo, self._cent_hi,
                self._cent_lo_lbl, self._cent_hi_lbl,
                self._cent_min, self._cent_max, ".0f", lo=False,
            )
        )
        panel_row.addLayout(
            _range_col("Spectral Centroid (Hz)", self._cent_lo, self._cent_hi,
                       self._cent_lo_lbl, self._cent_hi_lbl),
            stretch=3,
        )

        root.addWidget(panel)

        # ── Table ─────────────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels([
            "Filename", "Duration (s)", "RMS Energy",
            "Spectral Centroid (Hz)", "Zero Crossing Rate",
        ])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 5):
            hh.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSortIndicatorShown(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        root.addWidget(self._table)

        # ── Bottom buttons ────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._toggle_btn = QPushButton("Select All")
        self._toggle_btn.setFixedWidth(120)
        self._toggle_btn.clicked.connect(self._toggle_selection)

        apply_btn = QPushButton("Apply to Main View")
        apply_btn.clicked.connect(self._on_apply)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        btn_row.addWidget(self._toggle_btn)
        btn_row.addStretch()
        btn_row.addWidget(apply_btn)
        btn_row.addSpacing(8)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        self._apply_stylesheet()

    # ── Slider/range helpers ──────────────────────────────────────

    def _on_range(self, lo_sl, hi_sl, lo_lbl, hi_lbl, dmin, dmax, fmt, lo):
        """Clamp, update labels, re-filter after a slider move."""
        if lo and lo_sl.value() > hi_sl.value():
            hi_sl.setValue(lo_sl.value())
        elif not lo and hi_sl.value() < lo_sl.value():
            lo_sl.setValue(hi_sl.value())
        lo_lbl.setText(f"{_scale(lo_sl.value(), dmin, dmax):{fmt}}")
        hi_lbl.setText(f"{_scale(hi_sl.value(), dmin, dmax):{fmt}}")
        self._apply_filters()

    # ── Table population ──────────────────────────────────────────

    def _populate_table(self):
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(self._all_paths))

        for row, path in enumerate(self._all_paths):
            vec      = self._features[path]
            duration = self._durations.get(path, 0.0)
            rms      = float(vec[_IDX_RMS])
            centroid = float(vec[_IDX_CENTROID])
            zcr      = float(vec[_IDX_ZCR])

            name_item = QTableWidgetItem(path.name)
            name_item.setData(Qt.ItemDataRole.UserRole, path)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            self._table.setItem(row, 0, name_item)
            self._table.setItem(row, 1, _NumItem(f"{duration:.2f}", duration))
            self._table.setItem(row, 2, _NumItem(f"{rms:.5f}",      rms))
            self._table.setItem(row, 3, _NumItem(f"{centroid:.1f}", centroid))
            self._table.setItem(row, 4, _NumItem(f"{zcr:.5f}",      zcr))

            for col in range(1, 5):
                item = self._table.item(row, col)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

        self._table.setSortingEnabled(True)

    # ── Filtering ─────────────────────────────────────────────────

    def _apply_filters(self):
        search  = self._search_box.text().lower()
        rms_lo  = _scale(self._rms_lo.value(),  self._rms_min,  self._rms_max)
        rms_hi  = _scale(self._rms_hi.value(),  self._rms_min,  self._rms_max)
        cent_lo = _scale(self._cent_lo.value(), self._cent_min, self._cent_max)
        cent_hi = _scale(self._cent_hi.value(), self._cent_min, self._cent_max)

        for row in range(self._table.rowCount()):
            name_item = self._table.item(row, 0)
            rms_item  = self._table.item(row, 2)
            cent_item = self._table.item(row, 3)
            if not name_item:
                continue

            visible = (
                (not search or search in name_item.text().lower())
                and rms_lo  <= float(rms_item.text())  <= rms_hi
                and cent_lo <= float(cent_item.text()) <= cent_hi
            )
            self._table.setRowHidden(row, not visible)

    # ── Selection toggle ──────────────────────────────────────────

    def _toggle_selection(self):
        visible = [r for r in range(self._table.rowCount())
                   if not self._table.isRowHidden(r)]
        if not visible:
            return

        sel_model    = self._table.selectionModel()
        selected_set = {idx.row() for idx in self._table.selectedIndexes()}
        all_selected = all(r in selected_set for r in visible)

        sel_model.clearSelection()
        if not all_selected:
            for row in visible:
                left  = self._table.model().index(row, 0)
                right = self._table.model().index(row, self._table.columnCount() - 1)
                sel_model.select(
                    QItemSelection(left, right),
                    QItemSelectionModel.SelectionFlag.Select,
                )
            self._toggle_btn.setText("Deselect All")
        else:
            self._toggle_btn.setText("Select All")

    # ── Apply ─────────────────────────────────────────────────────

    def _on_apply(self):
        selected_rows = {idx.row() for idx in self._table.selectedIndexes()}

        # Nothing explicitly selected → use all visible rows
        if not selected_rows:
            selected_rows = {
                r for r in range(self._table.rowCount())
                if not self._table.isRowHidden(r)
            }

        paths = []
        for row in selected_rows:
            item = self._table.item(row, 0)
            if item:
                path = item.data(Qt.ItemDataRole.UserRole)
                if path is not None:
                    paths.append(path)

        if paths:
            self.apply_selection.emit(paths)
        self.accept()

    # ── Stylesheet ────────────────────────────────────────────────

    def _apply_stylesheet(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {_BG};
                color: {_TEXT};
                font-family: "Segoe UI", sans-serif;
                font-size: 13px;
            }}
            QWidget#filtersPanel {{
                background-color: #232323;
                border: 1px solid #363636;
                border-radius: 5px;
            }}
            QLabel {{
                color: {_TEXT};
                background: transparent;
            }}
            QLabel[role="hdr"] {{
                font-size: 11px;
                font-weight: 600;
                color: {_SUB};
            }}
            QLineEdit {{
                background-color: #2a2a2a;
                color: {_TEXT};
                border: 1px solid {_SPIN};
                border-radius: 4px;
                padding: 5px 8px;
                font-size: 12px;
            }}
            QLineEdit:focus {{
                border-color: {_BLUE};
            }}
            QTableWidget {{
                background-color: #222222;
                color: {_TEXT};
                border: 1px solid {_SPIN};
                border-radius: 4px;
                gridline-color: #2e2e2e;
                font-size: 12px;
            }}
            QTableWidget::item {{
                padding: 4px 8px;
            }}
            QTableWidget::item:selected {{
                background-color: #2e4a70;
                color: #ffffff;
            }}
            QHeaderView::section {{
                background-color: #2a2a2a;
                color: {_TEXT};
                border: none;
                border-right: 1px solid {_SPIN};
                border-bottom: 1px solid {_SPIN};
                padding: 6px 8px;
                font-size: 12px;
                font-weight: 600;
            }}
            QHeaderView::section:hover {{
                background-color: #383838;
            }}
            QScrollBar:vertical {{
                background: #2a2a2a;
                width: 10px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: #4a4a4a;
                border-radius: 4px;
                min-height: 24px;
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: #3a3a3a;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {_BLUE};
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {_BLUE};
                border-radius: 2px;
            }}
            QPushButton {{
                background-color: #2e2e2e;
                color: {_TEXT};
                border: 1px solid #4a4a4a;
                border-radius: 6px;
                padding: 7px 16px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: #3c3c3c;
                border-color: #6e6e6e;
                color: #ffffff;
            }}
            QPushButton:pressed {{
                background-color: #252525;
            }}
        """)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _hdr(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("role", "hdr")
    return lbl


def _slider() -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setMinimum(0)
    s.setMaximum(1000)
    return s


def _scale(v: int, dmin: float, dmax: float) -> float:
    return dmin + (v / 1000.0) * (dmax - dmin)


def _range_col(title, lo_sl, hi_sl, lo_lbl, hi_lbl) -> QVBoxLayout:
    col = QVBoxLayout()
    col.setSpacing(4)
    col.addWidget(_hdr(title))
    for prefix, sl, lbl in (("Min", lo_sl, lo_lbl), ("Max", hi_sl, hi_lbl)):
        row = QHBoxLayout()
        row.addWidget(QLabel(prefix))
        row.addWidget(sl, stretch=1)
        row.addWidget(lbl)
        col.addLayout(row)
    return col


class _NumItem(QTableWidgetItem):
    """QTableWidgetItem that sorts by numeric value, not string."""

    def __init__(self, display: str, value: float):
        super().__init__(display)
        self._value = value

    def __lt__(self, other) -> bool:
        if isinstance(other, _NumItem):
            return self._value < other._value
        try:
            return self._value < float(other.text())
        except ValueError:
            return super().__lt__(other)
