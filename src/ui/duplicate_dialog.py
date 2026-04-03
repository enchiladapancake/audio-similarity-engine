"""Duplicate / near-duplicate detection dialog.

Layout
------
  Threshold slider (0.900 – 1.000, live updates)
  Scroll area with group cards:
    Group header (colored per group)
    Per-file rows: filename | duration | similarity | ▶ play | Keep radio
    "Flag non-kept for deletion" toggle button
  Bottom: "Export flagged list…"  +  "Rescan"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.duplicate_detector import find_duplicates

# ── Theme (mirrors main_window) ──────────────────────────────────────────────
_BG       = "#1e1e1e"
_PANEL_BG = "#252525"
_TEXT     = "#dddddd"
_SUBTLE   = "#888888"
_SPINE    = "#444444"

# Shared palette — UMAP rings and card borders use the same colours
DUP_PALETTE: list[str] = [
    "#ff5555",   # red
    "#55dd55",   # green
    "#ffaa33",   # amber
    "#bb55ff",   # violet
    "#33ffcc",   # cyan
    "#ff55aa",   # pink
    "#55aaff",   # sky
    "#ffff55",   # yellow
    "#ff8855",   # orange
    "#55ffee",   # teal
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


def _fmt_dur(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


# ── Dialog ───────────────────────────────────────────────────────────────────

class DuplicatesDialog(QDialog):
    """Near-duplicate detection dialog.

    Signals
    -------
    groups_changed(object)  — emits list[list[Path]] whenever detection reruns
                              (used by main window to highlight UMAP dots)
    play_file(object)       — emits Path when a row's play button is pressed
    """

    groups_changed = pyqtSignal(object)
    play_file      = pyqtSignal(object)

    def __init__(
        self,
        features:  dict[Path, np.ndarray],
        durations: dict[Path, float],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Find Duplicates")
        self.resize(740, 580)

        self._features       = features
        self._durations      = durations
        self._threshold      = 0.97
        self._current_groups: list[list[Path]] = []
        # Per-group mutable state; rebuilt on each detection run
        self._group_states:  list[dict] = []

        self._build_ui()
        self._apply_style()
        self._run_detection()   # populate before first show

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def current_groups(self) -> list[list[Path]]:
        """Groups from the most recent detection run."""
        return self._current_groups

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setSpacing(10)
        main.setContentsMargins(14, 14, 14, 14)

        # ── Threshold row ─────────────────────────────────────────────────────
        thresh_row = QHBoxLayout()

        self._thresh_label = QLabel(f"Similarity threshold:  {self._threshold:.3f}")
        self._thresh_label.setObjectName("threshLabel")
        self._thresh_label.setMinimumWidth(230)

        self._thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._thresh_slider.setObjectName("threshSlider")
        # Range 0-100 maps linearly to 0.900-1.000 (step 0.001)
        self._thresh_slider.setRange(0, 100)
        self._thresh_slider.setValue(70)           # default 0.970
        self._thresh_slider.setTickInterval(10)
        self._thresh_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._thresh_slider.valueChanged.connect(self._on_threshold_change)

        thresh_hint = QLabel("0.90")
        thresh_hint.setObjectName("threshHint")
        thresh_max  = QLabel("1.00")
        thresh_max.setObjectName("threshHint")

        thresh_row.addWidget(self._thresh_label)
        thresh_row.addWidget(thresh_hint)
        thresh_row.addWidget(self._thresh_slider, stretch=1)
        thresh_row.addWidget(thresh_max)

        # ── Scroll area for group cards ───────────────────────────────────────
        self._groups_widget = QWidget()
        self._groups_widget.setObjectName("groupsWidget")
        self._groups_layout = QVBoxLayout(self._groups_widget)
        self._groups_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._groups_layout.setSpacing(10)
        self._groups_layout.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._groups_widget)
        scroll.setObjectName("groupsScroll")

        # ── Bottom buttons ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self._btn_export = QPushButton("Export flagged list…")
        self._btn_export.clicked.connect(self._export_flagged)

        self._btn_rescan = QPushButton("Rescan")
        self._btn_rescan.setFixedWidth(90)
        self._btn_rescan.clicked.connect(self._run_detection)

        btn_row.addWidget(self._btn_export, stretch=1)
        btn_row.addSpacing(8)
        btn_row.addWidget(self._btn_rescan)

        main.addLayout(thresh_row)
        main.addWidget(scroll, stretch=1)
        main.addLayout(btn_row)

    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QDialog, QWidget {{
                background-color: {_BG};
                color: {_TEXT};
                font-family: "Segoe UI", sans-serif;
                font-size: 13px;
            }}
            QLabel#threshLabel {{
                font-size: 12px;
                color: {_TEXT};
            }}
            QLabel#threshHint {{
                font-size: 10px;
                color: {_SUBTLE};
                min-width: 28px;
            }}
            QScrollArea#groupsScroll {{
                border: 1px solid {_SPINE};
                border-radius: 4px;
                background-color: {_PANEL_BG};
            }}
            QWidget#groupsWidget {{
                background-color: {_PANEL_BG};
            }}
            QPushButton {{
                background-color: #2e2e2e;
                color: {_TEXT};
                border: 1px solid #4a4a4a;
                border-radius: 5px;
                padding: 7px 14px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: #3c3c3c;
                border-color: #6e6e6e;
                color: #ffffff;
            }}
            QPushButton:pressed {{
                background-color: #252525;
            }}
            QPushButton#playBtn {{
                background-color: #2a2a3e;
                border: 1px solid #4a4a6a;
                padding: 0px;
                min-width: 26px;
                max-width: 26px;
                min-height: 26px;
                max-height: 26px;
                font-size: 11px;
                border-radius: 4px;
            }}
            QPushButton#playBtn:hover {{
                background-color: #3a3a5e;
                border-color: #7070bb;
                color: #ffffff;
            }}
            QSlider#threshSlider::groove:horizontal {{
                height: 4px;
                background: #3a3a4a;
                border-radius: 2px;
            }}
            QSlider#threshSlider::handle:horizontal {{
                background: #5b9bd5;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }}
            QSlider#threshSlider::sub-page:horizontal {{
                background: #5b9bd5;
                border-radius: 2px;
            }}
            QRadioButton {{
                color: {_TEXT};
                font-size: 11px;
                spacing: 4px;
            }}
        """)

    # ── Detection & UI rebuild ────────────────────────────────────────────────

    def _on_threshold_change(self, value: int) -> None:
        self._threshold = 0.90 + value * 0.001
        self._thresh_label.setText(f"Similarity threshold:  {self._threshold:.3f}")
        self._run_detection()

    def _run_detection(self) -> None:
        groups = find_duplicates(self._features, self._threshold)
        self._current_groups = groups
        self._rebuild_groups_ui(groups)
        self.groups_changed.emit(groups)

    def _rebuild_groups_ui(self, groups: list[list[Path]]) -> None:
        layout = self._groups_layout

        # Remove all existing widgets
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self._group_states = []

        if not groups:
            lbl = QLabel("No duplicate groups found at this threshold.")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                f"color: {_SUBTLE}; font-style: italic; padding: 24px;"
            )
            layout.addWidget(lbl)
            return

        for g_idx, group_paths in enumerate(groups):
            color = DUP_PALETTE[g_idx % len(DUP_PALETTE)]
            card  = self._build_group_card(g_idx, group_paths, color)
            layout.addWidget(card)

        layout.addStretch()

    def _build_group_card(
        self,
        g_idx:       int,
        group_paths: list[Path],
        color:       str,
    ) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background-color: #202020; "
            f"border: 1px solid {color}55; border-radius: 6px; }}"
        )

        cv = QVBoxLayout(card)
        cv.setContentsMargins(10, 8, 10, 8)
        cv.setSpacing(3)

        # ── Group header ──────────────────────────────────────────────────────
        header = QLabel(f"Group {g_idx + 1}  —  {len(group_paths)} files")
        header.setStyleSheet(
            f"color: {color}; font-weight: 600; font-size: 12px; "
            f"padding: 0 0 4px 0; background: transparent; border: none;"
        )
        cv.addWidget(header)

        # ── Column-header labels ──────────────────────────────────────────────
        col_hdr = QHBoxLayout()
        col_hdr.setContentsMargins(4, 0, 4, 2)
        col_hdr.setSpacing(8)
        for txt, width, align in [
            ("File",        0,  Qt.AlignmentFlag.AlignLeft),
            ("Duration",   46,  Qt.AlignmentFlag.AlignRight),
            ("Similarity", 60,  Qt.AlignmentFlag.AlignRight),
            ("",           34,  Qt.AlignmentFlag.AlignCenter),   # play column
            ("Keep",       48,  Qt.AlignmentFlag.AlignCenter),
        ]:
            lbl = QLabel(txt)
            lbl.setStyleSheet(
                f"color: {_SUBTLE}; font-size: 9px; "
                f"background: transparent; border: none;"
            )
            lbl.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
            if width:
                lbl.setFixedWidth(width)
            else:
                lbl.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Preferred,
                )
            col_hdr.addWidget(lbl)
        cv.addLayout(col_hdr)

        # Thin divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"color: {_SPINE}; background: {_SPINE}; border: none; max-height: 1px;")
        cv.addWidget(div)

        # ── File rows ─────────────────────────────────────────────────────────
        ref_vec = self._features.get(group_paths[0])

        btn_group  = QButtonGroup(card)
        btn_group.setExclusive(True)
        row_frames: list[QFrame] = []

        for f_idx, path in enumerate(group_paths):
            # Similarity to group[0]
            if f_idx == 0 or ref_vec is None:
                score_str = "ref"
                score_color = _SUBTLE
            else:
                vec = self._features.get(path)
                sim = _cosine_sim(ref_vec, vec) if vec is not None else 0.0
                score_str   = f"{sim:.4f}"
                score_color = "#88cc88"

            dur_str = _fmt_dur(self._durations.get(path, 0.0))

            row = QFrame()
            row.setStyleSheet("background: transparent; border: none;")
            rh  = QHBoxLayout(row)
            rh.setContentsMargins(4, 3, 4, 3)
            rh.setSpacing(8)

            name_lbl = QLabel(path.name)
            name_lbl.setStyleSheet(
                f"color: {_TEXT}; font-size: 11px; background: transparent; border: none;"
            )
            name_lbl.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            name_lbl.setToolTip(str(path))

            dur_lbl = QLabel(dur_str)
            dur_lbl.setFixedWidth(46)
            dur_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            dur_lbl.setStyleSheet(
                f"color: {_SUBTLE}; font-size: 11px; background: transparent; border: none;"
            )

            score_lbl = QLabel(score_str)
            score_lbl.setFixedWidth(60)
            score_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            score_lbl.setStyleSheet(
                f"color: {score_color}; font-size: 11px; "
                f"background: transparent; border: none;"
            )

            play_btn = QPushButton("▶")
            play_btn.setObjectName("playBtn")
            play_btn.setFixedSize(26, 26)
            play_btn.clicked.connect(
                lambda _checked, p=path: self.play_file.emit(p)
            )

            keep_radio = QRadioButton("Keep")
            keep_radio.setChecked(f_idx == 0)
            keep_radio.setStyleSheet(
                f"color: {_TEXT}; font-size: 11px; background: transparent; border: none;"
            )
            btn_group.addButton(keep_radio, f_idx)

            rh.addWidget(name_lbl,  stretch=1)
            rh.addWidget(dur_lbl)
            rh.addWidget(score_lbl)
            rh.addWidget(play_btn)
            rh.addWidget(keep_radio)

            cv.addWidget(row)
            row_frames.append(row)

        # ── Flag button ───────────────────────────────────────────────────────
        flag_btn = QPushButton("Flag non-kept for deletion")
        flag_btn.setStyleSheet(self._flag_btn_style(flagged=False))
        flag_btn.clicked.connect(
            lambda _c, gi=g_idx, btn=flag_btn, bg=btn_group, rows=row_frames:
                self._toggle_flag(gi, btn, bg, rows)
        )
        cv.addSpacing(4)
        cv.addWidget(flag_btn)

        self._group_states.append({
            "paths":        group_paths,
            "button_group": btn_group,
            "row_frames":   row_frames,
            "flag_btn":     flag_btn,
            "flagged":      False,
        })

        return card

    # ── Flag logic ────────────────────────────────────────────────────────────

    @staticmethod
    def _flag_btn_style(flagged: bool) -> str:
        if flagged:
            return (
                "QPushButton { background-color: #2a3a2a; border: 1px solid #4a7a4a; "
                "color: #aaffaa; border-radius: 5px; padding: 5px 12px; font-size: 11px; }"
                "QPushButton:hover { background-color: #3a4a3a; border-color: #6a9a6a; }"
            )
        return (
            "QPushButton { background-color: #3a1a1a; border: 1px solid #7a3a3a; "
            "color: #ffaaaa; border-radius: 5px; padding: 5px 12px; font-size: 11px; }"
            "QPushButton:hover { background-color: #4a2020; border-color: #aa5555; }"
        )

    def _toggle_flag(
        self,
        g_idx:    int,
        flag_btn: QPushButton,
        bg:       QButtonGroup,
        rows:     list[QFrame],
    ) -> None:
        state    = self._group_states[g_idx]
        flagging = not state["flagged"]
        state["flagged"] = flagging

        kept_idx = max(0, bg.checkedId())

        for f_idx, row in enumerate(rows):
            if flagging and f_idx != kept_idx:
                row.setStyleSheet(
                    "background-color: #3a1010; border: none; border-radius: 3px;"
                )
            else:
                row.setStyleSheet("background: transparent; border: none;")

        flag_btn.setText("Unflag group" if flagging else "Flag non-kept for deletion")
        flag_btn.setStyleSheet(self._flag_btn_style(flagged=flagging))

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_flagged(self) -> None:
        flagged: list[Path] = []
        for state in self._group_states:
            if not state["flagged"]:
                continue
            kept_idx = max(0, state["button_group"].checkedId())
            for f_idx, path in enumerate(state["paths"]):
                if f_idx != kept_idx:
                    flagged.append(path)

        if not flagged:
            QMessageBox.information(
                self, "Export Flagged List",
                "No files are currently flagged for deletion.\n\n"
                'Use "Flag non-kept for deletion" on one or more groups first.',
            )
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Flagged File List", "flagged_for_deletion.txt",
            "Text Files (*.txt);;All Files (*)",
        )
        if not save_path:
            return

        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(
                "# Files flagged for deletion\n"
                "# Review carefully before deleting — nothing is deleted by this export.\n\n"
            )
            for path in flagged:
                fh.write(str(path) + "\n")

        QMessageBox.information(
            self, "Export Complete",
            f"Exported {len(flagged)} file path(s) to:\n{save_path}",
        )
