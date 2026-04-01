"""Main application window — full pipeline wired.

Left panel
----------
  Title → directory path → file count → description
  Status label (shown during background work)
  Load Folder / Sort & Filter / Check & Score buttons
  Similarity results list (visible after Check & Score)

Right panel
-----------
  Live matplotlib UMAP/PCA cluster map (always visible)
"""
import logging
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
    QInputDialog, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

from src.audio_loader import load_wav_files, AudioLoaderError
from src.feature_extractor import extract_features_batch
from src.similarity import score_similarity
from src import umap_reducer

logger = logging.getLogger(__name__)

# ── Theme constants ──────────────────────────────────────────────────────────
_BG         = "#1e1e1e"
_PANEL_BG   = "#252525"
_TEXT       = "#dddddd"
_SUBTLE     = "#888888"
_SPINE      = "#444444"
_NEUTRAL_PT = "#5b9bd5"   # steel blue — default points
_QUERY_PT   = "#ffff00"   # bright yellow — selected file
_LABEL_FS   = 7
_MAX_LABEL  = 18          # chars before truncation


# ── Background worker ────────────────────────────────────────────────────────

class _ExtractionWorker(QThread):
    """Runs feature extraction then UMAP/PCA reduction off the main thread.

    Signals
    -------
    progress(str)   — status message safe to display in the UI
    finished(object) — emits tuple (features: dict, coords: dict)
    error(str)      — human-readable error if the whole run fails
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, wav_paths: list[Path]):
        super().__init__()
        self._wav_paths = wav_paths

    def run(self):
        try:
            self.progress.emit("Extracting features…")
            features = extract_features_batch(self._wav_paths)

            if not features:
                self.error.emit("No features could be extracted from the loaded files.")
                return

            self.progress.emit("Computing layout…")
            coords = umap_reducer.reduce(features)

            self.finished.emit((features, coords))
        except Exception as exc:                    # noqa: BLE001
            logger.exception("Worker failed")
            self.error.emit(str(exc))


# ── Vector diagram ───────────────────────────────────────────────────────────

class _VectorDiagram(QWidget):
    """Matplotlib scatter plot embedded in a QWidget.

    Public methods
    --------------
    plot_neutral(coords)                — all points in steel blue
    plot_scored(coords, ranked, query)  — query=yellow, others red→green
    """

    def __init__(self):
        super().__init__()
        self._figure = Figure(facecolor=_BG, tight_layout=True)
        self._ax     = self._figure.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._draw_empty()

    # ── Styling ──────────────────────────────────────────────────

    def _style_axes(self):
        ax = self._ax
        ax.set_facecolor(_BG)

        # Spines: keep but tone down
        for spine in ax.spines.values():
            spine.set_color(_SPINE)

        # Remove tick marks and tick numbers — UMAP coordinates are meaningless
        ax.tick_params(
            bottom=False, left=False,
            labelbottom=False, labelleft=False,
        )

        # Semantic axis labels
        _axis_label = "← more different  |  more similar →"
        ax.set_xlabel(_axis_label, color=_SUBTLE, fontsize=8, labelpad=6)
        ax.set_ylabel(_axis_label, color=_SUBTLE, fontsize=8, labelpad=6)

        # Chart title
        ax.set_title(
            "Audio Similarity Map",
            color=_TEXT, fontsize=11, pad=10, fontweight="normal",
        )

        # Subtle grid — drawn behind everything
        ax.set_axisbelow(True)
        ax.grid(True, color="#2e2e2e", linewidth=0.6, alpha=0.8, zorder=0)

    def _draw_empty(self):
        self._ax.clear()
        self._style_axes()
        self._ax.text(
            0.5, 0.5, "Load a folder to see the cluster map",
            transform=self._ax.transAxes,
            ha="center", va="center",
            color="#555555", fontsize=10,
        )
        self._canvas.draw()

    @staticmethod
    def _truncate(name: str) -> str:
        return name if len(name) <= _MAX_LABEL else name[:_MAX_LABEL - 1] + "…"

    # ── Public plot API ──────────────────────────────────────────

    def plot_neutral(self, coords: dict[Path, tuple[float, float]]):
        """Render all files as steel-blue points."""
        ax = self._ax
        ax.clear()
        self._style_axes()

        if not coords:
            self._draw_empty()
            return

        xs = [x for x, _ in coords.values()]
        ys = [y for _, y in coords.values()]

        ax.scatter(
            xs, ys,
            c=_NEUTRAL_PT, s=85, zorder=2,
            edgecolors="#ffffff22", linewidths=0.5,
        )
        for path, (x, y) in coords.items():
            ax.annotate(
                self._truncate(path.stem), (x, y),
                xytext=(7, 5), textcoords="offset points",
                color=_TEXT, fontsize=_LABEL_FS, zorder=3,
            )

        self._canvas.draw()

    def plot_scored(
        self,
        coords: dict[Path, tuple[float, float]],
        ranked: list[tuple[Path, float]],
        query_path: Path,
    ):
        """Recolor: query=yellow, others on red(0)→green(1) gradient."""
        score_map = {path: score for path, score in ranked}
        ax = self._ax
        ax.clear()
        self._style_axes()

        # ── Non-query points (batched for performance) ───────────
        others = [(p, xy) for p, xy in coords.items() if p != query_path]
        if others:
            other_paths, other_xys = zip(*others)
            other_xs = [xy[0] for xy in other_xys]
            other_ys = [xy[1] for xy in other_xys]
            # Red (score=0) → green (score=1); slight blue tint keeps
            # low-score points from being pure red on a dark background.
            other_colors = [
                (1.0 - score_map.get(p, 0.0),
                 score_map.get(p, 0.0) * 0.85,
                 0.15)
                for p in other_paths
            ]
            ax.scatter(
                other_xs, other_ys,
                c=other_colors, s=85, zorder=2,
                edgecolors="#ffffff22", linewidths=0.5,
            )
            for path, (x, y) in zip(other_paths, other_xys):
                ax.annotate(
                    self._truncate(path.stem), (x, y),
                    xytext=(7, 5), textcoords="offset points",
                    color=_TEXT, fontsize=_LABEL_FS, zorder=3,
                )

        # ── Query point (drawn on top) ───────────────────────────
        if query_path in coords:
            qx, qy = coords[query_path]
            ax.scatter(
                qx, qy,
                c=_QUERY_PT, s=160, zorder=4,
                edgecolors="#ffffff66", linewidths=1.0,
            )
            ax.annotate(
                self._truncate(query_path.stem), (qx, qy),
                xytext=(7, 5), textcoords="offset points",
                color=_QUERY_PT, fontsize=_LABEL_FS + 1,
                fontweight="bold", zorder=5,
            )

        # ── Legend (only shown in scored state) ──────────────────
        legend_handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=_QUERY_PT, markersize=8,
                   label="Selected file"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=(0.15, 0.85 * 0.85, 0.15), markersize=8,
                   label="Most similar"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=(0.85, 0.15 * 0.85, 0.15), markersize=8,
                   label="Least similar"),
        ]
        legend = ax.legend(
            handles=legend_handles,
            loc="lower right",
            fontsize=7,
            framealpha=0.75,
            facecolor="#2a2a2a",
            edgecolor=_SPINE,
            labelcolor=_TEXT,
            handletextpad=0.5,
            borderpad=0.7,
        )
        legend.get_frame().set_linewidth(0.5)

        self._canvas.draw()


# ── Main window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Similarity Engine")
        self.resize(1200, 700)

        self._wav_files: list[Path]               = []
        self._features:  dict[Path, np.ndarray]   = {}
        self._coords:    dict[Path, tuple[float, float]] = {}
        self._worker:    _ExtractionWorker | None = None

        self._build_ui()
        self._apply_dark_theme()

    # ── UI construction ──────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        h_layout = QHBoxLayout(root)
        h_layout.setSpacing(12)
        h_layout.setContentsMargins(12, 12, 12, 12)

        # ── Left panel ───────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(280)
        lv = QVBoxLayout(left)
        lv.setAlignment(Qt.AlignmentFlag.AlignTop)
        lv.setSpacing(6)
        lv.setContentsMargins(0, 0, 8, 0)

        self.title_label      = QLabel("Audio Similarity Engine")
        self.title_label.setObjectName("title")

        self.dir_label        = QLabel("No directory loaded")
        self.dir_label.setWordWrap(True)
        self.dir_label.setObjectName("dir")

        self.file_count_label = QLabel("")
        self.file_count_label.setObjectName("subtle")

        self.desc_label = QLabel(
            "Load a folder of WAV files to explore\n"
            "their audio features and similarity."
        )
        self.desc_label.setObjectName("subtle")

        self.status_label = QLabel("")
        self.status_label.setObjectName("status")
        self.status_label.setVisible(False)

        self.btn_load_folder  = QPushButton("Load Folder…")
        self.btn_sort_filter  = QPushButton("Sort & Filter")
        self.btn_check_score  = QPushButton("Check & Score")
        self.btn_sort_filter.setEnabled(False)
        self.btn_check_score.setEnabled(False)

        self.btn_load_folder.clicked.connect(self._on_load_folder)
        self.btn_sort_filter.clicked.connect(self._open_sort_filter)
        self.btn_check_score.clicked.connect(self._open_check_score)

        self.results_header = QLabel("Similarity scores")
        self.results_header.setObjectName("subtle")
        self.results_header.setVisible(False)

        self.results_list = QListWidget()
        self.results_list.setVisible(False)
        self.results_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        lv.addWidget(self.title_label)
        lv.addSpacing(4)
        lv.addWidget(self.dir_label)
        lv.addWidget(self.file_count_label)
        lv.addWidget(self.desc_label)
        lv.addSpacing(8)
        lv.addWidget(self.status_label)
        lv.addWidget(self.btn_load_folder)
        lv.addWidget(self.btn_sort_filter)
        lv.addWidget(self.btn_check_score)
        lv.addSpacing(12)
        lv.addWidget(self.results_header)
        lv.addWidget(self.results_list)
        lv.addStretch()

        # ── Right: vector diagram ────────────────────────────────
        self.vector_diagram = _VectorDiagram()

        h_layout.addWidget(left)
        h_layout.addWidget(self.vector_diagram, stretch=1)

    def _apply_dark_theme(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {_BG};
                color: {_TEXT};
                font-family: "Segoe UI", sans-serif;
                font-size: 13px;
            }}
            QLabel#title {{
                font-size: 17px;
                font-weight: bold;
                color: #ffffff;
            }}
            QLabel#dir {{
                color: #aaaaaa;
                font-size: 11px;
            }}
            QLabel#subtle {{
                color: {_SUBTLE};
                font-size: 11px;
            }}
            QLabel#status {{
                color: #f0c040;
                font-size: 12px;
            }}
            QPushButton {{
                background-color: #3a3a3a;
                color: {_TEXT};
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px 12px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: #4a4a4a;
                border-color: #888888;
            }}
            QPushButton:pressed {{
                background-color: #2e2e2e;
            }}
            QPushButton:disabled {{
                background-color: #2a2a2a;
                color: #555555;
                border-color: #333333;
            }}
            QListWidget {{
                background-color: {_PANEL_BG};
                color: {_TEXT};
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                font-size: 11px;
                font-family: "Consolas", monospace;
            }}
            QListWidget::item {{
                padding: 3px 6px;
            }}
            QListWidget::item:selected {{
                background-color: #3a5a8a;
            }}
            QDialog {{
                background-color: {_BG};
                color: {_TEXT};
            }}
            QDialogButtonBox QPushButton {{
                min-width: 80px;
                text-align: center;
            }}
        """)

    # ── Folder loading ───────────────────────────────────────────

    def _on_load_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select WAV folder", "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if not folder:
            return

        try:
            wav_files = load_wav_files(folder)
        except AudioLoaderError as exc:
            QMessageBox.warning(self, "Load Error", str(exc))
            return

        # Reset state from any previous load
        self._wav_files = wav_files
        self._features  = {}
        self._coords    = {}
        self.results_list.clear()
        self.results_list.setVisible(False)
        self.results_header.setVisible(False)
        self.btn_sort_filter.setEnabled(False)
        self.btn_check_score.setEnabled(False)
        self.vector_diagram.plot_neutral({})

        self.dir_label.setText(folder)
        self.file_count_label.setText(f"{len(wav_files)} WAV file(s) found")

        self._start_worker(wav_files)

    def _start_worker(self, wav_paths: list[Path]):
        self.status_label.setVisible(True)
        self.btn_load_folder.setEnabled(False)

        self._worker = _ExtractionWorker(wav_paths)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_worker_progress(self, message: str):
        self.status_label.setText(message)

    def _on_worker_done(self, result: object):
        features, coords = result
        self._features = features
        self._coords   = coords

        n_loaded  = len(self._wav_files)
        n_ok      = len(features)
        skipped   = n_loaded - n_ok

        self.file_count_label.setText(
            f"{n_ok} of {n_loaded} file(s) processed"
            + (f" · {skipped} skipped" if skipped else "")
        )
        self.status_label.setVisible(False)
        self.btn_load_folder.setEnabled(True)
        self.btn_sort_filter.setEnabled(True)
        self.btn_check_score.setEnabled(True)

        self.vector_diagram.plot_neutral(coords)

    def _on_worker_error(self, message: str):
        self.status_label.setText("Failed.")
        self.btn_load_folder.setEnabled(True)
        QMessageBox.critical(self, "Pipeline Error", message)

    # ── Check & Score ────────────────────────────────────────────

    def _open_check_score(self):
        if not self._features:
            return

        names = [p.name for p in self._features]
        chosen, ok = QInputDialog.getItem(
            self, "Check & Score",
            "Select a file to score against all others:",
            names, 0, False,
        )
        if not ok or not chosen:
            return

        query_path = next((p for p in self._features if p.name == chosen), None)
        if query_path is None:
            return

        ranked = score_similarity(self._features[query_path], self._features)

        self.vector_diagram.plot_scored(self._coords, ranked, query_path)
        self._show_results(ranked, query_path)

    def _show_results(
        self,
        ranked: list[tuple[Path, float]],
        query_path: Path,
    ):
        self.results_list.clear()

        for path, score in ranked:
            if path == query_path:
                text = f"  ★  {path.name}"
                item = QListWidgetItem(text)
                item.setForeground(QColor(_QUERY_PT))
            else:
                bar  = _score_bar(score)
                text = f"{bar} {path.name}  {score:.3f}"
                item = QListWidgetItem(text)
                # Tint the text colour toward green/red to match the diagram
                r = int((1.0 - score) * 200)
                g = int(score * 170)
                item.setForeground(QColor(r, g, 38))

            self.results_list.addItem(item)

        self.results_header.setText(f"Scores vs: {query_path.name}")
        self.results_header.setVisible(True)
        self.results_list.setVisible(True)

    # ── Sort & Filter (stub) ─────────────────────────────────────

    def _open_sort_filter(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Sort & Filter")
        dlg.setMinimumWidth(320)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        msg = QLabel("Sort & Filter coming soon.")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(msg)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btns.accepted.connect(dlg.accept)
        layout.addWidget(btns)

        dlg.exec()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _score_bar(score: float, width: int = 8) -> str:
    """Unicode block progress bar, e.g. '█████░░░' for score=0.625."""
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)
