"""Main application window — full pipeline wired.

Left panel
----------
  Title → directory path → file count → description
  Status label (shown during background work, with live elapsed timer)
  Load Folder / Sort & Filter / Check & Score buttons
  Similarity results list (visible after Check & Score)

Right panel
-----------
  Live matplotlib UMAP/PCA cluster map (always visible)
"""
import logging
import time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.cm
import matplotlib.lines
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout,
    QInputDialog, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

from src.audio_loader import load_wav_files, AudioLoaderError
from src.feature_extractor import extract_features_batch, get_duration
from src.similarity import score_similarity
from src import umap_reducer
from src.ui.sort_filter_page import SortFilterDialog

logger = logging.getLogger(__name__)

# ── Theme constants ──────────────────────────────────────────────────────────
_BG         = "#1e1e1e"
_PANEL_BG   = "#252525"
_TEXT       = "#dddddd"
_SUBTLE     = "#888888"
_SPINE      = "#444444"
_NEUTRAL_PT = "#5b9bd5"   # steel blue — default points
_QUERY_PT   = "#00aaff"   # bright blue — selected file
_LABEL_FS   = 7
_MAX_LABEL  = 18          # chars before truncation


# ── Background worker ────────────────────────────────────────────────────────

class _ExtractionWorker(QThread):
    """Runs feature extraction, duration collection, then UMAP/PCA reduction.

    Signals
    -------
    finished(object)  — emits tuple (features: dict, coords: dict, durations: dict)
    error(str)        — human-readable error if the whole run fails
    """
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, wav_paths: list[Path]):
        super().__init__()
        self._wav_paths = wav_paths

    def run(self):
        try:
            features = extract_features_batch(self._wav_paths)

            if not features:
                self.error.emit("No features could be extracted from the loaded files.")
                return

            # Collect per-file durations for the Sort & Filter table
            durations: dict[Path, float] = {}
            for path in features:
                try:
                    durations[path] = get_duration(path)
                except Exception:          # noqa: BLE001
                    durations[path] = 0.0

            coords = umap_reducer.reduce(features)

            self.finished.emit((features, coords, durations))
        except Exception as exc:           # noqa: BLE001
            logger.exception("Worker failed")
            self.error.emit(str(exc))


# ── Vector diagram ───────────────────────────────────────────────────────────

_CMAP = matplotlib.colormaps["RdYlGn"]
_NORM = Normalize(vmin=0.0, vmax=1.0)


class _VectorDiagram(QWidget):
    """Matplotlib scatter plot embedded in a QWidget.

    Public methods
    --------------
    plot_neutral(coords)                — all points in steel blue
    plot_scored(coords, ranked, query)  — RdYlGn colormap, colorbar, hover
    """

    def __init__(self):
        super().__init__()
        self._figure = Figure(facecolor=_BG, tight_layout=True)
        self._ax     = self._figure.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._colorbar  = None
        self._tooltip   = None
        # Each entry: (data_x, data_y, display_name, score_str | "")
        self._hover_pts: list[tuple[float, float, str, str]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._draw_empty()

    # ── Helpers ──────────────────────────────────────────────────

    def _remove_colorbar(self):
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None

    def _style_axes(self):
        ax = self._ax
        ax.set_facecolor(_BG)
        for spine in ax.spines.values():
            spine.set_color(_SPINE)

        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

        ax.set_title("Audio Similarity Map",
                     color=_TEXT, fontsize=11, fontweight="normal", pad=34)
        ax.annotate(
            "Points closer together sound more alike",
            xy=(0.5, 1.0), xytext=(0, 14),
            xycoords="axes fraction", textcoords="offset points",
            ha="center", va="bottom",
            color=_SUBTLE, fontsize=7.5, style="italic",
            annotation_clip=False,
        )
        ax.annotate(
            "Axes represent computed audio dimensions, not single properties",
            xy=(0.5, 1.0), xytext=(0, 2),
            xycoords="axes fraction", textcoords="offset points",
            ha="center", va="bottom",
            color="#5a5a5a", fontsize=6.5, style="italic",
            annotation_clip=False,
        )

        ax.set_xlabel("Acoustic Character (A)", color=_SUBTLE, fontsize=8, labelpad=4)
        ax.set_ylabel("Acoustic Character (B)", color=_SUBTLE, fontsize=8, labelpad=4)

        ax.set_axisbelow(True)
        ax.grid(True, color="#2e2e2e", linewidth=0.6, alpha=0.8, zorder=0)

    def _setup_tooltip(self):
        self._tooltip = self._ax.annotate(
            "", xy=(0, 0), xytext=(14, 14),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.45", fc="#222222",
                      ec=_SPINE, lw=0.8, alpha=0.93),
            color=_TEXT, fontsize=8,
            visible=False, zorder=10,
            annotation_clip=False,
        )

    def _draw_empty(self):
        self._remove_colorbar()
        self._hover_pts = []
        self._ax.clear()
        self._style_axes()
        self._ax.text(
            0.5, 0.5, "Load a folder to see the cluster map",
            transform=self._ax.transAxes,
            ha="center", va="center",
            color="#555555", fontsize=10,
        )
        self._setup_tooltip()
        self._canvas.draw()

    @staticmethod
    def _truncate(name: str) -> str:
        return name if len(name) <= _MAX_LABEL else name[:_MAX_LABEL - 1] + "…"

    # ── Public plot API ──────────────────────────────────────────

    def plot_neutral(self, coords: dict[Path, tuple[float, float]]):
        """Render all files as steel-blue points."""
        self._remove_colorbar()
        self._hover_pts = []
        ax = self._ax
        ax.clear()
        self._style_axes()

        if not coords:
            self._draw_empty()
            return

        xs = [x for x, _ in coords.values()]
        ys = [y for _, y in coords.values()]

        ax.scatter(xs, ys, c=_NEUTRAL_PT, s=85, zorder=2,
                   edgecolors="#ffffff22", linewidths=0.5)

        for path, (x, y) in coords.items():
            ax.annotate(self._truncate(path.stem), (x, y),
                        xytext=(7, 5), textcoords="offset points",
                        color=_TEXT, fontsize=_LABEL_FS, zorder=3)
            self._hover_pts.append((x, y, path.stem, ""))

        self._setup_tooltip()
        self._canvas.draw()

    def plot_scored(
        self,
        coords:     dict[Path, tuple[float, float]],
        ranked:     list[tuple[Path, float]],
        query_path: Path,
    ):
        """Recolor via RdYlGn colormap; query=yellow/white-ring; colorbar on right."""
        self._remove_colorbar()
        self._hover_pts = []
        score_map = {path: score for path, score in ranked}

        ax = self._ax
        ax.clear()
        self._style_axes()

        # ── Non-query points ─────────────────────────────────────
        others = [(p, xy) for p, xy in coords.items() if p != query_path]
        if others:
            other_paths, other_xys = zip(*others)
            other_xs     = [xy[0] for xy in other_xys]
            other_ys     = [xy[1] for xy in other_xys]
            other_scores = [score_map.get(p, 0.0) for p in other_paths]
            other_colors = [_CMAP(_NORM(s)) for s in other_scores]

            ax.scatter(other_xs, other_ys, c=other_colors, s=85, zorder=2,
                       edgecolors="#00000033", linewidths=0.4)

            for path, (x, y), score in zip(other_paths, other_xys, other_scores):
                ax.annotate(self._truncate(path.stem), (x, y),
                            xytext=(7, 5), textcoords="offset points",
                            color=_TEXT, fontsize=_LABEL_FS, zorder=3)
                self._hover_pts.append((x, y, path.stem, f"{score:.3f}"))

        # ── Query point ──────────────────────────────────────────
        if query_path in coords:
            qx, qy = coords[query_path]
            ax.scatter(qx, qy, c=_QUERY_PT, s=190, zorder=5,
                       edgecolors="white", linewidths=2.5)
            ax.annotate(self._truncate(query_path.stem), (qx, qy),
                        xytext=(7, 5), textcoords="offset points",
                        color=_QUERY_PT, fontsize=_LABEL_FS + 1,
                        fontweight="bold", zorder=6)
            self._hover_pts.append((qx, qy, query_path.stem, "selected"))

        # ── Colorbar ─────────────────────────────────────────────
        sm = matplotlib.cm.ScalarMappable(cmap=_CMAP, norm=_NORM)
        sm.set_array([])
        self._colorbar = self._figure.colorbar(
            sm, ax=ax, fraction=0.025, pad=0.02, aspect=28,
        )
        self._colorbar.set_label("similarity", color=_SUBTLE, fontsize=7, labelpad=4)
        self._colorbar.ax.tick_params(colors=_SPINE, labelcolor=_SUBTLE, labelsize=7)
        self._colorbar.outline.set_edgecolor(_SPINE)
        self._colorbar.ax.set_facecolor(_BG)

        # ── Legend ───────────────────────────────────────────────
        selected_handle = matplotlib.lines.Line2D(
            [], [], marker="o", color="none",
            markerfacecolor=_QUERY_PT,
            markeredgecolor="white", markeredgewidth=1.5,
            markersize=9, label="Selected file",
        )
        leg = ax.legend(
            handles=[selected_handle],
            loc="lower left",
            fontsize=7,
            framealpha=0.75,
            facecolor="#2a2a2a",
            edgecolor=_SPINE,
            labelcolor=_TEXT,
            handletextpad=0.5,
            borderpad=0.7,
        )
        leg.get_frame().set_linewidth(0.5)

        self._setup_tooltip()
        self._canvas.draw()

    # ── Hover tooltip ────────────────────────────────────────────

    def _on_mouse_move(self, event):
        if self._tooltip is None or not self._hover_pts:
            return

        if event.inaxes != self._ax:
            if self._tooltip.get_visible():
                self._tooltip.set_visible(False)
                self._canvas.draw_idle()
            return

        ax   = self._ax
        hit  = None
        best = float("inf")

        for x, y, name, score_str in self._hover_pts:
            px, py = ax.transData.transform((x, y))
            dist   = ((px - event.x) ** 2 + (py - event.y) ** 2) ** 0.5
            if dist < 14.0 and dist < best:
                best = dist
                hit  = (x, y, name, score_str)

        changed = False
        if hit:
            x, y, name, score_str = hit
            if score_str == "selected":
                text = f"{name}\n(selected)"
            elif score_str:
                text = f"{name}\nsimilarity: {score_str}"
            else:
                text = name
            self._tooltip.xy = (x, y)
            self._tooltip.set_text(text)
            if not self._tooltip.get_visible():
                self._tooltip.set_visible(True)
            changed = True
        elif self._tooltip.get_visible():
            self._tooltip.set_visible(False)
            changed = True

        if changed:
            self._canvas.draw_idle()


# ── Main window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Similarity Engine")
        self.resize(1200, 700)

        self._wav_files:  list[Path]               = []
        self._features:   dict[Path, np.ndarray]   = {}
        self._coords:     dict[Path, tuple[float, float]] = {}
        self._durations:  dict[Path, float]         = {}
        self._query_path: Path | None               = None
        self._worker:     _ExtractionWorker | None  = None

        # Timer state (initialised in _start_worker)
        self._elapsed_timer: QTimer | None = None
        self._extraction_start: float      = 0.0

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
        left.setFixedWidth(288)
        lv = QVBoxLayout(left)
        lv.setAlignment(Qt.AlignmentFlag.AlignTop)
        lv.setSpacing(0)
        lv.setContentsMargins(4, 8, 12, 8)

        self.title_label = QLabel("Audio Similarity Engine")
        self.title_label.setObjectName("appTitle")

        info_box = QFrame()
        info_box.setObjectName("infoBox")
        ib = QVBoxLayout(info_box)
        ib.setContentsMargins(9, 7, 9, 7)
        ib.setSpacing(3)

        self.dir_label = QLabel("No directory loaded")
        self.dir_label.setWordWrap(True)
        self.dir_label.setObjectName("dirPath")

        self.file_count_label = QLabel("")
        self.file_count_label.setObjectName("fileCount")

        ib.addWidget(self.dir_label)
        ib.addWidget(self.file_count_label)

        self.desc_label = QLabel(
            "Load a folder of audio files to explore\n"
            "their features and similarity."
        )
        self.desc_label.setObjectName("desc")

        self.status_label = QLabel("")
        self.status_label.setObjectName("status")
        self.status_label.setVisible(False)

        # && renders as a literal & in Qt button labels
        self.btn_load_folder = QPushButton("Load Folder…")
        self.btn_sort_filter = QPushButton("Sort && Filter")
        self.btn_check_score = QPushButton("Check && Score")
        self.btn_sort_filter.setEnabled(False)
        self.btn_check_score.setEnabled(False)

        self.btn_load_folder.clicked.connect(self._on_load_folder)
        self.btn_sort_filter.clicked.connect(self._open_sort_filter)
        self.btn_check_score.clicked.connect(self._open_check_score)

        self.results_header = QLabel("Similarity Scores")
        self.results_header.setObjectName("sectionHeader")
        self.results_header.setVisible(False)

        self.results_list = QListWidget()
        self.results_list.setVisible(False)
        self.results_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        lv.addWidget(self.title_label)
        lv.addSpacing(10)
        lv.addWidget(info_box)
        lv.addSpacing(8)
        lv.addWidget(self.desc_label)
        lv.addSpacing(14)
        lv.addWidget(self.status_label)
        lv.addWidget(self.btn_load_folder)
        lv.addSpacing(5)
        lv.addWidget(self.btn_sort_filter)
        lv.addSpacing(3)
        lv.addWidget(self.btn_check_score)
        lv.addSpacing(18)
        lv.addWidget(self.results_header)
        lv.addSpacing(4)
        lv.addWidget(self.results_list)
        lv.addStretch()

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
            QLabel#appTitle {{
                font-size: 20px;
                font-weight: 700;
                color: #ffffff;
                letter-spacing: 0.3px;
            }}
            QFrame#infoBox {{
                background-color: #232323;
                border: 1px solid #363636;
                border-radius: 5px;
            }}
            QLabel#dirPath {{
                font-family: "Consolas", "Courier New", monospace;
                font-size: 10px;
                color: #999999;
                background: transparent;
            }}
            QLabel#fileCount {{
                font-size: 10px;
                color: #666666;
                background: transparent;
            }}
            QLabel#desc {{
                color: #707070;
                font-size: 11px;
                font-style: italic;
            }}
            QLabel#status {{
                color: #f0c040;
                font-size: 11px;
                padding-bottom: 4px;
            }}
            QLabel#sectionHeader {{
                color: #5a5a5a;
                font-size: 9px;
                font-weight: 600;
                letter-spacing: 1.2px;
                text-transform: uppercase;
                padding-bottom: 1px;
            }}
            QPushButton {{
                background-color: #2e2e2e;
                color: {_TEXT};
                border: 1px solid #4a4a4a;
                border-radius: 6px;
                padding: 8px 14px;
                text-align: center;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: #3c3c3c;
                border-color: #6e6e6e;
                color: #ffffff;
            }}
            QPushButton:pressed {{
                background-color: #252525;
                border-color: #555555;
            }}
            QPushButton:disabled {{
                background-color: #252525;
                color: #484848;
                border-color: #303030;
            }}
            QListWidget {{
                background-color: #222222;
                color: {_TEXT};
                border: 1px solid #363636;
                border-radius: 5px;
                font-size: 11px;
                font-family: "Consolas", "Courier New", monospace;
                outline: none;
            }}
            QListWidget::item {{
                padding: 5px 8px;
                border-bottom: 1px solid #2a2a2a;
            }}
            QListWidget::item:last-child {{
                border-bottom: none;
            }}
            QListWidget::item:selected {{
                background-color: #2e4a70;
                color: #ffffff;
            }}
            QListWidget::item:hover {{
                background-color: #2a2a2a;
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

        self._wav_files  = wav_files
        self._features   = {}
        self._coords     = {}
        self._durations  = {}
        self._query_path = None
        self.results_list.clear()
        self.results_list.setVisible(False)
        self.results_header.setVisible(False)
        self.btn_sort_filter.setEnabled(False)
        self.btn_check_score.setEnabled(False)
        self.vector_diagram.plot_neutral({})

        self.dir_label.setText(folder)
        self.file_count_label.setText(f"{len(wav_files)} audio file(s) found")

        self._start_worker(wav_files)

    def _start_worker(self, wav_paths: list[Path]):
        self._extraction_start = time.monotonic()

        self.status_label.setText("Extracting features... 0:00 elapsed")
        self.status_label.setVisible(True)
        self.btn_load_folder.setEnabled(False)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._elapsed_timer.start()

        self._worker = _ExtractionWorker(wav_paths)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _update_elapsed(self):
        elapsed = time.monotonic() - self._extraction_start
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        self.status_label.setText(
            f"Extracting features... {mins}:{secs:02d} elapsed"
        )

    def _on_worker_done(self, result: object):
        elapsed = time.monotonic() - self._extraction_start
        if self._elapsed_timer is not None:
            self._elapsed_timer.stop()

        features, coords, durations = result
        self._features  = features
        self._coords    = coords
        self._durations = durations

        n_loaded = len(self._wav_files)
        n_ok     = len(features)
        skipped  = n_loaded - n_ok

        count_text = (
            f"{n_ok} of {n_loaded} file(s) processed (took {elapsed:.1f}s)"
        )
        if skipped:
            count_text += f" · {skipped} skipped"
        self.file_count_label.setText(count_text)

        self.status_label.setVisible(False)
        self.btn_load_folder.setEnabled(True)
        self.btn_sort_filter.setEnabled(True)
        self.btn_check_score.setEnabled(True)

        self.vector_diagram.plot_neutral(coords)

    def _on_worker_error(self, message: str):
        if self._elapsed_timer is not None:
            self._elapsed_timer.stop()
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

        self._query_path = query_path
        ranked = score_similarity(self._features[query_path], self._features)

        self.vector_diagram.plot_scored(self._coords, ranked, query_path)
        self._show_results(ranked, query_path)

    def _show_results(
        self,
        ranked:     list[tuple[Path, float]],
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
                r = int((1.0 - score) * 200)
                g = int(score * 170)
                item.setForeground(QColor(r, g, 38))

            self.results_list.addItem(item)

        self.results_header.setText("SIMILARITY SCORES")
        self.results_header.setVisible(True)
        self.results_list.setVisible(True)

    # ── Sort & Filter ────────────────────────────────────────────

    def _open_sort_filter(self):
        if not self._features:
            return

        dlg = SortFilterDialog(self._features, self._durations, parent=self)
        dlg.apply_selection.connect(self._apply_filter_to_view)
        dlg.exec()

    def _apply_filter_to_view(self, selected_paths: list):
        """Update the diagram to show only the user-selected subset."""
        if not selected_paths:
            return

        active = set(selected_paths)
        filtered_coords = {p: c for p, c in self._coords.items() if p in active}

        if self._query_path and self._query_path in active:
            # Re-score within the filtered subset
            active_features = {
                p: self._features[p] for p in active if p in self._features
            }
            ranked = score_similarity(
                self._features[self._query_path], active_features
            )
            self.vector_diagram.plot_scored(filtered_coords, ranked, self._query_path)
            self._show_results(ranked, self._query_path)
        else:
            self._query_path = None
            self.vector_diagram.plot_neutral(filtered_coords)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _score_bar(score: float, width: int = 8) -> str:
    """Unicode block progress bar, e.g. '█████░░░' for score=0.625."""
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)
