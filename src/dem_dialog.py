from __future__ import annotations

from pathlib import Path

import matplotlib

TITLE = "Выберите файл рельефа (GeoTIFF)"

_qt_app = None


class DialogUnavailable(Exception):
    pass


def _ask_qt(initial_dir: str) -> str:
    global _qt_app
    try:
        from matplotlib.backends.qt_compat import QtWidgets
    except ImportError as e:
        raise DialogUnavailable(e) from e
    try:
        from matplotlib.backends.backend_qt import _create_qApp
        _qt_app = _create_qApp()
    except Exception:
        _qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None, TITLE, initial_dir,
        "GeoTIFF (*.tif *.tiff);;Все файлы (*)",
    )
    return path


def _ask_tk(initial_dir: str) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
    except Exception as e:
        raise DialogUnavailable(e) from e
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            parent=root, title=TITLE, initialdir=initial_dir,
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("Все файлы", "*.*")],
        )
    finally:
        root.destroy()
    return path


def ask_dem_path(initial_dir: str | Path | None = None) -> Path | None:
    initial = str(initial_dir) if initial_dir else ""
    openers = [_ask_qt, _ask_tk]
    if "qt" not in matplotlib.get_backend().lower():
        openers.reverse()
    for opener in openers:
        try:
            path = opener(initial)
        except DialogUnavailable:
            continue
        return Path(path) if path else None
    print("Диалог выбора файла недоступен (нет ни Qt, ни tkinter).")
    return None
