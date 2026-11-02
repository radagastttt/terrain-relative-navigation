from pathlib import Path

import pytest

from src import dem_dialog
from src.dem_dialog import DialogUnavailable, ask_dem_path


def _unavailable(initial_dir):
    raise DialogUnavailable("нет")


@pytest.fixture
def calls(monkeypatch):
    log = []

    def fake(name, result):
        def opener(initial_dir):
            log.append((name, initial_dir))
            if isinstance(result, Exception):
                raise result
            return result
        return opener

    def install(qt, tk, backend="qtagg"):
        monkeypatch.setattr(dem_dialog, "_ask_qt", fake("qt", qt))
        monkeypatch.setattr(dem_dialog, "_ask_tk", fake("tk", tk))
        monkeypatch.setattr(dem_dialog.matplotlib, "get_backend", lambda: backend)
    install.log = log
    return install


def test_returns_selected_path(calls):
    calls(qt="/data/dem.tif", tk="")
    assert ask_dem_path("/data") == Path("/data/dem.tif")
    assert calls.log == [("qt", "/data")]


def test_cancel_returns_none(calls):
    calls(qt="", tk="/other.tif")
    assert ask_dem_path() is None
    assert [n for n, _ in calls.log] == ["qt"]


def test_falls_back_when_unavailable(calls):
    calls(qt=DialogUnavailable("no qt"), tk="/dem.tif")
    assert ask_dem_path() == Path("/dem.tif")
    assert [n for n, _ in calls.log] == ["qt", "tk"]


def test_tk_first_on_non_qt_backend(calls):
    calls(qt="/qt.tif", tk="/tk.tif", backend="TkAgg")
    assert ask_dem_path() == Path("/tk.tif")
    assert [n for n, _ in calls.log] == ["tk"]


def test_none_when_no_dialog_available(calls):
    calls(qt=DialogUnavailable("a"), tk=DialogUnavailable("b"))
    assert ask_dem_path() is None
