import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent, NavigationToolbar2

from src.zoom import (
    disable_rect_zoom,
    enable_scroll_zoom,
    limit_view_to_image,
    zoom_axes,
)


def _scroll(fig, ax, x, y, step):
    px, py = ax.transData.transform((x, y))
    event = MouseEvent("scroll_event", fig.canvas, px, py, step=step)
    fig.canvas.callbacks.process("scroll_event", event)


@pytest.fixture
def fig_ax():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.canvas.draw()
    yield fig, ax
    plt.close(fig)


def test_zoom_axes_keeps_center():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    zoom_axes(ax, 25, 75, 0.5)
    assert ax.get_xlim() == pytest.approx((12.5, 62.5))
    assert ax.get_ylim() == pytest.approx((37.5, 87.5))
    plt.close(fig)


def test_scroll_up_zooms_in_under_cursor(fig_ax):
    fig, ax = fig_ax
    enable_scroll_zoom(fig, step=2.0)
    _scroll(fig, ax, 50, 50, step=1)
    assert ax.get_xlim() == pytest.approx((25, 75))
    assert ax.get_ylim() == pytest.approx((25, 75))


def test_scroll_down_zooms_out(fig_ax):
    fig, ax = fig_ax
    enable_scroll_zoom(fig, step=2.0)
    _scroll(fig, ax, 10, 10, step=-1)
    assert ax.get_xlim() == pytest.approx((-10, 190))
    assert ax.get_ylim() == pytest.approx((-10, 190))


def test_scroll_outside_axes_ignored(fig_ax):
    fig, ax = fig_ax
    enable_scroll_zoom(fig)
    event = MouseEvent("scroll_event", fig.canvas, 0, 0, step=1)
    fig.canvas.callbacks.process("scroll_event", event)
    assert ax.get_xlim() == pytest.approx((0, 100))


def test_disable_rect_zoom():
    disable_rect_zoom()
    assert all(item[0] != "Zoom" for item in NavigationToolbar2.toolitems)
    assert matplotlib.rcParams["keymap.zoom"] == []


@pytest.fixture
def map_ax():
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((10, 20)), origin="lower", extent=[0, 200, 0, 100])
    limit_view_to_image(ax)
    enable_scroll_zoom(fig, step=2.0)
    fig.canvas.draw()
    yield fig, ax
    plt.close(fig)


def test_map_zoom_out_stays_within_image(map_ax):
    fig, ax = map_ax
    _scroll(fig, ax, 30, 30, step=-1)
    assert ax.get_xlim() == pytest.approx((0, 200))
    assert ax.get_ylim() == pytest.approx((0, 100))


def test_map_zoom_out_near_edge_shifts_inside(map_ax):
    fig, ax = map_ax
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 25)
    _scroll(fig, ax, 10, 10, step=-1)
    assert ax.get_xlim() == pytest.approx((0, 100))
    assert ax.get_ylim() == pytest.approx((0, 50))


def test_map_pan_clamped(map_ax):
    fig, ax = map_ax
    ax.set_xlim(150, 250)
    ax.set_ylim(-20, 30)
    assert ax.get_xlim() == pytest.approx((100, 200))
    assert ax.get_ylim() == pytest.approx((0, 50))


def test_map_inverted_axis_keeps_orientation():
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((10, 10)), origin="upper", extent=[0, 10, 10, 0])
    limit_view_to_image(ax)
    ax.set_ylim(12, 4)
    assert ax.get_ylim() == pytest.approx((10, 2))
    plt.close(fig)


def test_disable_rect_zoom_strips_backend_toolbars(monkeypatch):
    from src import zoom

    class BackendToolbar(NavigationToolbar2):
        toolitems = [("Home", "", "home", "home"), ("Zoom", "", "zoom", "zoom")]

    monkeypatch.setattr(zoom, "_rect_zoom_disabled", False)
    disable_rect_zoom()
    assert BackendToolbar.toolitems == [("Home", "", "home", "home")]
