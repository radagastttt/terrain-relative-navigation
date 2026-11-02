from __future__ import annotations

import matplotlib as mpl
from matplotlib.backend_bases import NavigationToolbar2

ZOOM_STEP = 1.2

_rect_zoom_disabled = False


def _without_zoom(items):
    return type(items)(item for item in items if item[0] != "Zoom")


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def disable_rect_zoom():
    global _rect_zoom_disabled
    if _rect_zoom_disabled:
        return
    NavigationToolbar2.toolitems = _without_zoom(NavigationToolbar2.toolitems)
    for cls in _subclasses(NavigationToolbar2):
        if "toolitems" in cls.__dict__:
            cls.toolitems = _without_zoom(cls.toolitems)
    mpl.rcParams["keymap.zoom"] = []
    _rect_zoom_disabled = True


def _zoom_axis(axis, get_lim, set_lim, center, scale):
    tr = axis.get_transform()
    lo, hi = tr.transform(get_lim())
    c = tr.transform([center])[0]
    new = tr.inverted().transform([c - (c - lo) * scale, c + (hi - c) * scale])
    set_lim(*new)


def zoom_axes(ax, x, y, scale):
    _zoom_axis(ax.xaxis, ax.get_xlim, ax.set_xlim, x, scale)
    _zoom_axis(ax.yaxis, ax.get_ylim, ax.set_ylim, y, scale)


def enable_scroll_zoom(fig, step: float = ZOOM_STEP):
    def on_scroll(event):
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return
        if not ax.get_navigate():
            return
        toolbar = fig.canvas.toolbar
        if toolbar is not None and toolbar._nav_stack() is None:
            toolbar.push_current()
        zoom_axes(ax, event.xdata, event.ydata, step ** (-event.step))
        if toolbar is not None:
            toolbar.push_current()
        fig.canvas.draw_idle()

    return fig.canvas.mpl_connect("scroll_event", on_scroll)


def _clamp_interval(a, b, lo_bound, hi_bound):
    lo, hi = min(a, b), max(a, b)
    width = hi - lo
    if width >= hi_bound - lo_bound:
        lo, hi = lo_bound, hi_bound
    elif lo < lo_bound:
        lo, hi = lo_bound, lo_bound + width
    elif hi > hi_bound:
        lo, hi = hi_bound - width, hi_bound
    return (lo, hi) if a <= b else (hi, lo)


def _image_bounds(ax):
    extents = [im.get_extent() for im in ax.get_images()]
    if not extents:
        return None
    xs = [v for e in extents for v in e[:2]]
    ys = [v for e in extents for v in e[2:]]
    return min(xs), max(xs), min(ys), max(ys)


def limit_view_to_image(ax):
    def clamp(_ax):
        bounds = _image_bounds(ax)
        if bounds is None:
            return
        x0, x1, y0, y1 = bounds
        xlim = _clamp_interval(*ax.get_xlim(), x0, x1)
        ylim = _clamp_interval(*ax.get_ylim(), y0, y1)
        if xlim != ax.get_xlim():
            ax.set_xlim(xlim, emit=False)
        if ylim != ax.get_ylim():
            ax.set_ylim(ylim, emit=False)

    ax.callbacks.connect("xlim_changed", clamp)
    ax.callbacks.connect("ylim_changed", clamp)
    clamp(ax)
