from __future__ import annotations

from contextlib import contextmanager
import pyvista as pv
from typing import Iterator

@contextmanager
def maybe_plotter(p: pv.Plotter | None = None, **kwargs) -> Iterator[pv.Plotter]:
    show = p is None
    p = p or pv.Plotter(**kwargs)
    yield p
    if show:
        p.show()


def iter_subplots(
        plotter: pv.Plotter | None = None,
        link_views=True,
        **kwargs
) -> Iterator[tuple[tuple[int, int], pv.Plotter]]:
    with maybe_plotter(plotter, **kwargs) as p:
        nr, nc = p.shape
        for i in range(nr):
            for j in range(nc):
                p.subplot(i, j)
                yield (i, j), p

        if link_views:
            p.link_views()