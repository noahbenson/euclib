#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# docs/euclib/euclib_viz.py
#
# Figure helpers for the euclib example pages.
#
# The documentation uses two plotting backends:
#
#  * ``k3d`` for interactive, rotatable 3D views, and
#  * ``matplotlib`` for 2D images and for the *static* 3D figures that are
#    committed under ``examples/_figures/``.
#
# The reason for the split is that a ``k3d.Plot`` is a live Jupyter widget: it
# serializes to an ipywidgets mime bundle and needs the widgets JavaScript
# runtime (and, for snapshots, a live kernel) to draw anything. A statically
# built Jupyter Book has no such runtime, so a displayed ``k3d.Plot`` renders
# as nothing at all. Every 3D figure therefore goes through ``show3d`` here,
# which uses k3d when an interactive front-end is available and otherwise
# writes and displays a matplotlib PNG. Keeping the choice in one function
# means a page never has to think about which backend it is running under.
#
# Set ``EUCLIB_DOCS_INTERACTIVE=1`` in the environment to force the k3d path
# (useful when running the pages locally in Jupyter). The docs build does not
# set it, so published pages get the PNGs.
#
# by Noah C. Benson

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

FIGDIR = Path(__file__).resolve().parent / 'examples' / '_figures'


def interactive():
    """Returns whether the k3d (interactive) backend should be used.

    The value is taken from the environment variable
    ``EUCLIB_DOCS_INTERACTIVE``; when it is unset or a false-looking value,
    the static matplotlib backend is used.
    """
    val = os.environ.get('EUCLIB_DOCS_INTERACTIVE')
    if val is None:
        return False
    return val.strip().lower() not in ('', '0', 'false', 'no', 'off')


def _as_coords(geom):
    "Returns the ``(D, N)`` coordinate matrix of a geometry as a NumPy array."
    coords = np.asarray(geom.coords, dtype=float)
    return coords


def _tri_arrays(geom):
    """Returns ``(coords, faces)`` for a triangle-mesh-like geometry.

    ``coords`` is a ``(3, N)`` matrix of vertex positions and ``faces`` a
    ``(3, M)`` matrix of triangle corners suitable for a Poly3DCollection or
    for ``k3d.mesh``. Geometry types that are not built from triangles raise
    a ``TypeError``.
    """
    coords = _as_coords(geom)
    indices = np.asarray(geom.topo.indices)
    if coords.shape[0] != 3 or indices.shape[0] != 3:
        raise TypeError(
            'show3d can render triangle-mesh geometries (TriMesh and the '
            'triangle geometries of PrismMesh); construct the object with '
            'triangles or use show2d')
    return coords, np.asarray(indices, dtype='int64')


def _face_values(geom, color_by, faces):
    "Returns one scalar per face, taken from a property or a coordinate axis."
    if color_by is None:
        return None
    if color_by in ('x', 'y', 'z') and color_by not in geom.properties:
        axis = 'xyz'.index(color_by)
        return np.asarray(_as_coords(geom)[axis], dtype=float)[faces].mean(axis=0)
    values = np.asarray(geom[color_by], dtype=float)
    return values[faces].mean(axis=0)


def _figure_path(name, ext='png'):
    "Returns the path of a committed figure file, creating the directory."
    FIGDIR.mkdir(parents=True, exist_ok=True)
    return FIGDIR / f'{name}.{ext}'


def _display_image(path):
    "Displays a saved image in a notebook, or returns the path outside one."
    try:
        from IPython.display import Image, display
    except ImportError:  # pragma: no cover - only hit outside IPython
        return path
    display(Image(filename=str(path)))
    return path


def show3d(geom, color_by=None, name='figure', interactive=None, elev=22,
           azim=-58, cmap='viridis', square=True, points=None,
           pointcolor='crimson', pointsize=26):
    """Renders a triangle-mesh-type geometry in 3D and displays the result.

    When ``interactive()`` is true (or the ``interactive`` argument is given),
    a ``k3d`` plot is built and returned so that a live notebook front-end can
    rotate it. Otherwise a matplotlib 3D figure is written to
    ``examples/_figures/<name>.png`` and displayed as an image, which is what
    the published, statically built book shows.

    Parameters
    ----------
    geom : euclib.types.TriMesh or similar
        The geometry to render. It must be built from triangles: ``coords``
        must have shape ``(3, N)`` and ``topo.indices`` shape ``(3, M)``.
    color_by : str or None, optional
        The name of a property (or one of ``'x'``, ``'y'``, ``'z'``) whose
        value colors each triangle. When ``None`` the mesh is drawn in a
        single flat color.
    name : str, optional
        The file stem used for the committed PNG.
    interactive : bool or None, optional
        Forces the use of k3d (``True``) or matplotlib (``False``). When
        ``None``, ``interactive()`` decides.
    elev, azim : float, optional
        The elevation and azimuth of the static matplotlib camera.
    cmap : str, optional
        The matplotlib colormap used to color triangles.
    square : bool, optional
        Whether to force equal axis scaling in the static figure.
    points : array-like or None, optional
        An optional ``(3, K)`` matrix of positions to draw as markers over the
        mesh, such as the query points of a nearest-position example.
    pointcolor : str, optional
        The marker color for ``points``.
    pointsize : float, optional
        The marker size for ``points``.
    """
    coords, faces = _tri_arrays(geom)
    if interactive is None:
        interactive = globals()['interactive']()
    if interactive:
        import k3d
        plot = k3d.plot(name=name)
        kw = {}
        if color_by is not None:
            vals = np.asarray(geom[color_by], dtype=float)
            kw['attribute'] = vals
            kw['color_map'] = k3d.colormaps.matplotlib_color_maps.Viridis
            if np.isfinite(vals).any():
                lo = float(np.nanmin(vals))
                hi = float(np.nanmax(vals))
                if hi > lo:
                    kw['color_range'] = [lo, hi]
        plot += k3d.mesh(coords.T, faces.T, **kw)
        if points is not None:
            plot += k3d.points(np.asarray(points, dtype=float).T,
                               point_size=0.15, color=0xff3030)
        plot.display()
        return plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vals = _face_values(geom, color_by, faces)
    if vals is None:
        facecolors = np.array([[0.45, 0.62, 0.85, 1.0]])
    else:
        norm = Normalize(vmin=float(np.nanmin(vals)), vmax=float(np.nanmax(vals)))
        facecolors = getattr(cm, cmap)(norm(vals))
    fig = plt.figure(figsize=(5.2, 4.4))
    ax = fig.add_subplot(111, projection='3d')
    poly = Poly3DCollection(coords.T[faces.T], facecolors=facecolors,
                            edgecolor='white', linewidths=0.2)
    ax.add_collection3d(poly)
    if points is not None:
        pts = np.asarray(points, dtype=float)
        ax.scatter(pts[0], pts[1], pts[2], c=pointcolor, s=pointsize,
                   depthshade=False, zorder=5)
    # Fit the axes to the geometry with a small margin.
    lo, hi = coords.min(axis=1), coords.max(axis=1)
    if points is not None:
        pts = np.asarray(points, dtype=float)
        lo = np.minimum(lo, pts.min(axis=1))
        hi = np.maximum(hi, pts.max(axis=1))
    mid, span = (lo + hi) / 2, (hi - lo).max() / 2 or 0.5
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    if square:
        ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if vals is not None:
        mappable = cm.ScalarMappable(norm=Normalize(
            vmin=float(np.nanmin(vals)), vmax=float(np.nanmax(vals))), cmap=cmap)
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label=color_by)
    fig.tight_layout()
    path = _figure_path(name)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return _display_image(path)


def show2d(image=None, name='figure', cmap='viridis', extent=None, vmin=None,
           vmax=None, colorbar=True, label=None, points=None, lines=None,
           markercolor='w', linewidth=1.6, size=18, equal=True):
    """Renders a 2D image, a set of 2D lines, or 2D points, and displays it.

    This is the 2D counterpart of ``show3d``: it always uses matplotlib, which
    renders natively in a static Jupyter Book, and writes the figure to
    ``examples/_figures/<name>.png`` so it is committed alongside the page.

    Any combination of the three sources may be given; the axes then show all
    of them together.

    Parameters
    ----------
    image : array-like or None, optional
        The 2D array to display, such as a grid geometry's image-valued
        property. When ``None`` no image is drawn.
    name : str, optional
        The file stem used for the committed PNG.
    cmap : str, optional
        The matplotlib colormap.
    extent : sequence or None, optional
        The ``(left, right, bottom, top)`` extent of the image in the units of
        the grid geometry (see ``euclib.types.Grid.origin``/``spacing``).
    vmin, vmax : float or None, optional
        The color limits.
    colorbar : bool, optional
        Whether to draw a colorbar (only meaningful with an image).
    label : str or None, optional
        A label for the colorbar.
    points : array-like or None, optional
        An optional ``(2, K)`` matrix of 2D points to draw as markers.
    lines : list of array-like or None, optional
        An optional list of ``(2, K)`` matrices, each drawn as a polyline. This
        is how a 2D `SegPath`'s coordinates are shown.
    markercolor : str, optional
        The marker color for ``points``.
    linewidth : float, optional
        The width of the polylines in ``lines``.
    size : float, optional
        The marker size for ``points``.
    equal : bool, optional
        Whether to force equal axis scaling, so curves are not distorted.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    if image is not None:
        arr = np.asarray(image, dtype=float)
        if arr.ndim > 2:
            arr = arr[0]
        im = ax.imshow(arr, cmap=cmap, origin='lower', extent=extent,
                       vmin=vmin, vmax=vmax)
        if colorbar:
            fig.colorbar(im, ax=ax, label=label)
    if lines is not None:
        for line in lines:
            pts = np.asarray(line, dtype=float)
            # Close the loop if the caller passed an open polyline's points.
            closed = np.concatenate([pts, pts[:, :1]], axis=1)
            ax.plot(closed[0], closed[1], '-', color='tab:blue',
                    linewidth=linewidth, zorder=2)
    if points is not None:
        pts = np.asarray(points, dtype=float)
        ax.scatter(pts[0], pts[1], c=markercolor, s=size, edgecolor='black',
                   linewidths=0.4, zorder=3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if equal:
        ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    path = _figure_path(name)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return _display_image(path)
