---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python (euclib)
  language: python
  name: euclib
---
# Distance, Separation, and Containment

Three related questions are easy to run together: *how far is each of these
points from that surface*, *how close do these two objects ever come*, and *is
this point inside the object*. `euclib` answers them with three separate
operations, and — importantly — treats "inside" differently for a surface and
for a volume.

:::{admonition} What this demonstrates
:class: tip
- `euclib.distance`: a distance per query position, not a single number.
- `euclib.separation`: the *smallest* distance between two geometries.
- `euclib.contains`: on-or-within a geometry, whose meaning depends on
  whether the geometry encloses a volume.
- Why `euclib` has no signed distance: the sign is carried by choosing the
  right geometry type, not by a convention about negative numbers.
:::

## Distance from each of many points

`el.distance(a, b)` measures from `b`'s positions to geometry `a`, and returns
one value per position. Given the cube below and a query cloud, it reports how
far each query is from the cube's surface:

```{code-cell}
import numpy as np
import euclib as el

corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=float).T
tris = np.array([[0, 1, 4], [0, 4, 2], [3, 5, 7], [3, 7, 6],
                 [0, 1, 5], [0, 5, 3], [2, 4, 7], [2, 7, 6],
                 [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]], dtype='int64').T
cube = el.trimesh(corners, tris)

points = [[0.5, 0.5, 2.0],   # 1 unit above the top face
          [2.0, 0.5, 0.5],   # 1 unit beyond the +x face
          [0.5, 0.5, 0.5]]   # at the center of the cube
# Coordinates are stored as (D, N): one column per point.
queries = np.array(points, dtype=float).T
cloud = el.points(queries)

distances = np.asarray(el.distance(cube, cloud)).ravel()
print('distance to the cube surface:', np.round(distances, 4))
```

The center of the cube is half a unit from every face, which is why its distance
to the *surface* is `0.5` — the surface does not "know" it is enclosed. That is
the distinction the rest of the page is about.

## How close two objects come

`el.separation(a, b)` collapses all the pairwise distances to the single
smallest one — the closest approach between two geometries. Two clouds that are
far apart in general can still have one close pair:

```{code-cell}
near = el.points(np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 0.0]]))
far = el.points(np.array([[5.0], [10.0], [0.0]]))
print('separation between the two clouds:', round(float(el.separation(near, far)), 4))
```

## Inside a surface vs inside a volume

`el.contains` asks whether positions lie on or within a geometry. For a
triangle mesh there is no interior to be within, so only points on the surface
count:

```{code-cell}
on_surface = np.array([[0.5, 0.0], [0.5, 0.5], [1.0, 0.5]])   # (3, 2)
inside = np.array([[0.5], [0.5], [0.5]])
print('on the surface :', el.contains(cube, on_surface))
print('at the center  :', el.contains(cube, inside))
```

To ask the enclosure question, the geometry must enclose a volume — which in
`euclib` means a tetrahedral mesh. The cube splits into six tetrahedra:

```{code-cell}
tets = np.array([[0, 1, 2, 5], [0, 2, 5, 6], [0, 1, 5, 4],
                 [0, 3, 6, 7], [0, 4, 5, 7], [0, 5, 6, 7]], dtype='int64').T
solid = el.tetmesh(corners, tets)
print('volume       :', round(float(np.sum(solid.measures)), 6))

probes = np.array([[0.4, 2.0, -0.5], [0.4, 0.5, 0.5], [0.4, 0.5, 0.5]])
print('inside a solid:', el.contains(solid, probes))
```

The center is now *within* the geometry and reports `True`, while a point just
outside the -x face reports `False`.

## Figure

The cube is drawn with the query points from the distance section (in red) above,
beside, and inside it:

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

euclib_viz.show3d(cube, color_by='x', name='distance', points=queries)
```

:::{admonition} There is no signed distance here, on purpose
:class: tip
Trimesh's `signed_distance` returns a negative number for points inside and a
positive one outside. `euclib` avoids that convention: `el.distance` is always
non-negative, and the inside/outside question is asked separately with
`el.contains` on a geometry that has a volume. This keeps a distance a
distance, and makes the inside/outside answer an explicit choice of geometry
type rather than a sign to remember.
:::

:::{admonition} Provenance
:class: note
**Upstream:** `trimesh/trimesh` — `tests/test_proximity.py`, methods
`test_coplanar_signed_distance` (line 151) and `test_noncoplanar_signed_distance`
(line 169).

**Pinned revision:** [`fcf660f`](https://github.com/trimesh/trimesh/blob/fcf660feb0a14c68fd3945789e8ed77e260f9167/tests/test_proximity.py#L151).

**How this was adapted.** The upstream tests take a box primitive and call
`signed_distance` on points outside it, asserting that the result is negative
and that a point coplanar with a face counts as outside. No code is copied. The
adaptation keeps the box and the points-in-and-around-it setup, but splits the
single signed query into `euclib`'s two operations: `el.distance` for the
magnitude, and `el.contains` for the inside/outside decision. Because the
pyvista-style `solids` of trimesh do not exist in `euclib`, the enclosure case
requires an explicit tetrahedralization of the cube, which is written out in
the page. The upstream coplanar boundary convention is preserved and made
explicit in the surface case, where a point exactly on a face counts as
"contained" — the same choice the upstream test documents.
:::

:::{seealso}
- [Nearest points on a triangle mesh](nearest-points.md) for *where* the
  closest point is, not just how far.
- [Tetrahedral meshes](../primitives/tetrahedral-meshes.md) for building the
  volumetric geometry this page's enclosure test needs.
:::
