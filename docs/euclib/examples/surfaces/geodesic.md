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
# Geodesic Distance along a Surface

How far apart are two points *along a surface*? A straight line through space
answers a different question once the surface is curved, because the straight
line may leave the surface entirely. `euclib.geodesic` measures distance
along the surface's own edges, from one or more source points to every other
point. This page measures geodesic distance on a flat mesh and on a dome, and
shows where the two kinds of distance part ways.

:::{admonition} What this demonstrates
:class: tip
- `euclib.geodesic(geom, sources)`: one distance per coordinate, measured
  along the geometry's edges from the sources.
- Marking sources with a boolean mask over the coordinates.
- On a flat surface, geodesic distance can be compared directly with Euclidean
  distance; on a curved one it cannot.
- Using a property to carry the result so a figure can color by it.
:::

## A flat surface

The mesh is a flat grid. Geodesic distance from a corner is the shortest path
along the edges, which on this triangulation means the graph distance to the
source.

```{code-cell}
import numpy as np
import euclib as el


def grid_mesh(n=8, size=1.0):
    "Returns a triangulated n-by-n grid over a square of the given size."
    xs, ys = np.meshgrid(np.linspace(0, size, n), np.linspace(0, size, n))
    coords = np.array([xs.ravel(), ys.ravel(), np.zeros(n * n)])
    corners = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, i * n + j + n, i * n + j + n + 1
            corners += [[a, b, d], [a, d, c]]
    return el.trimesh(coords, np.array(corners, dtype='int64').T)


flat = grid_mesh()
sources = np.zeros(flat.coord_count, dtype=bool)
sources[0] = True                      # the corner at the origin

distance = np.asarray(el.geodesic(flat, sources))
print('distance field (rows are x, columns y):')
print(distance.reshape(8, 8).astype(int))
```

Because the surface is flat, geodesic distance is the same as Euclidean, once
you accept that travel is restricted to the mesh's two edge directions:

```{code-cell}
coords = np.asarray(flat.coords)
euclidean = np.linalg.norm(coords - coords[:, 0:1], axis=0)
print('geodesic  >= euclidean everywhere:', bool(np.all(distance >= euclidean - 1e-9)))
print('geodesic / euclidean at the far corner:',
      round(float(distance[-1] / euclidean[-1]), 4), '(sqrt(2) would be 1.414)')
```

The ratio is not the `sqrt(2)` that a straight cut across the square would give:
the path may only follow edges, so it zig-zags.

## A curved surface

Bowing the same grid upward changes the answer, because a path along the surface
is now longer than the straight line between its endpoints:

```{code-cell}
dome = grid_mesh()
bowed = np.asarray(dome.coords).copy()
bowed[2] = 0.4 * np.sin(np.pi * bowed[0]) * np.sin(np.pi * bowed[1])
dome = el.trimesh(bowed, np.asarray(dome.topo.indices))

apex = int(np.argmax(bowed[2]))
sources = np.zeros(dome.coord_count, dtype=bool)
sources[apex] = True

geodesic = np.asarray(el.geodesic(dome, sources))
straight = np.linalg.norm(bowed - bowed[:, apex:apex + 1], axis=0)
print('shortest geodesic path  :', round(float(np.max(geodesic)), 4))
print('shortest straight line  :', round(float(np.max(straight)), 4))
print('the surface route is longer:', bool(np.max(geodesic) > np.max(straight)))
```

## Figure

The distance field is attached to the mesh as an ordinary property, so the
figure simply colors by it:

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

dome = dome.withprop('geodesic', geodesic)
euclib_viz.show3d(dome, color_by='geodesic', name='geodesic')
```

:::{admonition} Provenance
:class: note
**Upstream:** `pyvista/pyvista` — `tests/core/test_polydata.py`, function
`test_geodesic_distance` (line 354).

**Pinned revision:** [`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_polydata.py#L354).

**How this was adapted.** The upstream test picks two vertices of a sphere and
asserts that the geodesic path between them is shorter than the path through a
third waypoint, and that both endpoints report the right distance. No code is
copied. The adaptation keeps the central idea — distance measured along a
surface rather than through space — but replaces the sphere-with-waypoints
setup with two meshes chosen to isolate the point: a flat grid, where geodesic
and Euclidean distance can be compared directly, and a dome, where they
diverge. The waypoint comparison is not reproduced, because `euclib.geodesic`
returns a distance field rather than a single path; the field makes the same
distinction visible everywhere at once, and the page reads the longest geodesic
path out of it to compare against the longest straight line.
:::

:::{seealso}
- [Nearest points on a triangle mesh](../queries/nearest-points.md) for the
  Euclidean counterpart of this question.
- [Distance, separation, and containment](../queries/distance-and-separation.md)
  for distances between whole geometries.
:::
