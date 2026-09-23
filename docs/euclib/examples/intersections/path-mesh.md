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
# Where a Path Crosses a Surface

A path that passes through a surface crosses it one or more times, and knowing
*where* it crosses — and on which triangle — is what turns a crossing into a
usable result. `euclib.path_crossings` reports the crossing points together with
the path segment and the mesh triangle each one belongs to.

:::{admonition} What this demonstrates
:class: tip
- `euclib.path_crossings(path, mesh)`, which returns three parallel arrays: the
  crossing points, the path segment each lies on, and the mesh triangle it
  crossed.
- That a crossing is reported per *triangle*, so a path that crosses a shared
  edge can be reported more than once at the same place.
- A path that misses the surface entirely, which reports no crossings rather
  than raising.
:::

## A zig-zag through a plane

The surface is a flat grid; the path alternates above and below it, so each
segment crosses once.

```{code-cell}
import numpy as np
import euclib as el

n = 5
xs, ys = np.meshgrid(np.linspace(0, 2, n), np.linspace(0, 2, n))
coords = np.array([xs.ravel(), ys.ravel(), np.zeros(n * n)])
corners = []
for i in range(n - 1):
    for j in range(n - 1):
        a, b, c, d = i * n + j, i * n + j + 1, i * n + j + n, i * n + j + n + 1
        corners += [[a, b, d], [a, d, c]]
plane = el.trimesh(coords, np.array(corners, dtype='int64').T)

# Alternating z values make each segment pierce the plane once.
zigzag = el.segpath(np.array([[0.3, 0.6, 1.1, 1.4],
                              [0.8, 0.4, 1.4, 1.0],
                              [1.0, -1.0, 1.0, -1.0]]))

points, segments, triangles = el.path_crossings(zigzag, plane)
print(f'{points.shape[1]} crossings')
print('points   :', np.round(points, 4).tolist())
print('segment  :', segments)
print('triangle :', triangles)
```

Each crossing lies on the plane (`z = 0`) at the midpoint of its segment, and
the triangle index says which triangle was pierced:

```{code-cell}
print('all crossings are on the plane:', bool(np.allclose(points[2], 0.0)))
for k in range(points.shape[1]):
    tri = np.asarray(plane.topo.indices)[:, triangles[k]]
    print(f'crossing {k} on triangle {triangles[k]} {tri.tolist()} '
          f'at {np.round(points[:, k], 3).tolist()}')
```

## A crossing is per triangle

The result has one entry per *(path segment, mesh triangle)* pair, not one per
location. That distinction shows up when a path crosses exactly along an edge
shared by two triangles: both triangles are then reported, at the same point.

```{code-cell}
# The mesh's diagonals run along x = y, so a path that crosses there hits the
# shared edge of two triangles at once.
along_edge = el.segpath(np.array([[0.3, 0.7, 1.1, 1.5],
                                  [0.3, 0.7, 1.1, 1.5],
                                  [1.0, -1.0, 1.0, -1.0]]))
edge_points, _, _ = el.path_crossings(along_edge, plane)
print('segments in the path      :', along_edge.topo.simplex_count[1])
print('crossings reported        :', edge_points.shape[1])
print('unique crossing locations :', np.unique(np.round(edge_points, 4), axis=1).shape[1])
```

Four crossings are reported for three segments because the middle segments
straddle a diagonal.

## A path that misses

A path that never reaches the surface crosses nothing, and the result says so
by being empty rather than by failing:

```{code-cell}
above = el.segpath(np.array([[0.3, 1.5], [0.3, 1.5], [1.0, 1.0]]))
misses, _, _ = el.path_crossings(above, plane)
print('crossings for a path that stays above the plane:', misses.shape[1])
```

## Figure

The plane is drawn with the zig-zag's three crossing points marked in red:

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

euclib_viz.show3d(plane, color_by='y', name='path-mesh', points=points)
```

:::{admonition} Provenance
:class: note
**Upstream:** `trimesh/trimesh` — `tests/test_ray.py`, function `test_rps`
(line 27).

**Pinned revision:** [`fcf660f`](https://github.com/trimesh/trimesh/blob/fcf660feb0a14c68fd3945789e8ed77e260f9167/tests/test_ray.py#L27).

**How this was adapted.** The upstream test fires ten thousand parallel rays at
a subdivided sphere and times how quickly trimesh's intersectors return the hit
triangle indices and locations; it is a performance benchmark whose assertions
only check that the results are arrays. No code is copied. The adaptation keeps
the operation it measures — a segment meeting a triangle — but drops the ray
abstraction and the timing, and uses a single explicit path instead of ten
thousand rays, because `euclib` exposes crossings as one call over a whole path
rather than as a ray-casting object with trees and engines. What the upstream
test takes for granted and this page makes explicit is the structure of the
answer: that the crossing triangle is returned at all, and that it is returned
once per triangle, which is why a path over a shared edge reports a location
twice.
:::

:::{seealso}
- [Where two surfaces meet](mesh-mesh.md), whose crossing curve is built from
  the same segment-triangle test.
- [Where two paths cross](path-path.md) for crossings without a surface.
:::
