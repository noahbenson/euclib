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
# Nearest Points on a Triangle Mesh

"Where on this surface is the closest point to here?" is the query behind
clamping, snapping, registration, and surface sampling. This page asks it of a
triangle mesh and shows the three things `euclib` can report about the answer:
the nearest *position*, its *distance*, and its *local coordinates* — the
triangle and barycentric weights it landed on.

:::{admonition} What this demonstrates
:class: tip
- `euclib.nearest`: the closest position on a geometry to each query point.
- `euclib.distance`: the distance from a geometry to each query point.
- `Geometry.to_local`, which returns a `TriLoc` naming the triangle and the two
  barycentric weights of the projected point.
- The three cases a projection can fall into: inside a face, on an edge, or at a
  vertex.
:::

## A single triangle to reason about

```{code-cell}
import numpy as np
import euclib as el

corners = np.array([[0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0]])
mesh = el.trimesh(corners, np.array([[0], [1], [2]], dtype='int64'))
print('triangle area:', float(np.sum(mesh.measures)))
```

The query points are given as a `(3, Q)` matrix, exactly like a geometry's
coordinates. Each one probes a different case:

```{code-cell}
points = [[0.25, 0.25, 2.0],   # above the middle of the face
          [0.25, 0.25, 5.0],   # higher above the same spot
          [2.0, -3.0, 0.0]]    # far outside, nearest a corner
# Coordinates are stored as (D, N): one column per point.
queries = np.array(points, dtype=float).T
query_cloud = el.points(queries)
```

## The nearest position

`el.nearest` returns, for each query, a position on the mesh:

```{code-cell}
nearest = np.asarray(el.nearest(mesh, query_cloud))
for i in range(queries.shape[1]):
    print(f'query {queries[:, i]} -> nearest {np.round(nearest[:, i], 4)}')
```

The first two queries project straight down onto the same spot; the third is
pulled back to the nearest corner, `(1, 0, 0)`, because a triangle has no
surface beyond its edges.

## The distance

`el.distance` returns the distance to that nearest position:

```{code-cell}
distances = np.asarray(el.distance(mesh, query_cloud)).ravel()
for i in range(queries.shape[1]):
    print(f'query {i}: distance {distances[i]:.4f}')

print('\noutside query is sqrt(1^2 + 3^2) =', np.sqrt(10))
```

The in-plane distance for the first query is exactly zero — the point is on the
surface — which is a useful check that the projection is exact rather than
merely close.

## The local coordinates

Where the nearest position sits is described by a `TriLoc`: which triangle
(`index`) and, within it, the two barycentric weights (`weight`). A point inside
the face has weights summing to at most one; a point clamped to a corner has a
single weight of one:

```{code-cell}
loc = mesh.to_local(queries)
print('triangle index per query:', np.asarray(loc.index).ravel())
print('barycentric weights:\n', np.round(np.asarray(loc.weight), 4))
```

The third query's weights are `(0, 1)`, naming the triangle's second corner,
`(1, 0, 0)` — the same corner the nearest position landed on. Local coordinates
are how a result on one geometry can be reused on another geometry that shares
its topology.

## Figure

The triangle is drawn with the three query points (in red) above and around it:

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

euclib_viz.show3d(mesh, color_by='z', name='nearest', points=queries)
```

:::{admonition} Provenance
:class: note
**Upstream:** `trimesh/trimesh` — `tests/test_proximity.py`, method
`ProximityTest.test_nearest_naive` (line 45), which is driven by the helper
`check_nearest_point_function` (line 69).

**Pinned revision:** [`fcf660f`](https://github.com/trimesh/trimesh/blob/fcf660feb0a14c68fd3945789e8ed77e260f9167/tests/test_proximity.py#L45).

**How this was adapted.** The upstream helper builds a single-triangle mesh and
a ring of query points, computes the closest point with both trimesh's own code
and shapely as a reference, and checks that the two distances agree and that
the result is unchanged by a rigid transform. No code is copied. The
adaptation keeps the single-triangle setup and the spirit of the reference
comparison, but replaces the shapely cross-check with hand-computed values
(the off-plane point's distance is `sqrt(1+3^2)`); the ring of queries becomes
three queries chosen to hit the interior, the projection, and the vertex cases
explicitly, so the page can show *why* each answer is what it is. The
frame-invariance check, which the upstream helper uses to catch
coordinate-frame bugs, is not reproduced here but is implicit in `euclib`'s
`to_local`/`to_global` split; the third output it reports — the local
coordinates — is exercised instead, and has no direct shapely analogue.
:::

:::{seealso}
- [Distance, separation, and containment](distance-and-separation.md) for the
  scalar distance between two geometries.
- [Interpolating a property within a geometry](../properties/interpolation.md)
  for what local coordinates are used for.
:::
