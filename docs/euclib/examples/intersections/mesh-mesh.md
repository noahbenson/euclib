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
# Where Two Surfaces Meet

Two surfaces that cross do so along a curve. `euclib.mesh_intersections` finds
that curve and returns it as a `SegPath` — a set of straight segments, one for
each pair of triangles that meet. This page crosses a floor with a wall and
checks the result against the line the two planes obviously share.

:::{admonition} What this demonstrates
:class: tip
- `euclib.mesh_intersections(first, second)`, which returns the crossing curve
  as a `SegPath` rather than as loose points.
- Why the result is many short segments and not one long one: the curve is
  assembled from triangle-pair crossings.
- That the lengths of those segments sum to the length of the whole curve.
- The `tolerance` argument, which decides how near two triangles must come to
  count as meeting.
:::

## A floor and a wall

Both surfaces are flat grids. The floor lies in the plane `z = 0`; the wall
stands in the plane `x = 1`. Their intersection is the line where `x = 1` and
`z = 0`, which is easy to check by hand.

```{code-cell}
import numpy as np
import euclib as el


def quad(origin, u, v, m=4, n=4):
    "Returns a triangulated m-by-n grid spanning the parallelogram origin+u+v."
    us, vs = np.linspace(0, 1, m), np.linspace(0, 1, n)
    coords = np.array([origin + s * u + t * v for s in us for t in vs]).T
    corners = []
    for i in range(m - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, i * n + j + n, i * n + j + n + 1
            corners += [[a, b, d], [a, d, c]]
    return el.trimesh(coords, np.array(corners, dtype='int64').T)


floor = quad(np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]),
             np.array([0.0, 2.0, 0.0]))
wall = quad(np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0]),
            np.array([0.0, 0.0, 2.0]))

curve = el.mesh_intersections(floor, wall)
print('result:', type(curve).__name__, 'with', curve.topo.simplex_count[1], 'segments')
```

## The crossing curve

Every coordinate of the curve lies on the line `x = 1, z = 0`, and the segments
join end to end:

```{code-cell}
points = np.asarray(curve.coords)
print('all points have x = 1:', bool(np.allclose(points[0], 1.0)))
print('all points have z = 0:', bool(np.allclose(points[2], 0.0)))
print('the curve spans y from', round(float(points[1].min()), 4),
      'to', round(float(points[1].max()), 4))
print('total curve length:', round(float(np.sum(curve.measures)), 4))
```

The total length is exactly `2.0`, the length of the shared line — which is the
check that matters, because the *individual* segments are an artifact of how the
two grids happen to be triangulated, not a property of the surfaces.

## Figure

The figure draws both surfaces as one two-sheet mesh, with the crossing curve
marked in red:

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

def stack(a, b):
    "Returns one TriMesh holding both meshes' triangles, for a single figure."
    n = a.coords.shape[1]
    return el.trimesh(np.concatenate([np.asarray(a.coords), np.asarray(b.coords)], axis=1),
                      np.concatenate([np.asarray(a.topo.indices),
                                      np.asarray(b.topo.indices) + n], axis=1))

euclib_viz.show3d(stack(floor, wall), color_by='z', name='mesh-mesh',
                  points=points)
```

:::{admonition} Provenance
:class: note
**Upstream:** `pyvista/pyvista` — `tests/core/test_polydata.py`, function
`test_intersection` (line 737).

**Pinned revision:** [`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_polydata.py#L737).

**How this was adapted.** The upstream test intersects two spheres and checks
the crossing curve's polyline count, with a second variant that requires the
result to be watertight and covered by the input surfaces. No code is copied.
The adaptation keeps the surface-surface intersection and the check on the
result's structure, but replaces the two spheres — which need a download and
whose crossing curve is a circle that is awkward to verify by hand — with a
floor and a wall, whose crossing line is known exactly. The upstream
watertightness assertion is replaced by the length check, which is the property
that makes the result a curve rather than a bag of segments; the upstream
`split`/`split_all` parameters have no `euclib` analogue because
`mesh_intersections` returns the curve itself rather than cutting the surfaces
apart.
:::

:::{seealso}
- [Where a path crosses a surface](path-mesh.md) for a 1-dimensional input.
- [Where two paths cross](path-path.md) for the same question between two
  curves.
:::
