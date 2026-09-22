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
# Interpolating a Property within a Geometry

A property is stored at a geometry's vertices, but it is usually wanted
*between* them. This page attaches a property to a triangle mesh, reads it back
at arbitrary positions, and shows the three things that control the answer:
the interpolation order, the mask, and whether extrapolation is allowed.

:::{admonition} What this demonstrates
:class: tip
- `prop(name, at=positions)`: reading a property at global positions rather
  than at the vertices that store it.
- Linear (order 1) versus nearest (order 0) interpolation, and why a linear
  property is reproduced exactly by linear interpolation.
- Property metadata: `interp`, `mask`, `null`, and `extrap`.
- Extrapolation being *off* by default, so a position outside the geometry
  yields "not available" rather than a guess.
:::

## A property on a flat mesh

The mesh is a flat grid whose property is an exactly linear function of
position. A linear function is the case where linear interpolation can be
checked against a formula:

```{code-cell}
import numpy as np
import euclib as el
from euclib import types as et

xs, ys = np.meshgrid(np.arange(4.0), np.arange(4.0))
coords = np.array([xs.ravel(), ys.ravel(), np.zeros(16)], dtype=float)
tris = []
for i in range(3):
    for j in range(3):
        a, b, c, d = i * 4 + j, i * 4 + j + 1, i * 4 + j + 4, i * 4 + j + 5
        tris += [[a, b, d], [a, d, c]]
mesh = et.TriMesh(coords, et.TriTopology(np.array(tris, dtype='int64').T))

values = 2 * coords[0] + 3 * coords[1]
mesh = mesh.withprop('plane', values)
mesh = mesh.withprop('plane_nearest', values, interp=0)
print('property shape:', mesh['plane'].shape)
```

## Reading the property between the vertices

The query positions are ordinary global coordinates. With linear interpolation
the result matches the analytic value exactly, because the property is linear
within every triangle:

```{code-cell}
points = [[0.25, 1.7, 2.5],
          [1.25, 2.7, 0.5],
          [0.0, 0.0, 0.0]]
# Coordinates are stored as (D, N): one column per query position.
queries = np.array(points, dtype=float).T

analytic = 2 * queries[0] + 3 * queries[1]
linear = np.asarray(mesh.prop('plane', at=queries))
nearest = np.asarray(mesh.prop('plane_nearest', at=queries))

print('analytic           :', analytic)
print('linear interpolation:', np.round(linear, 6))
print('nearest neighbour   :', nearest)
```

Nearest-neighbour interpolation returns the value at whichever vertex is
closest, so it snaps to the grid; linear interpolation blends the triangle's
corners. The order is metadata stored with the property, set either at
`withprop` time or by requesting a different order at read time.

## Masked values

A mask marks values that should not be used. Interpolating *inside* a triangle
that has a masked corner returns the property's *null* value instead — `NaN`
here, chosen explicitly rather than left to the default:

```{code-cell}
mask = np.ones(16, dtype=bool)
mask[0] = False                       # invalidate the first corner
masked = mesh.withprop('plane', values, mask=mask, null=np.nan)

# (0.25, 0.25) lies inside the triangle whose corner (0, 0) is masked.
inside = np.array([[0.25], [0.25], [0.0]])
print('interpolated near a masked corner:', masked.prop('plane', at=inside))
```

The mask affects interpolation, not the stored vertices: the masked vertex's
own value is still there if it is read directly. Masking is `euclib`'s way of
recording that a measurement is missing without discarding it.

## Outside the geometry

By default a property does not extrapolate, so a position outside the mesh has
no value. Passing `extrap=0` allows the nearest point's value to be returned
instead:

```{code-cell}
outside = np.array([[10.0], [10.0], [0.0]])

print('default (no extrapolation):', mesh.prop('plane', at=outside))

mesh_extrap = mesh.withprop('plane', values, extrap=0)
print('with extrapolation        :', mesh_extrap.prop('plane', at=outside))
print('nearest corner is (3, 3), whose value is 2*3 + 3*3 =',
      2 * 3 + 3 * 3)
```

:::{admonition} Figure
:class: tip
The property rides with the geometry, so rendering the mesh colored by it uses
the same object the queries were made against:

```python
euclib_viz.show3d(mesh, color_by='plane', name='interpolation')
```
:::

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

euclib_viz.show3d(mesh, color_by='plane', name='interpolation')
```

:::{admonition} Provenance
:class: note
**Upstream:** `pyvista/pyvista` — `tests/core/test_dataset_filters.py`, function
`test_sample_over_line` (line 2273).

**Pinned revision:** [`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_dataset_filters.py#L2273).

**How this was adapted.** The upstream test puts a linearly spaced scalar on a
line dataset and samples it over a sub-segment at a given resolution, checking
that the sampled values are the interpolated ones. No code is copied. The
adaptation keeps the idea — a known function sampled at positions between the
points that store it — but moves it from a 1D line to a triangle mesh, because
interpolation *within a triangle* is the case that requires barycentric weights
and is therefore the more interesting one in `euclib`. The upstream
`sample_over_line` call becomes `prop(name, at=positions)`, and its single
resolution parameter becomes `euclib`'s property metadata, so the page can show
the order, mask, null value, and extrapolation controls that pyvista expresses
through separate filters. The upstream equality check survives as a comparison
against `2x + 3y`.
:::

:::{seealso}
- [Transferring a property between geometries](transfer.md) for sampling a
  property from a *different* geometry.
- [Nearest points on a triangle mesh](../queries/nearest-points.md) for the
  local coordinates that interpolation uses internally.
:::
