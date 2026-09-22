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
# Design Principles

`euclib` makes a few strong choices about how geometry should be represented and
manipulated. This page summarizes them; understanding them explains most of the
library's interface.

## Data are immutable

`euclib` prefers to treat data — matrices of points and the properties assigned
to them — as immutable, and it uses explicitly immutable data structures and
collections wherever possible. An object is never modified in place: operations
return a new object, and the new object shares all of the untouched,
lazily-computed data with the old one.

```{code-cell}
import numpy as np
import euclib as el
from euclib import types as et

coords = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
topo = et.TriTopology(np.array([[0], [1], [2]], dtype='int64'))
mesh = et.TriMesh(coords, topo)

# `withprop` returns a new mesh; `mesh` is unchanged.
shifted = mesh.withprop('dz', coords[2] + 1.0)
print('shifted has dz:', 'dz' in shifted.properties)
print('mesh    has dz:', 'dz' in mesh.properties)
```

## Computation is lazy

When possible, `euclib` provides lazily-loaded interfaces to data: a mesh does
not calculate its surface area until its surface area has been requested.
Results are cached in memory, so a lazy computation is only ever performed once.
The machinery for this comes from [`immlib`](https://noahbenson.github.io/immlib/),
whose `plan`s, `plandict`s, and `planobject`s represent computations as
directed acyclic graphs of cached, lazy units.

## Properties travel with their geometry

Geometric objects encourage and enable the use of *properties*: numerical values
attached to the geometry itself. A point cloud might carry a `'temperature'`
per point; a mesh's vertices might carry `'curvature'` while its faces carry
`'surface_area'`. Properties are easily interpolated between representations of
different geometric objects, which is what makes it practical to measure
something on one kind of geometry and use it on another.

## NumPy, PyTorch, and units

`euclib` is designed to work with both NumPy and PyTorch, and with `pint`
quantities and units. It uses `immlib`'s utilities to abstract over the numeric
backend. Operations are meant to be compatible with PyTorch gradient descent, so
they can appear inside models and loss functions.

## Channel-first shapes

Because the library values PyTorch interoperability, it orders its arrays with
their shapes as `(C..., X...)`, where `C...` are channel dimensions and `X...`
are spatial dimensions. A geometry with `N` coordinates therefore stores a
coordinate matrix of shape `(D, N)`, with `D` (2 or 3) the single channel
dimension and `N` the single spatial dimension. A property that encodes an
`A...`-shaped value per coordinate has shape `(A..., N)`. For a voxel image of
shape `(R, C, S)`, a property holding a `2x3` Jacobian per voxel has shape
`(2, 3, R, C, S)`. PyTorch typically wants `(N, C, X...)` — with `N` a batch
dimension — and `euclib` assumes a batch holds several geometric objects whose
properties make up the channels and extra dimensions.
