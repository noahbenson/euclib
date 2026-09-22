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
# Point Clouds and Their Properties

A point cloud — `euclib.types.VertexSet` — is the simplest geometry: a set of
positions with no connections between them. What makes it useful is that it can
carry a value at each point. This page builds a point cloud, attaches a
computed property to it, and extracts parts of that property with `euclib`'s
indexing rules.

:::{admonition} What this demonstrates
:class: tip
- `VertexSet` and `VertexTopology`, whose indices are simply a `1 x N` range.
- Attaching a coordinate property with `withprop`, and the shape rule: a
  value per coordinate means a final dimension of `N`.
- Extracting a property by name (`cloud['value']`) and selecting a subset of its
  positions (`cloud['value', indices]`) without changing the property's shape.
- Removing a property with `dropprop`, and how indexing behaves for a
  point-valued lookup.
:::

## A cloud of points

A point cloud's topology is trivial: one vertex per coordinate, in order. It is
still an explicit object, because `euclib` keeps coordinates and connectivity
separate for every geometry:

```{code-cell}
import numpy as np
import euclib as el
from euclib import types as et

rng = np.random.default_rng(0)
coords = rng.normal(size=(3, 40))

cloud = et.VertexSet(coords, et.VertexTopology(np.arange(40)[None, :]))
print('coordinates:', cloud.coords.shape)
print('vertices   :', cloud.topo.vertex_count)
```

## Attaching a property

A property attached to a cloud has one value per coordinate, so its final
dimension must equal the number of coordinates. Here the value is a scalar
computed from each point's position:

```{code-cell}
radial = np.linalg.norm(coords, axis=0)
cloud = cloud.withprop('radial', radial)

print('property shape:', cloud['radial'].shape)
print('first three   :', np.round(cloud['radial'][:3], 4))
print('range         :', round(float(cloud['radial'].min()), 3),
      'to', round(float(cloud['radial'].max()), 3))
```

The original object is untouched — `withprop` returned a new cloud — so a
property can never surprise code that is holding an older reference:

```{code-cell}
without = cloud.dropprop('radial')
print('cloud has radial   :', 'radial' in cloud.properties)
print('without has radial :', 'radial' in without.properties)
```

## Selecting positions

Properties are indexed with the same syntax as coordinates: a trailing index
selects *positions*, and the property keeps its shape in the other dimensions.
Selecting three points of a scalar property therefore yields three values:

```{code-cell}
chosen = cloud['radial', [0, 1, 2]]
print('selected values:', np.round(np.asarray(chosen).ravel(), 4))

# A boolean mask works the same way.
near_origin = radial < 1.0
print('points within one unit of the origin:', int(np.sum(near_origin)))
print('their values:', np.round(np.asarray(cloud['radial', near_origin]).ravel(), 4))
```

This is the behavior that lets a property of arbitrary shape ride along with a
geometry: the *spatial* dimension is always last, so indexing the last
dimension never disturbs the value's own shape.

## Figure

The cloud is three-dimensional; the figure shows its x-y projection, so the
points scatter across the plane:

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

euclib_viz.show2d(name='point-cloud', points=coords[:2])
```

:::{admonition} Provenance
:class: note
**Upstream:** `pyvista/pyvista` — `tests/core/test_pointset.py`, function
`test_pointset_basic` (line 16).

**Pinned revision:** [`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_pointset.py#L16).

**How this was adapted.** The upstream test builds a `pv.PointSet` from an
array of points and checks that the reported point count and the stored points
agree, including after wrapping the points in a `DataSet` subclass. No code is
copied. `euclib`'s point cloud requires an explicit topology, so the adaptation
adds the `VertexTopology` that pyvista's `PointSet` infers, and keeps the
upstream "the count is what I asked for" check as a printout of
`topo.vertex_count`. The bulk of the page is then new: pyvista's `PointSet`
carries scalar fields through a separate `point_data` dictionary, whereas
`euclib`'s properties are part of the geometry, so the page demonstrates the
`withprop`/`dropprop`/indexing behavior that is the reason `euclib` attaches
values to geometry at all.
:::

:::{seealso}
- [Interpolating a property within a geometry](interpolation.md) for reading a
  property *between* its points.
- [Nearest points on a triangle mesh](../queries/nearest-points.md) for the
  query machinery a point cloud also supports.
:::
