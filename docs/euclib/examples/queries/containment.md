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
# Containment: on a Surface, or within a Volume

"Is this position part of the object?" sounds like one question, but it is two.
For a surface, a position is part of the object when it lies *on* it. For a
volume, a position is part of the object when it lies *within* it. `euclib.contains`
answers whichever question the geometry you hand it implies.

:::{admonition} What this demonstrates
:class: tip
- `euclib.contains(geom, points, tolerance=None)`, and the one difference that
  decides the answer: whether the geometry is a surface or a volume.
- The same cube faces as a `TriMesh` and as a `TetMesh` giving *different*
  answers for a position at the cube's center.
- The `tolerance` argument, which decides how near counts as on.
- A trap worth knowing: scaling points about the origin leaves the origin fixed,
  so a corner at the origin stays "contained" however the other corners move.
:::

## The same cube, as a surface and as a volume

Both geometries have the same eight corners. The first is the twelve triangles
of the cube's surface; the second splits the cube's interior into six
tetrahedra.

```{code-cell}
import numpy as np
import euclib as el

corners = np.array([[0, 1, 0, 0, 1, 1, 0, 1],
                    [0, 0, 1, 0, 1, 0, 1, 1],
                    [0, 0, 0, 1, 0, 1, 1, 1]], dtype=float)
faces = np.array([[0, 1, 4], [0, 4, 2], [3, 5, 7], [3, 7, 6],
                  [0, 1, 5], [0, 5, 3], [2, 4, 7], [2, 7, 6],
                  [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]], dtype='int64').T
tets = np.array([[0, 1, 2, 5], [0, 2, 5, 6], [0, 1, 5, 4],
                 [0, 3, 6, 7], [0, 4, 5, 7], [0, 5, 6, 7]], dtype='int64').T

surface = el.trimesh(corners, faces)
solid = el.tetmesh(corners, tets)

center = np.array([[0.5], [0.5], [0.5]])
print('surface says the center is contained :', el.contains(surface, center))
print('solid   says the center is contained :', el.contains(solid, center))
```

The corners themselves are contained by both, because they lie on the surface
and therefore also on the boundary of the volume:

```{code-cell}
print('surface contains the corners:', el.contains(surface, corners))
print('solid   contains the corners:', el.contains(solid, corners))
```

## Points inside, on, and outside

Testing points at three scales makes the difference concrete. Scaling the
corners inward and outward about the origin probes inside and outside:

```{code-cell}
print('solid, corners scaled down by 1.5 (inside) :', el.contains(solid, corners / 1.5))
print('solid, corners scaled up by 1.5 (outside)  :', el.contains(solid, corners * 1.5))
print()
print('surface, corners scaled down by 1.5 (inside):', el.contains(surface, corners / 1.5))
```

The solid accepts the inward-scaled points and rejects the outward-scaled ones;
the surface rejects both, because neither set lies *on* it.

:::{admonition} The origin does not move
:class: tip
The first corner is at `(0, 0, 0)`, and scaling about the origin leaves it
there. So the first entry in each `* 1.5` row is `True` even though every other
corner has left the cube — a reminder that these probes scale about the origin
rather than about the cube's center.
:::

## How near counts as on

A point just off the surface is not on it, unless a tolerance says otherwise.
This is the same tolerance the intersection operations accept:

```{code-cell}
just_above = np.array([[0.5], [0.5], [1.0 + 1e-9]])
print('default tolerance:', el.contains(surface, just_above))
print('tolerance 1e-6   :', el.contains(surface, just_above, tolerance=1e-6))
```

## Figure

The cube surface is drawn with the center (inside the volume, but not on the
surface) marked:

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

euclib_viz.show3d(surface, color_by='z', name='containment', points=center)
```

:::{admonition} Provenance
:class: note
**Upstream:** `trimesh/trimesh` — `tests/test_ray.py`, function `test_contains`
(line 95).

**Pinned revision:** [`fcf660f`](https://github.com/trimesh/trimesh/blob/fcf660feb0a14c68fd3945789e8ed77e260f9167/tests/test_ray.py#L95).

**How this was adapted.** The upstream test takes a unit-cube mesh and asks
whether points are inside it: the cube's own vertices (which count as on the
surface), the vertices scaled down by 1.5 (inside), scaled up by 1.5 (outside),
random points far away (outside), and the centroid (inside). No code is copied.
The adaptation keeps the cube and the scaled-point probes exactly, because they
are a good test set, but where trimesh answers all of them with one ray-casting
method on one mesh, `euclib` splits the question in two: the same cube built as
a `TriMesh` and as a `TetMesh` disagrees about the centroid, and that
disagreement is the page's subject. The upstream centroid assertion — that the
center of mass is inside — therefore holds here only for the volumetric
geometry, which is stated rather than hidden. The far-away random points are
not reproduced, since the scaled-outward corners already cover the outside case,
and the tolerance demonstration is added because `euclib` exposes tolerance as
an argument where trimesh does not.
:::

:::{seealso}
- [Distance, separation, and containment](../queries/distance-and-separation.md)
  for containment alongside the distances it is usually paired with.
- [Tetrahedral meshes and volumes](../primitives/tetrahedral-meshes.md) for the
  volumetric geometry that makes "within" meaningful.
:::
