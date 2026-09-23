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
# Tetrahedral Meshes and Volumes

A tetrahedral mesh is the 3-dimensional simplex geometry: the volumetric
counterpart of a triangle mesh. It is what `euclib` uses when a value belongs to
a *region* of space rather than to a surface. This page decomposes a cube into
tetrahedra, checks that the volumes are exact, and shows the volume-preserving
conversion from a prism mesh.

:::{admonition} What this demonstrates
:class: tip
- `TetMesh` and `TetTopology`: four corner indices per tetrahedron.
- `TetMesh.measures`, one volume per tetrahedron, and summing to a total.
- A volume measured from a decomposition is exact, unlike the surface area of a
  tessellated curve.
- `PrismMesh.to_tetmesh()`, which converts a layered sheet into a volume
  without changing the total.
:::

## Decomposing a cube

A cube splits into six tetrahedra by choosing two opposite corners and
connecting them through the faces. The eight corners are the same as the
[cube page](cube.md)'s; only the connectivity differs:

```{code-cell}
import numpy as np
import euclib as el

corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=float).T

# Four corner indices per tetrahedron.
tets = np.array([[0, 1, 2, 5], [0, 2, 5, 6], [0, 1, 5, 4],
                 [0, 3, 6, 7], [0, 4, 5, 7], [0, 5, 6, 7]], dtype='int64').T

cube = el.tetmesh(corners, tets)
cube
```

## Volumes

Each tetrahedron reports its own volume. Every one must be positive (a
negative volume means the corners are wound the wrong way), and the total must
be the cube's volume of exactly one:

```{code-cell}
volumes = cube.measures
print('per-tetrahedron volumes:', np.round(volumes, 6))
print('all positive           :', bool(np.all(volumes > 0)))
print('total volume           :', round(float(np.sum(volumes)), 10))
```

Unlike a tessellated sphere's *surface area*, a tetrahedral decomposition's
volume is exact: the tetrahedra tile the cube's interior without gaps or
overlap, so their volumes sum to the true volume.

:::{admonition} Why the decomposition matters
:class: tip
A tetrahedral mesh gives every point a well-defined region, which is what makes
`el.contains` answer "inside" for a `TetMesh` but not for a `TriMesh` — see
[distance, separation, and containment](../queries/distance-and-separation.md).
:::

## From a prism mesh

A `PrismMesh` converts to a `TetMesh` by splitting each prism into three
tetrahedra. The total volume is unchanged by the conversion, which is the check
worth making:

```{code-cell}
bottom = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
top = bottom + np.array([[0.5], [0.0], [1.5]])
corners = np.array([[0], [1], [2]], dtype='int64')

sheet = el.prismmesh(np.array([bottom, top]), corners)
solid = sheet.to_tetmesh()
print('prism total :', round(float(np.sum(sheet.measures)), 6))
print('tet total   :', round(float(np.sum(solid.measures)), 6))
print('tetrahedra  :', solid.topo.simplex_count[3], '(3 per prism)')
```

## Figure

The figure shows the two faces of the prism mesh, whose tetrahedralization is
what the volume above was measured from:

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

tri = np.asarray(corners)
both = el.trimesh(np.concatenate([bottom, top], axis=1),
                  np.concatenate([tri, tri + 3], axis=1))
euclib_viz.show3d(both, color_by='z', name='tetrahedra')
```

:::{admonition} Provenance
:class: note
**Upstream:** `pyvista/pyvista` — `tests/core/test_geometric_objects.py`,
function `test_solid_sphere` (line 203).

**Pinned revision:** [`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_geometric_objects.py#L203).

**How this was adapted.** The upstream test builds a *volumetric* sphere — as
opposed to a sphere surface — computes each cell's volume, and asserts that
every volume is positive and that their total matches `4/3 * pi * r^3`. No code
is copied. `euclib` has no solid-primitive generators, so the adaptation keeps
the two things the upstream test actually checks — positivity of each cell's
volume and exactness of the total — but demonstrates them on a cube split into
six tetrahedra, whose volume is known by hand. The volume-preserving
`PrismMesh`→`TetMesh` conversion is added, because it is how a volumetric mesh
arises in `euclib` without a primitive generator, and the positive-volume check
is exactly what guards such a conversion against mis-wound tetrahedra.
:::

:::{seealso}
- [Prism sweeps](prism-sweeps.md) for the two-sided sheet that converts to the
  mesh used here.
- [Distance, separation, and containment](../queries/distance-and-separation.md)
  for the different meaning of "inside" that a volumetric mesh enables.
:::
