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
# Spheres and How Tessellation Limits Accuracy

A sphere is the standard example of a shape that a triangle mesh can only
approximate. This page builds a sphere as a `euclib.types.TriMesh` from a
latitude/longitude grid of samples, checks that every vertex really lies at the
requested radius, and shows how the measured surface area approaches — but
never reaches — the exact value.

:::{admonition} What this demonstrates
:class: tip
- Parameterizing a curved surface as a `TriMesh` from a grid of points.
- Verifying a construction with an invariant (every vertex is `radius` from the
  center) rather than by eye.
- `TriMesh.measures`: the per-triangle surface areas, summed to a total.
- Why a mesh's area is always a little short of the analytic value.
:::

## A grid of samples

The sphere is sampled at `n_lat + 1` latitudes and `n_lon` longitudes. The
longitude values exclude their endpoint, because the first and last longitude
coincide; the latitudes include theirs, because the poles do not.

```{code-cell}
import numpy as np
import euclib as el
from euclib import types as et

n_lat, n_lon, radius = 8, 12, 1.0

lat = np.linspace(0, np.pi, n_lat + 1)
lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)

points = np.array([[radius * np.sin(t) * np.cos(p),
                    radius * np.sin(t) * np.sin(p),
                    radius * np.cos(t)]
                   for t in lat for p in lon], dtype=float).T
points.shape
```

Each of the `n_lat * n_lon` grid cells is split into two triangles by
connecting its corners to the next latitude and the next longitude (wrapping
around at the seam):

```{code-cell}
faces = []
for i in range(n_lat):
    for j in range(n_lon):
        a = i * n_lon + j
        b = i * n_lon + (j + 1) % n_lon
        c = (i + 1) * n_lon + j
        d = (i + 1) * n_lon + (j + 1) % n_lon
        faces += [[a, b, d], [a, d, c]]

sphere = et.TriMesh(points, et.TriTopology(np.array(faces, dtype='int64').T))
sphere
```

## Checking the invariant

A rendered sphere can look correct while its vertices are wrong, so the
construction is checked directly: every vertex must lie exactly `radius` from
the center.

```{code-cell}
radii = np.linalg.norm(points, axis=0)
print('all vertices at the requested radius:', np.allclose(radii, radius))
```

## How much area the tessellation loses

`TriMesh.measures` holds one surface area per triangle. Their sum is the area of
the approximation, which is always less than the exact `4*pi*r^2` because the
flat triangles lie inside the curved surface.

```{code-cell}
triangles = sphere.measures
exact = 4 * np.pi * radius**2
print(f'{len(triangles)} triangles')
print(f'measured area: {np.sum(triangles):.5f}')
print(f'exact area   : {exact:.5f}')
print(f'shortfall    : {(exact - np.sum(triangles)) / exact:.2%}')
```

:::{admonition} Refining the mesh
:class: tip
Increasing `n_lat` and `n_lon` shrinks the shortfall without changing any of the
code above, which is the usual way to trade memory for accuracy in a mesh.
:::

## Figure

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

euclib_viz.show3d(sphere, color_by='z', name='sphere')
```

:::{admonition} Provenance
:class: note
**Upstream:** `trimesh/trimesh` — `tests/test_creation.py`, function
`test_spheres` (line 138).

**Pinned revision:** [`fcf660f`](https://github.com/trimesh/trimesh/blob/fcf660feb0a14c68fd3945789e8ed77e260f9167/tests/test_creation.py#L138).

**How this was adapted.** The upstream test builds both a UV sphere and an
icosphere and checks that every vertex is at radius 1, that the surfaces are
watertight and convex, and that face colors are assigned. Nothing is copied.
The adaptation keeps the part that translates to `euclib` — the radius
invariant — and expresses it on a sphere built by hand from a latitude/longitude
grid. The icosphere comparison is dropped because generating an icosphere is a
subdivision algorithm that `euclib` does not provide; the watertightness and
convexity assertions are dropped because they are mesh-topology properties
(closedness, self-intersection) that `euclib` does not test. In their place the
page adds the `measures`-based area comparison, which makes the same point the
upstream test makes about both tessellations: that they approximate the sphere
from the inside.
:::

:::{seealso}
- [Surfaces of revolution](revolution.md) for another way to sweep out a curved
  surface.
- [A cube as a triangle mesh](cube.md) for the simplest possible construction.
:::
