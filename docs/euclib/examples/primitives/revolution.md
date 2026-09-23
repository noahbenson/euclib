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
# Surfaces of Revolution: Tori and Tubes

Many curved surfaces are swept out by spinning a flat profile around an axis. A
torus comes from revolving a circle; a tube or cylinder comes from revolving a
line. Both are the same construction, so this page writes one small `revolve`
helper and uses it twice, checking each result against its analytic surface
area.

:::{admonition} What this demonstrates
:class: tip
- Building a `TriMesh` by sweeping a profile along a circular path.
- The two ways a swept surface can close: across the profile (a torus) or only
  around the sweep (a tube).
- Validating a curved surface against an exact value rather than a rendering.
- `TriMesh.measures` summarized as a total surface area.
:::

## One helper, two shapes

```{code-cell}
import numpy as np
import euclib as el


def revolve(profile, n_theta, close_profile):
    """Revolves a (2, P) profile of (r, z) pairs around the z axis.

    Parameters
    ----------
    profile : array-like
        A ``(2, P)`` matrix whose rows are the radius and height of each
        profile point.
    n_theta : int
        The number of steps around the axis.
    close_profile : bool
        Whether the profile's last point should join back to its first; True
        for a closed cross-section such as a torus's circle, False for an open
        one such as a cylinder wall.

    Returns
    -------
    euclib.types.TriMesh
        The swept surface.
    """
    r, z = np.asarray(profile, dtype=float)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    # One point per (profile, angle) pair, laid out profile-major.
    grid = np.array([[ri * np.cos(t), ri * np.sin(t), zi]
                     for zi, ri in zip(z, r) for t in theta]).T
    P = len(r)

    steps = P if close_profile else P - 1
    faces = []
    for i in range(steps):
        for j in range(n_theta):
            a = i * n_theta + j
            b = i * n_theta + (j + 1) % n_theta
            c = ((i + 1) % P) * n_theta + j
            d = ((i + 1) % P) * n_theta + (j + 1) % n_theta
            faces += [[a, b, d], [a, d, c]]
    return el.trimesh(grid, np.array(faces, dtype='int64').T)
```

## A torus

Revolving a small circle whose center is `R` from the axis produces a torus.
Its exact surface area is `4 * pi^2 * R * r`:

```{code-cell}
R, r, n_profile, n_theta = 1.0, 0.25, 32, 48

phi = np.linspace(0, 2 * np.pi, n_profile, endpoint=False)
tube = np.array([R + r * np.cos(phi), r * np.sin(phi)])   # (2, P) profile

torus = revolve(tube, n_theta, close_profile=True)
approx = np.sum(torus.measures)
exact = 4 * np.pi**2 * R * r
print(f'torus: {torus.topo.simplex_count[2]} triangles')
print(f'measured {approx:.5f}   exact {exact:.5f}   shortfall {(exact-approx)/exact:.2%}')
```

## A tube

Revolving a straight, open profile gives a tube wall. The profile runs from
`z = 0` to `z = h` at radius `R`, so the exact lateral area is `2 * pi * R * h`:

```{code-cell}
h, m = 2.0, 24
wall = np.array([[R, R], [0.0, h]])          # an open profile: two points

tube_mesh = revolve(wall, m, close_profile=False)
approx = np.sum(tube_mesh.measures)
exact = 2 * np.pi * R * h
print(f'tube wall: {tube_mesh.topo.simplex_count[2]} triangles')
print(f'measured {approx:.5f}   exact {exact:.5f}')
```

The same box would be produced by the [prism sweep](prism-sweeps.md) page's
construction, which is the general form of this idea; revolving is the special
case where the sweep path is a circle.

## Figure

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

euclib_viz.show3d(torus, color_by='z', name='torus')
```

:::{admonition} Provenance
:class: note
**Upstream:** `trimesh/trimesh` — `tests/test_creation.py`, functions
`test_revolve` (line 384), `test_torus` (line 424), and `test_annulus`
(line 242).

**Pinned revision:** [`fcf660f`](https://github.com/trimesh/trimesh/blob/fcf660feb0a14c68fd3945789e8ed77e260f9167/tests/test_creation.py#L384).

**How this was adapted.** The upstream `test_revolve` revolves a rectangle into
a cylinder and checks that the volume is `pi * r^2 * h` and that a half
revolution has half that volume; `test_torus` builds a full and a partial torus
and checks the minor radius; `test_annulus` builds an annular cylinder across a
range of transforms. No code is copied. The adaptation keeps the shared idea —
revolving a profile — but factors it into one `revolve` helper whose
`close_profile` flag is exactly the difference between the torus and the
cylinder cases the three upstream tests cover separately. Because `euclib`'s
`TriMesh` measures areas rather than enclosed volumes, the checks compare
surface area against `4*pi^2*R*r` and `2*pi*R*h` instead of the upstream
volume assertions. The partial-torus (angle-limited) case and the annulus's
transform matrix are not reproduced; partial sweeps follow from passing a
`theta` range, and transforms are the subject of the
[transforms pages](../transforms/affine-transforms.md).
:::

:::{seealso}
- [Prism sweeps](prism-sweeps.md) for sweeping a profile along an arbitrary
  path rather than a circle.
- [Spheres and tessellation](spheres.md) for the same accuracy-versus-fineness
  trade-off on a different curved surface.
:::
