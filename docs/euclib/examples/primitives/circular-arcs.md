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
# Circular Arcs: Length along a Curve vs Distance across It

An arc is a `euclib.types.SegPath` that is *not* closed. This page builds one
and compares two numbers that are easy to confuse: the length *along* the curve
— the sum of the segment measures — and the straight-line distance *across* it,
between the two endpoints.

:::{admonition} What this demonstrates
:class: tip
- An open `SegPath`, contrasted with the closed loops of the
  [previous page](circles-and-ellipses.md).
- Summing segment measures to get arc length.
- The difference between a geodesic quantity (length along the curve) and a
  Euclidean one (distance between endpoints).
- How a finer tessellation converges on the exact arc length.
:::

## An arc

The arc spans `angle` radians of a circle of the given radius. Unlike the circle
page, the endpoints are *not* joined to each other, so the topology has one
fewer segment than there are points:

```{code-cell}
import numpy as np
import euclib as el

radius, angle, n = 1.0, np.pi / 2, 16
theta = np.linspace(0, angle, n)
coords = np.array([radius * np.cos(theta), radius * np.sin(theta)])

# n points joined by n - 1 segments; no wrap-around.
segments = np.array([np.arange(n - 1), np.arange(1, n)], dtype='int64')
arc = el.segpath(coords)
arc.topo.simplex_count[1]
```

## Length along the arc

The arc length is the sum of the segment measures:

```{code-cell}
measured = np.sum(arc.measures)
exact = radius * angle
print(f'measured arc length: {measured:.6f}')
print(f'exact  arc length  : {exact:.6f}')
```

## Distance across the arc

The straight-line distance between the endpoints is a different, shorter
quantity. It is what `euclib.separation` reports between two point
geometries — but here it is simplest to read directly from the coordinates:

```{code-cell}
starts, ends = coords[:, 0], coords[:, -1]
chord = np.linalg.norm(ends - starts)
endpoints = np.column_stack([starts, ends])
print(f'chord (endpoints) : {chord:.6f}')
print(f'arc / chord ratio : {measured / chord:.4f}')
```

:::{admonition} The quarter-circle case
:class: tip
For a quarter circle of radius 1 the arc is `pi/2 ~= 1.5708` and the chord is
`sqrt(2) ~= 1.4142`; the ratio is about `1.1107`. The ratio grows without bound
as the arc sweeps further around, which is why "distance" must always be
qualified as either along a surface or through space.
:::

## Refining the arc

More segments bring the measured length closer to the exact value without
changing the code:

```{code-cell}
for count in (4, 8, 16, 64):
    t = np.linspace(0, angle, count)
    c = np.array([radius * np.cos(t), radius * np.sin(t)])
    s = np.array([np.arange(count - 1), np.arange(1, count)], dtype='int64')
    total = np.sum(el.segpath(c).measures)
    print(f'{count:3d} segments: {total:.6f}  (error {exact - total:.2e})')
```

## Figure

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

euclib_viz.show2d(name='arc', lines=[coords],
                  points=np.column_stack([starts, ends]))
```

:::{admonition} Provenance
:class: note
**Upstream:** `pyvista/pyvista` — `tests/core/test_geometric_objects.py`,
function `test_circular_arc` (line 907).

**Pinned revision:** [`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_geometric_objects.py#L907).

**How this was adapted.** The upstream test builds a `pv.CircularArc` between
two points (and, in a second variant, from a center, normal, polar vector, and
angle), then checks that the arc's `'Distance'` scalar increases monotonically
and that the endpoint-to-endpoint span matches expectations. No code is copied.
Since `euclib` has no arc generator, the samples are computed from a polar
parameterization and joined into an open `SegPath`; the upstream per-point
`'Distance'` scalar is replaced by the segment measures that `euclib` provides
natively, summed for the arc length. The upstream monotonicity check is
implicit here in the definition of length along a path, so the adaptation
substitutes the more informative comparison the upstream test gestures at: arc
length against chord length, plus the convergence of the measured length as the
tessellation refines. The second upstream variant (arc from a center and a
normal) is not reproduced, because it is the same construction in a rotated
frame, which the [transforms pages](../transforms/affine-transforms.md) cover.
:::

:::{seealso}
- [Circles and ellipses](circles-and-ellipses.md) for closed curves.
- [Measures: lengths, areas, and volumes](../queries/measures.md) for the
  general treatment of measures.
:::
