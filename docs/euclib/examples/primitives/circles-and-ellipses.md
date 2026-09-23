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
# Circles and Ellipses as Segment Paths

A closed curve is the simplest object built from 1-dimensional simplices. This
page builds a circle and an ellipse as `euclib.types.SegPath` objects — closed
loops of line segments — and checks their geometry through the segment measures
rather than by inspecting coordinates.

:::{admonition} What this demonstrates
:class: tip
- `SegPath` and `SegTopology`: a `(2, M)` matrix of segment endpoints, and how a
  closed loop is written by wrapping the last segment back to the first point.
- How a curved shape is approximated by straight segments.
- Segment measures: `SegPath.measures` gives one length per segment.
- A uniform parameterization gives equal segment lengths on a circle but
  *unequal* ones on an ellipse.
:::

## A circle

The circle is sampled at equally spaced angles, and the segment matrix joins
each sample to the next, wrapping the final sample back to index 0:

```{code-cell}
import numpy as np
import euclib as el

n, radius = 24, 1.0
theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

coords = np.array([radius * np.cos(theta), radius * np.sin(theta)])
segments = np.array([np.arange(n), (np.arange(n) + 1) % n], dtype='int64')

circle = el.segpath(coords, vertices=list(range(n)) + [0])
circle.topo.simplex_count[1]
```

The circle's diameter can be read off its bounding box, which should be
`2 * radius` in each of x and y:

```{code-cell}
lo, hi = el.utils.bounds_of(coords)
print('bounds        :', np.round(np.column_stack([lo, hi]), 3).tolist())
print('measured width:', np.round(hi - lo, 3))
```

Because the samples are equally spaced in angle, every segment has the same
length — a property the ellipse below will *not* share:

```{code-cell}
lengths = circle.measures
print(f'{len(lengths)} segments, each {lengths[0]:.5f} long')
print('all equal:', np.allclose(lengths, lengths[0]))
print(f'total length: {np.sum(lengths):.5f}   (exact circumference {2 * np.pi:.5f})')
```

## An ellipse

The ellipse uses the same construction with different semi-axes. Its bounding
box has width `2a` and height `2b`:

```{code-cell}
a, b = 2.0, 0.5
ell_coords = np.array([a * np.cos(theta), b * np.sin(theta)])

ellipse = el.segpath(ell_coords, vertices=list(range(n)) + [0])
lo, hi = el.utils.bounds_of(ell_coords)
print('semi-axes from bounds:', (hi - lo) / 2)
ell_lengths = ellipse.measures
print('segment lengths equal:', np.allclose(ell_lengths, ell_lengths[0]))
print('shortest / longest  :', round(ell_lengths.min(), 4),
      round(ell_lengths.max(), 4))
```

:::{admonition} Why this matters
:class: tip
Equal steps in the parameter `theta` are not equal steps along the curve. A
circle hides this because it is symmetric; an ellipse exposes it. This is the
same reason `euclib` distinguishes *local* coordinates — the parameter within a
simplex — from global positions in space.
:::

## Figure

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

euclib_viz.show2d(name='curves', lines=[coords, ell_coords])
```

:::{admonition} Provenance
:class: note
**Upstream:** `pyvista/pyvista` — `tests/core/test_geometric_objects.py`,
functions `test_circle` (line 1054) and `test_ellipse` (line 1069).

**Pinned revision:** [`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_geometric_objects.py#L1054).

**How this was adapted.** The upstream tests generate `pv.Circle` and
`pv.Ellipse` and check that the measured diameter equals twice the requested
radius or semi-axis, and that the polyline's segments are of equal length. No
code is copied. `euclib` has no circle generator, so the samples are written
out explicitly and joined into a `SegPath`; the upstream polyline becomes a
segment path with an explicit wrap-around segment. The diameter assertion is
re-expressed through `el.utils.bounds_of`, and the equal-length assertion is kept
for the circle but deliberately *inverted* for the ellipse — where it does not
hold — since showing where a uniform parameterization stops being uniform is
more useful than asserting it. The upstream tests check equality of segment
lengths for both shapes; the ellipse case here therefore documents a real
difference between a curve's parameter and its arc length.
:::

:::{seealso}
- [Circular arcs](circular-arcs.md) for open curves and arc length.
- [Measures: lengths, areas, and volumes](../queries/measures.md) for the
  general treatment of simplex measures.
:::
