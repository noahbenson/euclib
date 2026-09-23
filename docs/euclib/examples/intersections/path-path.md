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
# Where Two Paths Cross

Two curves cross where a segment of one meets a segment of the other.
`euclib.path_intersections` finds those meeting points and reports, for each
one, which segment of each path it belongs to — the information needed to
locate the crossing on either curve.

:::{admonition} What this demonstrates
:class: tip
- `euclib.path_intersections(first, second)`, which returns the crossing points
  and the segment indices of both paths.
- That a curved crossing is found by testing straight segments against each
  other, so a path's tessellation affects how exactly the crossing is located.
- Two paths that do not meet, which report no crossings rather than raising.
- Using a `SegPath` as a set of segments rather than as a route to travel.
:::

## A zig-zag and a line

One path zig-zags; the other is a straight line. Where the line's height equals
a zig-zag segment's, the two cross.

```{code-cell}
import numpy as np
import euclib as el

# A zig-zag path through (0, 0) -> (1, 3) -> (2, 0) -> (3, 3).
zigzag = el.segpath(np.array([[0.0, 1.0, 2.0, 3.0],
                              [0.0, 3.0, 0.0, 3.0]]))

# A straight horizontal line at y = 1.5.
line = el.segpath(np.array([[-0.5, 3.5],
                            [1.5, 1.5]]))

points, zig_segments, line_segments = el.path_intersections(zigzag, line)
print(f'{points.shape[1]} crossings')
print('points        :', np.round(points, 4).tolist())
print('zigzag segment:', zig_segments)
print('line segment  :', line_segments)
```

Each crossing lies at `y = 1.5`, once on each of the zig-zag's three segments —
which the segment indices confirm:

```{code-cell}
print('all crossings at y = 1.5:', bool(np.allclose(points[1], 1.5)))
for k in range(points.shape[1]):
    seg = np.asarray(zigzag.coords)[:, zig_segments[k]:zig_segments[k] + 2]
    print(f'crossing {k} at {np.round(points[:, k], 3).tolist()} '
          f'on zigzag segment {zig_segments[k]} from '
          f'{np.round(seg[:, 0], 2).tolist()} to {np.round(seg[:, 1], 2).tolist()}')
```

## Two paths that miss

A crossing is a precise condition, so two paths that pass near each other
without meeting report nothing. Here one path sits entirely above the other:

```{code-cell}
lower = el.segpath(np.array([[0.0, 1.0],
                             [0.0, 0.0]]))
upper = el.segpath(np.array([[0.0, 1.0],
                             [1.0, 1.0]]))

none, _, _ = el.path_intersections(lower, upper)
print('crossings between parallel horizontal paths:', none.shape[1])
```

:::{admonition} Why the answer is approximate
:class: tip
The paths are stored as straight segments, so the crossing is found by
segment-against-segment tests. If a curve is represented coarsely, the crossing
is located on the coarse segments rather than on the true curve — a finer
`SegPath` gives a more accurate answer. This is the same trade-off the
[spheres page](../primitives/spheres.md) shows for surface area.
:::

## Figure

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

euclib_viz.show2d(name='path-path',
                  lines=[np.asarray(zigzag.coords), np.asarray(line.coords)],
                  points=points)
```

:::{admonition} Provenance
:class: note
**Upstream:** `trimesh/trimesh` — `tests/test_intersect.py`, function
`test_line_line` (line 6).

**Pinned revision:** [`fcf660f`](https://github.com/trimesh/trimesh/blob/fcf660feb0a14c68fd3945789e8ed77e260f9167/tests/test_intersect.py#L6).

**How this was adapted.** The upstream test calls trimesh's `line_line` on two
lines in 2D, in 3D, and in 3D with a plane normal supplied, and checks the
meeting point each time — including a case where the wrong normal makes the two
lines count as missing each other. No code is copied. The adaptation keeps the
line-against-line test and the "they may simply not meet" case, but raises the
unit of work from two infinite lines to two polylines: `euclib.path_intersections`
tests every segment pair and returns the segment indices, so the page can show
*where on each curve* the crossing happened, which a bare line-line predicate
does not. The upstream `plane_normal` variant is not reproduced, because
`euclib` defines a path's segments directly and has no separate plane to
disambiguate a 3D near-miss.
:::

:::{seealso}
- [Where a path crosses a surface](path-mesh.md) for crossings against a mesh.
- [Where two surfaces meet](mesh-mesh.md) for the two-dimensional analogue.
:::
