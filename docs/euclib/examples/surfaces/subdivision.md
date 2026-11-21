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
# Subdividing a Mesh and Simplifying a Path

:::{admonition} Warning: This page was written with AI assistance.
:class: warning

This page was written using AI assistance and has not yet been carefully
reviewed by a human. Its code runs and its claims are checked against the test
suite, but a person has not read it end to end. Treat it as an unreviewed draft:
verify the behaviour you depend on against the API reference and the tests, and
report anything that looks wrong.
:::

Two operations move a geometry's detail in opposite directions.
`euclib.subdivide` replaces every triangle of a mesh with four, adding corners
where a drawing or a simulation needs more of them, without changing the
surface the mesh describes. `euclib.merge_colinear` removes the corners of a
path that lie on a straight run, leaving the curve the same with fewer
segments.

:::{admonition} What this demonstrates
:class: tip
- `el.subdivide(mesh)` on a closed triangle mesh, and that its area, volume,
  and closure are unchanged while its triangle count is multiplied by four.
- That `iterations` repeats the refinement and `max_edge` chooses enough
  repetitions to reach a length.
- That edge midpoints are shared between the triangles that meet at an edge, so
  the refined surface has no cracks.
- `el.merge_colinear(path)`, which replaces a straight run of segments with one
  segment, and leaves a bend, a reversal, a junction, and a distinct curve
  alone.
- That a tolerance decides how far a corner may lie from the line through its
  neighbours and still count as straight.
:::

## Subdividing a cube

The cube is the one the [cube page](../primitives/cube.md) builds. Its twelve
triangles become forty-eight, and its eight corners become twenty-six: one for
each old corner and one for each of its eighteen edges, because a shared edge
gets one midpoint rather than one per triangle.

```{code-cell}
import numpy as np
import euclib as el

def cube(low=(0.0, 0.0, 0.0), high=(1.0, 1.0, 1.0)):
    pattern = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
               (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    coords = np.array([[low[k] + bit[k] * (high[k] - low[k])
                        for k in range(3)] for bit in pattern], float).T
    faces = np.array([[0, 4, 1], [0, 2, 4], [3, 5, 7], [3, 7, 6],
                      [0, 1, 5], [0, 5, 3], [2, 7, 4], [2, 6, 7],
                      [0, 3, 6], [0, 6, 2], [1, 4, 7], [1, 7, 5]]).T
    return el.trimesh(coords, faces)

def volume(mesh):
    coords = np.asarray(mesh.coords)
    corners = np.asarray(mesh.topo.indices)
    total = 0.0
    for k in range(corners.shape[1]):
        a = coords[:, corners[0, k]]
        b = coords[:, corners[1, k]]
        c = coords[:, corners[2, k]]
        total += float(a @ np.cross(b - a, c - a))
    return total / 6.0

def closed(mesh):
    corners = np.asarray(mesh.topo.indices)
    seen = {}
    for k in range(corners.shape[1]):
        a, b, c = (int(v) for v in corners[:, k])
        for edge in ((a, b), (b, c), (c, a)):
            seen[edge] = seen.get(edge, 0) + 1
    return all(n == 1 and (e[1], e[0]) in seen for (e, n) in seen.items())

block = cube()
fine = el.subdivide(block)
print('corners  :', block.coord_count, '->', fine.coord_count)
print('triangles:', block.topo.simplex_count[2], '->',
      fine.topo.simplex_count[2])
print('area     :', round(float(np.sum(block.measures)), 6), '->',
      round(float(np.sum(fine.measures)), 12))
print('volume   :', round(volume(block), 6), '->', round(volume(fine), 12))
print('closed   :', closed(fine))
```

The area and volume are unchanged to the last decimal the printout shows. Each
child triangle has a quarter of its parent's area, and there are four of them;
the volume is the same sum over the same surface, cut into smaller pieces. The
midpoints are computed as exact averages of the corners, so a shared edge's
midpoint is one position used by both of its triangles --- which is why the
surface still closes.

## Repetitions and a maximum edge

`iterations` refines every triangle again, multiplying the triangle count by
four each time. A `max_edge` asks instead for a longest edge, and the operation
takes enough uniform repetitions to reach it.

```{code-cell}
for iterations in range(4):
    refined = el.subdivide(block, iterations=iterations)
    print(f'{iterations} iterations: {refined.topo.simplex_count[2]:>4} '
          f'triangles, {refined.coord_count:>4} corners')

short = el.subdivide(block, max_edge=0.4)
edges = []
for k in range(short.topo.simplex_count[2]):
    (a, b, c) = (int(v) for v in np.asarray(short.topo.indices)[:, k])
    for (u, v) in ((a, b), (b, c), (c, a)):
        step = np.asarray(short.coords)[:, u] - np.asarray(short.coords)[:, v]
        edges.append(float(np.sqrt(step @ step)))
print('longest edge under 0.4:', round(max(edges), 6))
print('area                    :', round(float(np.sum(short.measures)), 12))
```

The default one repetition already halves the longest edge, from
$\sqrt{2}$ to $\sqrt{2}/2 \approx 0.7071$; a second brings it to about 0.3536,
under the 0.4 asked for. The refinement is uniform: every face is split the
same number of times, even one whose own edges were already short, which is
what lets the result stay conforming without a second pass.

## Simplifying a path

A path sampled along a straight line carries a corner at every sample. None of
them changes the curve, and `merge_colinear` drops them.

```{code-cell}
line = el.segpath(np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]]))
simple = el.merge_colinear(line)
print('corners  :', line.coord_count, '->', simple.coord_count)
print('segments :', line.topo.simplex_count[1], '->',
      simple.topo.simplex_count[1])
print('length   :', round(float(np.sum(simple.measures)), 12))
```

The length is the same four units. A bend is not removed, because the three
corners around it do not lie on one line; neither is a reversal, where the path
doubles back and the middle corner is a turning point rather than a redundant
one.

```{code-cell}
bend = el.segpath(np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]))
spike = el.segpath(np.array([[0.0, 2.0, 1.0], [0.0, 0.0, 0.0]]))
print('bend  :', el.merge_colinear(bend).topo.simplex_count[1], 'segments')
print('spike :', el.merge_colinear(spike).topo.simplex_count[1], 'segments')
```

## Loops and junctions

A closed loop with a corner in the middle of each side simplifies back to its
four sides: the midpoints are removed and the corners stay. A junction, where
three segments meet at one corner, is left alone, because that corner is not an
interior point of a single curve.

```{code-cell}
square = np.array([[0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5]])
loop = el.SegPath(square, el.SegTopology(
    np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]]),
    coord_count=8))
round_loop = el.merge_colinear(loop)
print('loop    :', loop.topo.simplex_count[1], '->',
      round_loop.topo.simplex_count[1], 'segments,',
      'length', round(float(np.sum(round_loop.measures)), 12))

star = el.SegPath(np.array([[0.0, 1.0, -1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]]),
                  el.SegTopology(np.array([[0, 0, 0], [1, 2, 3]]),
                                 coord_count=4))
print('junction:', el.merge_colinear(star).topo.simplex_count[1], 'segments')
```

The loop keeps four corners and its perimeter is still four. The junction keeps
all three segments: the library cannot know which pair of them is the curve
that continues through the center, so it does not guess.

## A tolerance

A corner that is almost on the line counts as on it when the tolerance is wide
enough. The default tolerance is a small fraction of the path's extent, and the
keyword replaces it.

```{code-cell}
almost = el.segpath(np.array([[0.0, 1.0, 2.0], [0.0, 0.001, 0.0]]))
print('wide   :', el.merge_colinear(almost, tolerance=0.01)
      .topo.simplex_count[1], 'segment')
print('narrow :', el.merge_colinear(almost, tolerance=1e-6)
      .topo.simplex_count[1], 'segments')
```

:::{admonition} What subdivision does not carry
:class: note
Subdivision returns the refined geometry, not the input's properties: a
property attached to the old corners has no value at the new midpoints until a
caller supplies one, and `euclib` does not invent one. Attach properties to the
result, or transfer them from the old mesh with `euclib.ops.transfer`, which
interpolates them at the new positions. Both operations read tensor coordinates
and return NumPy arrays, and both carry an input's units through to the result.
:::

## Provenance

Adapted from trimesh's
[`tests/test_remesh.py::test_subdivide`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_remesh.py#L8)
and
[`tests/test_simplify.py::test_merge_colinear`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_simplify.py#L80).

The upstream subdivision test runs on three meshes --- a soup of random
triangles, a multi-body mesh, and a single-body mesh --- and checks that the
area is unchanged, that the face count grows, that `subdivide_to_size` leaves
every edge under the requested length, that every face has a parent index, and
that each new corner lies inside its parent triangle by barycentric
coordinates. The page makes the first three checks on a cube, where the
volume, the corner count, and the closure can be stated exactly as well.
The upstream simplification test builds a hundred collinear points and checks
that `merge_colinear` reduces them to two; the page does the same on a
five-point line and adds the cases the upstream test does not: a bend, a
reversal, a junction, and a closed loop.

The differences are in the interface. trimesh's `subdivide` returns vertices,
faces, and optionally a parent index; `euclib.subdivide` returns a `TriMesh`,
so the result is a geometry that can be measured, drawn, or fed into another
operation, and the parent index is not reported. trimesh's `subdivide_to_size`
refines only the faces whose edges are too long and then conforms the
neighbouring faces; `euclib` refines every face the same number of times,
which can add more triangles than the adaptive method but needs no conformity
pass. trimesh's `merge_colinear` operates on an array of points, while
`euclib.merge_colinear` operates on a `SegPath` and returns one, so the result
keeps its topology and units.
