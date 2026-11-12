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
# Clipping and Sectioning a Volume

:::{admonition} Warning: This page was written with AI assistance.
:class: warning

This page was written using AI assistance and has not yet been carefully
reviewed by a human. Its code runs and its claims are checked against the test
suite, but a person has not read it end to end. Treat it as an unreviewed draft:
verify the behaviour you depend on against the API reference and the tests, and
report anything that looks wrong.
:::

A plane divides space in two, and it divides a solid into two solids.
`euclib.clip` returns the part on one side, closed with a flat cap where the
plane cut it, so the answer is a solid like the input. `euclib.section` returns
the curve where the plane meets the surface, as a `SegPath` whose segments meet
at shared corners. The two are asked from the same plane --- an origin and a
normal --- but they answer different questions, and only one of them needs the
surface to enclose anything.

:::{admonition} What this demonstrates
:class: tip
- `el.clip(mesh, origin, normal)` on a closed triangle mesh, and the volume and
  surface the clipped solid has.
- `el.section(mesh, origin, normal)` on the same mesh, and the length of the
  curve it returns.
- That the two halves of a cut solid add back up to the whole, and that each
  half is a closed surface with Euler characteristic 2.
- That the plane's normal names the kept side, and that a plane which misses the
  mesh leaves it whole or leaves nothing.
- That a section is defined for an open surface even though a clip is not, and
  that a clip of an open surface is refused with a message that says why.
- That a plane lying in a face finds that face's boundary, while the triangles
  in the plane itself contribute no segments.
:::

## A cube cut in half

The cube is the one the [cube page](../primitives/cube.md) builds, wound so that
its triangles face outward. Clipping it at `z = 0.5` with a normal of `+z` keeps
the upper half; the plane's normal points into the half that is kept.

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

block = cube()
top = el.clip(block, np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 1.0]))
print('volume   :', round(volume(top), 4))
print('corners  :', top.coord_count)
print('triangles:', top.topo.simplex_count[2])
```

The volume is half the cube's, 0.5. The corner and triangle counts are larger
than half of the cube's because the cut does not fall along existing edges: it
passes through the middle of the four side faces, so those faces are split into
new triangles and the cap adds its own.

## The two halves make the whole

Clipping the same cube with the opposite normal keeps the other half. The two
solid volumes must add back to the cube's.

```{code-cell}
bottom = el.clip(block, np.array([0.0, 0.0, 0.5]),
                 np.array([0.0, 0.0, -1.0]))
print('top half   :', round(volume(top), 6))
print('bottom half:', round(volume(bottom), 6))
print('together   :', round(volume(top) + volume(bottom), 12))
print('the cube   :', round(volume(block), 6))
```

This identity is the sharpest check of a clip's volume, because the two halves
share every cut corner but no volume: an error in the cap of one would leave the
other unable to make up for it.

## A closed and outward-wound surface

A clip is built as an intersection of the solid with a half-space, and it
repairs the input's winding on the way, so the result is a solid the boolean
operations can take as an input. Two checks from the triangles alone say so:
every directed edge is matched by a reverse edge, and Euler's formula for a
surface without holes is $V - E + F = 2$.

```{code-cell}
def closed(mesh):
    corners = np.asarray(mesh.topo.indices)
    seen = {}
    for k in range(corners.shape[1]):
        a, b, c = (int(v) for v in corners[:, k])
        for edge in ((a, b), (b, c), (c, a)):
            seen[edge] = seen.get(edge, 0) + 1
    return all(n == 1 and (e[1], e[0]) in seen for (e, n) in seen.items())

def euler(mesh):
    corners = np.asarray(mesh.topo.indices)
    edges = set()
    for k in range(corners.shape[1]):
        a, b, c = (int(v) for v in corners[:, k])
        for (u, v) in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    return mesh.coord_count - len(edges) + corners.shape[1]

print('closed :', closed(top))
print('euler  :', euler(top))
print('area   :', round(float(np.sum(top.measures)), 4))
```

The area is 4: the cap is a unit square, the top face is a unit square, and the
four side faces remain as halves of unit squares.

## Cutting at an angle

Nothing requires the plane to be a coordinate plane. A plane through the cube's
center with a normal of $(1, 1, 1)$ cuts it in half, and the section is a regular
hexagon.

```{code-cell}
origin = np.array([0.5, 0.5, 0.5])
normal = np.array([1.0, 1.0, 1.0])
tilted = el.clip(block, origin, normal)
path = el.section(block, origin, normal)
print('volume      :', round(volume(tilted), 6))
print('closed      :', closed(tilted))
print('section     :', path.topo.simplex_count[1], 'segments,',
      round(float(np.sum(path.measures)), 6), 'long')
print('3 sqrt(2)   :', round(3.0 * np.sqrt(2.0), 6))
```

The hexagon's side is $1/\sqrt{2}$, so its perimeter is $6/\sqrt{2} = 3\sqrt{2}$.
The normal need not be a unit vector: `clip` and `section` normalize it, so the
same plane can be named by any nonzero multiple of its normal.

## A sphere's equator

On a curved surface every latitude is a different polygon, and the cut's cap is
a polygon too. The sphere is the one the
[sphere page](../primitives/spheres.md) builds, at eight latitudes and twelve
longitudes.

```{code-cell}
def sphere(n_lat=8, n_lon=12):
    lat = np.linspace(0.0, np.pi, n_lat + 1)
    lon = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    coords = np.array([[np.sin(t) * np.cos(p), np.sin(t) * np.sin(p),
                        np.cos(t)] for t in lat for p in lon]).T
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + (j + 1) % n_lon
            faces += [[a, b, d], [a, d, c]]
    return el.trimesh(coords, np.array(faces).T)

ball = sphere()
half = el.clip(ball, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
equator = el.section(ball, np.array([0.0, 0.0, 0.0]),
                     np.array([0.0, 0.0, 1.0]))
print('the sphere        :', round(volume(ball), 4), 'signed')
print('a half            :', round(volume(half), 6))
print('half the sphere   :', round(abs(volume(ball)) / 2.0, 6))
print('equator           :', equator.coord_count, 'corners,',
      round(float(np.sum(equator.measures)), 6), 'long')
print('twelve chords     :',
      round(12.0 * 2.0 * np.sin(np.pi / 12.0), 6))
print('half area + cap   :', round(float(np.sum(ball.measures)) / 2.0
                                  + 12.0 / 2.0 * np.sin(2.0 * np.pi / 12.0),
                                  6))
print('clipped area      :', round(float(np.sum(half.measures)), 6))
```

The sphere's signed volume is negative, because the [sphere
page](../primitives/spheres.md)'s triangles wind into the sphere rather than out
of it. The clip turns them the right way round before cutting, which is why the
half's volume is positive. Its surface area is half the sphere's plus the cap:
the cap is a regular twelve-sided polygon of area $12/2 \sin(\pi/6) = 3$, and
the equator section is the same polygon's boundary, twelve chords of the unit
circle.

## A plane that misses

A plane can pass entirely outside the solid, or entirely beyond it with nothing
left on its kept side. Neither is an error, and neither builds a mesh: the whole
solid comes back unchanged, or an empty mesh does.

```{code-cell}
whole = el.clip(block, np.array([0.0, 0.0, -1.0]),
                np.array([0.0, 0.0, 1.0]))
none = el.clip(block, np.array([0.0, 0.0, 2.0]),
               np.array([0.0, 0.0, 1.0]))
missed = el.section(block, np.array([0.0, 0.0, 2.0]),
                    np.array([0.0, 0.0, 1.0]))
print('the mesh itself :', whole is block)
print('empty clip      :', none.coord_count, none.topo.simplex_count[2])
print('empty section   :', missed.coord_count, missed.topo.simplex_count[1])
```

`whole is block` is `True` rather than merely equal: when there is nothing to
cut, there is nothing to rebuild.

## A plane lying in a face

The bottom face of the cube lies in the plane `z = 0`. Its two triangles meet
the plane over an area, not along a curve, so they contribute no segments. The
four side faces do: each meets the plane along the bottom edge of the face.

```{code-cell}
edge = el.section(block, np.array([0.0, 0.0, 0.0]),
                  np.array([0.0, 0.0, 1.0]))
print('segments:', edge.topo.simplex_count[1])
print('length  :', round(float(np.sum(edge.measures)), 6))
print('all in the plane:',
      bool(np.allclose(np.asarray(edge.coords)[2], 0.0)))
```

The answer is the square's boundary, four unit segments, even though the square
itself is not part of it.

## Sections of open surfaces

A section does not need the surface to enclose anything: a plane crossing a
sheet crosses it in a curve whether or not the sheet has an inside. A clip does
need it, because the part of an open surface on one side of a plane is not a
solid.

```{code-cell}
sheet = el.trimesh(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                             [0.0, 0.0, 1.0]]),
                   np.array([[0], [1], [2]]))
line = el.section(sheet, np.array([0.0, 0.0, 0.5]),
                  np.array([0.0, 0.0, 1.0]))
print('section length:', round(float(np.sum(line.measures)), 4))
try:
    el.clip(sheet, np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 1.0]))
except ValueError as error:
    print('clip refuses  :', error)
```

The section is the segment halfway up the triangle, of length 0.5. The clip's
refusal names the property that failed --- the surface does not close --- rather
than returning a wrong solid.

:::{admonition} Tensors and units
:class: note
Both operations accept coordinates that are PyTorch tensors, and both return
NumPy arrays: which triangles the plane crosses, and where it crosses them, is a
combinatorial answer with no derivative to carry. An input's units, on the other
hand, are carried through --- a mesh measured in meters gives a clipped mesh in
meters and a section whose lengths are meters --- while the axes and the plane's
normal remain unitless directions.
:::

:::{admonition} About the tolerance
:class: note
A corner that lies within a small distance of the plane counts as lying on it.
The default distance is a small fraction of the mesh's extent, and the
`tolerance` keyword overrides it for callers who know their coordinates' scale.
It affects whether a corner is treated as on the plane, not where the plane is.
:::

## Provenance

Adapted from pyvista's
[`tests/core/test_dataset_filters.py::test_clip_closed_surface`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_dataset_filters.py#L447)
and from trimesh's
[`tests/test_section.py::test_section`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_section.py#L8)
and
[`::test_slice`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_section.py#L181),
with
[`::test_slice_onplane`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_section.py#L263).

The pyvista test clips a closed sphere and asserts that the result has no open
edges, then clips the result with the uncapped operation and asserts that
clipping *that* open surface is refused. Both checks are made here: the clipped
half is checked to be closed, edge by edge, and an open sheet is checked to be
refused. The trimesh tests section a mesh at many heights and at slightly
rotated orientations, and slice a box at a corner and at a face, checking the
cut-off solid's bounds, its face count, its watertightness, and that the two
halves' volumes add to the original's. The pages make the same checks on a cube
and on a sphere, where the cut's area and the section's length can also be
written down in closed form.

The difference is in how the cap is made. pyvista delegates clipping to VTK and
trimesh caps with an ear-clipping triangulator; `euclib` has no external CSG
engine, so a clip is built as an intersection of the mesh with a box that has
one face in the cutting plane and extends only over the kept side. That reuses
the boolean implementation, its winding repair, and its welding of cut corners,
which is why the sphere's inward-wound triangles are accepted. trimesh's
`section` returns a path in the *2D* plane by default and offers
`to_planar()`; `euclib.section` returns the segments in the mesh's own
three-dimensional space, with the segment ends welded, and leaves the projection
to the caller.
