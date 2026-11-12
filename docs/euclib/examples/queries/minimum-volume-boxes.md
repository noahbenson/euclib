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
# Minimum-Volume Oriented Bounding Boxes

:::{admonition} Warning: This page was written with AI assistance.
:class: warning

This page was written using AI assistance and has not yet been carefully
reviewed by a human. Its code runs and its claims are checked against the test
suite, but a person has not read it end to end. Treat it as an unreviewed draft:
verify the behaviour you depend on against the API reference and the tests, and
report anything that looks wrong.
:::

`euclib.minimum_volume_bounding_box` finds the smallest box of *any*
orientation that holds every point. It is a different question from the one
`euclib.oriented_bounding_box` answers: that box follows the points' principal
axes, and its faces can be turned away from the orientation that would make the
box smaller. The minimum-volume box is the one whose volume cannot be reduced by
turning it, and for a shape that is not aligned with its principal axes the
difference is measurable.

:::{admonition} What this demonstrates
:class: tip
- `el.minimum_volume_bounding_box(points)`, which returns a center, a frame of
  axes, and one extent per axis, for points in two or three dimensions.
- The difference between the minimum-volume box and the principal-axis box on a
  triangular prism, where the minimum is the prism's true box and the principal
  box is slightly larger.
- That every point lies inside the returned box, that the axes are unit vectors
  forming a right-handed frame, and that the box turns with the points.
- That a set of points which does not span its space is refused rather than
  given a flat box.
:::

## A triangular prism

The prism's cross-section is a right triangle with legs 4 and 1, extruded one
unit along `z`. Its true bounding box is 4 by 1 by 1, so its volume is exactly
4. The principal axes of its six corners do not line up with those sides: the
triangle pulls the frame diagonally.

```{code-cell}
import numpy as np
import euclib as el

lower = np.array([[0.0, 4.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
upper = lower + np.array([[0.0], [0.0], [1.0]])
points = np.concatenate([lower, upper], axis=1)

(center, axes, extents) = el.minimum_volume_bounding_box(points)
(pcenter, paxes, pextents) = el.oriented_bounding_box(points)
print('minimum  : center', center.round(3), 'extents', extents.round(3),
      'volume', round(float(np.prod(extents)), 4))
print('principal: center', pcenter.round(3), 'extents', pextents.round(3),
      'volume', round(float(np.prod(pextents)), 4))
```

The minimum-volume box reports the prism's own extents, 4, 1, and 1. The
principal-axis box is turned a few degrees: one of its in-plane extents reaches
farther than the prism's 4, the other falls just short of 1, and the product is
4.0613 --- a little over one percent larger. That gap is not an approximation:
the minimum-volume box is the smallest one, and the principal box is a good
guess that this shape defeats.

## Inside the box

A point is inside when its offset from the center reaches no farther than half
an extent along each axis. The axes are unit vectors, and they are ordered by
descending extent and wound right-handed.

```{code-cell}
offsets = axes.T @ (points - center[:, None])
print('lowest reach :', offsets.min(axis=1).round(3))
print('highest reach:', offsets.max(axis=1).round(3))
print('half extents :', (extents / 2.0).round(3))
print('axis lengths :', np.linalg.norm(axes, axis=0).round(3))
print('right-handed :', round(float(np.linalg.det(axes)), 6))
```

The reaches are exactly half the extents on the low and high sides of every
axis, because the box is defined by the points that reach farthest. The
determinant is 1, so the frame is a rotation rather than a reflection.

## A box that is already a box

A set of eight corners of an axis-aligned box 4 by 2 by 1, centered at
$(1, 2, 3)$, is its own minimum-volume box. The extents come back ordered by
descending extent rather than in the order of the coordinate axes.

```{code-cell}
box = np.array([[x, y, z] for x in (1.0 - 2.0, 1.0 + 2.0)
                           for y in (2.0 - 1.0, 2.0 + 1.0)
                           for z in (3.0 - 0.5, 3.0 + 0.5)]).T
(c, a, e) = el.minimum_volume_bounding_box(box)
print('center :', c.round(3))
print('extents:', e.round(3))
print('volume :', round(float(np.prod(e)), 6))
```

The center is the box's center and the extents are 4, 2, and 1, so the volume is
8. There is nothing to improve on: a box around a box cannot be smaller than the
box itself.

## A box that has been turned

Turning the points turns the box with them, and the box recovers the turn. The
rotation here is a random proper rotation, fixed by a seed so that the page's
output is reproducible.

```{code-cell}
rng = np.random.default_rng(5)
(q, r) = np.linalg.qr(rng.normal(size=(3, 3)))
q = q @ np.diag(np.sign(np.diag(r)))
if np.linalg.det(q) < 0:
    q[:, -1] = -q[:, -1]
turned = q @ (box - np.array([[1.0], [2.0], [3.0]])) \
    + np.array([[1.0], [2.0], [3.0]])
(c, a, e) = el.minimum_volume_bounding_box(turned)
print('center :', c.round(3))
print('extents:', e.round(3))
print('axes against the turn:\n', np.abs(a.T @ q).round(3))
print('right-handed:', round(float(np.linalg.det(a)), 6))
```

The extents are still 4, 2, and 1, and the matrix of dot products between the
box's axes and the rotation is the identity: the box found the rotation that was
applied. That is the property the upstream test checks by transforming a mesh
into its box's frame and back again.

## A box in two dimensions

In a plane the box is the smallest rectangle, and its "volume" is an area. The
same search applies, with the hull's edges as the candidate directions. An
L-shaped set of points is a shape the principal axes misjudge: it pulls the frame
toward the diagonal, while the smallest rectangle is the 2 by 2 square the L was
drawn inside.

```{code-cell}
ell = np.array([[0.0, 2.0, 2.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]])
(c2, a2, e2) = el.minimum_volume_bounding_box(ell)
(p2, pa2, pe2) = el.oriented_bounding_box(ell)
print('minimum  : center', c2.round(3), 'extents', e2.round(3),
      'area', round(float(np.prod(e2)), 4))
print('principal: center', p2.round(3), 'extents', pe2.round(3),
      'area', round(float(np.prod(pe2)), 4))
print('right-handed:', round(float(np.linalg.det(a2)), 6))
```

The minimum rectangle's area is 4; the principal-axis rectangle's is 6, half
again as large, because its axes follow the L's diagonal rather than its sides.

## How the box is found

A minimum-volume box always has one face flush with a face of the points'
convex hull: if none were, the box could be turned slightly about an axis to
shrink it. That theorem turns an infinite search over orientations into a
finite one. The operation builds the hull, takes each of its face normals as a
candidate orientation, projects the hull onto the plane of that face, finds the
smallest rectangle around the projection by trying the hull's edge directions,
and keeps the box with the smallest volume. In two dimensions the candidates
are the hull's edges and the rectangle is found directly.

The search is exact in real arithmetic and is carried out in double precision,
with face normals that coincide to within rounding treated as one candidate. It
is also combinatorial: which face's box wins is a decision, so a tensor input is
read but the result is a NumPy array and carries no gradient.

:::{admonition} When the box is not unique
:class: note
A shape with symmetry does not have one minimum-volume box. A cube, a square,
and a regular polygon all admit a family of equally small boxes, and the one
returned is whichever candidate the search reaches first. It is still a true
minimum-volume box; it is just not the only one. The axes are signed so that
each one's largest-magnitude component is positive, ordered by descending
extent, and made right-handed, so the answer is reproducible even when the
choice is arbitrary.
:::

## When the points do not span their space

A box in two dimensions needs points that leave a line, and a box in three needs
points that leave a plane. A flat set would only have a flat box, so it is
refused.

```{code-cell}
for (name, flat) in (
        ('three collinear points',
         np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])),
        ('four coplanar points',
         np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 0.0, 0.0]]))):
    try:
        el.minimum_volume_bounding_box(flat)
        print(name, 'was boxed')
    except ValueError as error:
        print(name, 'is refused:', error)
```

The refusals say which condition failed rather than returning a box with a zero
extent, which would be a two-dimensional answer to a three-dimensional question.

:::{admonition} Points, meshes, and units
:class: note
The operation takes either a `(D, Q)` array of positions or a geometry whose
data positions are used, so a `TriMesh` can be passed directly. The center and
the extents carry the units of the input; the axes are unitless directions
without units, as a frame should be.
:::

## Provenance

Adapted from trimesh's
[`tests/test_bounds.py::test_obb_mesh`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_bounds.py#L27),
with
[`::test_obb_mesh_large`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_bounds.py#L12).

The upstream test walks each mesh in its fixture set through six transforms and
checks two things about the mesh's oriented bounding box each time: the
determinant of the box's rotation is 1, and after transforming the mesh by the
inverse of the box's transform every coordinate lies between $-1$ and $1$ times
half the box's extent. It also checks that applying the box's transform leaves
an axis-aligned bounding box with the same extents. This page checks the same
two properties directly: the determinant is 1, and the offsets of every point
along the box's axes are within half the extents. The last property --- that the
box's frame is the frame the points were drawn in --- is checked by turning a
box by a known rotation and comparing the box's axes with it.

The difference is in the interface and in what "oriented" means. trimesh
attaches the box to a mesh as a 4 by 4 transform and repeats the test after
mutating the mesh; `euclib` returns the box's center, axes, and extents and
leaves the geometry to the caller. And trimesh's test exercises one oriented
box, while `euclib` has two operations that answer different questions: the
principal-axis box, documented on the [principal axes
page](principal-axes.md), and the minimum-volume box on this one. The prism and
the L above are chosen because the two disagree on them, which is what makes the
minimum-volume search worth having.
