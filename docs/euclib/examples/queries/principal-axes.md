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
# Principal Axes and Oriented Bounding Boxes

:::{admonition} Warning: This page was written with AI assistance.
:class: warning

This page was written using AI assistance and has not yet been carefully
reviewed by a human. Its code runs and its claims are checked against the test
suite, but a person has not read it end to end. Treat it as an unreviewed draft:
verify the behaviour you depend on against the API reference and the tests, and
report anything that looks wrong.
:::

`euclib.principal_axes` finds the directions a set of points is stretched
along, and `euclib.oriented_bounding_box` wraps those directions in the
smallest box whose faces follow them. Both are the same decomposition the line
and plane fits use --- the singular value decomposition of the centred
positions --- which is why the first axis is the direction `fit_line` reports
and the last is the normal `fit_plane` reports.

:::{admonition} What this demonstrates
:class: tip
- `euclib.principal_axes(points)`, which returns the mean of the points and a
  matrix whose columns are the directions they spread along, ordered from most
  to least.
- `euclib.oriented_bounding_box(points)`, which returns a box's center, axes,
  and extents.
- That the axes are the ones the line and plane fits report, seen together.
- That the axes form a rotation rather than a reflection, so a caller can use
  them as one.
- That every point of a set lies inside its box, and that the box recovers the
  box a set of corners was drawn from.
- That a set of points that does not determine its axes is refused rather than
  given an arbitrary frame.
:::

## The directions a cloud is stretched along

A cloud of two thousand points drawn in a frame turned 30° about the `z` axis
and stretched by 3, 1.5, and 0.6 along the frame's own axes.

```{code-cell}
import numpy as np
import euclib as el

rng = np.random.default_rng(7)
turn = np.deg2rad(30.0)
rotation = np.array([[np.cos(turn), -np.sin(turn), 0.0],
                     [np.sin(turn),  np.cos(turn), 0.0],
                     [0.0, 0.0, 1.0]])
stretch = np.array([3.0, 1.5, 0.6])
centre = np.array([1.0, 2.0, 3.0])
cloud = rotation @ (stretch[:, None] * rng.normal(size=(3, 2000))) \
    + centre[:, None]

(origin, axes) = el.principal_axes(cloud)
print('origin:', origin.round(3))
print('axes:\n', axes.round(3))
```

The axes are that frame, to within the sampling noise of two thousand points.
Each is a column of the matrix, and they are ordered by how far the points
reach along them --- which is also how the stretches were ordered.

```{code-cell}
print('the axes against the frame they were drawn in:\n',
      np.abs(axes.T @ rotation).round(3))
print('the spread along each axis:',
      np.std(axes.T @ (cloud - origin[:, None]), axis=1).round(3))
```

The spreads are the stretches the cloud was drawn with, 3, 1.5, and 0.6, and
the matrix of dot products is the identity: the frame was recovered, not
merely approximated by another one.

## The same axes the fits report

`fit_line` takes the direction of greatest spread and `fit_plane` the direction
of least, so the first and last of these axes are what those two operations
report --- one looking along the line and one looking across the plane.

```{code-cell}
(_, direction) = el.fit_line(cloud)
(_, normal) = el.fit_plane(cloud)
print('the first axis is the line\'s direction:',
      round(abs(float(axes[:, 0] @ direction)), 6))
print('the last axis is the plane\'s normal   :',
      round(abs(float(axes[:, 2] @ normal)), 6))
```

A sign is compared as an absolute value because the fits and the axes each
choose one on their own; the *direction* is what they agree on.

## The box around the points

A box aligned with those axes is the smallest box whose faces follow the
directions the points themselves suggest. It is returned as a center, the same
axes, and one extent per axis --- the full width along it.

```{code-cell}
# The eight corners of a box 4 by 2 by 1, turned by the same rotation.
half = np.array([2.0, 1.0, 0.5])
signs = np.array([[sx, sy, sz]
                  for sx in (-1.0, 1.0)
                  for sy in (-1.0, 1.0)
                  for sz in (-1.0, 1.0)]).T
corners = centre[:, None] + rotation @ (signs * half[:, None])

(box_center, box_axes, extents) = el.oriented_bounding_box(corners)
print('center :', box_center.round(3))
print('extents:', extents.round(3))
print('axes:\n', box_axes.round(3))
```

The box is the one the corners came from: its extents are 4, 2, and 1, and its
axes are the rotation's columns. Every point is inside it, which is the
statement that each point reaches no farther than half an extent along each
axis.

```{code-cell}
offsets = box_axes.T @ (corners - box_center[:, None])
print('how far the farthest point reaches:',
      np.abs(offsets).max(axis=1).round(3))
print('half of each extent:              ', (extents / 2).round(3))
```

:::{admonition} It is not the smallest box of any orientation
:class: note
An oriented bounding box follows the points' principal axes; it is the
smallest box *among those*, and a smaller box turned away from the axes can
exist. Finding that one is a different problem --- it is a search over
orientations, not a decomposition. The name is exact: the box is oriented
along the points' own axes.

A second thing to know is that the axes are not unique when two spreads are
equal. A cube's corners, a square's, or a cloud drawn with isotropic noise has
no preferred frame, so the axes it reports are one of many equally good ones.
It reports a right-handed frame either way.
:::

## A box in two dimensions

A plane is a hyperplane, so the same operation works in two dimensions, where
it returns a rectangle.

```{code-cell}
rectangle = np.array([[0.0, 4.0, 4.0, 0.0], [0.0, 0.0, 2.0, 2.0]])
tilt = np.deg2rad(35.0)
tilted = np.array([[np.cos(tilt), -np.sin(tilt)],
                   [np.sin(tilt), np.cos(tilt)]]) @ rectangle

(center2, axes2, extents2) = el.oriented_bounding_box(tilted)
print('center :', center2.round(3))
print('extents:', extents2.round(3))
print('axes:\n', axes2.round(3))
```

The extents are the rectangle's own sides, 4 and 2, and the axes are the
rotation's.

## When the points do not determine the axes

A single point spreads nowhere, so no direction is better than any other. Three
points on a line fix one axis and leave the other two free: any pair orthogonal
to the line and to each other would do. Neither is answered with an invented
frame.

```{code-cell}
for (name, points) in (
        ('a single point', np.array([[1.0], [2.0]])),
        ('three collinear points',
         np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]))):
    try:
        el.principal_axes(points)
        print(name, 'was analyzed')
    except ValueError as error:
        print(name, 'is refused:', error)
```

:::{admonition} Tensors and gradients
:class: note
Both operations accept a PyTorch tensor and return tensors, computed with
`torch.linalg.svd` and with tensor maxima and minima, so a gradient flows back
to the coordinates. As with the fits, the derivative of a singular vector is
not defined when two spreads are equal --- a rotationally symmetric cloud, for
instance --- because that is a property of the decomposition. The answer is
still computed; it is the gradient that is undefined there.
:::

## Provenance

Adapted from pyvista's
[`test_utilities.py::test_principal_axes`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_utilities.py#L1625)
and
[`test_dataset_filters.py::test_oriented_bounding_box`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_dataset_filters.py#L4807),
with
[`::test_bounding_box_return_meta`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_dataset_filters.py#L4817).

The upstream principal-axes test pins the axes of four small point sets and
checks that they are unit vectors and that the frame is right-handed. It skips
several of those cases on some NumPy versions and on ARM Macs, which is the
sign and ordering of a decomposition varying between LAPACK builds. `euclib`
makes both explicit: the axes come back ordered by spread and as a right-handed
frame, and a set that does not determine them is refused --- pyvista answers an
empty or single-point set with the identity axes instead. pyvista also reports
the axes as the *rows* of its matrix and offers the spreads alongside them;
`euclib` returns them as columns, one per axis, and leaves the spreads to the
singular values a caller can ask for from `euclib.fit_line` and
`euclib.fit_plane`.

The bounding-box tests turn a `pv.Cube` and check that the fitted box has the
same bounds, and that its axes agree with `principal_axes` up to sign. `euclib`
returns the box's parameters rather than a mesh, since a caller who wants the
box's geometry can build it from the center, the axes, and the extents; the
check made here, that every corner reaches exactly half an extent along each
axis, is the same property the upstream bounds check states.
