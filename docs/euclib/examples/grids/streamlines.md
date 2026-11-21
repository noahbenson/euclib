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
# Integrating a Vector Field into Streamlines

:::{admonition} Warning: This page was written with AI assistance.
:class: warning

This page was written using AI assistance and has not yet been carefully
reviewed by a human. Its code runs and its claims are checked against the test
suite, but a person has not read it end to end. Treat it as an unreviewed draft:
verify the behaviour you depend on against the API reference and the tests, and
report anything that looks wrong.
:::

`euclib.streamlines` follows a velocity field across a grid. Starting from a
seed position, it advances a little at a time along the direction the field
gives there, and returns the curve the seed traces as a `SegPath`. The step is
a distance rather than a quantity of time --- the velocity is normalized at
every stage --- so a fast field and a slow one with the same direction produce
the same curve.

:::{admonition} What this demonstrates
:class: tip
- `el.streamlines(grid, vector, start=...)` on a two-dimensional grid whose
  property is a velocity field.
- That a constant field produces a straight line whose length is the number of
  steps times the step length.
- That the fourth-order rule follows a rotating field more closely than the
  second-order one, and that both keep the radius of a circular streamline.
- That several seeds produce several curves in one `SegPath`, each a connected
  run of segments.
- That the walk stops when it leaves the grid or reaches a position where the
  field vanishes, and that a zero field produces no segments at all.
:::

## A uniform flow

A small grid with a velocity of one unit along `x` at every cell. The seed is
at the left, and each step advances half a unit.

```{code-cell}
import numpy as np
import euclib as el

def grid2(shape=(3, 3), spacing=1.0, origin=(0.0, 0.0, 0.0)):
    affine = np.eye(len(shape) + 1)
    for axis in range(len(shape)):
        affine[axis, axis] = spacing
        affine[axis, -1] = origin[axis]
    return el.grid(shape, affine=affine)

def constant(vector=(1.0, 0.0), shape=(3, 3)):
    values = np.zeros((len(vector),) + tuple(shape))
    values[:] = np.array(vector)[:, None, None]
    return grid2(shape).withprop('v', values)

flow = constant()
path = el.streamlines(flow, 'v', start=np.array([0.0, 1.0]), step=0.5,
                      max_steps=10)
print('corners  :', path.coord_count)
print('segments :', path.topo.simplex_count[1])
print('length   :', round(float(np.sum(path.measures)), 12))
print('last     :', np.asarray(path.coords)[:, -1])
```

The path has six corners and five segments, and its length is 2.5. The grid's
region runs from half a cell before the first center to half a cell past the
last, so the line from `x = 0` can go as far as `x = 2.5`; the sixth step would
leave the grid, and the walk stops there. Every step is the same length, which
is what makes the total exactly five times 0.5.

## Two integrators

A rotating field, `(-y, x)`, on a grid covering `[-2, 2]²`. A particle released
at `(1, 0)` should travel around the unit circle forever. The fourth-order
Runge--Kutta rule follows it closely; the second-order midpoint rule does too,
but with a visible drift over a hundred steps.

```{code-cell}
coordinates = np.linspace(-2.0, 2.0, 21)
(xs, ys) = np.meshgrid(coordinates, coordinates, indexing='ij')
turn = grid2((21, 21), spacing=0.2, origin=(-2.0, -2.0))
turn = turn.withprop('v', np.stack([-ys, xs], axis=0))

for order in (2, 4):
    curve = el.streamlines(turn, 'v', start=np.array([1.0, 0.0]),
                           step=0.1, order=order, max_steps=100)
    offsets = np.asarray(curve.coords)
    radius = np.sqrt((offsets * offsets).sum(axis=0))
    print(f'order {order}: corners {curve.coord_count}, '
          f'length {float(np.sum(curve.measures)):.6f}, '
          f'worst radius error {np.abs(radius - 1.0).max():.2e}')
```

Both rules keep the radius within a small fraction of a unit, and the
fourth-order rule's worst error is about a thousand times smaller. The length
is slightly under 10 for the fourth-order rule rather than exactly ten because
the curve has drifted inward by a part in a million; a streamline is an
approximation to the flow, not an exact solution of it.

## Several seeds

A seed may be a single position or a matrix of them. Each seed traces its own
polyline, and the segments of all of them come back in one `SegPath`. The
segments of one seed are consecutive and share corners; the segments of
different seeds do not.

```{code-cell}
seeds = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 2.0]])
bundle = el.streamlines(flow, 'v', start=seeds, step=0.25, max_steps=200)
indices = np.asarray(bundle.topo.indices)
print('corners  :', bundle.coord_count)
print('segments :', indices.shape[1])
for (low, high, base) in ((0, 10, 0), (10, 20, 11), (20, 30, 22)):
    used = set(indices[:, low:high].reshape(-1))
    print(f'run at {low:>2}: corners {min(used)} to {max(used)}, '
          f'connected={used == set(range(base, base + 11))}')
```

Each seed contributes eleven corners and ten segments. The three runs have
disjoint corner ranges, so a caller can tell which curve a segment belongs to
by the run it falls in --- the same information a drawing would show as three
separate lines.

## Where a walk stops

The walk stops at the first step it cannot complete: the next position would
lie outside the grid, or a stage of the step would land where the field is
zero. A field that vanishes over part of the grid therefore ends the curves
that run into it, and a field that is zero everywhere gives nothing at all.

```{code-cell}
values = np.zeros((2, 3, 3))
values[0, 0, :] = 1.0
one_column = grid2().withprop('v', values)
stopped = el.streamlines(one_column, 'v', start=np.array([0.0, 0.0]),
                         step=0.25, max_steps=200)
print('length past one column  :',
      round(float(np.sum(stopped.measures)), 6))
print('last corner             :', np.asarray(stopped.coords)[:, -1])

still = el.streamlines(constant(vector=(0.0, 0.0)), 'v',
                       start=np.array([1.0, 1.0]))
print('a zero field            :', still.coord_count, 'corners,',
      still.topo.simplex_count[1], 'segments')
```

The velocity is one at the first column and zero at the second, so the field
falls from one to zero across the first cell; the walk stops in the middle of
that fall, before the fourth-order rule can take a step whose stages all lie
where the field is positive.

## Seeds, steps, and direction

Three defaults make the common calls short. The seed is the center of the grid,
the step is half the shortest cell, and the direction is along the field. The
`direction=-1` keyword walks against it.

```{code-cell}
centered = el.streamlines(flow, 'v', step=0.25, max_steps=200)
print('center seed :', np.asarray(centered.coords)[:, 0],
      'length', round(float(np.sum(centered.measures)), 6))
backward = el.streamlines(flow, 'v', start=np.array([2.0, 1.0]),
                          step=0.25, direction=-1, max_steps=200)
print('against     :', np.asarray(backward.coords)[:, -1],
      'length', round(float(np.sum(backward.measures)), 6))
coarse = el.streamlines(constant(shape=(3, 3)), 'v', start=np.array([0., 0.]),
                        max_steps=1)
print('default step:', round(float(np.sum(coarse.measures)), 6))
```

The default step on a grid whose cells are one unit apart is 0.5; the default
seed is the center of cell (1, 1).

:::{admonition} How the step is taken
:class: note
A step evaluates the field at the seed and at the points half a step and a
whole step along it, and combines those directions with the weights of the
classical fourth-order Runge--Kutta rule. Each evaluation is a grid
interpolation of the property, so the curve follows the same field `grid.prop`
reports. The field's magnitude is divided out at every stage: two cells may
carry different speeds, but neither makes the curve advance farther than the
step says.

The integration is combinatorial in the same way a boolean is: which steps are
taken and which cell a position falls in are decisions. A tensor field is read
and the result is NumPy; differentiate the interpolated field itself if a
derivative with respect to its values is wanted.
:::

:::{admonition} What is not here
:class: note
pyvista's filter spaces streamlines automatically, stops them when they come
too close to one another, and detects closed loops. `euclib.streamlines` does
none of those: it integrates the seeds it is given, and a caller who wants even
spacing can choose the seeds from a grid and drop the curves that overlap. It
also does not compute vorticity or ribbon geometry; the result is the curve,
which is the part a caller can measure or draw.
:::

## Provenance

Adapted from pyvista's
[`tests/core/test_dataset_filters.py::test_streamlines_evenly_spaced_2d`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_dataset_filters.py#L2210),
with the fixture `mesh_2d_velocity` beside it and the sibling tests for the
start position, the integrator type, and the interpolator type.

The upstream fixture builds a plane of 100 by 100 cells and sets a velocity of
one along `x` at every point; the test then checks that the result has points
and cells. The refinement tests change the separating distance, the start
position, the integrator, and the interpolator, and one test rotates the plane
out of the `xy` plane and expects a refusal. The page makes the same
constant-field check and adds the properties the upstream test does not state:
the line is straight, its length is the step count times the step, and the
default seed is the grid's center. The rotating field is the page's own, chosen
because its radius is a quantity a reader can predict without running the code.

The differences are in what the operation is. pyvista wraps VTK's
`vtkEvenlySpacedStreamlines2D`, which seeds, spaces, loops, and refuses a
dataset that is not in the `xy` plane; `euclib` takes explicit seeds on a grid
of any dimension and returns a `SegPath` rather than a `PolyData`. pyvista
offers integrator types 2 and 4 and step units in cell lengths or lengths;
`euclib` offers the same two orders but always measures the step as a distance,
with the default of half a cell. pyvista's filter is limited to two dimensions
because VTK's is; the integration here has no such limit, and a three-dimensional
seed walks through the volume in the same way.
