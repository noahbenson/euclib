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
# Interpolating a Grid: Nearest and Linear

A grid is not a mesh. Its cells are the unit boxes of an **index
space**, and that space is related to where the cells actually are by an
**affine transformation** — which may scale, shear, rotate or reflect. This page
is about the two simplest ways to read a value between the cells, and about the
one thing every grid method depends on and none of the mesh methods did: **which
space the interpolation happens in**.

:::{admonition} What this demonstrates
:class: tip
- The index space, and where a cell's sample sits in it: an index names its
  cell's **centre**, and the cell's region runs half a step either side.
- Nearest-neighbour (order 0) and multilinear (order 1) interpolation.
- Why the affine is applied to the *query* and never to the kernel.
- That a sheared, rotated or scaled grid gives the same answer as an upright one
  for the same **index-space** position — because the answer is a function of
  that and of nothing else.
:::

## The index space

A grid stores one value per cell and an affine matrix, and the affine carries a
cell's **index** to that cell's **centre** in global coordinates. Everything the
interpolation does happens in the index space, where the grid is upright and its
samples sit at the integers:

$$ s = (s_x, s_y, \ldots) \quad\text{and the cell } (i, j, \ldots)
   \text{ has its sample at } s = (i, j, \ldots) . $$

A cell's *region*, then, runs half a step to either side of its sample, so a grid
of extent $(3, 2)$ covers $[-0.5, 2.5] \times [-0.5, 1.5]$ in its own space. That
is what `contains` tests and what the bounding box reports, and the local
coordinate of a position is its index-space position, whatever the affine is.

:::{admonition} A choice worth naming
:class: note
There is a second convention in common use, in which the cell's **corner** is at
the integer and its centre at $i + ½$. It describes the same grid and differs
only in where the origin is put, but the two cannot be mixed: reading one as the
other is a half-cell error everywhere, silently.

`euclib` uses the first, for three reasons. The affine already carries the index
to the *centre*, so nothing has to be shifted to reach it. `contains`, the
bounding box and the nearest-cell rule already agree on it. And the standard
formulas for every kernel in this family are written for samples at integers —
$v_{m} = u(m)$ — so adopting the other convention would mean adding $½$ to every
kernel argument, for no gain.
:::

```{code-cell}
import numpy as np
import euclib as el

# A 5x4 grid with a scale of 2 and 3 and an offset of 10 and 20: an affine that
# is not the identity, so "index" and "position" are plainly different things.
affine = np.array([[2.0, 0.0, 10.0],
                   [0.0, 3.0, 20.0],
                   [0.0, 0.0, 1.0]])
grid = el.grid((5, 4), affine=affine)

index = grid.topo.Loc(sx=np.array([0.0, 2.0]), sy=np.array([0.0, 3.0]))
print("where the indices (0,0) and (2,3) actually are:")
print("  ", np.round(grid.to_global(index).T, 3).tolist())
print("and the bounding box, which is the centres plus half a step either side:")
box = np.asarray(getattr(grid.bbox, 'magnitude', grid.bbox), dtype=float)
print("  ", np.round(box, 3).tolist())
print(f"  the extent is the shape {grid.shape} less half a step at each end:")
print("  ", [grid.shape[0] - 0.5, grid.shape[1] - 0.5])
```

## The property, and the two methods

A grid's property has one value per cell, with the property's channel dimensions
leading, so its shape is the grid's shape with the channels in front. Reading it
at an index-space position is the whole of order 0 and order 1:

- **order 0 (`'nearest'`)** is the value of the cell whose sample is nearest,
  which is the cell whose index is the position rounded.
- **order 1 (`'polynomial' 1` or `'bezier' 1`)** is the multilinear blend of the
  $2^D$ cells around the position — and the two names are the same method at
  order 1, as they are on every element.

```{code-cell}
values = np.arange(grid.shape[0] * grid.shape[1],
                   dtype=float).reshape(grid.shape)
carried = grid.withprop('v', values)

def read(method, sx, sy):
    return float(np.ravel(np.asarray(carried.prop(
        'v', at=grid.topo.Loc(sx=np.array([sx]), sy=np.array([sy])),
        interp=method)))[0])

print("the values are the cell numbers:")
print(values.astype(int))
print()
print("  nearest at (0.6, 0.4) is cell (1, 0):", read(('nearest', 0), 0.6, 0.4))
print("  nearest at (0.4, 0.4) is cell (0, 0):", read(('nearest', 0), 0.4, 0.4))
print("  linear  at (0.5, 0.5) is the mean of cells (0,0), (1,0), (0,1), (1,1):",
      read(('polynomial', 1), 0.5, 0.5),
      "=", values[:2, :2].mean())
print("  linear  at (1.0, 1.0) lands on a sample, so it is that sample:",
      read(('polynomial', 1), 1.0, 1.0))
```

**It reproduces an affine function.** Away from the boundary, multilinear
interpolation is exact for a function that is affine in the index coordinates —
that is what gives it its approximation order 1 — and it is exact *because* it is
a convex combination of the cell's samples, which is also why it can never
overshoot the data.

```{code-cell}
# An affine function of the index coordinates, sampled at the cells.
def field(sx, sy):
    return 0.7 + 0.4 * sx - 0.9 * sy

(ix, iy) = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]),
                       indexing='ij')
carried = grid.withprop('f', field(ix, iy).astype(float)[None])

worst = 0.0
rng = np.random.default_rng(0)
for _ in range(200):
    (sx, sy) = (rng.uniform(0.0, grid.shape[0] - 1.0),
                rng.uniform(0.0, grid.shape[1] - 1.0))
    got = float(np.ravel(np.asarray(carried.prop(
        'f', at=grid.topo.Loc(sx=np.array([sx]), sy=np.array([sy])),
        interp=('polynomial', 1))))[0])
    worst = max(worst, abs(got - field(sx, sy)))
print(f"  worst difference from the affine function, over 200 positions:"
      f" {worst:.3e}")
```

## The affine is applied to the query, not to the kernel

This is the whole reason a grid method is simple. The kernel is evaluated on
index-space positions, which it knows nothing about the grid's placement to
compute; the affine's only job is to carry the global position of the query into
that space, and `to_local` is the matrix that does it. So a sheared, rotated or
scaled grid needs no separate code path — and the check below is the one that
would fail if the affine were applied anywhere else. The same values on a grid
whose affine shears, rotates and scales must give the same answer as the upright
one **for the same index-space position**, however far apart the affine has moved
those positions in space.

```{code-cell}
# The same values on a grid whose affine shears, rotates and scales.
turn = np.array([[0.6, -0.8], [0.8, 0.6]])          # a rotation
shear = np.array([[1.0, 0.45], [0.0, 1.0]])         # a shear
scale = np.array([[2.5, 0.0], [0.0, 0.4]])          # a scale, one axis each way
linear = turn @ shear @ scale
other = np.array([[linear[0, 0], linear[0, 1], 7.0],
                  [linear[1, 0], linear[1, 1], -3.0],
                  [0.0, 0.0, 1.0]])
upright = grid.withprop('u', values[None])
bent = el.grid(grid.shape, affine=other).withprop('u', values[None])

def ask(here, sx, sy):
    """The value a grid gives for an index-space position, reached through the
    grid's own affine."""
    at = here.to_global(here.topo.Loc(sx=np.array([sx]), sy=np.array([sy])))
    return float(np.ravel(np.asarray(here.prop(
        'u', at=at, interp=('polynomial', 1))))[0])

worst = 0.0
rng = np.random.default_rng(1)
for _ in range(200):
    (sx, sy) = (rng.uniform(0.0, grid.shape[0] - 1.0),
                rng.uniform(0.0, grid.shape[1] - 1.0))
    worst = max(worst, abs(ask(upright, sx, sy) - ask(bent, sx, sy)))
print(f"  the upright grid and the sheared one, asked for the same"
      f" index-space positions: worst difference {worst:.3e}")
print("  ...and the positions themselves are far apart, because the affine"
      " moved them:")
print("   ", np.round(bent.to_global(
    bent.topo.Loc(sx=np.array([0.0, 4.0]), sy=np.array([0.0, 3.0]))).T,
    3).tolist())
```

Note what the two grids did *not* share: the positions. The affine moved them,
by more than a cell in places, and the two grids do not even overlap everywhere.
What they shared is the index-space position of every query — and the answer is a
function of that alone, which is why no method in this family has to know what
the affine is.

## At the boundary

A kernel reaches *outside* the grid whenever the query is near an edge: a
position $½$ of a step inside the last cell's centre still wants a cell that does
not exist. Linear interpolation's stencil is small, so it only ever wants one —
but it does want one, and what it gets is a choice rather than a detail. The
order-1 path as it stands clamps its stencil to the grid, which is one policy
among several. The standard three — constant, half-sample symmetric, and
whole-sample symmetric — and how a property chooses between them are the subject
of [the boundary page](grid-boundary.md), which is the one that follows this.

:::{admonition} Where this comes from
:class: note
The methods and the terms for them are Pascal Getreuer's, *Linear Methods for
Image Interpolation*, Image Processing On Line **1** (2011), 238–259,
[doi:10.5201/ipol.2011.g_lmii](https://doi.org/10.5201/ipol.2011.g_lmii), which
treats nearest, bilinear, bicubic, splines, Lanczos and sinc together and is the
reference the rest of this family follows. The multinomial naming euclib uses —
`'polynomial'` for the monomial least-squares fit and `'bezier'` for the
control-value construction — is [Interpolating a Property within a
Geometry](interpolation.md)'s, and applies here unchanged at order 1, where the
two are the same method.
:::

:::{seealso}
- [Interpolating a Property within a Geometry](interpolation.md) for the metadata
  that chooses an order, and for what happens outside the object.
- [The roadmap](../roadmap.md) for what is not built yet.
:::
