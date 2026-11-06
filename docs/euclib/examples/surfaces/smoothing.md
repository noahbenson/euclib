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
# Smoothing a Field over a Mesh or an Image

:::{admonition} Warning: This page was written with AI assistance.
:class: warning

This page was written using AI assistance and has not yet been carefully
reviewed by a human. Its code runs and its claims are checked against the test
suite, but a person has not read it end to end. Treat it as an unreviewed draft:
verify the behaviour you depend on against the API reference and the tests, and
report anything that looks wrong.
:::

`euclib.ops.smooth` takes a property --- or any value array --- and returns it
smoothed: one step of the diffusion equation over a mesh's elements, or a kernel
along each axis of an image. The mesh side is the cotangent Laplacian with
Meyer's mixed mass; the image side is a separable kernel, or a bilateral one
that does not smooth across a step.

:::{admonition} What this demonstrates
:class: tip
- `euclib.ops.smooth(geom, values, loglambda=...)`, which returns a `Property`
  and leaves the geometry alone.
- That `loglambda` is the logarithm of the step, so zero is one step and a
  linear change in it is multiplicative in the smoothing.
- That the operator is mass-normalised, so a step means the same thing on every
  mesh.
- That boundary vertices keep their values by default, and `fixboundary=False`
  lets them move along their interior edges.
- That an image smooths with a gaussian, a box, or a bilateral kernel, and that
  `sigma` means the same reach in all three.
:::

## A field on a mesh

The mesh is a four by four grid of points, triangulated; the field is a bump
whose maximum is at the middle and whose minimum at each corner.

```{code-cell}
import numpy as np
import euclib as el

points = np.asarray([0., 1., 2., 3.])
(X, Y) = np.meshgrid(points, points, indexing='ij')
coords = np.stack([X.ravel(), Y.ravel()])
def vid(i, j):
    return i * 4 + j
tris = []
for i in range(3):
    for j in range(3):
        tris.append([vid(i, j), vid(i+1, j), vid(i, j+1)])
        tris.append([vid(i+1, j), vid(i+1, j+1), vid(i, j+1)])
mesh = el.trimesh(coords, np.asarray(tris).T)
bump = (coords[0] - 1.5) ** 2 + (coords[1] - 1.5) ** 2
held = mesh.withprop('b', bump)
print('the interior four points read', bump[[5, 6, 9, 10]].round(3))
```

## One step, and more

`loglambda` is the logarithm of the step: zero is one, negative is less, positive
is more. Each call returns a new `Property`; the mesh it came from is untouched.

```{code-cell}
for loglambda in (-2, 0, 2):
    smoothed = el.ops.smooth(held, 'b', loglambda=loglambda)
    inside = np.asarray(smoothed.value)[[5, 6, 9, 10]]
    print(f'loglambda {loglambda:+d}: the interior reads {inside.round(3)}')
```

The bump's minimum rises toward the rim each time: smoothing pulls a value
toward its neighbours, and the rim is held where it is.

## Holding the boundary

By default boundary vertices keep their values, which is what makes the rim the
fixed condition the diffusion runs against.

```{code-cell}
smoothed = np.asarray(el.ops.smooth(held, 'b').value)
rim = np.ones(len(bump), dtype=bool)
rim[[5, 6, 9, 10]] = False
print('the rim is unchanged :', np.allclose(smoothed[rim], bump[rim]))
print('the interior moved   :', not np.allclose(smoothed[[5, 6, 9, 10]],
                                               bump[[5, 6, 9, 10]]))
```

With `fixboundary=False` the rim vertices are smoothed along their interior
edges too, so the whole field relaxes.

```{code-cell}
free = np.asarray(el.ops.smooth(held, 'b', fixboundary=False).value)
print('rim, held :', smoothed[rim].round(3)[:4], '...')
print('rim, free :', free[rim].round(3)[:4], '...')
```

## An image

A grid's cells are boxes rather than simplices, so its smoothing is a kernel
along each axis rather than a Laplacian over elements. A gaussian blurs a step;
a bilateral filter does not, because a neighbour whose value is far from the
centre's contributes little however near it is.

```{code-cell}
step = np.zeros((5, 5, 5))
step[2:] = 10.0
image = el.grid((5, 5, 5), np.eye(4)).withprop('s', step)
blurred = np.asarray(el.ops.smooth(image, 's', kernel='gaussian',
                                   sigma=1.0).value)
kept = np.asarray(el.ops.smooth(image, 's', kernel='bilateral', sigma=1.0,
                                rangesigma=0.5).value)
print('gaussian  across the step:', blurred[:, 2, 2].round(3))
print('bilateral across the step:', kept[:, 2, 2].round(3))
```

:::{admonition} The boundary is a condition, not a defect
:class: note
The Laplacian's natural boundary condition is ``df/dn = 0``, so a field with a
gradient across the boundary is not left alone there unless the boundary is
held. A plane is preserved by the default --- which holds it, a Dirichlet
condition --- and moves without it. Both behaviours are tested, so this is the
operator's specification rather than a surprise.
:::

## Provenance

Adapted from trimesh's
[`smoothing.py::filter_laplacian`](https://github.com/mikedh/trimesh/blob/main/trimesh/smoothing.py)
and pyvista's
[`test_polydata.py::test_smooth`](https://github.com/pyvista/pyvista/blob/main/tests/core/test_polydata.py).

Both smooth a mesh's *coordinates*; this smooths a *property* over a mesh, which
is the same operator applied to a field rather than to the geometry. The image
half adapts `scipy.ndimage`'s `gaussian_filter` and the bilateral filter of
Tomasi and Manduchi, whose weights depend on the values and so cannot be
separated into axes.
