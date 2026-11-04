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
# The Gradient and Hessian of an Interpolated Field

:::{admonition} Warning: This page was written with AI assistance.
:class: warning

This page was written using AI assistance and has not yet been carefully
reviewed by a human. Its code runs and its claims are checked against the test
suite, but a person has not read it end to end. Treat it as an unreviewed draft:
verify the behaviour you depend on against the API reference and the tests, and
report anything that looks wrong.
:::

`prop` reads a property's interpolated field at a set of positions. `propgrad`
reads that same field's gradient, and `prophess` its hessian --- the derivative
of whatever fit the property's interpolation asks for, in the geometry's own
coordinates.

:::{admonition} What this demonstrates
:class: tip
- `geom.propgrad(name, at=...)` and `geom.prophess(name, at=...)`, shaped
  `(C..., D, Q)` and `(C..., D, D, Q)`.
- That a linear field's gradient is its plane's, everywhere.
- That a quadratic's gradient is that quadratic's, because a tetrahedron's
  values and gradients determine it --- so the fit reproduces the field.
- That a `nearest` field has no gradient between its regions, and a linear one
  no hessian.
- That the answer is exact on a *skewed* element, which is the only case that
  tests the chain rule: on the standard tetrahedron the element's Jacobian is
  the identity and a wrong transpose would pass.
:::

## A skewed tetrahedron

The tetrahedron is deliberately irregular. Its Jacobian is then not orthogonal,
so the step that carries a gradient from the element's own coordinates into the
global ones has to be right rather than merely present.

```{code-cell}
import numpy as np
import euclib as el

coords = np.array([[0.0, 1.3, 0.2, -0.4],
                   [0.0, 0.5, 1.1,  0.3],
                   [0.0, 0.2, 0.4,  1.2]])
tet = el.tetmesh(coords, np.array([[0], [1], [2], [3]]))
centre = coords.mean(axis=1).reshape(3, 1)
print('the centroid is at', centre.ravel())
```

## A linear field

A plane's gradient is the plane's, at every position. The interpolation does not
change it.

```{code-cell}
plane = 2 * coords[0] - 3 * coords[1] + coords[2]
linear = tet.withprop('f', plane, interp=('polynomial', 1))
print('the gradient is', linear.propgrad('f', at=centre).ravel())
print('...and it is     [2. -3.  1.] everywhere')
```

## A quadratic field

A quadratic is determined by a tetrahedron's values *and* its corners'
gradients, so supplying both makes the fit the field itself --- and then the
gradient the field's own. Over the unit tetrahedron the field `x^2` has gradient
`2x`.

```{code-cell}
gradient = np.zeros((3, 4))
gradient[0] = 2 * coords[0]
square = tet.withprop('sq', coords[0] ** 2, gradient=gradient,
                      interp=('polynomial', 2))
print('2x at the centroid is', 2 * centre[0, 0])
print('propgrad gives       ', square.propgrad('sq', at=centre).ravel())
```

The hessian of `x^2` is constant, `diag(2, 0, 0)`, and `prophess` reports it.

```{code-cell}
print('the hessian is\n', square.prophess('sq', at=centre)[:, :, 0])
```

## What has no derivative

A `nearest` field is piecewise constant: flat within each region a vertex is
nearest to, and discontinuous between them. `propgrad` reports zero, which is
the true derivative wherever the field is differentiable at all.

```{code-cell}
near = tet.withprop('n', coords[0], interp=0)
print('a nearest field has gradient', near.propgrad('n', at=centre).ravel())
```

A linear field has no second derivative, and `prophess` reports zero for it too
--- for a different reason, since the field exists everywhere and is simply
straight.

```{code-cell}
print('a linear field has hessian\n',
      linear.prophess('f', at=centre)[:, :, 0])
```

:::{admonition} Where the derivative comes from
:class: note
Each interpolation method states its fit; the gradient is that fit's derivative
and the hessian its second. Below the quadratic order there is no fit to
differentiate, so the answers are exact by construction rather than by a fit:
a linear field's gradient is its corner differences carried through the
element's Jacobian. Above it, the fit is differentiated in the element's own
barycentric coordinates and carried back the same way.
:::

## Provenance

Adapted from pyvista's
[`test_dataset_filters.py::test_compute_gradient`](https://github.com/pyvista/pyvista/blob/main/tests/test_dataset_filters.py)
and trimesh's
[`test_proximity.py::test_signed_distance`](https://github.com/mikedh/trimesh/blob/main/tests/test_proximity.py).

Both compute gradients *at points*, which is what `propgrad` does; neither
reports the derivative of an interpolation it chose itself. The example uses a
skewed tetrahedron for a reason neither upstream test insists on: on a regular
one a wrong Jacobian transpose is invisible.
