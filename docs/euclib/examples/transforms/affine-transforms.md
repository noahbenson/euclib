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
# Affine Transforms: Translation, Scale, and Rotation

:::{admonition} Warning: This page was written with AI assistance.
:class: warning

This page was written using AI assistance and has not yet been carefully
reviewed by a human. Its code runs and its claims are checked against the test
suite, but a person has not read it end to end. Treat it as an unreviewed draft:
verify the behaviour you depend on against the API reference and the tests, and
report anything that looks wrong.
:::

An affine transformation is the general "move, resize, and turn" operation on
space. `euclib` represents one as an `euclib.types.Affine`, a `(D+1, D+1)`
matrix that can be applied to positions, composed with other transforms, and
inverted. This page builds a translation and a scale, applies them to a
geometry, and shows that composing and inverting are exact inverses of each
other.

:::{admonition} What this demonstrates
:class: tip
- The `affine_translation`, `affine_scaling`, and `affine_identity` helpers.
- Building a rotation as an explicit matrix and wrapping it in an `Affine`.
- `Transform.compose` and its `@` operator shorthand.
- `Transform.inverse`, and why `t @ t.inverse` is the identity transform.
- `Affine.decompose`, which splits a matrix into translation, rotation, scale,
  and shear, and the product that rebuilds the matrix from them.
- `Geometry.transformed`: applying a transform to an immutable geometry.
:::

## The building blocks

The helpers produce the common cases. Each returns an `Affine` whose matrix can
be inspected directly:

```{code-cell}
import numpy as np
import euclib as el

translation = el.types.affine_translation([1.0, 2.0, 3.0])
scaling = el.types.affine_scaling([2.0, 2.0, 2.0])

print('translation matrix\n', translation.matrix)
print('\nscaling matrix\n', scaling.matrix)
```

A rotation has no helper, because a rotation is a full matrix; it is constructed
by wrapping a rotation matrix in an `Affine`. The matrix's last row must be
`[0, ..., 0, 1]`, which is what makes it *affine* rather than merely linear:

```{code-cell}
def affine_matrix(linear, translation=None):
    "Wraps a (D, D) linear part and optional offset into an Affine."
    linear = np.asarray(linear, dtype=float)
    D = linear.shape[0]
    matrix = np.eye(D + 1)
    matrix[:D, :D] = linear
    if translation is not None:
        matrix[:D, D] = translation
    return el.types.Affine(matrix)


theta = np.pi / 2
rotation = affine_matrix([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta),  np.cos(theta), 0],
                          [0, 0, 1]])
print('rotation sends (1, 0, 0) to', np.round(rotation.apply([1.0, 0.0, 0.0]).ravel(), 6))
```

## Composing transforms

`compose` applies one transform after another, and `@` is shorthand for it. The
result is itself an `Affine`, so composition does not accumulate a chain of
matrix multiplications to replay later:

```{code-cell}
combined = translation.compose(scaling)
print('translation @ scaling\n', combined.matrix)
print('same via @ :', np.allclose((translation @ scaling).matrix, combined.matrix))
```

## Inverting a transform

Every transform knows its inverse. Composing a transform with its inverse gives
the identity, and applying it to a point returns the point:

```{code-cell}
point = np.array([1.0, 1.0, 1.0])
forward = combined.apply(point)
back = combined.inverse.apply(forward)
print('point        :', point)
print('after transform:', np.round(forward.ravel(), 6))
print('after inverse  :', np.round(back.ravel(), 6))
print('round trip exact:', np.allclose(back.ravel(), point))
```

## Transforming a geometry

Because geometries are immutable, transforming one returns a new geometry with
new coordinates and the *same* topology. Here a translation moves a point cloud
without disturbing anything else about it:

```{code-cell}
coords = np.array([[0.0, 1.0, 0.0, 1.0],
                   [0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 0.0, 1.0]])
cloud = el.points(coords)

moved = cloud.transformed(translation)
print('original corner 3:', cloud.coords[:, 3])
print('moved    corner 3:', moved.coords[:, 3])
print('topology shared  :', moved.topo is cloud.topo
      or np.array_equal(np.asarray(moved.topo.indices), np.asarray(cloud.topo.indices)))
```

## Taking a transform apart

A matrix is a compact way to carry a transform, but it does not say which
rotation, which scale, and which shear produced it. `Affine.decompose` splits
the linear part into an orthogonal rotation, a positive scale along each axis,
and a unit upper-triangular shear, and reports the translation beside them. The
parts rebuild the matrix exactly.

```{code-cell}
(t, r, s, k) = combined.decompose()
print('translation:', t)
print('rotation   :\n', np.round(r, 6))
print('scale      :', s)
print('shear      :\n', np.round(k, 6))

def rebuild(transform):
    "Rebuilds an affine matrix from its decomposition."
    (t, r, s, k) = transform.decompose()
    d = transform.dim
    matrix = np.eye(d + 1)
    matrix[:d, :d] = r @ (s[:, None] * k)
    matrix[:d, d] = t
    return matrix

print('rebuilt exactly:', np.allclose(rebuild(combined), combined.matrix))
```

The rotation is orthogonal, and its determinant says whether the transform
reflects space: a pure rotation has determinant 1, and a reflection comes back
as a rotation with determinant -1 rather than as a fifth factor. A shear is
reported by the matrix that produces it, and a scale is always positive.

```{code-cell}
shear = affine_matrix([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
(_, shear_rotation, shear_scale, shear_part) = shear.decompose()
print('a shear keeps the axes :', np.allclose(shear_rotation, np.eye(3)))
print('a shear has unit scale :', np.allclose(shear_scale, np.ones(3)))
print('the reported shear     :\n', shear_part)

mirror = el.types.affine_scaling([-1.0, 1.0, 1.0])
(_, mirror_rotation, mirror_scale, _) = mirror.decompose()
print('a reflection rotates with determinant',
      round(float(np.linalg.det(mirror_rotation)), 6),
      'and keeps positive scales', mirror_scale)
```

:::{admonition} Why an affine matrix and not a list of operations
:class: tip
Keeping the transform as one invertible matrix means it composes associatively,
can be inverted exactly, and can be taken apart when a caller needs the
individual operations. The decomposition is the QR factorization of the linear
part, with the signs moved from the triangular factor into the rotation so that
the scale is positive; it is unique for a nonsingular transform.
:::

## Figure

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

triangle = el.trimesh(np.array([[0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [0.0, 0.0, 0.0]]),
                      np.array([[0], [1], [2]], dtype='int64'))
moved_triangle = triangle.transformed(combined)
euclib_viz.show3d(moved_triangle, color_by='z', name='affine-triangle')
```

:::{admonition} Provenance
:class: note
**Upstream:** `pyvista/pyvista` — `tests/core/test_utilities.py`, functions
`test_transform_translate` (line 1971), `test_transform_scale` (line 1956),
`test_transform_rotate` (line 2014), and `test_transform_decompose` (line 2730).

**Pinned revision:** [`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_utilities.py#L1956).

**How this was adapted.** The upstream tests drive pyvista's fluent `Transform`
API — `transform.translate(...)`, `transform.rotate_vector(...)`, and so on —
and `test_transform_decompose` composes a shear, scale, reflection, rotation,
and translation and then decomposes the result back into its parts. No code is
copied. `euclib` has no in-place fluent transform object, so the adaptation
uses the immutable `Affine` plus `compose`/`@`/`inverse`, building each transform
from its matrix instead of mutating one. The upstream decomposition test
composes a shear, scale, reflection, rotation, and translation and checks that
the five parts it gets back rebuild the matrix; the page does the same, but
`euclib` reports four parts rather than five. pyvista returns the reflection as
a separate scalar that multiplies the whole matrix; `euclib` folds it into the
rotation, whose determinant is then -1, which is the same information in a form
that also works in an even-dimensional space. The `Geometry.transformed`
demonstration is not in the upstream tests, which operate on transform objects
rather than on geometry.
:::

:::{seealso}
- [Rotation about an axis, and reflection](axis-angle-and-reflection.md) for
  non-trivial affine matrices.
- [Prism sweeps](../primitives/prism-sweeps.md) for a geometry whose two faces
  are related by a translation.
:::
