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
# Rotation about an Arbitrary Axis, and Reflection

A transform does not have to be one of the familiar axis-aligned moves. This
page builds two less obvious affine transforms — a rotation about an arbitrary
axis through an arbitrary point, and a reflection across an arbitrary plane —
and checks each by an invariant it must satisfy rather than by its matrix
entries.

:::{admonition} What this demonstrates
:class: tip
- Constructing an `Affine` from a rotation matrix derived by the
  axis-angle (Rodrigues) formula.
- Rotation about a point that is not the origin, via a translate-rotate-
  translate composition.
- Reflection as the linear map `I - 2 * n n^T / (n . n)`.
- Testing transforms by invariants: repeated rotation returns to the start, a
  reflection is its own inverse.
:::

```{code-cell}
import numpy as np
import euclib as el


def affine_matrix(linear, translation=None):
    "Wraps a (D, D) linear part and optional offset into an Affine."
    linear = np.asarray(linear, dtype=float)
    D = linear.shape[0]
    matrix = np.eye(D + 1)
    matrix[:D, :D] = linear
    if translation is not None:
        matrix[:D, D] = translation
    return el.types.Affine(matrix)


def rotation_about(axis, angle):
    "Returns the (3, 3) rotation matrix for a rotation about `axis`."
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]],
                  [n[2], 0, -n[0]],
                  [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
```

## Rotation about a body diagonal

Rotating `2*pi/3` about the diagonal `(1, 1, 1)` cyclically permutes the three
coordinate axes, because that rotation is a symmetry of a cube:

```{code-cell}
R = affine_matrix(rotation_about([1.0, 1.0, 1.0], 2 * np.pi / 3))
for axis, name in [(np.array([1.0, 0, 0]), 'x'),
                   (np.array([0, 1.0, 0]), 'y'),
                   (np.array([0, 0, 1.0]), 'z')]:
    print(f'unit {name} ->', np.round(R.apply(axis).ravel(), 4))
```

Four successive quarter-turns about any axis must return to the identity, which
is easy to get wrong by a sign in the formula:

```{code-cell}
quarter = affine_matrix(rotation_about([0.0, 0.0, 1.0], np.pi / 2))
product = quarter
for _ in range(3):
    product = product @ quarter
print('four quarter-turns are the identity:',
      np.allclose(product.matrix, np.eye(4)))
```

## Rotating about a point

A rotation matrix turns space about the *origin*. To turn about some other
point, translate that point to the origin, rotate, and translate back — three
`Affine`s composed with `@`:

```{code-cell}
def rotation_about_point(point, axis, angle):
    point = np.asarray(point, dtype=float)
    to_origin = el.types.affine_translation(-point)
    back = el.types.affine_translation(point)
    spin = affine_matrix(rotation_about(axis, angle))
    return back @ spin @ to_origin


pivot = np.array([1.0, 0.0, 0.0])
turn = rotation_about_point(pivot, [0.0, 0.0, 1.0], np.pi / 2)

# The pivot is on the axis, so it must not move.
print('pivot stays put  :', np.allclose(turn.apply(pivot).ravel(), pivot))
# A point one unit to the pivot's right swings one unit above it.
print('pivot + x -> y   :', np.round(turn.apply(pivot + [1, 0, 0]).ravel(), 4))
```

## Reflection across a plane

A reflection across a plane through the origin with unit normal `n` is the
linear map `I - 2 n n^T`. Unlike a rotation it reverses handedness, so its
determinant is `-1`:

```{code-cell}
def reflection_matrix(normal):
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return np.eye(len(n)) - 2 * np.outer(n, n)


reflect = affine_matrix(reflection_matrix([1.0, -1.0, 0.0]))   # plane x = y
n = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)

print('a point          :', np.array([2.0, 0.0, 1.0]))
print('its reflection   :', np.round(reflect.apply([2.0, 0.0, 1.0]).ravel(), 4))
print('normal reversed  :', np.round(reflect.apply(n).ravel(), 4))
print('determinant      :', round(float(np.linalg.det(reflect.matrix[:3, :3])), 6))

# Reflecting twice must return every point to where it started.
twice = reflect @ reflect
print('twice is identity:', np.allclose(twice.matrix, np.eye(4)))
```

:::{admonition} Where the "flip" appears
:class: tip
The determinant is the way to tell a reflection from a rotation when both are
orthogonal: a rotation has determinant `+1`, a reflection `-1`. `euclib`
carries the matrix directly, so this check costs nothing.
:::

## Figure

The reflected triangle is shown below; a reflection reverses its winding as well
as its position, which is the geometric meaning of the `-1` determinant:

```{code-cell}
import pathlib
import sys

_root = next((d for d in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
              if (d / 'myst.yml').exists()), pathlib.Path.cwd() / 'docs' / 'euclib')
sys.path.insert(0, str(_root))
import euclib_viz

shape = el.trimesh(np.array([[0.0, 2.0, 1.0],
                             [0.0, 0.0, 3.0],
                             [0.0, 0.0, 0.0]]),
                   np.array([[0], [1], [2]], dtype='int64'))
euclib_viz.show3d(shape.transformed(reflect), color_by='x', name='reflection')
```

:::{admonition} Provenance
:class: note
**Upstream:** `pyvista/pyvista` — `tests/core/test_utilities.py`, functions
`test_axis_angle_rotation` (line 1006) and `test_reflection` (line 1065).

**Pinned revision:** [`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_utilities.py#L1006).

**How this was adapted.** The upstream tests call
`pv.transformations.axis_angle_rotation` and pyvista's reflection helpers, then
check the resulting transforms against hand-computed values (rotating about the
body diagonal permutes the axes; four 90-degree turns return a unit matrix; a
reflection maps a point across a plane). No code is copied. `euclib` has no
transform-generator functions, so the rotation matrix is built here from the
Rodrigues formula and the reflection matrix from `I - 2 n n^T` — the same
mathematics the upstream helpers encapsulate — and each is wrapped in an
`Affine`. The upstream assertions are kept as invariants (the axis permutation,
the four-quarter-turn identity, the double-reflection identity), and the
rotation-about-a-point case is added because `euclib`'s `Affine` has no notion
of a pivot, so it must be assembled by composition.
:::

:::{seealso}
- [Affine transforms](affine-transforms.md) for the composition and inversion
  machinery used here.
- [A cube as a triangle mesh](../primitives/cube.md), whose diagonal symmetry
  the body-diagonal rotation exploits.
:::
