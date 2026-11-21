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
# Capsules, Cylinders, and Cones

:::{admonition} Warning: This page was written with AI assistance.
:class: warning

This page was written using AI assistance and has not yet been carefully
reviewed by a human. Its code runs and its claims are checked against the test
suite, but a person has not read it end to end. Treat it as an unreviewed draft:
verify the behaviour you depend on against the API reference and the tests, and
report anything that looks wrong.
:::

A cylinder, a cone, and a capsule are quadric surfaces: their sides are ruled
by straight lines, and their cross-sections are circles. `euclib` has no
primitive generator yet, so this page builds each one by placing rings of
points around an axis and joining them with triangles. The results are ordinary
`TriMesh` objects, which lets the page check the properties that define the
shapes: the extents, the closure, and the volume and area as the number of
sides grows.

:::{admonition} What this demonstrates
:class: tip
- A closed `TriMesh` built from rings of points, and the winding that makes its
  volume positive.
- The extents of a cylinder and a cone of diameter 2 and height 2, and of a
  capsule of radius 1 and cylindrical height 2.
- That the volume and area approach the closed-form formulas as the ring gains
  sides, and how quickly.
- Euler's characteristic and edge pairing for each shape, which say the surface
  closes.
:::

## Building the surfaces

The helper places a ring of `n` points at each height, joins consecutive rings
with two triangles per quad, and closes the ends with fans. It measures the
signed volume first and reverses every triangle if the surface came out inside
out, so the caller need not think about the winding.

```{code-cell}
import numpy as np
import euclib as el

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

def mesh_of(coords, faces):
    coords = np.array(coords, dtype=float)
    faces = np.array(faces, dtype=int)
    solid = el.trimesh(coords.T, faces.T)
    if volume(solid) < 0.0:
        solid = el.trimesh(coords.T, faces[:, [0, 2, 1]].T)
    return solid

def check(name, mesh):
    coords = np.asarray(mesh.coords)
    print(f'{name:>9}: V={mesh.coord_count:>4} '
          f'E={mesh.topo.simplex_count[1]:>4} '
          f'T={mesh.topo.simplex_count[2]:>4} '
          f'closed={closed(mesh)} '
          f'volume={volume(mesh):.4f} '
          f'area={float(np.sum(mesh.measures)):.4f}')
    print('           extents', np.round(coords.max(axis=1)
                                        - coords.min(axis=1), 4))
```

## A cylinder

Two rings at `z = -h/2` and `z = +h/2`, joined by quads, with a fan at each end.

```{code-cell}
def cylinder(radius, height, sides=64):
    angle = 2.0 * np.pi * np.arange(sides) / sides
    coords = []
    for z in (-height / 2.0, height / 2.0):
        coords += [[radius * np.cos(a), radius * np.sin(a), z]
                   for a in angle]
    coords += [[0.0, 0.0, -height / 2.0], [0.0, 0.0, height / 2.0]]
    bottom = 2 * sides
    top = 2 * sides + 1
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces += [[i, j, sides + j], [i, sides + j, sides + i],
                  [bottom, j, i], [top, sides + i, sides + j]]
    return mesh_of(coords, faces)

can = cylinder(1.0, 2.0)
check('cylinder', can)
print('formulas : volume', round(np.pi * 2.0, 6),
      'area', round(2.0 * np.pi * 2.0 + 2.0 * np.pi, 6))
```

The extents are 2 by 2 by 2: the diameter in `x` and `y`, and the height in
`z`. The computed volume is a little under $2\pi$ and the area a little under
$6\pi$, because a 64-gon is inscribed in the circle rather than being the
circle itself.

## A cone

One ring at the base and one apex, with a fan for the base and a triangle for
each side.

```{code-cell}
def cone(radius, height, sides=64):
    angle = 2.0 * np.pi * np.arange(sides) / sides
    coords = [[radius * np.cos(a), radius * np.sin(a), -height / 2.0]
              for a in angle]
    coords += [[0.0, 0.0, height / 2.0], [0.0, 0.0, -height / 2.0]]
    apex = sides
    base = sides + 1
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces += [[i, j, apex], [base, j, i]]
    return mesh_of(coords, faces)

tip = cone(1.0, 2.0)
check('cone', tip)
print('formulas : volume', round(np.pi * 2.0 / 3.0, 6),
      'area', round(np.pi * np.sqrt(5.0) + np.pi, 6))
```

The cone's volume is one third of the cylinder's, as it is for any cone with
the same base and height, and its area is the lateral surface
$\pi r \sqrt{r^2 + h^2}$ plus the base.

## A capsule

A capsule is a cylinder of height `h` with a hemisphere at each end, so its
total height is `h + 2r`. The rings run from the bottom pole, around the
bottom hemisphere, along the cylinder, around the top hemisphere, and to the
top pole.

```{code-cell}
def capsule(radius, height, sides=64, bands=8):
    angle = 2.0 * np.pi * np.arange(sides) / sides
    rings = []
    for t in np.linspace(-np.pi / 2.0, 0.0, bands + 1)[1:]:
        rings.append((radius * np.cos(t), -height / 2.0
                      + radius * np.sin(t)))
    rings.append((radius, height / 2.0))
    for t in np.linspace(0.0, np.pi / 2.0, bands + 1)[1:-1]:
        rings.append((radius * np.cos(t), height / 2.0
                      + radius * np.sin(t)))
    coords = []
    for (r, z) in rings:
        coords += [[r * np.cos(a), r * np.sin(a), z] for a in angle]
    bottom = len(coords)
    coords.append([0.0, 0.0, -height / 2.0 - radius])
    top = len(coords)
    coords.append([0.0, 0.0, height / 2.0 + radius])
    faces = []
    for i in range(sides):
        faces.append([bottom, (i + 1) % sides, i])
    for k in range(len(rings) - 1):
        a = k * sides
        b = (k + 1) * sides
        for i in range(sides):
            j = (i + 1) % sides
            faces += [[a + i, a + j, b + j], [a + i, b + j, b + i]]
    last = (len(rings) - 1) * sides
    for i in range(sides):
        faces.append([top, last + i, last + (i + 1) % sides])
    return mesh_of(coords, faces)

pill = capsule(1.0, 2.0)
check('capsule', pill)
print('formulas : volume', round(2.0 * np.pi + 4.0 * np.pi / 3.0, 6),
      'area', round(4.0 * np.pi + 4.0 * np.pi, 6))
```

The extents are 2 by 2 by 4: the radius in `x` and `y`, and `h + 2r` in `z`.
This is the check trimesh's capsule test makes on the same radius and height,
and the widest vertices reach the radius while the cylindrical wall spans the
two units between the hemispheres.

## More sides, less error

A polygon inscribed in a circle has an area less than the circle's; the error
falls as the square of the angle between its sides. Quadrupling the number of
sides quarters the relative error, which is what the cylinder's volume shows.

```{code-cell}
for sides in (8, 16, 32, 64, 128):
    approximate = volume(cylinder(1.0, 2.0, sides))
    print(f'{sides:>4} sides: volume {approximate:.8f}, '
          f'relative error {abs(approximate - 2.0 * np.pi)
                             / (2.0 * np.pi):.2e}')
```

The same holds for the cone and the capsule's curved surface. A caller who
needs an exact circle uses an analytic surface, not a mesh; a caller who needs
a mesh chooses the side count that keeps the error below what the application
can see.

## The surfaces close

Each helper's faces pair their edges and give Euler characteristic 2. The
printed `closed` values above are the same statement, and the counts can be
checked by hand: a closed triangle surface has $E = 3T/2$, and
$V - E + T = 2$ follows from the ring construction.

```{code-cell}
def euler(mesh):
    return (mesh.coord_count - mesh.topo.simplex_count[1]
            + mesh.topo.simplex_count[2])

for (name, shape) in (('cylinder', can), ('cone', tip), ('capsule', pill)):
    print(f'{name:>9}: V - E + T = {euler(shape)}; '
          f'3T/2 = {3 * shape.topo.simplex_count[2] / 2:.1f}, '
          f'E = {shape.topo.simplex_count[1]}')
```

:::{admonition} These are page helpers, not library operations
:class: note
`euclib` has no `cylinder`, `cone`, or `capsule` constructor; the helpers above
are written here so that the page can show the properties of the shapes. The
[roadmap](../roadmap.md) records primitive generators as an operation still to
come. If they arrive, the page will call them instead, and the checks --- the
extents, the closure, and the approach to the formulas --- will stay the same.
:::

## When a size is not a size

A radius and a height must be positive; a ring of zero radius is a line, and a
negative height turns the shape inside out before the winding fix can decide
what is outside.

```{code-cell}
for (name, radius, height) in (('a zero radius', 0.0, 1.0),
                               ('a negative height', 1.0, -2.0)):
    try:
        if radius <= 0.0 or height <= 0.0:
            raise ValueError('a radius and a height must be positive')
        cylinder(radius, height)
        print(name, 'was built')
    except ValueError as error:
        print(name, 'is refused:', error)
```

## Provenance

Adapted from trimesh's
[`tests/test_creation.py::test_cylinder`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_creation.py#L80),
[`::test_cone`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_creation.py#L27),
and
[`::test_capsule`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_creation.py#L123),
and from pyvista's
[`tests/core/test_geometric_objects.py::test_cylinder`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_geometric_objects.py#L88),
[`::test_capsule`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_geometric_objects.py#L736),
and
[`::test_capsule_center`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_geometric_objects.py#L745).

The trimesh tests build each primitive from a radius and a height and check
that it is a volume, that it has one body, that its extents are the expected
size, and (for the capsule) that the widest vertices reach the radius and span
the cylindrical height. pyvista's check that the points and faces exist, and
its check that a capsule and a cylinder given the same center and direction
share that center, are the upstream's lighter version of the same construction.
The page checks the extents on a radius of 1 and a height of 2 --- the same
sizes trimesh uses, where the capsule's extents are 2 by 2 by 4 --- and adds
the closure, the Euler characteristic, and the convergence to the volume and
area formulas that the upstream tests take for granted.

The difference is that neither upstream test writes the mesh. pyvista and
trimesh have primitive generators that build a quadric surface from a radius,
a height, and a tessellation count, and they return their own mesh types with
metadata such as the shape's name. `euclib` has no such generator yet, so the
page builds the rings and triangles explicitly and uses `el.trimesh`; the
result is a `TriMesh` like any other, and the page can measure it, close it,
and compare it with the formulas without needing a primitive's metadata.
