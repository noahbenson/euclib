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
# Metadata, Immutability, and Transient Edits

`euclib` geometries are immutable: an operation returns a new object and leaves
the old one alone. That discipline has two consequences worth seeing directly —
metadata that travels with an object, and a *transient* copy that can be edited
in place when making a new object for every small change would be wasteful.

:::{admonition} What this demonstrates
:class: tip
- `withmeta`, `dropmeta`, and `metadata`: arbitrary metadata attached to a
  geometry, preserved by later operations.
- That `withprop` and `transformed` return new geometries and never modify the
  original.
- `transient()` and `persistent()`: editing a copy in place and then freezing
  it, which is how a sequence of small edits avoids allocating a geometry each
  time.
- How this differs from libraries whose transforms are mutable objects.
:::

## Metadata travels with the geometry

Metadata is a mapping of arbitrary, hashable keys to values. It is not data
about the geometry's shape — it is what you want to remember *about* the object:
units, a source, a note.

```{code-cell}
import numpy as np
import euclib as el

mesh = el.trimesh(np.array([[0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0]]),
                  np.array([[0], [1], [2]], dtype='int64'))

tagged = mesh.withmeta(units='mm', source='hand-written')
print('metadata        :', dict(tagged.metadata))
print('the original    :', dict(mesh.metadata))

print('after dropmeta  :', dict(tagged.dropmeta('source').metadata))
```

Metadata is not lost when the geometry goes through an operation:

```{code-cell}
moved = tagged.transformed(el.types.affine_translation([1.0, 0.0, 0.0]))
print('metadata after transforming:', dict(moved.metadata))
print('second vertex moved from', np.round(np.asarray(tagged.coords)[:, 1], 3).tolist(),
      'to', np.round(np.asarray(moved.coords)[:, 1], 3).tolist())
```

## Operations never edit their input

Both `withprop` and `transformed` return a new geometry. The original keeps its
coordinates and its (empty) property set:

```{code-cell}
carrying = mesh.withprop('value', np.array([1.0, 2.0, 3.0]))
print('new geometry has value:', 'value' in carrying.properties)
print('original still has it :', 'value' in mesh.properties)
print('original still spans  :', np.round(np.asarray(mesh.coords).max(axis=1), 3).tolist())
```

## Editing in place: transients

When many small edits are needed, allocating a fresh immutable geometry each
time is wasteful. `transient()` returns a copy that *can* be edited in place;
`persistent()` freezes it back into an immutable object. The original is never
touched.

```{code-cell}
from euclib.abc import Property

working = mesh.transient()
working.coords = np.asarray(mesh.coords) * 2.0
working.properties = {'scaled': Property(np.array([1.0, 2.0, 3.0]), (3,))}

frozen = working.persistent()
print('frozen  area  :', round(float(np.sum(frozen.measures)), 4),
      'properties:', sorted(frozen.properties.keys()))
print('original area :', round(float(np.sum(mesh.measures)), 4),
      'properties:', sorted(mesh.properties.keys()))
```

Scaling the coordinates by two multiplies the triangle's area by four, and the
frozen copy has that larger area while the original is unchanged. A transient
is a thread-local scratch object, not a way to share mutation: once it is frozen
with `persistent()`, the result is an ordinary immutable geometry.

:::{admonition} Why not just mutate?
:class: tip
Because immutable objects can be shared freely between threads and cached by
value, `euclib` makes them the default and keeps mutation as an explicitly
opt-in, local convenience. A transform in some libraries is a mutable object you
nudge step by step; here the same sequence is written as a new object per step,
or as one transient edited in place and frozen at the end.
:::

:::{admonition} Provenance
:class: note
**Upstream:** `trimesh/trimesh` — `tests/test_meta.py`, method `test_glb`
(line 8); and `pyvista/pyvista` — `tests/core/test_utilities.py`, function
`test_transform_translate` (line 1971).

**Pinned revisions:** trimesh
[`fcf660f`](https://github.com/trimesh/trimesh/blob/fcf660feb0a14c68fd3945789e8ed77e260f9167/tests/test_meta.py#L8);
pyvista
[`85fbb5c`](https://github.com/pyvista/pyvista/blob/85fbb5c71c02943ebe9dd10051235fe4d05b39aa/tests/core/test_utilities.py#L1971).

**How this was adapted.** The trimesh test attaches metadata to a scene and to
each mesh and asserts that it survives an export/import round trip; the pyvista
test mutates a `Transform` object in place with successive calls and checks the
resulting matrix. Neither is copied. The two are combined here because `euclib`
answers both with one design: metadata is attached with `withmeta` and carried
through operations (the trimesh idea, checked by transforming rather than by
exporting, since `euclib` has no serialization), while the in-place mutation the
pyvista test relies on is available only through an explicit transient (the
pyvista idea, but opt-in and frozen at the end). The page therefore documents a
genuine difference in defaults rather than presenting the two libraries'
behavior as equivalent.
:::

:::{seealso}
- [Point clouds and their properties](point-clouds.md) for the property API the
  transient's `properties` field mirrors.
- [Affine transforms](../transforms/affine-transforms.md) for the transforms
  applied here.
:::
