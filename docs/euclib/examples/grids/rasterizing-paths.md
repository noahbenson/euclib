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
# Rasterizing a Two-Dimensional Path

:::{admonition} Warning: This page was written with AI assistance.
:class: warning

This page was written using AI assistance and has not yet been carefully
reviewed by a human. Its code runs and its claims are checked against the test
suite, but a person has not read it end to end. Treat it as an unreviewed draft:
verify the behaviour you depend on against the API reference and the tests, and
report anything that looks wrong.
:::

`euclib.rasterize` turns a two-dimensional `SegPath` into a `Grid` whose
property is a boolean image: a cell is `True` where the path covers it. A
closed loop fills its interior, a loop inside a loop leaves a hole, and an open
path encloses nothing but can still be drawn as an outline. The decisions are
made at the cells' centers, and the result is an image the rest of `euclib`
already knows how to read --- a grid of cells with a property on it.

:::{admonition} What this demonstrates
:class: tip
- `el.rasterize(path, pitch=..., origin=..., resolution=...)` returning a
  `Grid` with a boolean property.
- That a closed square fills its interior, and that a square inside a square
  leaves a hole.
- That the fill and the outline are separate choices, and that a raster can
  have either or both.
- That the covered area approaches the path's area as the cells get smaller.
- That an open path encloses nothing but can still be drawn.
- That the grid's affine puts each cell's center where the pitch and origin
  say it should.
:::

## A filled square

The square is written as a chain whose first corner is repeated at the end, so
the last segment closes it. The raster is 8 by 8 cells of half a unit, with its
low corner at the origin, and all 64 cells are inside.

```{code-cell}
import numpy as np
import euclib as el

def square(size=4.0, low=(0.0, 0.0)):
    (x, y) = low
    return el.segpath(np.array([[x, x + size, x + size, x, x],
                                [y, y, y + size, y + size, y]]))

def draw(image):
    "Prints the raster with one character per cell, y upward."
    values = np.asarray(image['raster'])
    for row in reversed(range(values.shape[1])):
        print(''.join('#' if values[column, row] else '.'
                      for column in range(values.shape[0])))

image = el.rasterize(square(), pitch=0.5, origin=np.array([0.0, 0.0]),
                     resolution=(8, 8))
print('shape    :', image.shape)
print('origin   :', np.asarray(image.origin))
print('spacing  :', np.asarray(image.spacing))
print('covered  :', int(np.asarray(image['raster']).sum()))
draw(image)
```

The affine maps cell indices to cell centers, so the first cell's center is at
`(0.25, 0.25)` and the last is at `(3.75, 3.75)`. The grid's `origin` is the
corner rather than the first center, which is the convention the
[grid page](grids-from-arrays.md) describes.

```{code-cell}
print('center of index (0, 0):',
      np.asarray(image.affine.apply(np.array([[0.0], [0.0]])))[:, 0])
print('property               :', image['raster'].dtype,
      image['raster'].shape)
```

## A hole

Two disjoint loops: a four-unit square and a two-unit square inside it. The
even-odd rule fills a cell inside an odd number of loops and leaves it empty
inside an even number, so the inner square is a hole.

```{code-cell}
def ring(outer=4.0, inner=1.0):
    (o, i) = (outer, inner)
    coords = np.array([[0.0, o, o, 0.0, i, o - i, o - i, i],
                       [0.0, 0.0, o, o, i, i, o - i, o - i]])
    segments = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                         [1, 2, 3, 0, 5, 6, 7, 4]])
    return el.SegPath(coords, el.SegTopology(segments, coord_count=8))

hollow = el.rasterize(ring(), pitch=0.5, origin=np.array([0.0, 0.0]),
                      resolution=(8, 8))
print('covered:', int(np.asarray(hollow['raster']).sum()), 'of 64')
draw(hollow)
```

The four-by-four block in the middle is the hole; the 48 covered cells are the
ring around it.

## Fill, outline, or both

`fill` decides whether closed interiors are covered, and `width` asks for an
outline that many cells wide. They are independent: an open path can only be an
outline, and a closed path can be a filled shape, its outline, or both.

```{code-cell}
def count(**kw):
    found = el.rasterize(ring(), pitch=0.5, origin=np.array([0.0, 0.0]),
                         resolution=(8, 8), **kw)
    return int(np.asarray(found['raster']).sum())

print('fill only     :', count(fill=True, width=None))
print('outline only  :', count(fill=False, width=1.0))
print('fill + outline:', count(fill=True, width=1.0))
```

The outline alone covers 56 cells: the ring's two boundaries, one cell thick.
Filling as well adds the four cells of the outline that reach into the hole,
for 60. The outline width is measured in cells of the smaller pitch, so a
rectangular grid's outline is the same physical thickness along both axes.

## Resolution and pitch

A polygon inscribed in a circle covers a little less than the circle; the error
falls as the cell gets smaller. A circle of radius 1, rasterized into a square
image of pitch `p`, covers an area that approaches $\pi$.

```{code-cell}
angle = np.linspace(0.0, 2.0 * np.pi, 200)
circle = el.segpath(np.array([np.cos(angle), np.sin(angle)]))
for pitch in (0.05, 0.02, 0.01):
    sides = int(2.0 / pitch)
    found = el.rasterize(circle, pitch=pitch,
                         origin=np.array([-1.0, -1.0]),
                         resolution=(sides, sides))
    covered = int(np.asarray(found['raster']).sum()) * pitch ** 2
    print(f'pitch {pitch:.2f}: {found.shape[0]:>3} by {found.shape[1]:>3}, '
          f'area {covered:.4f}, relative error '
          f'{abs(covered - np.pi) / np.pi:.2e}')
```

The relative error falls roughly with the square of the pitch --- a cell five
times smaller has about a thirty-fifth of the error where a square law would
give a twenty-fifth --- which is what a boundary one cell thick would do: the
error is the perimeter times the cell size, and the perimeter is fixed.

## An open path

A line encloses nothing, so `fill=True` covers no cells. Asking for an outline
draws the cells the line passes through.

```{code-cell}
line = el.segpath(np.array([[0.0, 1.0], [0.0, 1.0]]))
none = el.rasterize(line, pitch=0.1, origin=np.array([0.0, 0.0]),
                    resolution=(10, 10))
drawn = el.rasterize(line, pitch=0.1, origin=np.array([0.0, 0.0]),
                     resolution=(10, 10), fill=False, width=1.0)
print('filled  :', int(np.asarray(none['raster']).sum()))
print('outlined:', int(np.asarray(drawn['raster']).sum()))
draw(drawn)
```

The diagonal is the set of cells whose centers lie within half a cell of the
segment. A line that passes exactly through the corner between four cells is
answered by the same center rule as any other, which is why the outline is a
staircase rather than an unbroken ideal line.

:::{admonition} What the raster is, and is not
:class: note
The raster is a boolean image on a `Grid`, not a picture and not an
anti-aliased sample: a cell is either covered or not, and the decision is made
at its center. A caller who needs coverage fractions, colors, or a different
sampling rule can rasterize at a finer pitch and aggregate the cells. The
property is named `'raster'` unless the `name` keyword says otherwise. The
operation returns NumPy values, carries the path's units into the grid's
affine, and needs no image library: the crossings, the even-odd rule, and point
to segment distances are computed directly.
:::

:::{admonition} Overlapping loops
:class: note
The even-odd rule is what makes a nested loop a hole. It also means that two
loops that overlap without one containing the other leave their intersection
empty, where a union would fill it. A caller who needs the union of overlapping
regions can rasterize each loop separately and take the logical or of the
images.
:::

## Refusals

A raster is defined for a two-dimensional path with at least one segment, and
its lattice needs a positive pitch and at least one cell.

```{code-cell}
for (name, action) in (
        ('a three-dimensional path',
         lambda: el.rasterize(el.segpath(np.zeros((3, 2))))),
        ('a zero pitch',
         lambda: el.rasterize(square(), pitch=0.0)),
        ('a zero resolution',
         lambda: el.rasterize(square(), pitch=1.0, resolution=(0, 4)))):
    try:
        action()
        print(name, 'was rasterized')
    except (ValueError, TypeError) as error:
        print(name, 'is refused:', error)
```

## Provenance

Adapted from trimesh's
[`tests/test_raster.py::test_rasterize`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_raster.py#L8)
and
[`::test_nested`](https://github.com/mikedh/trimesh/blob/6c158cce226500fc64b12acd265af0cf01d1c212/tests/test_raster.py#L42).

The upstream tests rasterize a wrench outline at a fine pitch and check that
the filled image covers more cells than the outline, that filling and drawing
the outline together covers more than either alone, and that a different pitch
gives a different image. The nested test builds circles of several radii,
rasterizes them, and checks that the covered cell count times the cell area is
within one percent of the path's area. The page makes the same comparisons on
a square and on a ring --- where the outline, the fill, and both are visibly
different --- and makes the area check on a circle at three pitches, which is
the same property the nested test states.

The difference is in the machinery and the answer. trimesh delegates to
Pillow: it converts the path to pixel coordinates, draws lines and polygons
into a mode-`1` image, and returns the `PIL.Image`. `euclib` needs no image
library: it tests cell centers against the loops' edges and against the
segments, and returns a `Grid` with a boolean property, so the result can be
interpolated, measured, or combined with another grid. trimesh fills each
exterior polygon and then clears its children, using the path's enclosure
hierarchy; `euclib` uses the even-odd rule over the loops. The two agree on
nested loops, and differ on overlapping ones, which the note above states.
trimesh's `width` is in pixels; `euclib`'s is in cells.
