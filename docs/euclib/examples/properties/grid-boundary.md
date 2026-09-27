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
# Interpolating a Grid: the Boundary

[Nearest and linear interpolation](grid-linear.md) leave one question unanswered,
because they barely raise it: what does a kernel find when it reaches past the
grid's own edge? This page answers it for the whole family, and gives the answer
a name — the **boundary extension** — because every wider method needs the same
answer and none of them can be written without choosing one.

:::{admonition} What this demonstrates
:class: tip
- Why a grid's edge is a *rule* and not a detail: a kernel wider than one cell
  always wants a cell that does not exist.
- The three standard extensions — constant, half-sample symmetric, and
  whole-sample symmetric — and the index rule for each.
- That folding the *data* is the same as folding the *kernel*, which is what
  makes these three the natural ones and a general rule the wrong one.
- How a property chooses: the `border` field, its default, and the override.
:::

## Why there is a question at all

A grid covers half a cell past its last sample in each direction and no more —
that is what `contains` tests and what the bounding box reports. But a kernel
does not stop at the last sample: a position a quarter of a cell inside the last
sample's centre wants the cell beyond it, and a position half a cell past the
last sample's centre is still *inside* the grid while its stencil is entirely
outside it. So the interpolation has to be told what it finds there.

It is worth seeing how little room there is. On a five-cell axis the last sample
sits at index 4 and the grid ends at 4.5, so at order 1 — whose stencil is two
cells wide — the extension is only ever consulted over that final half-cell. At
order 3 the stencil is four cells wide, and the extension is consulted over a
cell and a half at each end. The wider the kernel, the more the answer depends on
this choice, which is why it is settled once, here, rather than per method.

## The three extensions

Each extension is a rule that carries an index *outside* the grid back inside:
the value at the outside index is taken to be the value at the index the rule
names. Writing $N$ for the number of cells along an axis and $E(n)$ for the index
that stands in for $n$:

| extension | rule | what the data looks like past the edge |
|---|---|---|
| `'constant'` | $\mathrm{clip}(n, 0, N-1)$ | `. . . a a b c d e e . . .` |
| `'half-symmetric'` | $\min(n \bmod 2N,\ (2N-1-n) \bmod 2N)$ | `. . . b a a b c d e e . . .` |
| `'whole-symmetric'` | $\min(n \bmod (2N-2),\ (2N-2-n) \bmod (2N-2))$ | `. . . b a b c d e d . . .` |

Constant repeats each edge value outward. Half-sample symmetric repeats the edge
value *once and then* reflects, so the edge sample appears twice — this is the
usual choice for resampling an image, and it is the one that commutes with
flipping the data left for right. Whole-sample symmetric reflects about the edge
value itself, so that value appears once per period and the data is an even
function about it.

The names, the rules and the three example sequences are Getreuer's.

```{code-cell}
import numpy as np
import euclib as el

def fold(index, size, border):
    """The real cell an index outside the grid stands for."""
    if border == 'constant':
        return int(np.clip(index, 0, size - 1))
    if border == 'half-symmetric':
        period = 2 * size
        return int(min(index % period, (period - 1 - index) % period))
    period = 2 * size - 2
    if period == 0:
        return 0
    return int(min(index % period, (period - index) % period))

BORDERS = ('constant', 'half-symmetric', 'whole-symmetric')
size = 5
print("which real cell an outside index stands for, on an axis of five cells:")
print("  virtual |" + "|".join(f"{b:^17}" for b in BORDERS))
print("  --------+ " + "+ ".join("-" * 16 for _ in BORDERS))
for n in range(-3, 9):
    mark = "  " if 0 <= n < size else "->"
    print(f"  {n:7d} {mark}|" + "|".join(f"{fold(n, size, b):^17d}"
                                         for b in BORDERS))
```

Reading down the columns: constant is flat outside, half-sample symmetric is flat
for one step and then reflects, and whole-sample symmetric reflects at once. All
three agree inside, which is the next check.

**Folding the data is folding the kernel.** The reason *these* rules and not, say,
continuing the last slope, is that each is a symmetry of the sampling: the
extended sequence is the same sequence read from the other end, so

$$ \sum_n v_{E(n)}\,K(p-n) \;=\; \sum_m \tilde{v}_m\,K(p-m), $$

where $\tilde{v}$ is the *periodic* data the rule produces. Interpolating the
folded indices and extending the data periodically are the same arithmetic, which
is what lets the rule be applied to indices alone — cheaply, once, without
materialising any extended array. A rule that continued a slope would not have
this property, and applying it would mean building the extension first.

```{code-cell}
# A kernel wider than one cell, so that the extensions have room to disagree:
# the cubic convolution kernel with the standard parameter, which reaches two
# cells out.  No method on this page uses it yet -- it is the subject of the
# next page -- but it is what shows the three apart.
def kernel(t):
    t = abs(t)
    if t <= 1.0:
        return 1.5 * t ** 3 - 2.5 * t ** 2 + 1.0
    if t < 2.0:
        return -0.5 * t ** 3 + 2.5 * t ** 2 - 4.0 * t + 2.0
    return 0.0

ramp = np.array([1., 2., 3., 4., 5.])

def wide(sx, border, size=5):
    """The cubic convolution of the ramp at a position, extended by a rule."""
    total = 0.0
    for k in range(int(np.floor(sx)) - 1, int(np.floor(sx)) + 3):
        total += ramp[fold(k, size, border)] * kernel(sx - k)
    return total

print("the ramp is 1 2 3 4 5, on samples at 0 . . . 4; the grid ends at 4.5.")
print("  position |" + "|".join(f"{b:^17}" for b in BORDERS))
for sx in (3.0, 4.0, 4.25, 4.5):
    inside = "  " if sx <= 4.0 else "->"
    print(f"  {sx:8.2f} {inside}|"
          + "|".join(f"{wide(sx, b):^17.4f}" for b in BORDERS))
print()
print("At a sample the three agree, as they must. Past the last one they do not,")
print("and the reason is the rule: half-sample repeats the edge value, so it")
print("holds it flat for a step, where whole-sample turns the data around.")
```

## Choosing one

The extension is a field on the property, and a default of half-sample symmetric.
Like the interpolation itself, it can be overridden at the point of the read,
which is what a resampling routine that wants a different rule will do.

```{code-cell}
# A five-cell axis.  The affine is the identity, so an index is a position.
axis = el.grid((5,)).withprop('v', ramp)

def read(**kw):
    """The value at a quarter of a cell inside the last sample's centre."""
    at = axis.topo.Loc(sx=np.array([4.25]))
    got = axis.prop('v', at=at, interp=('polynomial', 1), **kw)
    return float(np.ravel(np.asarray(got))[0])

print("the value a quarter of a cell inside the last sample's centre:")
for border in BORDERS:
    print(f"  {border:>17s}: {read(border=border):.4f}")
print(f"  {'(no argument)':>17s}: {read():.4f}"
      f"   <- the default, {el.abc.BORDER_DEFAULT}")
print()
carried = axis.withprop('w', ramp, border='whole-symmetric')
print("and a property may carry it, with a read overriding it:")
print("  the property's own:", float(np.ravel(np.asarray(carried.prop(
    'w', at=axis.topo.Loc(sx=np.array([4.25])),
    interp=('polynomial', 1))))[0]))
print("  overridden to constant:", float(np.ravel(np.asarray(carried.prop(
    'w', at=axis.topo.Loc(sx=np.array([4.25])), interp=('polynomial', 1),
    border='constant')))[0]))
```

**A coincidence worth knowing about.** At order 1, constant and half-sample
symmetric give the *same* answer everywhere. Both fold the single cell past the
end onto the edge cell, and a two-cell stencil never asks for anything further
out, so the two rules never get the chance to differ. They part company at order
3 and above, where the stencil reaches two cells out and the rules disagree about
the second one. The check below is the one that matters for the two methods built
so far: that the interior is untouched by the choice at all.

```{code-cell}
rng = np.random.default_rng(0)
worst = 0.0
same = 0.0
for _ in range(300):
    sx = rng.uniform(0.0, 4.5)
    at = axis.topo.Loc(sx=np.array([sx]))
    got = [float(np.ravel(np.asarray(axis.prop(
        'v', at=at, interp=('polynomial', 1), border=border)))[0])
        for border in BORDERS]
    if sx <= 4.0:
        # Inside the samples the three must agree, and must give the ramp.
        same = max(same, max(got) - min(got), abs(got[0] - (1.0 + sx)))
    else:
        worst = max(worst, abs(got[0] - got[1]))
print(f"  inside the samples: the three extensions differ by at most"
      f" {same:.3e}, and all give the ramp")
print(f"  past the last sample: constant and half-symmetric differ by at most"
      f" {worst:.3e}  (and whole-symmetric does its own thing)")
```

:::{admonition} Where this comes from
:class: note
The three extensions, their names, their index rules and the three example
sequences are Pascal Getreuer's, *Linear Methods for Image Interpolation*, Image
Processing On Line **1** (2011), 238–259, section 14.1,
[doi:10.5201/ipol.2011.g_lmii](https://doi.org/10.5201/ipol.2011.g_lmii), which
writes them as the index maps $E(n)$ this page uses and gives the matrix entries
they produce. The observation that folding the data is folding the kernel is the
standard justification for a symmetric extension and is derived above rather than
cited.
:::

:::{seealso}
- [Interpolating a Grid: Nearest and Linear](grid-linear.md) for the index space
  this page assumes, and for the two methods whose boundary this is.
- [Interpolating a Property within a Geometry](interpolation.md) for the metadata
  that chooses an order, and for what happens outside the object.
- [The roadmap](../roadmap.md) for what is not built yet.
:::
