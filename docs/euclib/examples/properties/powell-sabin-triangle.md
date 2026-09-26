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
# Powell–Sabin Interpolation on a Triangle

[Clough–Tocher](clough-tocher-triangle.md) makes the field's slope continuous
across a triangle mesh's edges by splitting each triangle into three cubics. It
pays for the smoothness with a **datum the vertices do not carry**: the
derivative across each edge at its midpoint, which the method has to estimate
from the surrounding mesh. That estimate is a choice, and the field carries the
consequences of it.

Powell–Sabin is the other answer to the same question, and it buys the same
smoothness for a different price: it splits each triangle into **six**
quadratics, and needs **nothing beyond the value and the gradient at each
vertex**. This page derives why that is possible, what the split is, and how the
numbers come out — and checks the claim that matters, that the slope across a
shared edge is continuous.

:::{admonition} What this demonstrates
:class: tip
- Why a quadratic on a whole triangle cannot be made smooth across its edges from
  corner data alone.
- The Powell–Sabin split: six mini-triangles, using the triangle's **incenter**
  and the points where the segments between neighbouring incenters cross the
  shared edges.
- The nineteen ordinates the split has, and why the arithmetic leaves exactly
  as much freedom as the data can fill.
- The check: the derivative across a shared edge is **one linear function** —
  not a piecewise one — and the two triangles produce the *same* function.
:::

## Why a quadratic is harder than a cubic

The Clough–Tocher page's argument was that a cubic on a triangle has ten control
values, nine of which the corners' values and gradients pin, leaving one — not
enough to force the slope across an edge to match the neighbour's. A quadratic is
worse off: it has **six** control values, and the two ends of an edge carry four
numbers between them (two values and two slopes), which are one too many. Write
$q(t)$ for the field restricted to an edge from $a$ to $b$:

$$ q(0) = v_a , \quad q(1) = v_b , \quad q'(0) = d_a , \quad q'(1) = d_b . $$

A quadratic has three coefficients, so four conditions on three unknowns is
over-determined: it has a solution only when the data satisfies

$$ v_b - v_a = \tfrac{1}{2}(d_a + d_b) , $$

which a general field does not. So a quadratic cannot even reproduce its own
endpoint slopes along an edge — and the attempt to *share* an edge between two
triangles inherits that failure directly.

The way out is not to give the quadratic more freedom but to change what has to
match. The amount that must agree across a shared edge is the derivative in the
direction **normal** to it, and for a quadratic that derivative, evaluated along
the edge, is a **linear** function of the edge's parameter. Two linear functions
that agree at two points agree everywhere, so if the two triangles can be made to
produce the same normal derivative at the edge's two **ends**, they produce the
same one along the whole edge — and at the ends the normal derivative is
$\nabla_a \cdot n$ and $\nabla_b \cdot n$, which *both* triangles already agree
about, because both are given the same vertex gradients.

So the whole difficulty collapses to this: arrange the split so that the normal
derivative across the edge is one linear function, rather than a different linear
function on each half of it. That is what the incenter does.

## The split

Join an interior point $Z$ of the triangle to its three corners and to a point
$S$ on each edge. That makes **six** mini-triangles — one per half-edge, each
with $Z$ as its third corner — and each carries a quadratic Bézier patch.

The two choices of point are not free. Powell and Sabin's is:

- $Z$ is the triangle's **incenter**.
- $S$ on an edge shared with a neighbouring triangle is where the segment joining
  the two triangles' incenters crosses the edge.

The second is what makes the method work on a *mesh* rather than on a single
triangle: both triangles sharing an edge compute the same $S$, since the segment
between their incenters is the same segment whichever one asks. The split is
therefore **conforming** — the two sides cut their common edge in the same place
— which is the same requirement the Clough–Tocher edge direction had to meet.

## The numbers, and how many there are

A quadratic patch has six ordinates: three at its corners and three at its edges'
midpoints. The six mini-triangles share corners and edge midpoints, so the
distinct numbers of the whole piecewise quadratic are one per **vertex** of the
split and one per **edge** of it:

| the split has | how many |
|---|---|
| vertices ($A$, $B$, $C$, $Z$, and the three $S$) | 7 |
| edges (six half-edges and six spokes) | 12 |
| **ordinates** | **19** |

Nineteen numbers, then — but they are not all free, and the count of what is
left is what the method turns on. Requiring the pieces to agree in *gradient*
across the six interior edges is a condition on each of them: the normal
derivative of a quadratic along an edge is linear in the edge's parameter, so
matching two pieces' linear functions is **two** conditions apiece — twelve in
all. They are not independent: the condition at one end of a spoke says what the
conditions at the corner it meets already said, and the arithmetic below measures
their rank at ten rather than twelve.

The space that survives is **nine-dimensional**, and the six-dimensional space of
global quadratics sits inside it, as it must — a polynomial is smooth across
everything. Nine is also exactly the value and the two gradient components at
each of three corners. The data is nine numbers and the freedom is nine
dimensions, so the data determines the interpolation — and *unlike*
Clough–Tocher there is nothing left over to choose.

That last sentence is the whole difference between the two methods. Clough–Tocher
spends its freedom on the cross-boundary datum, so it has to be estimated and the
estimate shows in the field. Powell–Sabin spends it on the split, which is
geometry rather than data, so the field is whatever the vertex values and
gradients say and nothing else.

## Where each ordinate comes from

Three of the four kinds have closed forms, and the fourth is a small linear solve.

**The corners hold their own values**, $b_A = v_A$ and so on, as always.

**The midpoint of a half-edge** — say the one between $A$ and $S$ — is where the
field's slope at $A$ in the direction of that edge shows up first:

$$ b = v_A + \tfrac{1}{2}\, \nabla_A \cdot (X_S - X_A) . $$

This is the same rule whatever the triangle looks like, and it holds because the
*only* thing the ordinate at the first control point away from a corner can
express is the derivative there.

**The point where the edge is split** is where the two halves of the edge meet,
and requiring *them* to agree — the tangential version of the same argument, since
the field along an edge should not crease at $S$ either — gives a weighted
average of the two half-edge ordinates next to it:

$$ b_S = (1-s)\, b_{A S} + s\, b_{S B}, \qquad s = \frac{|AS|}{|AB|} . $$

It is worth seeing why: the restriction of the field to the edge is a quadratic
spline with one knot at $S$, and the four numbers the two ends carry (their values
and their along-edge slopes) determine exactly one such spline. This is that
spline's ordinate at its knot.

**Everything else** — the ordinate at $Z$ and the six midpoints of the spokes —
follows from C¹ across the six interior edges, which is a linear system. It has a
unique solution, and the section after next shows what it produces.

```{code-cell}
import numpy as np

def incenter(corners):
    """The incenter of a triangle, from its side lengths."""
    (P, Q, R) = (corners['A'], corners['B'], corners['C'])
    (a, b, c) = (np.linalg.norm(Q - R), np.linalg.norm(R - P),
                 np.linalg.norm(P - Q))
    w = np.array([a, b, c]) / (a + b + c)
    return w[0] * P + w[1] * Q + w[2] * R

def crossing(one, two):
    """Where the segment between two triangles' incenters crosses their shared
    edge — which both triangles compute the same way, so the split conforms."""
    (P, Q) = (one['A'], one['B'])
    (Z1, Z2) = (incenter(one), incenter(two))
    n = np.array([-(Q - P)[1], (Q - P)[0]])
    s = -float(n @ (Z1 - P)) / float(n @ (Z2 - Z1))
    return Z1 + s * (Z2 - Z1)

def mid(p, q):
    return '-'.join(sorted((p, q)))

class Split:
    """The six-way split of one triangle, and its nineteen ordinates."""

    def __init__(self, corners, inner, edge_points):
        self.vert = dict(corners)
        self.vert['Z'] = np.asarray(inner, float)
        for name in ('AB', 'BC', 'CA'):
            self.vert['S_' + name] = np.asarray(edge_points[name], float)
        self.points = ['A', 'B', 'C', 'Z', 'S_AB', 'S_BC', 'S_CA']
        self.halves = [('A', 'S_AB'), ('S_AB', 'B'), ('B', 'S_BC'),
                       ('S_BC', 'C'), ('C', 'S_CA'), ('S_CA', 'A')]
        self.mini = [(p, q, 'Z') for (p, q) in self.halves]
        self.ordinates = (self.points
                          + [mid(p, q) for (p, q) in self.halves]
                          + [mid(p, 'Z') for p in self.points if p != 'Z'])
        self.where = {n: i for (i, n) in enumerate(self.ordinates)}
        self.count = len(self.ordinates)

    def place(self, name):
        if name in self.vert:
            return self.vert[name]
        (p, q) = name.split('-')
        return (self.vert[p] + self.vert[q]) / 2.0

    def slots(self, tri):
        (P, Q, R) = tri
        return [self.where[P], self.where[Q], self.where[R],
                self.where[mid(P, Q)], self.where[mid(P, R)],
                self.where[mid(Q, R)]]

    def bary(self, point, tri):
        (P0, P1, P2) = [self.vert[x] for x in tri]
        M = np.stack([P1 - P0, P2 - P0], axis=1)
        (s, t) = np.linalg.solve(M, np.asarray(point, float) - P0)
        return np.array([1.0 - s - t, s, t])

    def basis(self, u, v, w):
        return np.array([u*u, v*v, w*w, 2*u*v, 2*u*w, 2*v*w])

    def dbasis(self, tri, point):
        """The derivatives of the six basis functions at a point."""
        (P0, P1, P2) = [self.vert[x] for x in tri]
        M = np.stack([P1 - P0, P2 - P0], axis=1)
        dsv = np.linalg.inv(M)
        db = np.stack([-dsv.sum(axis=0), dsv[0], dsv[1]])
        (u, v, w) = self.bary(point, tri)
        return np.array([
            [2*u*db[0,0], 2*u*db[0,1]], [2*v*db[1,0], 2*v*db[1,1]],
            [2*w*db[2,0], 2*w*db[2,1]],
            [2*(u*db[1,0]+v*db[0,0]), 2*(u*db[1,1]+v*db[0,1])],
            [2*(u*db[2,0]+w*db[0,0]), 2*(u*db[2,1]+w*db[0,1])],
            [2*(v*db[2,0]+w*db[1,0]), 2*(v*db[2,1]+w*db[1,1])]]).T

    def value_row(self, tri, point):
        row = np.zeros(self.count)
        row[self.slots(tri)] = self.basis(*self.bary(point, tri))
        return row

    def gradient_row(self, tri, point, direction):
        row = np.zeros(self.count)
        row[self.slots(tri)] = self.dbasis(tri, point).T @ direction
        return row

    def interior(self):
        """Each interior edge, and the two mini-triangles that share it."""
        return [(mid(p, 'Z'),
                 [t for t in self.mini if p in t and 'Z' in t])
                for p in self.points if p != 'Z']

    def data_rows(self):
        """The data conditions: a value and a gradient at each of the corners."""
        rows = []
        for name in ('A', 'B', 'C'):
            for tri in self.mini:
                if name in tri:
                    rows.append(self.value_row(tri, self.vert[name]))
                    break
        for tri in self.mini:
            for name in tri:
                if name in ('A', 'B', 'C'):
                    for axis in (0, 1):
                        d = np.zeros(2); d[axis] = 1.0
                        rows.append(self.gradient_row(tri, self.vert[name], d))
        return np.array(rows)

    def interior_rows(self):
        """C¹ across each interior edge: the two pieces' normal derivatives must
        agree. Each one is linear along the edge, so agreeing at both of its ends
        is agreeing everywhere."""
        rows = []
        for (edge, tris) in self.interior():
            (p, q) = edge.split('-')
            n = self.vert[q] - self.vert[p]
            n = np.array([-n[1], n[0]]); n = n / np.linalg.norm(n)
            (one, two) = tris
            for point in (self.vert[p], self.vert[q]):
                rows.append(self.gradient_row(one, point, n)
                            - self.gradient_row(two, point, n))
        return np.array(rows)

    def solve(self, values, gradients):
        """The ordinates of the interpolant matching the data and C¹ here."""
        rows = []; rhs = []
        for name in ('A', 'B', 'C'):
            for tri in self.mini:
                if name in tri:
                    rows.append(self.value_row(tri, self.vert[name]))
                    rhs.append(values[name]); break
        for tri in self.mini:
            for name in tri:
                if name in ('A', 'B', 'C'):
                    for axis in (0, 1):
                        d = np.zeros(2); d[axis] = 1.0
                        rows.append(self.gradient_row(tri, self.vert[name], d))
                        rhs.append(gradients[name][axis])
        for row in self.interior_rows():
            rows.append(row); rhs.append(0.0)
        M = np.array(rows); b = np.array(rhs)
        got = np.linalg.lstsq(M, b, rcond=None)[0]
        return (got, float(np.abs(M @ got - b).max()))

    def gradient_at(self, ords, point):
        """The gradient of the interpolant at a point, from the piece holding it."""
        for tri in self.mini:
            (P0, P1, P2) = [self.vert[x] for x in tri]
            M = np.stack([P1 - P0, P2 - P0], axis=1)
            try:
                (s, t) = np.linalg.solve(M, point - P0)
            except np.linalg.LinAlgError:
                continue
            if s > -1e-9 and t > -1e-9 and s + t < 1 + 1e-9:
                return self.dbasis(tri, point) @ ords[self.slots(tri)]
        raise ValueError("no piece of the split covers that point")
```

## The dimension count

The claim worth checking first is the one the method rests on: that the ordinates
which survive the C¹ conditions are exactly as many as the data can supply. The
code below builds the C¹ conditions — writing each one out by evaluating it on
unit controls, so nothing is transcribed by hand — and measures the dimension of
the space they leave, and then the rank of the same conditions together with the
data.

```{code-cell}
triangle = np.array([[0.0, 0.0], [1.7, 0.1], [0.6, 1.9]])
corners = {'A': triangle[0], 'B': triangle[1], 'C': triangle[2]}
here = Split(corners, incenter(corners),
             {'AB': (triangle[0] + triangle[1]) / 2.0,
              'BC': (triangle[1] + triangle[2]) / 2.0,
              'CA': (triangle[2] + triangle[0]) / 2.0})

(interior, data) = (here.interior_rows(), here.data_rows())
free = here.count - np.linalg.matrix_rank(interior)
print(f"  the split: {here.count} ordinates in {len(here.mini)}"
      f" mini-triangles, held together across {len(here.interior())}"
      f" interior edges")
print(f"  requiring the pieces to agree in gradient across those:"
      f" {interior.shape[0]} conditions of rank"
      f" {np.linalg.matrix_rank(interior)}, so {free} dimensions remain")
print(f"  the global quadratics are among them, which needs "
      f"{(2 + 1) * (2 + 2) // 2} of those {free} dimensions")
print(f"  the data: {data.shape[0]} conditions of rank"
      f" {np.linalg.matrix_rank(data)}")
print(f"  the two together: rank"
      f" {np.linalg.matrix_rank(np.vstack([data, interior]))}, which is all"
      f" {here.count} of the ordinates --- so the data determines the"
      f" interpolation and nothing is left over")
```

## A worked example: two triangles across an edge

The mesh is a quadrilateral cut in two along a diagonal, in a plane, so that the
field has a formula to be compared against:

$$ f(x, y) = 0.6 + 0.5x - 0.4y + 0.7x^2 - 0.3xy + 0.2y^2 . $$

Its gradient is given at every corner, and the two triangles are cut in the same
place on the edge they share — which is the conformity the split needs.

```{code-cell}
A = np.array([0.0, 0.0]); B = np.array([1.6, 0.0])
C = np.array([0.3, 1.4]); D = np.array([0.9, -1.1])
one = {'A': A, 'B': B, 'C': C}
# The second triangle names its corners so that its edge AB is the same segment.
two = {'A': B, 'B': A, 'C': D}

shared = crossing(one, two)
def split_of(corner_set):
    return Split(corner_set, incenter(corner_set),
                 {'AB': shared,
                  'BC': (corner_set['B'] + corner_set['C']) / 2.0,
                  'CA': (corner_set['C'] + corner_set['A']) / 2.0})

(first, second) = (split_of(one), split_of(two))
print(f"  the split point on the shared edge is at"
      f" {float((shared - A) @ (B - A) / ((B - A) @ (B - A))):.4f}"
      f" of the way along it, and both triangles use it")

def f(p):
    return 0.6 + 0.5*p[0] - 0.4*p[1] + 0.7*p[0]**2 - 0.3*p[0]*p[1] + 0.2*p[1]**2

def df(p):
    return np.array([0.5 + 1.4*p[0] - 0.3*p[1], -0.4 - 0.3*p[0] + 0.4*p[1]])

def data(corner_set):
    return ({k: f(p) for (k, p) in corner_set.items()},
            {k: df(p) for (k, p) in corner_set.items()})

(first_ords, first_res) = first.solve(*data(one))
(second_ords, second_res) = second.solve(*data(two))
print(f"  the linear solve: worst residual {first_res:.2e} and {second_res:.2e}")
```

**It reproduces a quadratic.** The interpolant is piecewise quadratic and its
data pins nine numbers at three corners, which is more than the six a quadratic
has; and since the data determines the interpolation, the field comes back
exactly — inside the triangles as well as on them.

```{code-cell}
worst = 0.0
for (split, ords) in ((first, first_ords), (second, second_ords)):
    for tri in split.mini:
        # the piece's own centroid, as a point inside it
        point = np.mean([split.vert[x] for x in tri], axis=0)
        got = float(split.value_row(tri, point) @ ords)
        worst = max(worst, abs(got - f(point)))
print(f"  the difference between the interpolant and the quadratic, over the"
      f" centroids of all twelve pieces: {worst:.3e}")
```

**And the slope across the shared edge is continuous.** This is the check the
whole construction exists for. The derivative across the edge is measured from
*each* side, at points along it — including at the split point, where the single
triangle's own two pieces meet and would show a kink if the split were the wrong
one.

```{code-cell}
(P, Q) = (A, B)
n = np.array([-(Q - P)[1], (Q - P)[0]]); n = n / np.linalg.norm(n)
print("  the derivative across the shared edge, from each side of it:")
print("   position      from one        from two       difference")
worst = 0.0
for u in np.linspace(0.0, 1.0, 11):
    point = P + u * (Q - P)
    (g1, g2) = (first.gradient_at(first_ords, point),
                second.gradient_at(second_ords, point))
    worst = max(worst, abs(float(g1 @ n) - float(g2 @ n)))
    print(f"   {u:6.3f}   {float(g1 @ n):+.10f}   {float(g2 @ n):+.10f}"
          f"   {float(g1 @ n) - float(g2 @ n):+.2e}")
print(f"  worst disagreement anywhere on the edge: {worst:.2e}")
```

Note what the numbers are: they run *linearly* from one end of the edge to the
other, with nothing happening at the split point at $0.4486$ of the way along. A
piecewise-linear derivative would show a kink there; this is Farin's "each
cross-boundary derivative is just one linear function instead of being piecewise
linear", and it is the reason two triangles agree without either of them
estimating anything.

## Triangles in space

As with Clough–Tocher, nothing here is specific to a triangle drawn in a
coordinate plane. Every number above is either a **value** or a derivative in a
direction the geometry defines — a `direction` in the code is a vector in space,
and the 'across' direction is the edge turned a quarter turn *within the
triangle's plane*. A triangle in three-dimensional space has a plane of its own,
so the construction is unchanged; only the coordinates the arithmetic happens in
change. `euclib` builds the split in the plane the triangle lies in, whatever
dimension the geometry is embedded in, so a mesh standing in space is handled by
the same code as one lying flat.

:::{admonition} Where this comes from
:class: note
The element is M. J. D. Powell and M. A. Sabin's, from *Piecewise quadratic
approximations on triangles*, ACM Transactions on Mathematical Software **3**
(1977), 316–325,
[doi:10.1145/355759.355761](https://doi.org/10.1145/355759.355761), which is
where the six-way split, the incenter, and the claim that the interpolant is C¹
across the triangulation come from. The form used here — the quadratic Bézier
patch on a mini-triangle, the ordinates, and the C¹ condition across a shared
edge — is G. Farin's; he states the construction, and the "one linear function"
property of its cross-boundary derivative, in his textbook *Curves and Surfaces
for Computer-Aided Geometric Design* (Academic Press), section 18.6, and the
Bézier-triangle material that the ordinates are written in is his *Triangular
Bernstein-Bézier patches*, Computer Aided Geometric Design **3** (1986), 83–127,
[doi:10.1016/0167-8396(86)90016-6](https://doi.org/10.1016/0167-8396(86)90016-6).

**What is derived here and what is cited.** The closed forms for the half-edge
midpoints and for the edge's split point, the dimension count, and the whole of
the numerical check are worked out for this library and verified on this page;
`euclib`'s own implementation is tested in `euclib/test/types/test_ps.py`. What is
cited is the split itself — the incenter, the conforming edge point, and the
existence of a C¹ piecewise quadratic through vertex values and gradients.
:::

:::{seealso}
- [Clough–Tocher interpolation on a triangle](clough-tocher-triangle.md) for the
  other answer to the same problem, and for why it needs a datum this one does
  not.
- [Bézier interpolation on a triangle](bezier-triangle.md) for the control-value
  construction this page takes as given.
- [Interpolating a Property within a Geometry](interpolation.md) for the metadata
  that chooses an order, and for what happens outside the object.
- [The roadmap](../roadmap.md) for what is not built yet.
:::
