# -*- coding: utf-8 -*-
###############################################################################
# euclib/utils/_quadrature.py
'''Quadrature rules on the reference simplices.

A property's interpolation is a polynomial in an element's barycentric
coordinates --- `polynomial_fit` writes one of the element's own degree, and the
Bezier fits write Bernstein polynomials of the same degree --- so a quadrature
rule that is exact to that degree integrates the interpolated field over an
element exactly. That is what `euclib.ops.integrate` is built on: the field is
read at a handful of points and weighted, and no fit is reimplemented.

The rules are stated on the *reference* element --- the unit segment, the
triangle with corners at the axes, the tetrahedron spanned by the origin and the
three unit axes --- in barycentric coordinates, so that a rule is the same
whatever an element's shape or orientation is. A point's barycentric weights sum
to 1 and a rule's weights sum to 1, so that a simplex's measure scales the
weighted sum to its integral: the integral of ``f`` over an element is the
element's measure times ``sum(w * f(p))`` over the rule's points.

**Every rule here has positive weights.** The fewest-point rules at degree 3 have
one negative weight --- on a triangle at four points and on a tetrahedron at five
--- which is a needless risk for a library whose results are compared and cached,
so the degree-3 rules step up to rules with positive weights instead: six points
on a triangle, and a Gauss-Jacobi product on a tetrahedron. What that costs is
points, and what it buys is that no weight can cancel another.

Each rule is checked in the tests by integrating the monomials, which is the only
thing that says a rule is what it claims --- and it is what found that the
four-point tetrahedron rule below is degree 2 and not, as it is often listed,
degree 3.
'''

# Dependencies ###############################################################

from __future__ import annotations

from fractions import Fraction
from math import factorial

from numpy import asarray, zeros
from numpy.linalg import eigh


# Gauss-Jacobi ###############################################################

def _zip_pad(a, b, /):
    '''Zips two coefficient sequences, padding the shorter with zeros.'''
    n = max(len(a), len(b))
    return list(zip(list(a) + [0] * (n - len(a)), list(b) + [0] * (n - len(b))))


def _moment(k, alpha, /):
    '''The integral of ``t**k`` against ``(1-t)**alpha`` on the unit segment.

    Exact, and rational, which is what lets the recurrence below be built
    without a quadrature of its own. ``alpha`` is a whole number, since the
    weights the Duffy map produces are powers of ``(1-t)``.
    '''
    return Fraction(factorial(k) * factorial(alpha),
                    factorial(k + alpha + 1))


def _inner(p, q, alpha, /):
    '''The inner product of two polynomials under ``(1-t)**alpha`` on [0,1].

    A polynomial is a sequence of coefficients, lowest power first.
    '''
    res = Fraction(0)
    for (i, a) in enumerate(p):
        if a == 0:
            continue
        for (j, b) in enumerate(q):
            if b == 0:
                continue
            res += Fraction(a) * Fraction(b) * _moment(i + j, alpha)
    return res


def _poly_mul_monic(p, /):
    '''Multiplies a polynomial by ``t``.'''
    return [0] + list(p)


def _gauss_jacobi(n, alpha, /):
    '''Gauss-Jacobi nodes and weights on [0,1] for the weight ``(1-t)**alpha``.

    The rule is built from the three-term recurrence of the monic orthogonal
    polynomials, whose coefficients are the exact inner products above --- so
    the nodes are the eigenvalues of a symmetric tridiagonal matrix and the
    weights are its eigenvectors' first components, which is Golub and Welsch's
    construction. It is used rather than a table because a table for every
    degree the Duffy map needs is a table to get wrong, and because this way the
    weights are positive by construction at any degree.

    Parameters
    ----------
    n : int
        The number of nodes. The rule is exact for polynomials of degree
        ``2 * n - 1``.
    alpha : int
        The power of ``(1-t)`` in the weight.

    Returns
    -------
    nodes, weights : numpy.ndarray
        The nodes, ascending, and their weights, which sum to the weight's
        integral over [0,1].
    '''
    # The monic polynomials, by the recurrence p_{k+1} = (t - a_k) p_k
    # - b_k p_{k-1}, with a_k and b_k the exact inner products.
    polys = [[Fraction(1)]]
    (diag, off) = ([], [])
    for k in range(n):
        p = polys[k]
        pp = _poly_mul_monic(p)
        norm = _inner(p, p, alpha)
        a = _inner(pp, p, alpha) / norm
        diag.append(a)
        if k:
            # b_k is the Jacobi matrix's off-diagonal at (k-1, k), and the
            # recurrence's own coefficient for p_{k-1} --- so it belongs at
            # (0, 1) rather than at (0, 0), which is a_0 alone. It is computed
            # whenever p_k is, which is one more time than a p_{k+1} is built:
            # the last node's off-diagonal is the one that reaches it.
            off.append(norm / _inner(polys[k - 1], polys[k - 1], alpha))
        if k + 1 < n:
            nxt = [c - a * d for (c, d) in _zip_pad(pp, p)]
            if k:
                nxt = [c - off[-1] * d
                       for (c, d) in _zip_pad(nxt, polys[k - 1])]
            polys.append(nxt)
    # The symmetric tridiagonal Jacobi matrix; its eigenvalues are the nodes.
    m = zeros((n, n))
    for (i, a) in enumerate(diag):
        m[i, i] = float(a)
    for (i, b) in enumerate(off):
        m[i, i + 1] = m[i + 1, i] = float(b) ** 0.5
    (vals, vecs) = eigh(m)
    order = vals.argsort()
    nodes = vals[order]
    total = float(_inner(polys[0], polys[0], alpha))
    weights = total * vecs[0, order] ** 2
    return (nodes, weights)


# Rules ######################################################################

def _segment_rule(degree, /):
    '''A Gauss-Legendre rule on the unit segment.'''
    n = (degree + 2) // 2
    (nodes, weights) = _gauss_jacobi(n, 0)
    return (asarray([[1.0 - t, t] for t in nodes]), asarray(weights))


def _triangle_rule(degree, /):
    '''A rule on the reference triangle, in barycentric coordinates.

    The centroid, then the three mid-edge points, which is the quadratic; at
    degree 3 and above, the Duffy product described below --- four points where
    the fewest positive-weight rule at that degree needs six.
    '''
    if degree <= 1:
        return (asarray([[1 / 3, 1 / 3, 1 / 3]]), asarray([1.0]))
    if degree == 2:
        pts = [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6],
               [1 / 6, 1 / 6, 2 / 3]]
        return (asarray(pts), asarray([1 / 3, 1 / 3, 1 / 3]))
    # The Duffy product, as the tetrahedron's is: the reference triangle is the
    # unit square under ``x = u(1-v)``, ``y = v``, whose Jacobian is ``(1-v)``,
    # so a Gauss-Legendre rule in ``u`` and a Gauss-Jacobi one in ``v`` with
    # that weight integrate a polynomial of the degree exactly. Four points,
    # every weight positive.
    n = (degree + 2) // 2
    (us, wu) = _gauss_jacobi(n, 0)
    (vs, wv) = _gauss_jacobi(n, 1)
    pts = []
    wts = []
    for (i, u) in enumerate(us):
        for (j, v) in enumerate(vs):
            (x, y) = (u * (1 - v), v)
            pts.append([x, y, 1.0 - x - y])
            wts.append(wu[i] * wv[j])
    # The product's weights sum to the triangle's area, the Jacobian being part
    # of the rule; normalized here so that every rule's weights sum to 1.
    wts = asarray(wts)
    return (asarray(pts), wts / wts.sum())


def _tetrahedron_rule(degree, /):
    '''A rule on the reference tetrahedron, in barycentric coordinates.

    Below degree 3 the rule is the centroid, and then Keast's four-point rule at
    its degree-2 coordinates --- which is often listed as degree 3 and is not:
    it integrates a cubic to within a few parts in a thousand, which is the sort
    of error a test catches and an eye does not.

    At degree 3 and above the rule is the Duffy product: the reference
    tetrahedron is the unit cube under ``x = u(1-v)(1-w)``, ``y = v(1-w)``,
    ``z = w``, whose Jacobian is ``(1-v)(1-w)**2``, so a Gauss-Jacobi rule in
    each of the three coordinates integrates a polynomial of the degree exactly.
    It costs points --- eight where four would do with a negative weight --- and
    every one of its weights is positive.
    '''
    if degree <= 1:
        return (asarray([[1 / 4, 1 / 4, 1 / 4, 1 / 4]]), asarray([1.0]))
    if degree == 2:
        (a, b) = (0.5854101966249685, 0.1381966011250105)
        pts = []
        for i in range(4):
            one = [b, b, b, b]
            one[i] = a
            pts.append(one)
        return (asarray(pts), asarray([0.25] * 4))
    n = (degree + 2) // 2
    (us, wu) = _gauss_jacobi(n, 0)
    (vs, wv) = _gauss_jacobi(n, 1)
    (ws, ww) = _gauss_jacobi(n, 2)
    pts = []
    wts = []
    for (i, u) in enumerate(us):
        for (j, v) in enumerate(vs):
            for (k, w) in enumerate(ws):
                (x, y, z) = (u * (1 - v) * (1 - w), v * (1 - w), w)
                pts.append([x, y, z, 1.0 - x - y - z])
                wts.append(wu[i] * wv[j] * ww[k])
    # The Gauss-Jacobi weights sum to their weight's integral, so the product's
    # sum to the tetrahedron's volume --- the Jacobian is part of the rule. Every
    # other rule here sums to 1, and a rule's weights sum to 1 so that an
    # element's measure scales them to its integral, so this one is normalized
    # to match.
    wts = asarray(wts)
    return (asarray(pts), wts / wts.sum())


#: The rule for each simplex order, which is also its dimension.
_BUILDERS = {1: _segment_rule, 2: _triangle_rule, 3: _tetrahedron_rule}

#: The cache of built rules, since a rule is fixed by its order and degree.
_RULES = {}


# Functions ##################################################################

def quadrature(order, degree, /):
    '''The quadrature rule for a simplex of a given order and degree.

    A rule is chosen for the *highest* degree it must reach: asking for degree 2
    on a triangle gives the three-point rule, and asking for degree 3 gives the
    six-point one, which reaches degree 4 and so covers it.

    Parameters
    ----------
    order : int
        The simplex's order, which is its dimension: 1 for a segment, 2 for a
        triangle, 3 for a tetrahedron.
    degree : int
        The degree of polynomial the rule must integrate exactly.

    Returns
    -------
    points, weights : numpy.ndarray
        The rule's points as barycentric coordinates of the reference simplex,
        one row per point, and their weights, which sum to 1 --- so that a
        simplex's measure scales their sum to its integral.

    Raises
    ------
    ValueError
        If the order is not 1, 2, or 3, or if the degree is below 1.
    '''
    kind = {1: 'segment', 2: 'triangle', 3: 'tetrahedron'}.get(int(order))
    if kind is None:
        raise ValueError(
            f"a simplex has an order of 1, 2, or 3; found {order}")
    degree = int(degree)
    if degree < 1:
        raise ValueError(
            f"a quadrature rule reaches degree 1 or more; found {degree}")
    if (order, degree) not in _RULES:
        _RULES[(order, degree)] = _BUILDERS[int(order)](degree)
    return _RULES[(order, degree)]
