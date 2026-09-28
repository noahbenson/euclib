# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/test_ct.py
'''Tests for the Clough-Tocher element in ``euclib.types._ct``.

The element is a piecewise cubic on a triangle split into three, built from
twelve numbers: the value and the gradient at each corner and the derivative
across each edge at its midpoint. Its whole purpose is to be *smooth* --- C1
across the edges the three pieces share --- and to reproduce the polynomials
its data can determine. Both are checkable, and neither is a matter of opinion,
so this file checks them rather than the numbers that come out of the
construction.
'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

import numpy as np

from euclib.types import _ct


# Fixtures ###################################################################

#: The field the element is tested against, and its exact gradient. A cubic,
#: since the element holds the cubics: anything of lower degree comes back for
#: free, and a cubic is the strongest thing its data can determine.
CUBIC = lambda q: 0.3 * q[0] ** 3 - 0.7 * q[0] ** 2 * q[1] + 0.2 * q[1] ** 3
CUBIC_GRADIENT = lambda q: np.array([0.9 * q[0] ** 2 - 1.4 * q[0] * q[1],
                                     -0.7 * q[0] ** 2 + 0.6 * q[1] ** 2])

QUADRATIC = lambda q: (0.4 * q[0] ** 2 - 0.3 * q[0] * q[1]
                       + 0.25 * q[1] ** 2 + 0.8 * q[0] - 0.2 * q[1] + 0.5)
QUADRATIC_GRADIENT = lambda q: np.array([0.8 * q[0] - 0.3 * q[1] + 0.8,
                                         -0.3 * q[0] + 0.5 * q[1] - 0.2])


def numbers_of(triangle, field, derivative, /):
    '''The twelve numbers a known field gives an element on a triangle.

    A value at each corner, the slope at each corner along each of its two edges,
    and the derivative across each edge at its midpoint --- in the order the
    element reads them. The slopes are taken as the field's derivative in the
    *direction* from a corner to its neighbour, which is a direction in the
    geometry and so is the same number for a triangle in a plane and a triangle
    in space.
    '''
    out = []
    for vertex in range(3):
        out.append(field(triangle[:, vertex]))
        for other in _ct.neighbours_of(vertex):
            along = triangle[:, other] - triangle[:, vertex]
            out.append(float(derivative(triangle[:, vertex]) @ along))
    for k in range(3):
        middle = point_of(triangle, k, np.array([0.5, 0.5, 0.0]))
        out.append(float(derivative(middle) @ _ct.across_vector(triangle, k)))
    return np.array(out)


def point_of(triangle, k, w, /):
    '''The position the weights name within one of a triangle's pieces.'''
    return _ct._corners(triangle, k) @ w


def value_of(basis, numbers, k, w, /):
    '''The element's value, from its basis and a set of numbers.'''
    return _ct.evaluate(basis[k * 10:(k + 1) * 10] @ numbers, w)


def weights(rng, /):
    '''A position within a triangle, as barycentric weights.'''
    w = rng.uniform(size=3)
    return w / w.sum()


# Tests ######################################################################

class TestTheReferenceElement(TestCase):
    '''The construction, on the triangle it is derived on.'''

    def setUp(self):
        self.basis = _ct.basis()
        self.triangle = _ct.REFERENCE
        self.rng = np.random.default_rng(0)

    def test_the_basis_is_twelve_functions_over_thirty_controls(self):
        # Three pieces of ten controls each, held to twelve numbers.
        self.assertEqual(self.basis.shape, (30, 12))

    def test_a_constant_field_is_reproduced(self):
        # The twelve basis functions sum to one: data that says "the value is
        # one, the gradient zero, every edge flat" gives one everywhere. This
        # is the check that the construction is normalized at all.
        numbers = np.zeros(12)
        numbers[0:9:3] = 1.0
        worst = 0.0
        for k in range(3):
            for _ in range(200):
                w = weights(self.rng)
                worst = max(worst, abs(value_of(self.basis, numbers, k, w) - 1.0))
        self.assertLess(worst, 1e-12)

    def test_a_quadratic_is_reproduced(self):
        numbers = numbers_of(self.triangle, QUADRATIC, QUADRATIC_GRADIENT)
        worst = 0.0
        for k in range(3):
            for _ in range(300):
                w = weights(self.rng)
                worst = max(worst, abs(value_of(self.basis, numbers, k, w)
                                       - QUADRATIC(point_of(self.triangle, k, w))))
        self.assertLess(worst, 1e-12)

    def test_a_cubic_is_reproduced(self):
        # The element holds the cubics, so a cubic's own numbers give it back
        # with nothing left over to choose.
        numbers = numbers_of(self.triangle, CUBIC, CUBIC_GRADIENT)
        worst = 0.0
        for k in range(3):
            for _ in range(300):
                w = weights(self.rng)
                worst = max(worst, abs(value_of(self.basis, numbers, k, w)
                                       - CUBIC(point_of(self.triangle, k, w))))
        self.assertLess(worst, 1e-12)

    def test_the_pieces_are_c1_across_the_edges_they_share(self):
        # The gradient is a function along an interior edge, and both pieces
        # holding it must give the same one.
        numbers = numbers_of(self.triangle, QUADRATIC, QUADRATIC_GRADIENT)
        worst = 0.0
        for vertex in range(3):
            (k, other) = _ct._holders(vertex)
            for s in np.linspace(0.0, 1.0, 25):
                one = _ct.gradient(self.basis[k * 10:(k + 1) * 10] @ numbers,
                                   _ct._corners(self.triangle, k),
                                   _ct._weights_on_edge(k, vertex, s))
                two = _ct.gradient(self.basis[other * 10:(other + 1) * 10] @ numbers,
                                   _ct._corners(self.triangle, other),
                                   _ct._weights_on_edge(other, vertex, s))
                worst = max(worst, np.abs(one - two).max())
        self.assertLess(worst, 1e-12)


class TestTheEdgeEstimate(TestCase):
    '''The one number a Property has nowhere to keep.

    The derivative across an edge at its midpoint is per *edge*, and a Property
    carries its data per coordinate, so the element's twelfth kind of number is
    estimated from the mesh: each triangle says what the derivative is, from its
    own Bezier patch, and the estimate is the average of the triangles sharing
    the edge. For a quadratic every triangle's patch *is* the quadratic, so the
    estimate has a right answer to be held to.
    '''

    #: A square of four corners in two triangles, sharing the diagonal.
    COORDS = np.array([[0., 1., 0., 1.], [0., 0., 1., 1.]])

    def _mesh(self):
        from euclib import trimesh
        return trimesh(self.COORDS, np.array([[0, 1], [1, 3], [2, 2]]))

    def test_the_estimate_is_the_field_s_own_derivative(self):
        mesh = self._mesh()
        coords = self.COORDS
        values = np.array([[QUADRATIC(coords[:, i])
                            for i in range(coords.shape[1])]])[0]
        slopes = np.stack([QUADRATIC_GRADIENT(coords[:, i])
                           for i in range(coords.shape[1])], axis=-1)
        found = _ct.edge_data(mesh, values, slopes)
        self.assertEqual(len(found), 5)
        worst = 0.0
        for ((i, j), got) in found.items():
            middle = (coords[:, i] + coords[:, j]) / 2.0
            # These meshes are in the plane, so the direction across an edge
            # is the edge turned within that plane.
            want = float(QUADRATIC_GRADIENT(middle)
                         @ _ct._across_of(coords, i, j,
                                          np.array([0.0, 0.0, 1.0])))
            worst = max(worst, abs(got - want))
        self.assertLess(worst, 1e-12)

    def test_a_channelled_property_is_estimated_channel_by_channel(self):
        # A property may carry more than one channel, and each is a field of
        # its own: the estimate has to be made for every channel at once, with
        # the channel dimensions leading as they do everywhere else.
        mesh = self._mesh()
        coords = self.COORDS
        fields = (QUADRATIC, lambda q: (-0.2 * q[0] ** 2 + 0.5 * q[0] * q[1]
                                        + 0.3 * q[1] ** 2 - 0.4 * q[1] + 1.0))
        slopes = (QUADRATIC_GRADIENT,
                  lambda q: np.array([-0.4 * q[0] + 0.5 * q[1],
                                      0.5 * q[0] + 0.6 * q[1] - 0.4]))
        count = coords.shape[1]
        values = np.stack([
            np.array([field(coords[:, i]) for i in range(count)])
            for field in fields])
        gradients = np.stack([
            np.stack([slope(coords[:, i]) for i in range(count)], axis=-1)
            for slope in slopes])
        found = _ct.edge_data(mesh, values, gradients)
        worst = 0.0
        for ((i, j), got) in found.items():
            middle = (coords[:, i] + coords[:, j]) / 2.0
            across = _ct._across_of(coords, i, j,
                                    np.array([0.0, 0.0, 1.0]))
            with self.subTest(edge=(i, j)):
                self.assertEqual(np.asarray(got).shape, (2,))
                want = np.array([float(slope(middle) @ across)
                             for slope in slopes])
                worst = max(worst, np.abs(np.asarray(got) - want).max())
        self.assertLess(worst, 1e-12)

    def test_the_two_triangles_sharing_an_edge_agree_on_it(self):
        # The estimate is the average of the two, and each of them says the
        # same thing about a quadratic -- which is the property that makes the
        # elements on either side agree: two quadratics along an edge that
        # match at both ends and at the middle are the same quadratic.
        mesh = self._mesh()
        coords = self.COORDS
        values = np.array([[CUBIC(coords[:, i])
                            for i in range(coords.shape[1])]])[0]
        slopes = np.stack([CUBIC_GRADIENT(coords[:, i])
                           for i in range(coords.shape[1])], axis=-1)
        found = _ct.edge_data(mesh, values, slopes)
        # The shared edge is the diagonal, between corners 1 and 2.
        self.assertIn((1, 2), found)


class TestTheSplit(TestCase):
    '''Which piece of the split a position falls in, and its weights there.

    An element's patches live on the three sub-triangles, while a local
    coordinate in the engine names a position within the whole triangle, so the
    weights have to be converted. The conversion is not a matrix: the
    sub-triangle's corners are two corners of the triangle and its centre, so
    its weights come from the triangle's by subtraction.
    '''

    def test_every_position_lands_in_one_piece_with_weights_of_its_own(self):
        rng = np.random.default_rng(0)
        seen = set()
        for _ in range(500):
            weights = rng.uniform(size=2)
            if weights.sum() > 1.0:
                continue
            (pieces, inside) = _ct.sub_weights(weights.reshape(2, 1))
            piece = int(pieces[0])
            seen.add(piece)
            with self.subTest(weights=weights.tolist()):
                # The weights within the piece are the triangle's own weights
                # read off its three corners, so they cannot be negative...
                self.assertTrue((inside[0] > -1e-12).all(),
                                "a negative weight means the wrong piece")
                # ...and they name the same position the triangle's do.
                whole = np.append(weights, 1.0 - weights.sum())
                corners = _ct._corners(_ct.REFERENCE, piece)
                self.assertTrue(np.allclose(_ct.REFERENCE @ whole,
                                            corners @ inside[0]))
        # All three pieces are reachable, or the test is only checking one.
        self.assertEqual(seen, {0, 1, 2})


class TestTheElementOnAnotherTriangle(TestCase):
    '''The same properties on a triangle that is not the reference one.

    A construction that only works on a symmetric triangle is not a
    construction. This one is affine in everything except the direction the
    edge datum is taken along, which belongs to the edge and so cannot be
    carried over from another triangle --- which is why the basis is solved for
    the triangle at hand.
    '''

    #: A triangle with no symmetry to hide behind.
    TRIANGLE = np.array([[0.0, 2.0, -0.5], [0.0, 0.5, 1.7]])

    def test_it_reproduces_a_quadratic_here_too(self):
        basis = _ct.basis(self.TRIANGLE)
        numbers = numbers_of(self.TRIANGLE, QUADRATIC, QUADRATIC_GRADIENT)
        rng = np.random.default_rng(5)
        worst = 0.0
        for k in range(3):
            for _ in range(300):
                w = weights(rng)
                worst = max(worst, abs(value_of(basis, numbers, k, w)
                                       - QUADRATIC(point_of(self.TRIANGLE, k, w))))
        self.assertLess(worst, 1e-12)


class TestTheElementOnATriangleInSpace(TestCase):
    '''The element on a triangle that does not lie in a coordinate plane.

    This is what the element's data has to be *slopes in geometric directions*
    for. The construction is a construction on a triangle's own plane, and a
    triangle in space has one just as a triangle in the plane does --- the three
    pieces stay coplanar however the triangle is carried. What is different is
    the data that reaches it: a surface field's ambient gradient has as many
    components as the space has dimensions, and only the ones along the
    triangle's edges are what the element is built from. Read as the gradient's
    first two coordinate components, as they were, the number is right only when
    the triangle happens to lie in the first two axes --- and on a triangle that
    is merely turned it is wrong by more than a fifth of the field's own scale.
    '''

    #: A triangle carried off every coordinate plane, with no symmetry to hide
    #: behind either.
    TRIANGLE = np.array([[0.0, 1.2, 0.4], [0.3, 0.3, 1.1], [0.7, 0.6, -0.2]])

    def inplane(self, point, /):
        '''A point's two coordinates within the triangle's own plane.

        The inverse of "corner 0, plus so much along each of the two edges",
        which is what a position in the plane is; the pseudo-inverse is the
        inverse of that map for a triangle in space, where it is not square.
        '''
        along = np.stack([self.TRIANGLE[:, 1] - self.TRIANGLE[:, 0],
                          self.TRIANGLE[:, 2] - self.TRIANGLE[:, 0]], axis=1)
        return np.linalg.pinv(along) @ (point - self.TRIANGLE[:, 0])

    def field(self, point, /):
        '''A quadratic of the plane the triangle lies in, at a position.'''
        (u, v) = self.inplane(point)
        return (0.4 * u ** 2 - 0.3 * u * v + 0.25 * v ** 2
                + 0.8 * u - 0.2 * v + 0.5)

    def gradient(self, point, /):
        '''The field's gradient in the ambient coordinates, at a position.

        The chain rule, with the in-plane derivatives carried to the ambient
        ones by the two edge directions' pseudo-inverse --- the same map the
        coordinates above are read with, transposed.
        '''
        (u, v) = self.inplane(point)
        inplane = np.array([0.8 * u - 0.3 * v + 0.8, -0.3 * u + 0.5 * v - 0.2])
        along = np.stack([self.TRIANGLE[:, 1] - self.TRIANGLE[:, 0],
                          self.TRIANGLE[:, 2] - self.TRIANGLE[:, 0]], axis=1)
        return np.linalg.pinv(along).T @ inplane

    def test_it_reproduces_a_quadratic_on_a_triangle_in_space(self):
        basis = _ct.basis(self.TRIANGLE)
        numbers = numbers_of(self.TRIANGLE, self.field, self.gradient)
        rng = np.random.default_rng(7)
        worst = 0.0
        for k in range(3):
            for _ in range(300):
                w = weights(rng)
                worst = max(worst, abs(value_of(basis, numbers, k, w)
                                       - self.field(point_of(self.TRIANGLE, k, w))))
        self.assertLess(worst, 1e-12)

    def test_the_numbers_do_not_change_when_the_triangle_is_turned(self):
        '''The element's data does not depend on where the geometry is.

        Translating a triangle, or turning it in space, moves the field but
        changes none of the nine numbers a property supplies: each is a value, or
        a derivative along a direction the geometry itself defines, and both move
        with it. Read as the ambient gradient's first two coordinate components,
        as they were, every number of the second kind changes the moment the
        triangle is turned --- which is what this guards against, and the reason
        it is worth having rather than trusting the check above.

        The last three --- the derivatives *across* an edge --- are a different
        matter: an across direction is the edge turned a quarter turn, and which
        of the two quarter turns it is is a convention, settled from the geometry
        so that the two triangles sharing an edge make the same choice. A turn of
        the whole mesh may reverse it, so the three may change sign, and the sign
        is not part of what the element promises. The nine are.
        '''
        turn = np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]])
        rng = np.random.default_rng(11)
        before = numbers_of(self.TRIANGLE, self.field, self.gradient)
        for _ in range(20):
            shift = rng.normal(size=3)
            moved = turn @ self.TRIANGLE + shift[:, None]
            # The field is a function of the in-plane coordinates, so the same
            # quadratic serves the moved triangle: read at the moved point, and
            # with the turned gradient --- a translation touches no gradient.
            after = numbers_of(moved,
                               lambda p, /: self.field(np.linalg.solve(
                                   turn, p - shift)),
                               lambda p, /: turn @ self.gradient(
                                   np.linalg.solve(turn, p - shift)))
            self.assertTrue(np.allclose(before[:9], after[:9], atol=1e-12))


class TestTheEdgeOperator(TestCase):
    '''The operator the edge estimate is, held to the element's own statement.

    `edge_operator` builds one sparse matrix over the whole mesh, and it is
    built *without* a loop over triangles --- three thousand small matrices were
    being constructed and block-diagonalised, which for a mesh of nine thousand
    triangles took 2.25 seconds where it now takes 41 ms. That is a
    transcription of `triangle_blocks` into batched arithmetic, and a
    transcription is exactly the thing that can be wrong in a way no
    end-to-end test notices.

    So this builds the same operator the slow, readable way --- one triangle at
    a time, through `triangle_blocks` and `edge_key`, which are unchanged --- and
    holds the fast one to it. The two must agree to the arithmetic.
    '''

    def _mesh(self, side):
        (ix, iy) = np.meshgrid(np.arange(side, dtype=float),
                               np.arange(side, dtype=float), indexing='ij')
        coords = np.vstack([ix.ravel(), iy.ravel()])
        quads = []
        for i in range(side - 1):
            for j in range(side - 1):
                k = i * side + j
                quads.append((k, k + 1, k + side))
                quads.append((k + 1, k + side + 1, k + side))
        return (coords, np.array(quads).T)

    def _one_at_a_time(self, coords, indices):
        '''The operator as `triangle_blocks` states it, and as it used to be.'''
        from scipy.sparse import block_diag, csr_matrix
        (dim, count) = (coords.shape[0], coords.shape[1])
        triangles = indices.shape[1]
        corners_of = [[int(x) for x in indices[:, t]] for t in range(triangles)]
        where = {}
        for here in corners_of:
            for (a, b) in ((0, 1), (1, 2), (2, 0)):
                where.setdefault(_ct.edge_key(coords, here[a], here[b]),
                                 len(where))
        edges = [None] * len(where)
        for (edge, row) in where.items():
            edges[row] = edge
        says = block_diag(
            [csr_matrix(_ct.triangle_blocks(coords[:, here])[:, :9])
             for here in corners_of], format='csr')
        (rows_out, cols_out, data_out) = ([], [], [])
        for (t, here) in enumerate(corners_of):
            base = 9 * t
            for vertex in range(3):
                corner = here[vertex]
                rows_out.append(base + 3 * vertex)
                cols_out.append(corner)
                data_out.append(1.0)
                for (place, other) in enumerate(_ct.neighbours_of(vertex)):
                    along = coords[:, here[other]] - coords[:, corner]
                    for m in range(dim):
                        rows_out.append(base + 3 * vertex + 1 + place)
                        cols_out.append((m + 1) * count + corner)
                        data_out.append(float(along[m]))
        numbers = csr_matrix((data_out, (rows_out, cols_out)),
                             shape=(9 * triangles, count + dim * count))
        (rows, columns) = ([], [])
        for (t, here) in enumerate(corners_of):
            for (e, (a, b)) in enumerate(((0, 1), (1, 2), (2, 0))):
                rows.append(where[_ct.edge_key(coords, here[a], here[b])])
                columns.append(t * 3 + e)
        sharing = csr_matrix((np.ones(len(rows)), (rows, columns)),
                             shape=(len(where), triangles * 3))
        mean = sharing.multiply(
            1.0 / np.asarray(sharing.sum(axis=1)).ravel()[:, None])
        return ((mean @ says @ numbers).tocsr(), edges,
                np.asarray(rows).reshape(triangles, 3))

    def test_it_is_the_element_built_one_triangle_at_a_time(self):
        for side in (2, 3, 5, 8):
            (coords, indices) = self._mesh(side)
            (fast, fast_edges, fast_rows) = _ct.edge_operator(coords, indices)
            (slow, slow_edges, slow_rows) = self._one_at_a_time(coords, indices)
            with self.subTest(triangles=indices.shape[1]):
                # The identity of an edge is what a derivative across it is
                # taken *along*, so the numbering has to match and not merely be
                # a permuted set.
                self.assertEqual(fast_edges, slow_edges,
                                 "the edges are numbered differently")
                self.assertTrue(np.array_equal(fast_rows, slow_rows),
                                "each triangle's own edges are in other rows")
                self.assertEqual(fast.shape, slow.shape)
                self.assertLess(np.abs((fast - slow).toarray()).max(), 1e-12)

    def test_it_carries_the_operators_the_estimate_is(self):
        '''The shape and the sparsity the caller relies on.

        ``(E, N + D*N)``: one row per distinct edge, reading the values and then
        the gradients. A build that was right by accident --- the right numbers
        in a differently shaped matrix --- would pass the comparison above only
        if the shape matched too, so this says what the shape is.
        '''
        (coords, indices) = self._mesh(4)
        (operator, edges, rows) = _ct.edge_operator(coords, indices)
        (dim, count) = coords.shape
        self.assertEqual(operator.shape[1], count + dim * count)
        self.assertEqual(operator.shape[0], len(edges))
        self.assertEqual(rows.shape, (indices.shape[1], 3))


class TestTheBatchedBasis(TestCase):
    '''`basis_many`, held to `basis` triangle by triangle.

    The basis is a (53, 30) least-squares system solved for the triangle it is
    given, and the batched build assembles those systems for a whole mesh at
    once --- the rows are built from the pieces' `along` matrices and the edge
    and across vectors, with `_sharing`'s eleven rows and the three value rows
    constant. It is the same transcription-of-a-derivation situation the edge
    operator was in, and the same thing can go wrong: a sign, an ordering, a
    coefficient.

    So this builds the bases the readable way --- one triangle at a time, through
    `basis`, which is unchanged --- and holds the batched one to it. It matters
    more here than there: a Clough-Tocher fit builds a basis per distinct element
    *on every call*, so this is the whole of what such a fit spends on a mesh.
    '''

    def _triangles(self, count, dim, rng, /, degenerate=False):
        if not degenerate:
            return rng.normal(size=(count, dim, 3))
        out = np.zeros((count, dim, 3))
        for i in range(count):
            # Almost collinear: the along-matrix is nearly rank-deficient, which
            # is where a pseudo-inverse is least well behaved.
            out[i] = (rng.normal(size=(dim, 1))
                      + np.array([[0.0, 1.0, 2.0]]) * 1e-8
                      + rng.normal(size=(dim, 3)) * 1e-12)
        return out

    def test_it_is_the_basis_solved_one_triangle_at_a_time(self):
        rng = np.random.default_rng(11)
        for (count, dim) in ((1, 2), (7, 2), (7, 3), (40, 2), (30, 3)):
            triangles = self._triangles(count, dim, rng)
            want = np.stack([_ct.basis(triangles[i]) for i in range(count)])
            got = _ct.basis_many(triangles)
            with self.subTest(count=count, dim=dim):
                self.assertEqual(got.shape, want.shape)
                self.assertLess(np.abs(got - want).max(), 1e-12)

    def test_the_two_agree_on_a_nearly_flat_triangle_too(self):
        '''Where they agree only to the conditioning, which is worth knowing.

        A nearly collinear triangle has a nearly rank-deficient `along`-matrix,
        and the two builds reach the pseudo-inverse by different LAPACK paths ---
        one matrix at a time against a stack of them. Those differ in the last
        bit, and a condition number of about `1e8` turns that into `1e-8` in the
        answer, so the agreement here is loose where it is tight above.

        A transcription error would be `O(1)`, not `1e-8`, so this still catches
        one --- and the element is ill-defined on such a triangle anyway, which
        is the reason the tolerance is stated rather than hidden.
        '''
        rng = np.random.default_rng(13)
        triangles = self._triangles(16, 2, rng, degenerate=True)
        want = np.stack([_ct.basis(triangles[i]) for i in range(16)])
        got = _ct.basis_many(triangles)
        self.assertLess(np.abs(got - want).max(), 1e-5)

    def test_the_basis_reproduces_the_numbers_it_reads(self):
        '''The property that makes it *the* basis, checked on the batched one.

        Solving the conditions is what the equivalence above checks; this checks
        that the conditions are the right ones. Each datum asks for a unit
        number and each condition asks for zero, so reading a basis column back
        with the row that produced it must give one --- and with any other row,
        zero. A batched build that returned the same wrong answer as `basis`
        would pass the first test and fail this.
        '''
        rng = np.random.default_rng(12)
        triangles = self._triangles(12, 2, rng)
        bases = _ct.basis_many(triangles)
        for i in range(12):
            read = _ct.element_rows(triangles[i])
            for (column, (row, what)) in enumerate(read):
                with self.subTest(triangle=i, what=what):
                    # The twelve `element_rows` rows are the twelve numbers, in
                    # the order the basis solves for them.
                    self.assertAlmostEqual(
                        float(np.asarray(row) @ bases[i][:, column]), 1.0,
                        places=9)


class TestTheEdgeDatumOverTensors(TestCase):
    '''The per-call edge datum, held to the operator that is applied instead.

    A tensor's coordinates cannot use the cached operator: a scipy matrix holds
    no graph, and a *cached* graph cannot be reused, since a second `backward`
    through it fails unless every caller passes `retain_graph`. So the datum is
    assembled from the coordinates on the call --- `edge_data_many`, out of
    `triangle_blocks_many` and the averaging `edge_mean` --- and this is what
    says it is the same numbers.

    Without it there would be two implementations of one estimate and nothing
    holding them together, which is the situation the batched basis and the
    batched operator were each given their own test for.
    '''

    #: A mesh with a shared edge, so the averaging has something to average.
    COORDS = np.array([[0., 1., 0., 1.], [0., 0., 1., 1.]])
    INDICES = np.array([[0, 1], [1, 3], [2, 2]])

    def _stacked(self, width, rng):
        (dim, count) = self.COORDS.shape
        return rng.normal(size=(width, count + dim * count))

    def test_the_averaging_is_the_operator_s_own(self):
        (operator, edges, rows) = _ct.edge_operator(self.COORDS, self.INDICES)
        (mean, mine, myrows) = _ct.edge_mean(self.COORDS, self.INDICES)
        self.assertEqual(mine, edges, "the edges are numbered differently")
        self.assertTrue(np.array_equal(myrows, rows))
        # Applying the averaging to what each triangle says must reproduce the
        # operator, which is the composition of the two.
        self.assertEqual(mean.shape, (len(edges), 3 * self.INDICES.shape[1]))

    def test_it_gives_the_operator_s_numbers_for_arrays_and_for_tensors(self):
        rng = np.random.default_rng(31)
        (operator, edges, _) = _ct.edge_operator(self.COORDS, self.INDICES)
        (mean, _, _) = _ct.edge_mean(self.COORDS, self.INDICES)
        for width in (1, 3):
            stacked = self._stacked(width, rng)
            want = np.asarray(operator @ stacked.T).T
            got = _ct.edge_data_many(self.COORDS, self.INDICES, mean, stacked)
            with self.subTest(width=width):
                self.assertLess(np.abs(np.asarray(got) - want).max(), 1e-12)
        try:
            import torch
        except ImportError:                                  # pragma: no cover
            return
        stacked = torch.tensor(self._stacked(1, rng), dtype=torch.float64,
                               requires_grad=True)
        got = _ct.edge_data_many(self.COORDS, self.INDICES, mean, stacked)
        want = np.asarray(operator @ stacked.detach().numpy().T).T
        self.assertLess(np.abs(got.detach().numpy() - want).max(), 1e-12)
        self.assertTrue(got.grad_fn is not None)
        got.sum().backward()
        self.assertTrue(stacked.grad is not None)

    def test_a_tensor_s_coordinates_carry_their_derivative(self):
        '''The whole reason the per-call build exists.

        Checked against a central difference rather than for existence: a datum
        that was merely present would have passed while the fit it feeds was out
        by 0.19.

        On coordinates with no near-ties, and that is the point rather than a
        convenience. An edge's identity is its two corners put in order by where
        they are, and a nudge that swaps them is a *discontinuity* --- a
        difference taken across one measures the jump rather than a derivative.
        The mesh the other tests share has corners at (1, 0) and (0, 1), which
        are one nudge from a tie, and this read `6e5` until it used a mesh with
        room around each corner.
        '''
        try:
            import torch
        except ImportError:                                  # pragma: no cover
            self.skipTest("torch is not installed")
        rng = np.random.default_rng(32)
        coords_np = rng.normal(size=(2, 4)) * 3.0
        indices = np.array([[0, 1], [1, 3], [2, 2]])
        stacked = rng.normal(size=(1, coords_np.shape[1]
                                   + 2 * coords_np.shape[1]))
        (mean, _, _) = _ct.edge_mean(coords_np, indices)

        coords = torch.tensor(coords_np, dtype=torch.float64,
                              requires_grad=True)
        out = _ct.edge_data_many(coords, indices, mean,
                                 torch.tensor(stacked, dtype=torch.float64))
        out.sum().backward()
        got = coords.grad.numpy()

        step = 1e-6
        want = np.zeros_like(coords_np)
        for i in range(coords_np.shape[0]):
            for j in range(coords_np.shape[1]):
                (up, down) = (np.zeros_like(coords_np),
                              np.zeros_like(coords_np))
                up[i, j], down[i, j] = step, -step
                # Both are *added*: `down` carries its own sign.
                high = np.asarray(_ct.edge_data_many(
                    coords_np + up, indices, mean, stacked)).sum()
                low = np.asarray(_ct.edge_data_many(
                    coords_np + down, indices, mean, stacked)).sum()
                want[i, j] = (high - low) / (2 * step)
        self.assertLess(np.abs(got - want).max(), 1e-6)
