# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/test_interp.py
'''Tests for the interpolation engine in ``euclib.types._interp``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy.random import default_rng
from numpy import (allclose, arange, array, asarray, concatenate, cos, eye,
                   isfinite, isnan, nan, pi, sin, stack, zeros)

from euclib.types import (
    Grid, GridTopology, SegPath, SegTopology, TetMesh, TetTopology, TriMesh,
    TriTopology, VertexSet, VertexTopology)


# Fixtures ###################################################################

def _path():
    '''A two-segment path along the x axis, with a value at each node.'''
    path = SegPath(array([[0., 1., 2.], [0., 0., 0.]]),
                   SegTopology([[0, 1], [1, 2]]))
    return path.withprop('t', array([0., 10., 30.]), interp=1)


def _grid():
    '''A 4x4 grid of cells holding their own index.'''
    g = Grid(eye(3), GridTopology((4, 4)))
    return g.withprop('v', arange(16.).reshape(4, 4))


def _scalar(geom, name, at, **kw):
    '''Reads a scalar property and returns it as a float.'''
    res = geom.prop(name, at=at, **kw)
    return float(array(res).reshape(-1)[0])


# Tests ######################################################################

class TestLinearInterpolation(TestCase):
    '''Linear interpolation along a path.'''

    def test_a_position_between_two_nodes_is_blended(self):
        path = _path()
        self.assertAlmostEqual(_scalar(path, 't', array([[0.5], [0.]])), 5.0)
        self.assertAlmostEqual(_scalar(path, 't', array([[1.5], [0.]])), 20.0)

    def test_a_position_at_a_node_is_that_node_s_value(self):
        path = _path()
        self.assertAlmostEqual(_scalar(path, 't', array([[0.], [0.]])), 0.0)
        self.assertAlmostEqual(_scalar(path, 't', array([[2.], [0.]])), 30.0)

    def test_interpolation_is_the_identity_on_a_linear_field(self):
        # A field that is linear in position must come back exactly.
        path = SegPath(array([[0., 1., 2.], [0., 0., 0.]]),
                       SegTopology([[0, 1], [1, 2]]))
        p = path.withprop('x', array([0., 1., 2.]), interp=1)
        at = array([[0.25, 1.75], [0., 0.]])
        self.assertTrue(allclose(p.prop('x', at=at), [0.25, 1.75]))


class TestNearestInterpolation(TestCase):
    '''Order 0 takes the nearest component's value.'''

    def test_the_nearest_node_wins(self):
        path = _path().withprop('t', array([0., 10., 30.]), interp=0)
        self.assertAlmostEqual(_scalar(path, 't', array([[0.4], [0.]])), 0.0)
        self.assertAlmostEqual(_scalar(path, 't', array([[0.6], [0.]])), 10.0)

    def test_qualitative_data_defaults_to_nearest(self):
        path = SegPath(array([[0., 1., 2.], [0., 0., 0.]]),
                       SegTopology([[0, 1], [1, 2]]))
        labels = path.withprop('label', array([1, 2, 3]))
        self.assertEqual(labels.propinfo('label').interp, ('nearest', 0))
        self.assertAlmostEqual(_scalar(labels, 'label', array([[0.4], [0.]])),
                               1.0)


class TestLocalCoordinates(TestCase):
    '''Positions may be supplied as local coordinates.'''

    def test_a_loc_is_accepted(self):
        path = _path()
        loc = path.topo.Loc(array([0]), array([[0.5]]))
        self.assertAlmostEqual(_scalar(path, 't', loc), 5.0)

    def test_a_mapping_is_accepted(self):
        path = _path()
        at = {'index': array([1]), 'weight': array([[0.5]])}
        self.assertAlmostEqual(_scalar(path, 't', at), 20.0)

    def test_local_coordinates_are_never_outside(self):
        # Local coordinates name a position on the object by construction, so
        # they are answered even when extrapolation is not requested.
        path = _path()
        loc = path.topo.Loc(array([0]), array([[0.5]]))
        self.assertFalse(isnan(_scalar(path, 't', loc)))


class TestCoordinateNames(TestCase):
    '''A mapping's keys decide whether it names local or global coordinates.'''

    def setUp(self):
        affine = eye(3)
        affine[:2, :2] *= 2.
        g = Grid(affine, GridTopology((4, 4)))
        self.grid = g.withprop('v', arange(16.).reshape(4, 4))

    def test_looking_at_the_same_position_two_ways_agrees(self):
        # The affine doubles every index, so the local position (1.5, 2.5) is
        # the global position (3, 5).
        by_local = _scalar(self.grid, 'v', {'sx': 1.5, 'sy': 2.5})
        by_global = _scalar(self.grid, 'v', {'x': 3.0, 'y': 5.0})
        by_array = _scalar(self.grid, 'v', array([[3.], [5.]]))
        self.assertAlmostEqual(by_local, by_global)
        self.assertAlmostEqual(by_local, by_array)
        self.assertAlmostEqual(by_local, 8.5)

    def test_a_float_grid_has_the_expected_local_names(self):
        self.assertEqual(self.grid.topo.Loc._fields, ('sx', 'sy'))

    def test_an_unknown_coordinate_name_is_rejected(self):
        with self.assertRaises(ValueError):
            self.grid.prop('v', at={'u': 1.0, 'v': 2.0})


class TestExtrapolation(TestCase):
    '''What happens to positions outside the object.'''

    def test_no_extrapolation_reports_the_null_value(self):
        self.assertTrue(isnan(_scalar(_path(), 't', array([[0.], [5.]]))))

    def test_extrapolation_of_order_zero_reports_the_boundary(self):
        path = _path()
        at = array([[0.], [5.]])
        self.assertAlmostEqual(_scalar(path, 't', at, extrap=0), 0.0)
        at = array([[5.], [0.]])
        self.assertAlmostEqual(_scalar(path, 't', at, extrap=0), 30.0)

    def test_a_position_on_the_object_is_not_extrapolated(self):
        path = _path()
        self.assertAlmostEqual(_scalar(path, 't', array([[1.], [0.]])), 10.0)


class TestMasks(TestCase):
    '''A masked component poisons any result that would draw on it.'''

    def test_a_masked_contributor_poisons_the_result(self):
        path = SegPath(array([[0., 1., 2.], [0., 0., 0.]]),
                       SegTopology([[0, 1], [1, 2]]))
        p = path.withprop('t', array([0., 10., 30.]), interp=1,
                          mask=array([True, False, False]))
        self.assertTrue(isnan(_scalar(p, 't', array([[0.5], [0.]]))))
        # A position whose blend does not reach the masked node is unaffected.
        self.assertAlmostEqual(_scalar(p, 't', array([[1.5], [0.]])), 20.0)

    def test_the_null_value_may_be_overridden(self):
        path = SegPath(array([[0., 1., 2.], [0., 0., 0.]]),
                       SegTopology([[0, 1], [1, 2]]))
        p = path.withprop('t', array([0., 10., 30.]), interp=1,
                          mask=array([True, False, False]))
        self.assertEqual(_scalar(p, 't', array([[0.5], [0.]]), null=-1), -1.0)

    def test_nearest_ignores_a_mask_it_does_not_draw_on(self):
        path = SegPath(array([[0., 1., 2.], [0., 0., 0.]]),
                       SegTopology([[0, 1], [1, 2]]))
        p = path.withprop('t', array([0., 10., 30.]), interp=0,
                          mask=array([True, False, False]))
        # The nearest node to (0.6, 0) is node 1, which is not masked.
        self.assertAlmostEqual(_scalar(p, 't', array([[0.6], [0.]])), 10.0)
        # ...and to (0.4, 0) is node 0, which is.
        self.assertTrue(isnan(_scalar(p, 't', array([[0.4], [0.]]))))


class TestGridInterpolation(TestCase):
    '''Interpolation over a grid.'''

    def test_linear_interpolation_averages_the_surrounding_cells(self):
        self.assertAlmostEqual(_scalar(_grid(), 'v', array([[0.5], [0.5]])),
                               2.5)

    def test_a_position_at_a_cell_returns_that_cell(self):
        self.assertAlmostEqual(_scalar(_grid(), 'v', array([[2.], [3.]])),
                               11.0)

    def test_nearest_interpolation_takes_the_nearest_cell(self):
        g = _grid().withprop('v', arange(16.).reshape(4, 4), interp=0)
        self.assertAlmostEqual(_scalar(g, 'v', array([[1.4], [2.6]])), 7.0)
        self.assertAlmostEqual(_scalar(g, 'v', array([[0.4], [0.4]])), 0.0)

    def test_a_position_outside_the_grid_has_no_answer(self):
        self.assertTrue(isnan(_scalar(_grid(), 'v', array([[9.], [9.]]))))

    def test_a_masked_cell_poisons_the_blend(self):
        # The cell at (1, 1) holds 5; one of the four cells its neighbours
        # blend draws on.
        mask = zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        g = _grid().withprop('v', arange(16.).reshape(4, 4), mask=mask)
        self.assertTrue(isnan(_scalar(g, 'v', array([[1.1], [1.1]]))))
        # A position far from the masked cell is unaffected.
        self.assertFalse(isnan(_scalar(g, 'v', array([[3.], [3.]]))))


class TestUnimplementedOrders(TestCase):
    '''Orders 2 and 3 are refused rather than approximated.'''

    def test_a_higher_order_is_built_for_a_segment(self):
        # The polynomial method's higher orders are built one element at a time
        # and a segment's are done, so a path interpolates quadratically and
        # cubically; see TestSegmentPolynomial for what they compute.
        for order in (2, 3):
            with self.subTest(order=order):
                self.assertTrue(
                    isfinite(float(_path().prop('t', at=array([[0.5], [0.]]),
                                                interp=order)[0])))

    def test_a_higher_order_is_not_built_for_a_tetrahedron_yet(self):
        # Triangles are built; tetrahedra come next.
        mesh = TetMesh(array([[0., 1., 0., 0.], [0., 0., 1., 0.],
                              [0., 0., 0., 1.]]),
                       TetTopology([[0], [1], [2], [3]])).withprop(
                           't', array([1., 2., 3., 4.]))
        for order in (2, 3):
            with self.subTest(order=order):
                with self.assertRaises(NotImplementedError):
                    mesh.prop('t', at=array([[0.1], [0.1], [0.1]]), interp=order)


class TestSegmentPolynomial(TestCase):
    '''The polynomial method above linear, which a segment is the first to have.

    A segment's two values cannot determine a quadratic or a cubic: two
    conditions against three or four coefficients. The slopes at the ends supply
    the rest, and they come from the property when it carries them and are
    estimated from the values when it does not. The values themselves are always
    interpolated, so the field stays continuous where two segments meet.
    '''

    #: The three nodes of a path along the x axis, and the values that an
    #: analytic field takes there.
    COORDS = array([[0., 1., 2.], [0., 0., 0.]])

    def _path(self, values, gradient=None):
        '''A one-segment path from x=0 to x=1 carrying ``values``.'''
        path = SegPath(self.COORDS[:, :2], SegTopology([[0], [1]]))
        return path.withprop('t', values[:2], gradient=gradient)

    def _path3(self, values, gradient=None):
        '''A two-segment path over all three nodes.'''
        path = SegPath(self.COORDS, SegTopology([[0, 1], [1, 2]]))
        return path.withprop('t', values, gradient=gradient)

    def _at(self, path, x, order, y=0., **kw):
        return float(asarray(path.prop('t', at=array([[x], [y]]),
                                       interp=order, **kw))[0])

    def test_order_three_reproduces_a_cubic_from_supplied_slopes(self):
        # f(x) = x^3 on the nodes, with its exact derivative. A cubic has four
        # coefficients and value and slope at each end are four conditions, so
        # the fit is exact between them.
        values = array([0., 1.])
        gradient = array([[0., 3.], [0., 0.]])          # d/dx, d/dy
        path = self._path(values, gradient)
        for x in (0.25, 0.5, 0.75):
            with self.subTest(x=x):
                self.assertAlmostEqual(self._at(path, x, 3), x ** 3, places=12)

    def test_order_two_reproduces_a_quadratic_from_supplied_slopes(self):
        # f(x) = x^2, whose slopes are 0 and 2: the quadratic's free coefficient
        # is settled by them exactly, because the requested slopes straddle the
        # segment's own average slope by equal amounts.
        values = array([0., 1.])
        gradient = array([[0., 2.], [0., 0.]])
        path = self._path(values, gradient)
        for x in (0.25, 0.5, 0.75):
            with self.subTest(x=x):
                self.assertAlmostEqual(self._at(path, x, 2), x ** 2, places=12)

    def test_the_fit_passes_through_the_node_values(self):
        # The values are interpolated rather than merely fitted, whatever the
        # slopes say: this is what keeps the field continuous at a shared node,
        # where the two segments on either side must agree.
        path = self._path3(array([2., 5., 11.]))
        for order in (2, 3):
            with self.subTest(order=order):
                for (node, x) in ((0, 0.), (1, 1.), (2, 2.)):
                    self.assertAlmostEqual(
                        self._at(path, x, order), (2., 5., 11.)[node],
                        places=12)

    def test_the_estimate_reproduces_the_polynomial(self):
        # The stencil grows until it can determine a polynomial of the
        # interpolation's own order, so an estimate from values alone reproduces
        # a field of that order exactly --- on the end segments as much as the
        # interior ones. A one-ring estimate could not: it would be one-sided at
        # an end node and read 0.375 where x^2 is 0.25.
        path = self._path3(array([0., 1., 4.]))                # f(x) = x^2
        for order in (2, 3):
            with self.subTest(order=order):
                for x in (0.25, 0.5, 1.0, 1.5, 1.75):
                    self.assertAlmostEqual(self._at(path, x, order), x ** 2,
                                           places=12)

    def test_the_estimate_reproduces_a_cubic_when_the_order_needs_one(self):
        # A cubic fit needs slopes that come from a cubic estimate, which takes
        # four nodes; the stencil reaches that far on a four-node path.
        coords = array([[0., 1., 2., 3.], [0., 0., 0., 0.]])
        path = SegPath(coords, SegTopology([[0, 1, 2], [1, 2, 3]]))
        path = path.withprop('t', array([0., 1., 8., 27.]))    # f(x) = x^3
        for x in (0.25, 1.5, 2.75):
            self.assertAlmostEqual(self._at(path, x, 3), x ** 3, places=12)

    def test_the_gradient_components_come_back_in_axis_order(self):
        # The linear monomials are not ordered by axis, so picking the gradient
        # out of them has to be: getting it wrong transposes the components, and
        # a fit along a diagonal is where that shows.
        coords = array([[0., 1., 2., 3.], [0., 1., 2., 3.]])
        path = SegPath(coords, SegTopology([[0, 1, 2], [1, 2, 3]]))
        path = path.withprop('t', array([0., 2., 8., 18.]))   # 2 t^2 on t = x = y
        for t in (0.75, 1.5, 2.5):
            with self.subTest(t=t):
                self.assertAlmostEqual(self._at(path, t, 2, y=t), 2 * t * t,
                                       places=12)

    def test_a_geometry_with_too_few_nodes_still_answers(self):
        # The stencil cannot outgrow the geometry. With three nodes no estimate
        # can know what a cubic was, and the answer is the shortest fit through
        # the values there are; a field of an order the data *can* determine is
        # still reproduced.
        path = self._path3(array([0., 1., 2.]))                # a straight line
        for x in (0.5, 1.5):
            with self.subTest(x=x):
                self.assertAlmostEqual(self._at(path, x, 3), x, places=12)

    def test_a_supplied_gradient_is_used_instead_of_the_property_s(self):
        # f(x) = x^2 on three nodes. The estimate reproduces it, so asking with
        # the exact slopes changes nothing; asking with wrong ones changes the
        # answer, which is what shows the argument is the one that is used.
        path = self._path3(array([0., 1., 4.]))
        exact = array([[0., 2., 4.], [0., 0., 0.]])
        self.assertAlmostEqual(self._at(path, 0.5, 2), 0.25, places=12)
        self.assertAlmostEqual(self._at(path, 0.5, 2, gradient=exact), 0.25,
                               places=12)
        # Zero slopes leave the straight line the values anchor, which reads
        # 0.5 at the midpoint rather than 0.25.
        self.assertAlmostEqual(self._at(path, 0.5, 2, gradient=zeros((2, 3))),
                               0.5, places=12)
        # ...and the property is unchanged by the call.
        self.assertAlmostEqual(self._at(path, 0.5, 2), 0.25, places=12)

    def test_a_gradient_of_the_wrong_dimension_is_refused(self):
        path = self._path(array([0., 1.]))
        with self.assertRaises(ValueError):
            self._at(path, 0.5, 3, gradient=zeros((3, 2)))

    def test_a_mask_poisons_the_segments_that_draw_on_it(self):
        # A fit draws on both of its corners' values and slopes, so a masked
        # node makes every segment that touches it answer with the null value.
        # That is wider than linear interpolation's reach and it is allowed:
        # the rule is that a position nearest a masked node must be null, not
        # that nothing else may be.
        path = SegPath(self.COORDS, SegTopology([[0, 1], [1, 2]]))
        path = path.withprop('t', array([0., 1., 4.]),
                             mask=array([False, True, False]))
        self.assertTrue(isnan(self._at(path, 0.5, 2)))
        self.assertTrue(isnan(self._at(path, 1.5, 2)))
        # A node nothing masks still answers.
        path = path.withprop('t', array([0., 1., 4.]),
                             mask=array([False, False, True]))
        self.assertFalse(isnan(self._at(path, 0.5, 2)))
        self.assertTrue(isnan(self._at(path, 1.5, 2)))


class TestTrianglePolynomial(TestCase):
    '''The polynomial method on a triangle.

    A triangle's three values and three gradients are nine conditions where a
    quadratic takes six coefficients and a cubic takes ten, so the fit is built
    edge by edge: each shared edge gets its own one-dimensional fit, which is
    what keeps the field continuous across it, and a cubic's one interior value
    follows from the three edges. See
    ``docs/euclib/examples/properties/bezier-triangle.md`` for the derivation.
    '''

    #: The fan from that page: a central triangle with six around it. Six
    #: nodes is enough for the estimate to determine a quadratic.
    ANGLES = array([90., 210., 330.]) * pi / 180

    def _fan(self, gradient=None):
        inner = stack([cos(self.ANGLES), sin(self.ANGLES)])
        coords = concatenate([inner, 3.0 * inner], axis=1)
        triangles = ([(0, 1, 2)]
                     + [(k, (k + 1) % 3, 3 + (k + 2) % 3) for k in range(3)]
                     + [(k, 3 + (k + 1) % 3, 3 + (k + 2) % 3) for k in range(3)])
        mesh = TriMesh(coords, TriTopology(array(triangles).T))
        return mesh.withprop('f', self.F(coords[0], coords[1]), gradient=gradient)

    @staticmethod
    def F(x, y):
        '''A quadratic with a cross term, and its exact gradient.'''
        return 0.4 * x * x + 0.3 * y * y + 0.25 * x * y + 0.5 * x

    @staticmethod
    def gradient_of(coords):
        return stack([0.8 * coords[0] + 0.25 * coords[1] + 0.5,
                      0.6 * coords[1] + 0.25 * coords[0]])

    def _inside(self, mesh):
        '''A handful of positions inside the mesh, and their weights.'''
        tris = array(mesh.topo.indices).T
        rng = default_rng(3)
        found = []
        for _ in range(8):
            w = rng.uniform(size=3)
            w /= w.sum()
            (i, j, k) = tris[rng.integers(len(tris))]
            found.append((w, array(mesh.coords)[:, [i, j, k]] @ w))
        return found

    def test_a_quadratic_is_reproduced_with_supplied_gradients(self):
        coords = array([[0., 1., 0., 1.], [0., 0., 1., 1.]])
        mesh = TriMesh(coords, TriTopology([[0, 0], [1, 3], [3, 2]]))
        mesh = mesh.withprop('f', self.F(coords[0], coords[1]),
                             gradient=self.gradient_of(coords))
        for order in (2, 3):
            with self.subTest(order=order):
                for (w, point) in self._inside(mesh):
                    got = float(asarray(mesh.prop(
                        'f', at=point.reshape(2, 1), interp=order))[0])
                    self.assertAlmostEqual(got, self.F(point[0], point[1]),
                                           places=12)

    def test_the_estimate_reproduces_a_quadratic_too(self):
        # The stencil grows until the polynomial is determined by it, so an
        # estimated gradient is as good as a supplied one wherever the geometry
        # gives the neighbours to determine it.
        mesh = self._fan()
        for order in (2, 3):
            with self.subTest(order=order):
                for (w, point) in self._inside(mesh):
                    got = float(asarray(mesh.prop(
                        'f', at=point.reshape(2, 1), interp=order))[0])
                    self.assertAlmostEqual(got, self.F(point[0], point[1]),
                                           places=12)

    def test_the_shared_edge_is_interpolated_identically(self):
        # A position on an edge that two triangles share must be answered the
        # same way from either side. That is the edge-first construction's whole
        # purpose, and it is what the docs page's own check measures.
        inner = stack([cos(self.ANGLES), sin(self.ANGLES)])
        coords = concatenate([inner, 3.0 * inner], axis=1)
        triangles = array([(0, 1, 2)]
                          + [(k, (k + 1) % 3, 3 + (k + 2) % 3) for k in range(3)]
                          + [(k, 3 + (k + 1) % 3, 3 + (k + 2) % 3)
                             for k in range(3)])
        mesh = TriMesh(coords, TriTopology(triangles.T))
        mesh = mesh.withprop('f', self.F(coords[0], coords[1]),
                             gradient=self.gradient_of(coords))
        # Which triangles hold each edge, and where in each the edge's corners
        # sit, since a triangle's local weights are its own.
        sides = {}
        for (index, triangle) in enumerate(triangles):
            for (a, b) in ((0, 1), (1, 2), (2, 0)):
                sides.setdefault(tuple(sorted((triangle[a], triangle[b]))),
                                 []).append((index, a, b))
        shared = [(edge, held) for (edge, held) in sides.items()
                  if len(held) == 2]
        self.assertEqual(len(shared), 9)
        for order in (2, 3):
            for (edge, held) in shared:
                for s in (0.25, 0.5, 0.75):
                    answers = []
                    for (index, a, b) in held:
                        weights = zeros(3)
                        weights[a] = 1.0 - s
                        weights[b] = s
                        # A triangle's local weight holds the first two
                        # barycentric weights; the third is the remainder.
                        loc = mesh.topo.Loc(array([index]),
                                            weights[:2].reshape(2, 1))
                        answers.append(float(asarray(mesh.prop(
                            'f', at=loc, interp=order))[0]))
                    with self.subTest(order=order, edge=edge, s=s):
                        self.assertAlmostEqual(answers[0], answers[1],
                                               places=12)

    def test_a_channelled_property_interpolates_channel_by_channel(self):
        # The fits carry the value's channel dimensions through the control
        # values and the Bernstein sum, so a two-channel property has to come
        # back with two channels, each interpolated as the scalar case is.
        inner = stack([cos(self.ANGLES), sin(self.ANGLES)])
        coords = concatenate([inner, 3.0 * inner], axis=1)
        triangles = array([(0, 1, 2)]
                          + [(k, (k + 1) % 3, 3 + (k + 2) % 3) for k in range(3)]
                          + [(k, 3 + (k + 1) % 3, 3 + (k + 2) % 3)
                             for k in range(3)])
        mesh = TriMesh(coords, TriTopology(triangles.T))
        # Channel 0 is f; channel 1 is twice f, so both must interpolate.
        values = stack([self.F(coords[0], coords[1]),
                        2.0 * self.F(coords[0], coords[1])])
        gradient = stack([self.gradient_of(coords),
                          2.0 * self.gradient_of(coords)])
        mesh = mesh.withprop('f', values, gradient=gradient)
        for order in (2, 3):
            for (w, point) in self._inside(mesh):
                got = asarray(mesh.prop('f', at=point.reshape(2, 1),
                                        interp=order))
                self.assertEqual(got.shape, (2, 1))
                self.assertAlmostEqual(float(got[0, 0]),
                                       self.F(point[0], point[1]), places=12)
                self.assertAlmostEqual(float(got[1, 0]),
                                       2.0 * self.F(point[0], point[1]),
                                       places=12)

    def test_a_position_of_the_wrong_dimension_is_refused(self):
        # Without the check the query reaches the spatial index and fails inside
        # it with a broadcasting error about shapes, which tells the caller
        # nothing about what they did wrong.
        mesh = TriMesh(array([[0., 1., 0.], [0., 0., 1.]]),
                       TriTopology([[0], [1], [2]]))
        mesh = mesh.withprop('f', array([1., 2., 3.]))
        for bad in (zeros((3, 2)), array([0., 0., 0.])):
            with self.subTest(shape=bad.shape):
                with self.assertRaises(ValueError):
                    mesh.prop('f', at=bad)

    def test_the_values_at_the_corners_are_interpolated(self):
        mesh = self._fan()
        coords = array(mesh.coords)
        for order in (2, 3):
            with self.subTest(order=order):
                for node in range(6):
                    got = float(asarray(mesh.prop(
                        'f', at=coords[:, node].reshape(2, 1), interp=order))[0])
                    self.assertAlmostEqual(got, self.F(coords[0, node],
                                                       coords[1, node]),
                                           places=12)

    def test_a_supplied_gradient_overrides_the_property_s(self):
        mesh = self._fan()
        coords = array(mesh.coords)
        point = (coords[:, 0] + coords[:, 1] + coords[:, 2]) / 3.0
        exact = self.gradient_of(coords)
        with_estimate = float(asarray(mesh.prop(
            'f', at=point.reshape(2, 1), interp=2))[0])
        with_exact = float(asarray(mesh.prop(
            'f', at=point.reshape(2, 1), interp=2, gradient=exact))[0])
        with_zero = float(asarray(mesh.prop(
            'f', at=point.reshape(2, 1), interp=2, gradient=zeros((2, 6))))[0])
        self.assertAlmostEqual(with_estimate, with_exact, places=12)
        self.assertAlmostEqual(with_exact, self.F(point[0], point[1]),
                               places=12)
        # A wrong gradient gives a different answer, which is what shows the
        # argument is the one used.
        self.assertNotAlmostEqual(with_zero, with_exact, places=3)

    def test_a_gradient_of_the_wrong_dimension_is_refused(self):
        mesh = self._fan()
        with self.assertRaises(ValueError):
            mesh.prop('f', at=array([[0.], [0.]]), interp=2,
                      gradient=zeros((3, 6)))

    def test_a_mask_poisons_the_triangles_that_draw_on_it(self):
        # A triangle's fit draws on all three corners, so masking one makes
        # every triangle that touches it answer with the null value --- which
        # includes every position nearest that node, the rule's requirement.
        coords = array([[0., 1., 0., 1.], [0., 0., 1., 1.]])
        mesh = TriMesh(coords, TriTopology([[0, 0], [1, 3], [3, 2]]))
        mesh = mesh.withprop('f', self.F(coords[0], coords[1]),
                             gradient=self.gradient_of(coords),
                             mask=array([False, True, False, False]))
        for order in (2, 3):
            with self.subTest(order=order):
                # A position inside the triangle that has the masked corner,
                # and one inside the other triangle, which does not.
                self.assertTrue(isnan(float(asarray(mesh.prop(
                    'f', at=array([[0.2], [0.2]]), interp=order))[0])))
                self.assertTrue(isnan(float(asarray(mesh.prop(
                    'f', at=array([[0.8], [0.8]]), interp=order))[0])))


class TestPointCloudInterpolation(TestCase):
    '''A point cloud has points but no interior, so only its points are on it.'''

    def setUp(self):
        cloud = VertexSet(array([[0., 1., 2.], [0., 0., 0.]]),
                          VertexTopology([[0, 1, 2]]))
        self.cloud = cloud.withprop('v', array([1., 2., 3.]))

    def test_a_position_off_the_points_has_no_answer(self):
        self.assertTrue(isnan(_scalar(self.cloud, 'v', array([[1.9], [0.]]))))

    def test_a_position_at_a_point_is_that_point_s_value(self):
        self.assertAlmostEqual(
            _scalar(self.cloud, 'v', array([[1.], [0.]])), 2.0)
        self.assertAlmostEqual(
            _scalar(self.cloud, 'v', array([[2.], [0.]])), 3.0)

    def test_extrapolation_reports_the_nearest_point_s_value(self):
        self.assertAlmostEqual(
            _scalar(self.cloud, 'v', array([[1.9], [0.]]), extrap=0), 3.0)
        self.assertAlmostEqual(
            _scalar(self.cloud, 'v', array([[0.1], [0.]]), extrap=0), 1.0)

    def test_a_point_cloud_only_accepts_nearest_interpolation(self):
        with self.assertRaises(Exception):
            self.cloud.withprop('w', array([1., 2., 3.]), interp='polynomial')
        with self.assertRaises(Exception):
            self.cloud.withprop('w', array([1., 2., 3.]), interp=1)

    def test_the_default_interpolation_is_allowed_and_ignored(self):
        # A property that took the default is the geometry's business, and a
        # point cloud ignores it: the answer is still the nearest point's.
        self.assertFalse(self.cloud.propinfo('v').interp_specified)
        self.assertAlmostEqual(
            _scalar(self.cloud, 'v', array([[2.], [0.]])), 3.0)
