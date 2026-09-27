# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/test_interp.py
'''Tests for the interpolation engine in ``euclib.types._interp``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

import numpy as np
from numpy.random import default_rng
from numpy import (allclose, arange, array, asarray, concatenate, cos, eye,
                   isfinite, isnan, linspace, nan, ones, pi, ravel, sin,
                   stack, zeros)

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

    def test_order_one_is_the_same_under_either_method(self):
        # A degree-1 Bernstein patch is the barycentric blend itself, so the
        # two methods are one interpolation at order 1 and ('bezier', 1) is an
        # alias for ('polynomial', 1). They part company above it, where the
        # freedom the corners leave has to be spent somehow.
        path = _path()
        for x in (0.25, 0.5, 1.75):
            with self.subTest(x=x):
                at = array([[x], [0.]])
                self.assertEqual(
                    _scalar(path, 't', at, interp=('polynomial', 1)),
                    _scalar(path, 't', at, interp=('bezier', 1)))
                self.assertEqual(_scalar(path, 't', at, interp=('bezier', 1)),
                                 _scalar(path, 't', at, interp=1))

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
                                                interp=('bezier', order))[0])))

    def test_a_higher_order_is_built_for_a_tetrahedron(self):
        # The element-wise fits are built one element at a time, and a
        # tetrahedron's are done: it fits a quadratic and a cubic through its
        # values and its corners' gradients. A linear field is what every order
        # has to reproduce, so that is what both orders are held to.
        coords = array([[0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        mesh = TetMesh(coords, TetTopology([[0], [1], [2], [3]]))
        mesh = mesh.withprop('t', coords.sum(axis=0)[None, :],
                             gradient=ones((1, 3, 4)))
        for order in (2, 3):
            with self.subTest(order=order):
                self.assertAlmostEqual(
                    float(asarray(mesh.prop(
                        't', at=array([[0.2], [0.3], [0.1]]),
                        interp=('bezier', order))).ravel()[0]), 0.6, places=12)


class TestSegmentBezier(TestCase):
    '''The Bezier method above linear, which a segment is the first to have.

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
                                       interp=('bezier', order), **kw))[0])

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


class TestTriangleBezier(TestCase):
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
                        'f', at=point.reshape(2, 1), interp=('bezier', order)))[0])
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
                        'f', at=point.reshape(2, 1), interp=('bezier', order)))[0])
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
                            'f', at=loc, interp=('bezier', order)))[0]))
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
                                        interp=('bezier', order)))
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
                        'f', at=coords[:, node].reshape(2, 1), interp=('bezier', order)))[0])
                    self.assertAlmostEqual(got, self.F(coords[0, node],
                                                       coords[1, node]),
                                           places=12)

    def test_a_supplied_gradient_overrides_the_property_s(self):
        mesh = self._fan()
        coords = array(mesh.coords)
        point = (coords[:, 0] + coords[:, 1] + coords[:, 2]) / 3.0
        exact = self.gradient_of(coords)
        with_estimate = float(asarray(mesh.prop(
            'f', at=point.reshape(2, 1), interp=('bezier', 2)))[0])
        with_exact = float(asarray(mesh.prop(
            'f', at=point.reshape(2, 1), interp=('bezier', 2), gradient=exact))[0])
        with_zero = float(asarray(mesh.prop(
            'f', at=point.reshape(2, 1), interp=('bezier', 2), gradient=zeros((2, 6))))[0])
        self.assertAlmostEqual(with_estimate, with_exact, places=12)
        self.assertAlmostEqual(with_exact, self.F(point[0], point[1]),
                               places=12)
        # A wrong gradient gives a different answer, which is what shows the
        # argument is the one used.
        self.assertNotAlmostEqual(with_zero, with_exact, places=3)

    def test_a_gradient_of_the_wrong_dimension_is_refused(self):
        mesh = self._fan()
        with self.assertRaises(ValueError):
            mesh.prop('f', at=array([[0.], [0.]]), interp=('bezier', 2),
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
                    'f', at=array([[0.2], [0.2]]), interp=('bezier', order)))[0])))
                self.assertTrue(isnan(float(asarray(mesh.prop(
                    'f', at=array([[0.8], [0.8]]), interp=('bezier', order)))[0])))


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


class TestTheBatchedEstimate(TestCase):
    '''The gradient estimate, which takes the coordinates a block at a time.

    Every coordinate's stencil grows by a ring of the neighbourhood at a time, so
    the coordinates of one mesh are of many different sizes and the estimate
    takes them in runs of equal size. The answers must not depend on how they are
    grouped or how large a block is, and a fit on a mesh of two dimensions is the
    case where the frame and the design have real work to do --- a path's stencil
    spans one direction, and a mesh's spans two or three.
    '''

    def _grid(self, n):
        '''A triangulated ``n`` by ``n`` grid mesh.'''
        from euclib import trimesh
        from euclib.types import TriTopology
        axis = linspace(0.0, 1.0, n)
        gridx, gridy = np.meshgrid(axis, axis)
        points = stack([gridx.ravel(), gridy.ravel()])
        faces = []
        for (i, j) in np.ndindex(n - 1, n - 1):
            (a, b) = (i * n + j, i * n + j + 1)
            (c, d) = ((i + 1) * n + j, (i + 1) * n + j + 1)
            faces += [[a, b, d], [a, d, c]]
        return trimesh(points, array(faces).T)

    def test_it_reproduces_a_quadratic_on_a_mesh(self):
        # f(x, y) = x^2 + 2xy + 3y^2, whose gradient is (2x + 2y, 2x + 6y). A
        # stencil that determines a quadratic gives back that quadratic's own
        # derivative, whichever coordinate of the mesh it is taken at.
        from euclib.types._interp import estimate_gradient
        mesh = self._grid(12)
        points = np.asarray(mesh.coords)
        values = (points[0] ** 2 + 2.0 * points[0] * points[1]
                  + 3.0 * points[1] ** 2)[None, :]
        geom = mesh.withprop('f', values)
        prop = geom._prop_for('f', None)
        got = np.asarray(estimate_gradient(geom, prop, 2))
        want = stack([2.0 * points[0] + 2.0 * points[1],
                      2.0 * points[0] + 6.0 * points[1]])
        self.assertTrue(np.allclose(got, want, atol=1e-9),
                        f"the largest disagreement is"
                        f" {np.abs(got - want).max():.3e}")
        # And the interpolation built on it reproduces the field itself.
        for (x, y) in ((0.13, 0.81), (0.5, 0.5), (0.97, 0.02)):
            with self.subTest(x=x, y=y):
                self.assertAlmostEqual(
                    float(np.asarray(geom.prop(
                        'f', at=array([[x], [y]]), interp=('bezier', 2))).ravel()[0]),
                    x ** 2 + 2.0 * x * y + 3.0 * y ** 2, places=9)

    def test_a_channelled_property_is_estimated_channel_by_channel(self):
        # A property's channel dimensions are leading in its gradient as they
        # are in its values, and the solve's channel axis is leading too, so the
        # two are shaped alike and one is placed in the other. Putting the
        # solution in transposed --- which is what the corner axis of a *sample*
        # wants --- turns the channels and the dimensions the wrong way round for
        # a property with more than one, and a single-channel property cannot
        # tell the difference.
        from euclib.types._interp import estimate_gradient
        mesh = self._grid(10)
        points = np.asarray(mesh.coords)
        values = (points[0] ** 2 + 2.0 * points[0] * points[1])
        gradient = np.stack([2.0 * points[0] + 2.0 * points[1],
                             2.0 * points[0]])
        for channels in ((), (1,), (2,), (2, 3)):
            with self.subTest(channels=channels):
                geom = mesh.withprop(
                    'f', np.broadcast_to(values, channels + values.shape).copy(),
                    gradient=np.broadcast_to(
                        gradient, channels + gradient.shape).copy())
                prop = geom._prop_for('f', None)
                got = np.asarray(estimate_gradient(geom, prop, 2))
                self.assertEqual(got.shape, channels + (2,) + values.shape)
                self.assertTrue(np.allclose(
                    got, geom.propinfo('f').gradient, atol=1e-9))

    def test_the_block_size_does_not_change_the_answer(self):
        # The coordinates are taken a block at a time, and a block boundary
        # falls wherever the count puts it. Lowering the size to two puts a
        # boundary between nearly every pair of coordinates.
        from euclib.types import _interp
        from euclib.types._interp import estimate_gradient
        mesh = self._grid(10)
        points = np.asarray(mesh.coords)
        values = (points[0] ** 2 + 2.0 * points[0] * points[1]
                  + 3.0 * points[1] ** 2)[None, :]
        geom = mesh.withprop('f', values)
        prop = geom._prop_for('f', None)
        want = np.asarray(estimate_gradient(geom, prop, 2))
        saved = _interp._ESTIMATE_BLOCK
        try:
            _interp._ESTIMATE_BLOCK = 2
            got = np.asarray(estimate_gradient(geom, prop, 2))
        finally:
            _interp._ESTIMATE_BLOCK = saved
        self.assertTrue(np.array_equal(got, want))


class TestTetrahedronBezier(TestCase):
    '''The Bezier method on a tetrahedron.

    A quadratic or a cubic on a tetrahedron is not determined by its corners'
    values and gradients --- a cubic has twenty control values where the corners
    give sixteen conditions --- so a construction settles the rest. It is the
    triangle's, applied to each of the four faces: every edge carries its own
    one-dimensional fit, and a cubic's face values are the degree-elevation
    averages of those edges' degree-2 controls. What that buys is what these
    tests are about: a quadratic is reproduced exactly, and two tetrahedra that
    share a face agree on every point of it.
    '''

    #: Two tetrahedra sharing the face (c1, c2, c3).
    COORDS = array([[0., 1., 0., 0., 1.],
                    [0., 0., 1., 0., 1.],
                    [0., 0., 0., 1., 1.]])
    TETS = array([[0, 1], [1, 2], [2, 3], [3, 4]])

    #: The quadratic every order has to reproduce, and its gradient.
    F = staticmethod(lambda x, y, z: x ** 2 + 2 * x * y + 3 * y ** 2 + 4 * z ** 2)
    DF = staticmethod(lambda p: stack([2 * p[0] + 2 * p[1],
                                       2 * p[0] + 6 * p[1], 8 * p[2]]))

    def _mesh(self):
        mesh = TetMesh(self.COORDS, TetTopology(self.TETS))
        coords = array(mesh.coords)
        return mesh.withprop('f', self.F(coords[0], coords[1], coords[2])[None, :],
                             gradient=self.DF(coords)[None, :, :])

    #: Positions inside the first tetrahedron, and inside the second.
    INSIDE = ((0.2, 0.3, 0.1), (0.1, 0.1, 0.1), (0.05, 0.5, 0.05))

    def test_a_quadratic_is_reproduced(self):
        mesh = self._mesh()
        for order in (2, 3):
            with self.subTest(order=order):
                for (x, y, z) in self.INSIDE:
                    got = float(ravel(asarray(mesh.prop(
                        'f', at=array([[x], [y], [z]]),
                        interp=('bezier', order))))[0])
                    self.assertAlmostEqual(got, self.F(x, y, z), places=10)

    def _block(self, cubes):
        '''A tetrahedral mesh of a ``cubes``-cubed block of unit cubes.

        Each cube is cut into six tetrahedra by the Kuhn decomposition, which
        cuts a shared face the same way from both sides, so the pieces fit
        together. The corners belong to the mesh rather than to the cube, which
        is what gives a node inside the block the neighbours --- along the axes,
        across the faces, and through the body diagonal --- that a quadratic in
        three dimensions needs ten monomials to be determined by.
        '''
        coords = []
        where = {}

        def at(i, j, k):
            if (i, j, k) not in where:
                where[(i, j, k)] = len(coords)
                coords.append((i, j, k))
            return where[(i, j, k)]

        def corner(i, j, k, bits):
            return at(i + (bits & 1), j + ((bits >> 1) & 1),
                      k + ((bits >> 2) & 1))

        tets = []
        for i in range(cubes):
            for j in range(cubes):
                for k in range(cubes):
                    for (a, b) in ((0, 1), (0, 2), (1, 0),
                                   (1, 2), (2, 0), (2, 1)):
                        tets.append([corner(i, j, k, 0),
                                     corner(i, j, k, 1 << a),
                                     corner(i, j, k, (1 << a) + (1 << b)),
                                     corner(i, j, k, 7)])
        return TetMesh(array(coords, dtype=float).T,
                       TetTopology(array(tets, dtype=int).T))

    def test_the_estimate_reproduces_a_quadratic_on_a_mesh(self):
        # Two tetrahedra cannot say what a quadratic was --- five nodes do not
        # determine one --- so a mesh's worth of neighbours is what the estimate
        # needs. Given them, the fit uses an estimated gradient and reproduces
        # the quadratic from the values alone.
        mesh = self._block(2)
        coords = array(mesh.coords)
        mesh = mesh.withprop(
            'f', self.F(coords[0], coords[1], coords[2])[None, :])
        for (x, y, z) in ((0.5, 0.5, 0.5), (0.25, 0.75, 0.5), (1.5, 0.5, 0.25)):
            got = float(ravel(asarray(mesh.prop(
                'f', at=array([[x], [y], [z]]),
                interp=('bezier', 2))))[0])
            with self.subTest(point=(x, y, z)):
                self.assertAlmostEqual(got, self.F(x, y, z), places=9)

    def test_the_values_at_the_corners_are_interpolated(self):
        mesh = self._mesh()
        coords = array(mesh.coords)
        for order in (2, 3):
            with self.subTest(order=order):
                for node in range(5):
                    got = float(ravel(asarray(mesh.prop(
                        'f', at=coords[:, node].reshape(3, 1),
                        interp=('bezier', order))))[0])
                    self.assertAlmostEqual(
                        got, self.F(coords[0, node], coords[1, node],
                                    coords[2, node]), places=12)

    #: A face of the first tetrahedron, as the weights its own local coordinate
    #: stores: the shared face is corners 1, 2 and 3, so its weight on corner 0
    #: is zero and the last weight is what the three of them leave. The second
    #: tetrahedron's corners are (c1, c2, c3, c4), so the same face is opposite
    #: its *last* corner, which is the one a weight is implied for.
    def _on_the_face(self, one, two, three):
        return (array([0.0, one, two]).reshape(3, 1),
                array([one, two, three]).reshape(3, 1))

    def test_two_tetrahedra_agree_on_the_face_they_share(self):
        mesh = self._mesh()
        loc = mesh.topo.Loc
        for order in (2, 3):
            for (s, t) in ((0.2, 0.3), (0.5, 0.25), (0.1, 0.8), (0.6, 0.2)):
                with self.subTest(order=order, point=(s, t)):
                    (first, second) = self._on_the_face(s, t, 1.0 - s - t)
                    one = float(ravel(asarray(mesh.prop(
                        'f', at=loc(array([0]), first),
                        interp=('bezier', order))))[0])
                    two = float(ravel(asarray(mesh.prop(
                        'f', at=loc(array([1]), second),
                        interp=('bezier', order))))[0])
                    self.assertAlmostEqual(one, two, places=12)
                    # ...and both are the field's own value there, since it is
                    # a quadratic and the construction reproduces those.
                    (x, y, z) = (s, t, 1.0 - s - t)
                    self.assertAlmostEqual(one, self.F(x, y, z), places=10)


class TestPolynomialMethod(TestCase):
    '''The polynomial method: a monomial-basis least-squares fit.

    It is the same fit for every kind of element, because it makes no use of how
    an element's corners are arranged --- it writes a polynomial in the
    element's own local coordinates and finds its coefficients by least squares
    from the corner values and gradients. What it gives back depends on which of
    the two is the more numerous, and the tests below pin both cases.
    '''

    #: A quadratic with cross terms, and its exact gradient.
    F = staticmethod(lambda x, y, z: 0.4 * x ** 2 + 0.3 * y ** 2
                     + 0.2 * z ** 2 + 0.25 * x * y - 0.15 * y * z)
    DF = staticmethod(lambda x, y, z: np.array([0.8 * x + 0.25 * y,
                                                0.6 * y + 0.25 * x - 0.15 * z,
                                                0.4 * z - 0.15 * y]))

    def _square(self):
        '''A square split into two triangles, in the plane.'''
        from euclib import trimesh
        return trimesh(array([[0., 1., 0., 1.], [0., 0., 1., 1.]]),
                       array([[0, 1], [1, 3], [2, 2]]))

    def _cube(self):
        '''Two tetrahedra filling the unit cube.'''
        from euclib import tetmesh
        return tetmesh(array([[0., 1., 0., 0., 1.],
                              [0., 0., 1., 0., 1.],
                              [0., 0., 0., 1., 1.]]),
                       array([[0, 1], [1, 2], [2, 3], [3, 4]]))

    def _carrying(self, geom, dim):
        '''The geometry with a quadratic on it, and its exact gradient.

        The field is written in three dimensions and read in as many as the
        geometry has, so that the same quadratic serves a segment, a triangle,
        and a tetrahedron.
        '''
        coords = np.asarray(geom.coords)
        padded = np.vstack([coords, np.zeros((3 - dim, coords.shape[1]))])
        values = array([[self.F(padded[0, i], padded[1, i], padded[2, i])
                         for i in range(padded.shape[1])]])
        gradient = stack([self.DF(padded[0, i], padded[1, i], padded[2, i])[:dim]
                          for i in range(padded.shape[1])],
                         axis=-1)[None, :, :]
        return geom.withprop('f', values, gradient=gradient)

    def test_a_quadratic_is_reproduced_at_order_two(self):
        # At order 2 the conditions outnumber the coefficients on every element,
        # so the data is the polynomial and it comes back exactly.
        from euclib import SegPath
        from euclib.types import SegTopology
        for (label, geom, dim, places) in (
                ('segment',
                 SegPath(array([[0., 1., 2.], [0., 0., 0.]]),
                         SegTopology([[0, 1], [1, 2]])), 2,
                 ((0.5, 0.0), (1.5, 0.0), (0.2, 0.0))),
                ('triangle', self._square(), 2,
                 ((0.2, 0.3), (0.7, 0.1), (0.55, 0.4))),
                ('tetrahedron', self._cube(), 3,
                 ((0.2, 0.3, 0.1), (0.5, 0.2, 0.1), (0.1, 0.1, 0.6)))):
            with self.subTest(element=label):
                carried = self._carrying(geom, dim)
                for point in places:
                    at = array(point, dtype=float).reshape(dim, 1)
                    got = float(np.ravel(np.asarray(carried.prop(
                        'f', at=at, interp=('polynomial', 2))))[0])
                    self.assertAlmostEqual(got, self.F(*(point + (0.,) * (3 - dim))),
                                           places=12)

    def test_a_segment_reproduces_a_quadratic_at_order_three_too(self):
        # A segment's four conditions against four coefficients determine its
        # cubic exactly, so there is no freedom left for a least-norm choice to
        # spend and the quadratic comes back.
        from euclib import SegPath
        from euclib.types import SegTopology
        path = SegPath(array([[0., 1., 2.], [0., 0., 0.]]),
                       SegTopology([[0, 1], [1, 2]]))
        carried = self._carrying(path, 2)
        for x in (0.25, 1.0, 1.75):
            got = float(np.ravel(np.asarray(carried.prop(
                'f', at=array([[x], [0.]]), interp=('polynomial', 3))))[0])
            with self.subTest(x=x):
                self.assertAlmostEqual(got, self.F(x, 0.0, 0.0), places=12)

    def test_order_three_matches_the_data_and_not_the_polynomial(self):
        # A triangle's cubic has ten coefficients against nine conditions, so
        # the fit matches the data --- which it can, there being room --- and
        # then takes the solution of least norm. That is a different cubic from
        # the quadratic the data came from, and the difference is the point:
        # the Bezier method spends the same freedom on recovering the quadratic.
        carried = self._carrying(self._square(), 2)
        coords = array(carried.coords)
        for node in range(coords.shape[1]):
            at = coords[:, node].reshape(2, 1)
            for (method, places) in (('polynomial', 10), ('bezier', 12)):
                got = float(np.ravel(np.asarray(carried.prop(
                    'f', at=at, interp=(method, 3))))[0])
                with self.subTest(node=node, method=method):
                    self.assertAlmostEqual(got, self.F(coords[0, node],
                                                       coords[1, node], 0.0),
                                           places=places)
        # Between the corners the two differ, and the polynomial one is the
        # further from the quadratic.
        at = array([[0.55], [0.4]])
        mine = float(np.ravel(np.asarray(carried.prop(
            'f', at=at, interp=('polynomial', 3))))[0])
        theirs = float(np.ravel(np.asarray(carried.prop(
            'f', at=at, interp=('bezier', 3))))[0])
        want = self.F(0.55, 0.4, 0.0)
        self.assertAlmostEqual(theirs, want, places=12)
        self.assertGreater(abs(mine - want), 1e-4)

    def test_the_data_is_matched_where_there_is_room_for_it(self):
        # A cubic on a triangle can take any nine conditions, so it matches the
        # corner values and gradients whatever they are; a quadratic cannot, and
        # there the fit is a compromise rather than an interpolation.
        mesh = self._square()
        coords = array(mesh.coords)
        values = array([[self.F(x, y, 0.0) for (x, y) in coords.T]])
        gradient = stack([self.DF(x, y, 0.0)[:2] for (x, y) in coords.T],
                         axis=-1)[None, :, :]
        gradient[0, 0] *= 2.0                      # a slope the values forbid
        carried = mesh.withprop('f', values, gradient=gradient)
        for node in range(coords.shape[1]):
            at = coords[:, node].reshape(2, 1)
            cubic = float(np.ravel(np.asarray(carried.prop(
                'f', at=at, interp=('polynomial', 3))))[0])
            with self.subTest(node=node):
                self.assertAlmostEqual(cubic, values[0, node], places=10)

    def test_the_fit_is_the_same_for_every_kind_of_element(self):
        # The polynomial fit is one function, since it uses nothing about how
        # the corners are arranged beyond their count: a triangle's fit and a
        # tetrahedron's are found by the same code.
        from euclib.types._interp import _element_fit, polynomial_fit
        from euclib import SegPath
        from euclib.types import SegTopology
        path = SegPath(array([[0., 1., 2.], [0., 0., 0.]]),
                       SegTopology([[0, 1], [1, 2]]))
        for (label, geom) in (('segment', path), ('triangle', self._square()),
                              ('tetrahedron', self._cube())):
            with self.subTest(element=label):
                self.assertIs(_element_fit(geom, 'polynomial'), polynomial_fit)
                for order in (2, 3):
                    # ...and it answers for each of them, rather than raising.
                    self.assertTrue(np.isfinite(float(np.ravel(np.asarray(
                        self._carrying(geom, geom.dim).prop(
                            'f', at=np.asarray(geom.coords)[:, :1],
                            interp=('polynomial', order))))[0])))


class TestCloughTocher(TestCase):
    '''The Clough-Tocher method through the engine.

    The element itself is tested in ``test_ct``; what these check is that the
    engine reaches it, that it is offered where it belongs and nowhere else,
    and that the field it produces is the one the element was verified to
    compute. The scheme is cubic --- the piecewise *quadratic* one that is
    smooth across the split is Powell-Sabin's --- so order 3 is the only order
    it answers at.
    '''

    #: A square of four corners in two triangles, sharing the diagonal.
    COORDS = array([[0., 1., 0., 1.], [0., 0., 1., 1.]])

    QUADRATIC = staticmethod(lambda x, y: (0.4 * x ** 2 - 0.3 * x * y
                                           + 0.25 * y ** 2 + 0.8 * x
                                           - 0.2 * y + 0.5))
    GRADIENT = staticmethod(lambda x, y: array([0.8 * x - 0.3 * y + 0.8,
                                                -0.3 * x + 0.5 * y - 0.2]))

    def _mesh(self, gradient=True):
        from euclib import trimesh
        mesh = trimesh(self.COORDS, array([[0, 1], [1, 3], [2, 2]]))
        coords = self.COORDS
        count = coords.shape[1]
        values = array([[self.QUADRATIC(coords[0, i], coords[1, i])
                         for i in range(count)]])
        supplied = stack([self.GRADIENT(coords[0, i], coords[1, i])
                          for i in range(count)], axis=-1)[None, :, :]
        return mesh.withprop('f', values,
                             gradient=supplied if gradient else None)

    def test_a_triangle_offers_it_at_order_three_and_no_other(self):
        from euclib.abc import supported_interp
        from euclib import trimesh
        support = supported_interp(trimesh(self.COORDS,
                                           array([[0, 1], [1, 3], [2, 2]])).topo)
        self.assertIn(('clough-tocher', 3), support)
        for order in (0, 1, 2):
            with self.subTest(order=order):
                self.assertNotIn(('clough-tocher', order), support)

    def test_a_quadratic_is_reproduced(self):
        # The element holds the cubics and its data over-determines a
        # quadratic, so a quadratic comes back exactly.
        mesh = self._mesh()
        for (x, y) in ((0.2, 0.3), (0.7, 0.1), (0.55, 0.4), (0.05, 0.9),
                       (0.3, 0.3)):
            with self.subTest(point=(x, y)):
                got = float(ravel(asarray(mesh.prop(
                    'f', at=array([[x], [y]]),
                    interp=('clough-tocher', 3))))[0])
                self.assertAlmostEqual(got, self.QUADRATIC(x, y), places=12)

    def test_the_values_at_the_corners_are_interpolated(self):
        mesh = self._mesh()
        coords = self.COORDS
        for node in range(coords.shape[1]):
            with self.subTest(node=node):
                got = float(ravel(asarray(mesh.prop(
                    'f', at=coords[:, node].reshape(2, 1),
                    interp=('clough-tocher', 3))))[0])
                self.assertAlmostEqual(
                    got, self.QUADRATIC(coords[0, node], coords[1, node]),
                    places=12)

    def test_a_channelled_property_is_fitted_channel_by_channel(self):
        from euclib import trimesh
        mesh = trimesh(self.COORDS, array([[0, 1], [1, 3], [2, 2]]))
        coords = self.COORDS
        count = coords.shape[1]
        second = lambda x, y: -0.2 * x ** 2 + 0.5 * x * y + 0.1 * y + 1.0
        second_gradient = lambda x, y: array([-0.4 * x + 0.5 * y, 0.5 * x + 0.1])
        values = stack([array([self.QUADRATIC(coords[0, i], coords[1, i])
                               for i in range(count)]),
                        array([second(coords[0, i], coords[1, i])
                               for i in range(count)])])
        slopes = stack([
            stack([self.GRADIENT(coords[0, i], coords[1, i])
                   for i in range(count)], axis=-1),
            stack([second_gradient(coords[0, i], coords[1, i])
                   for i in range(count)], axis=-1)])
        carried = mesh.withprop('f', values, gradient=slopes)
        for (x, y) in ((0.25, 0.25), (0.6, 0.2)):
            got = ravel(asarray(carried.prop(
                'f', at=array([[x], [y]]), interp=('clough-tocher', 3))))
            with self.subTest(point=(x, y)):
                self.assertAlmostEqual(float(got[0]), self.QUADRATIC(x, y),
                                       places=12)
                self.assertAlmostEqual(float(got[1]), second(x, y), places=12)


class TestCloughTocherOnATriangleInSpace(TestCase):
    '''The method on a mesh that does not lie in a coordinate plane.

    A triangle in space has a plane of its own just as a triangle in the plane
    does --- its three pieces stay coplanar however it is carried --- so the
    construction is the same one and only the *data* has to be read aright: in
    directions the geometry defines rather than in the coordinates'. A surface
    field's ambient gradient has as many components as the space has
    dimensions, and the element is built from its derivatives along the
    triangle's edges alone. Read as the gradient's first two coordinate
    components, the element answered wrongly by more than a fifth of the
    field's own scale on a triangle that was merely turned; read as the
    derivation says, a quadratic of the triangle's own plane comes back to
    machine precision, which is what this checks.
    '''

    #: A triangle carried off every coordinate plane, with no symmetry to hide
    #: behind either.
    TRIANGLE = array([[0.0, 1.2, 0.4], [0.3, 0.3, 1.1], [0.7, 0.6, -0.2]])

    def _along(self):
        return np.stack([self.TRIANGLE[:, 1] - self.TRIANGLE[:, 0],
                         self.TRIANGLE[:, 2] - self.TRIANGLE[:, 0]], axis=1)

    def _inplane(self, point):
        '''A position's two coordinates within the triangle's own plane.'''
        return np.linalg.pinv(self._along()) @ (point - self.TRIANGLE[:, 0])

    def _field(self, point):
        (u, v) = self._inplane(point)
        return (0.4 * u ** 2 - 0.3 * u * v + 0.25 * v ** 2
                + 0.8 * u - 0.2 * v + 0.5)

    def _gradient(self, point):
        (u, v) = self._inplane(point)
        inplane = array([0.8 * u - 0.3 * v + 0.8, -0.3 * u + 0.5 * v - 0.2])
        return np.linalg.pinv(self._along()).T @ inplane

    def _mesh(self):
        from euclib import trimesh
        mesh = trimesh(self.TRIANGLE, array([[0], [1], [2]]))
        values = array([[self._field(self.TRIANGLE[:, c]) for c in range(3)]])
        slopes = stack([self._gradient(self.TRIANGLE[:, c])
                        for c in range(3)], axis=-1)[None, :, :]
        return mesh.withprop('q', values, gradient=slopes)

    def test_a_quadratic_of_the_triangles_own_plane_is_reproduced(self):
        carried = self._mesh()
        for (u, v) in ((0.2, 0.2), (0.5, 0.3), (0.3, 0.5), (0.1, 0.8)):
            with self.subTest(inplane=(u, v)):
                point = self.TRIANGLE[:, 0] + self._along() @ array([u, v])
                got = float(ravel(asarray(carried.prop(
                    'q', at=point.reshape(3, 1),
                    interp=('clough-tocher', 3))))[0])
                self.assertAlmostEqual(got, self._field(point), places=12)

    def test_it_is_offered_where_a_triangle_in_space_is(self):
        from euclib.abc import supported_interp
        from euclib import trimesh
        support = supported_interp(
            trimesh(self.TRIANGLE, array([[0], [1], [2]])).topo)
        self.assertIn(('clough-tocher', 3), support)


class TestPowellSabin(TestCase):
    '''The Powell-Sabin method through the engine.

    The element itself is tested in ``test_ps``; what these check is that the
    engine reaches it, that it is offered where it belongs and nowhere else, and
    that the field it produces is the one the element was verified to compute.
    The scheme is quadratic --- the piecewise *cubic* one that is C1 across a
    split of the same kind is Clough-Tocher's --- so order 2 is the only order it
    answers at.
    '''

    #: A square of four corners in two triangles, sharing the diagonal.
    COORDS = array([[0., 1., 0., 1.], [0., 0., 1., 1.]])

    QUADRATIC = staticmethod(lambda x, y: (0.4 * x ** 2 - 0.3 * x * y
                                           + 0.25 * y ** 2 + 0.8 * x
                                           - 0.2 * y + 0.5))
    GRADIENT = staticmethod(lambda x, y: array([0.8 * x - 0.3 * y + 0.8,
                                                -0.3 * x + 0.5 * y - 0.2]))

    def _mesh(self):
        from euclib import trimesh
        mesh = trimesh(self.COORDS, array([[0, 1], [1, 3], [2, 2]]))
        coords = self.COORDS
        count = coords.shape[1]
        values = array([[self.QUADRATIC(coords[0, i], coords[1, i])
                         for i in range(count)]])
        supplied = stack([self.GRADIENT(coords[0, i], coords[1, i])
                          for i in range(count)], axis=-1)[None, :, :]
        return mesh.withprop('f', values, gradient=supplied)

    def test_a_triangle_offers_it_at_order_two_and_no_other(self):
        from euclib.abc import supported_interp
        from euclib import trimesh
        support = supported_interp(trimesh(self.COORDS,
                                           array([[0, 1], [1, 3], [2, 2]])).topo)
        self.assertIn(('powell-sabin', 2), support)
        for order in (0, 1, 3):
            with self.subTest(order=order):
                self.assertNotIn(('powell-sabin', order), support)

    def test_a_quadratic_is_reproduced(self):
        # The element holds the quadratics and its nine numbers over-determine
        # one, so a quadratic comes back exactly.
        mesh = self._mesh()
        for (x, y) in ((0.2, 0.3), (0.7, 0.1), (0.55, 0.4), (0.05, 0.9),
                       (0.3, 0.3)):
            with self.subTest(point=(x, y)):
                got = float(ravel(asarray(mesh.prop(
                    'f', at=array([[x], [y]]),
                    interp=('powell-sabin', 2))))[0])
                self.assertAlmostEqual(got, self.QUADRATIC(x, y), places=12)

    def test_the_values_at_the_corners_are_interpolated(self):
        mesh = self._mesh()
        coords = self.COORDS
        for node in range(coords.shape[1]):
            with self.subTest(node=node):
                got = float(ravel(asarray(mesh.prop(
                    'f', at=coords[:, node].reshape(2, 1),
                    interp=('powell-sabin', 2))))[0])
                self.assertAlmostEqual(
                    got, self.QUADRATIC(coords[0, node], coords[1, node]),
                    places=12)

    def test_a_property_with_only_values_is_fitted_from_estimates(self):
        '''The element reads its triangle and nothing else, so nothing about the
        mesh can stop it answering.

        This is the structural difference from Clough-Tocher, and it is worth a
        test rather than a sentence. That element's twelfth datum is the
        derivative *across* an edge, which is a function of the whole mesh: its
        fit takes the property's values at every coordinate and refuses without
        them, because a triangle's patch is built from edges it does not own.
        This one takes its nine numbers from the triangle it is on, so the engine
        can fit a property that carries values and no gradients, and the corner
        values come back exactly whether the slopes were supplied or estimated.
        '''
        from euclib import trimesh
        mesh = trimesh(self.COORDS, array([[0, 1], [1, 3], [2, 2]]))
        coords = self.COORDS
        count = coords.shape[1]
        carried = mesh.withprop('f', array(
            [[self.QUADRATIC(coords[0, i], coords[1, i])
              for i in range(count)]]))
        for node in range(count):
            with self.subTest(node=node):
                got = float(ravel(asarray(carried.prop(
                    'f', at=coords[:, node].reshape(2, 1),
                    interp=('powell-sabin', 2))))[0])
                self.assertAlmostEqual(
                    got, self.QUADRATIC(coords[0, node], coords[1, node]),
                    places=12)


class TestPowellSabinOnATriangleInSpace(TestCase):
    '''The method on a mesh that does not lie in a coordinate plane.

    The element is barycentric throughout --- every number it is built from is a
    value or a derivative along a direction the triangle's own edges define ---
    so a triangle in space is the same problem as one in a plane, and the same
    answer. This is the check that it is: the Clough-Tocher element had to be
    rebuilt to earn this, and this one was built with it from the start.
    '''

    #: A triangle carried off every coordinate plane, with no symmetry to hide
    #: behind.
    TRIANGLE = array([[0.0, 1.2, 0.4], [0.3, 0.3, 1.1], [0.7, 0.6, -0.2]])

    def _along(self):
        return np.stack([self.TRIANGLE[:, 1] - self.TRIANGLE[:, 0],
                         self.TRIANGLE[:, 2] - self.TRIANGLE[:, 0]], axis=1)

    def _inplane(self, point):
        return np.linalg.pinv(self._along()) @ (point - self.TRIANGLE[:, 0])

    def _field(self, point):
        (u, v) = self._inplane(point)
        return (0.4 * u ** 2 - 0.3 * u * v + 0.25 * v ** 2
                + 0.8 * u - 0.2 * v + 0.5)

    def _gradient(self, point):
        (u, v) = self._inplane(point)
        inplane = array([0.8 * u - 0.3 * v + 0.8, -0.3 * u + 0.5 * v - 0.2])
        return np.linalg.pinv(self._along()).T @ inplane

    def test_a_quadratic_of_the_triangles_own_plane_is_reproduced(self):
        from euclib import trimesh
        mesh = trimesh(self.TRIANGLE, array([[0], [1], [2]]))
        values = array([[self._field(self.TRIANGLE[:, c]) for c in range(3)]])
        slopes = stack([self._gradient(self.TRIANGLE[:, c])
                        for c in range(3)], axis=-1)[None, :, :]
        carried = mesh.withprop('q', values, gradient=slopes)
        for (u, v) in ((0.2, 0.2), (0.5, 0.3), (0.3, 0.5), (0.1, 0.6)):
            with self.subTest(inplane=(u, v)):
                point = self.TRIANGLE[:, 0] + self._along() @ array([u, v])
                got = float(ravel(asarray(carried.prop(
                    'q', at=point.reshape(3, 1),
                    interp=('powell-sabin', 2))))[0])
                self.assertAlmostEqual(got, self._field(point), places=12)
