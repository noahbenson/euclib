# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/test_interp.py
'''Tests for the interpolation engine in ``euclib.types._interp``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

import numpy as np
from numpy import linalg
from numpy.random import default_rng
from numpy import (add, allclose, arange, array, asarray, concatenate, cos, eye,
                   floor, full, isfinite, isnan, linspace, meshgrid, nan, ones,
                   pi,
                   ravel, repeat, sin, sqrt, stack, tile, zeros)

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


class TestGridCubic(TestCase):
    '''Cubic convolution, the method between the samples.'''

    #: A wide axis, so that positions away from the boundary are the ones tested.
    COUNT = 40

    def _sampled(self, field, /):
        return Grid(eye(2), GridTopology((self.COUNT,))).withprop(
            'v', field(arange(self.COUNT, dtype=float)))

    def _read(self, field, positions, /, **kw):
        carried = self._sampled(field)
        at = carried.topo.Loc(sx=positions)
        return ravel(asarray(carried.prop('v', at=at, **kw)))

    def test_it_reproduces_the_quadratics(self):
        # What pins the kernel's free parameter down, and the whole reason the
        # method is worth having over linear interpolation.
        quadratic = lambda t: -0.7 * t ** 2 + 0.4 * t + 1.1
        probe = linspace(10.0, 30.0, 200)
        got = self._read(quadratic, probe, interp=('catmull-rom', 3))
        self.assertLess(np.abs(got - quadratic(probe)).max(), 1e-9)

    def test_it_does_not_reproduce_the_cubics(self):
        # Its approximation order is one less than the degree of its pieces, so
        # a cubic is close and not exact -- and *closer* than linear gets.
        cubic = lambda t: 0.3 * t ** 3 - 0.7 * t ** 2 + 0.4 * t + 1.1
        probe = linspace(10.0, 30.0, 200)
        got = self._read(cubic, probe, interp=('catmull-rom', 3))
        linear = self._read(cubic, probe, interp=('polynomial', 1))
        worst = np.abs(got - cubic(probe)).max()
        self.assertLess(worst, 0.1)
        self.assertGreater(np.abs(linear - cubic(probe)).max(), 10.0 * worst)

    def test_it_interpolates_the_samples(self):
        rng = default_rng(3)
        data = rng.normal(size=self.COUNT)
        carried = Grid(eye(2), GridTopology((self.COUNT,))).withprop('v', data)
        at = carried.topo.Loc(sx=arange(self.COUNT, dtype=float))
        got = ravel(asarray(carried.prop('v', at=at,
                                         interp=('catmull-rom', 3))))
        self.assertLess(np.abs(got - data).max(), 1e-10)

    def test_its_slope_is_continuous_across_a_sample(self):
        # The property the method is chosen for: linear interpolation's slope
        # jumps at every sample, this one's does not.  The measure is the second
        # difference across a sample, which for a smooth field is the curvature.
        smooth = lambda t: np.cos(t / 4.0)
        probe = linspace(10.0, 30.0, 400)
        got = self._read(smooth, probe, interp=('catmull-rom', 3))
        d2 = np.diff(got, 2) / (probe[1] - probe[0]) ** 2
        self.assertLess(np.abs(d2).max(), 0.1)
        linear = self._read(smooth, probe, interp=('polynomial', 1))
        d2 = np.diff(linear, 2) / (probe[1] - probe[0]) ** 2
        self.assertGreater(np.abs(d2).max(), 0.1)

    def test_the_boundary_rule_is_used_over_a_wider_span(self):
        # Four cells wide reaches two cells out, so a position a whole cell
        # inside the last sample already has a stencil past the end -- where the
        # two-cell linear stencil had half a cell of room.
        ramp = lambda t: 1.0 + t
        for sx in (38.0, 38.5, 39.0):
            with self.subTest(sx=sx):
                got = [float(self._read(ramp, array([sx]), interp=('catmull-rom', 3),
                                        border=one)[0])
                       for one in ('constant', 'half-symmetric')]
                self.assertAlmostEqual(got[0], got[1], places=5)
        # And past the last sample the three do differ, where the two-cell
        # linear stencil made two of them agree everywhere.
        (here, there) = (
            float(self._read(ramp, array([39.4]), interp=('catmull-rom', 3),
                             border='constant')[0]),
            float(self._read(ramp, array([39.4]), interp=('catmull-rom', 3),
                             border='whole-symmetric')[0]))
        self.assertGreater(abs(here - there), 1e-3)

    def test_a_triangle_reports_it_is_not_implemented(self):
        from euclib.abc import supported_interp
        mesh = TriMesh(array([[0., 1., 0.], [0., 0., 1.]]), TriTopology(
            array([[0], [1], [2]])))
        self.assertNotIn(('catmull-rom', 3), supported_interp(mesh.topo))


class TestGridBoundary(TestCase):
    '''What a grid's interpolation finds past its own edges.

    A position within half a cell of the edge straddles a cell that does not
    exist, and the three standard extensions say different things about what
    stands in for it. At order 1 two of them coincide --- the stencil is only
    two cells wide, and folding the one cell past the end gives the edge cell
    for *constant* and for *half-symmetric* alike --- which is a fact about
    narrow kernels and not a defect, and is what the first test pins down.
    '''

    #: A ramp, so that folding and continuing can be told apart.
    RAMP = array([1., 2., 3., 4., 5.])

    def _ramp(self, **kw):
        return Grid(eye(2), GridTopology((5,))).withprop('v', self.RAMP, **kw)

    def _at(self, geom, sx, /, **kw):
        return _scalar(geom, 'v', array([[sx]]), **kw)

    def test_the_interior_does_not_depend_on_the_extension(self):
        for sx in (0.0, 1.5, 3.25, 4.0):
            with self.subTest(sx=sx):
                got = [self._at(self._ramp(), sx, border=one)
                       for one in ('constant', 'half-symmetric',
                                   'whole-symmetric')]
                self.assertAlmostEqual(got[0], got[1])
                self.assertAlmostEqual(got[1], got[2])
                self.assertAlmostEqual(got[0], 1.0 + sx)

    def test_constant_and_half_symmetric_agree_at_order_one(self):
        # Both fold the one cell past the end onto the edge cell, so with a
        # two-cell stencil they are the same method. They part company as soon
        # as a kernel reaches two cells out, which is what the page computes.
        # A grid reaches only half a cell past its last sample, so that is as
        # far as these can be asked; a wider kernel reaches further, which is
        # what the page computes.
        for sx in (4.25, 4.4, 4.5):
            with self.subTest(sx=sx):
                self.assertAlmostEqual(
                    self._at(self._ramp(), sx, border='constant'),
                    self._at(self._ramp(), sx, border='half-symmetric'))

    def test_whole_symmetric_mirrors_the_data_about_the_edge(self):
        # The value a distance d past the last sample is the value d before it,
        # which for this ramp means the slope reverses rather than continues.
        for d in (0.25, 0.4, 0.5):
            with self.subTest(d=d):
                self.assertAlmostEqual(
                    self._at(self._ramp(), 4.0 + d, border='whole-symmetric'),
                    5.0 - d)

    def test_the_border_can_be_set_on_the_property_or_on_the_read(self):
        carried = self._ramp(interp=('polynomial', 1), border='whole-symmetric')
        # The property's own extension is used when none is given...
        self.assertAlmostEqual(self._at(carried, 4.5), 4.5)
        # ...and a read may override it.
        self.assertAlmostEqual(
            self._at(carried, 4.5, border='half-symmetric'), 5.0)

    def test_the_default_is_half_symmetric(self):
        from euclib.abc import BORDER_DEFAULT
        self.assertEqual(BORDER_DEFAULT, 'half-symmetric')
        self.assertAlmostEqual(self._at(self._ramp(), 4.5), 5.0)

    def test_an_unknown_extension_is_refused(self):
        from euclib.abc._property import normalize_border
        for bad in ('nonsense', 'reflect', ''):
            with self.subTest(border=bad):
                with self.assertRaises(ValueError):
                    normalize_border(bad)
        # ...and a read that asks for one is refused too, before it computes.
        with self.assertRaises(ValueError):
            self._at(self._ramp(), 4.5, border='nonsense')

    def test_a_two_dimensional_grid_extends_each_axis_on_its_own(self):
        # A corner is past two edges at once, and each axis folds by its own
        # rule. Half a cell out, the two extensions that repeat the edge value
        # fold both axes onto the corner cell; whole-sample reflection folds
        # each onto the cell before it, so the value comes from the four cells
        # meeting one step in.
        values = arange(16.).reshape(4, 4)
        g = Grid(eye(3), GridTopology((4, 4))).withprop('v', values)
        for border in ('constant', 'half-symmetric'):
            with self.subTest(border=border):
                self.assertAlmostEqual(
                    _scalar(g, 'v', array([[3.5], [3.5]]), border=border), 15.0)
        self.assertAlmostEqual(
            _scalar(g, 'v', array([[3.5], [3.5]]), border='whole-symmetric'),
            values[2:4, 2:4].mean())


class TestGridSpline(TestCase):
    '''B-splines, and the prefilter that makes them interpolate.'''

    COUNT = 60

    def _carried(self, field, /, **kw):
        return Grid(eye(2), GridTopology((self.COUNT,))).withprop(
            'v', field(arange(self.COUNT, dtype=float)), **kw)

    def _read(self, field, positions, /, **kw):
        carried = self._carried(field)
        at = carried.topo.Loc(sx=positions)
        return ravel(asarray(carried.prop('v', at=at, **kw)))

    def test_each_basis_reproduces_its_own_degree(self):
        # What the two orders are for. A quadratic basis reproduces quadratics
        # and not cubics; a cubic basis reproduces cubics. Getting one degree
        # more costs one more cell of support.
        probe = linspace(20.0, 40.0, 200)
        quadratic = lambda t: -0.7 * t ** 2 + 0.4 * t + 1.1
        cubic = lambda t: 0.3 * t ** 3 - 0.7 * t ** 2 + 0.4 * t + 1.1
        with self.subTest(degree=2, field='quadratic'):
            self.assertLess(np.abs(self._read(
                quadratic, probe, interp=('spline', 2))
                - quadratic(probe)).max(), 1e-9)
        with self.subTest(degree=3, field='cubic'):
            self.assertLess(np.abs(self._read(
                cubic, probe, interp=('spline', 3))
                - cubic(probe)).max(), 1e-6)
        with self.subTest(degree=2, field='cubic'):
            self.assertGreater(np.abs(self._read(
                cubic, probe, interp=('spline', 2))
                - cubic(probe)).max(), 1e-4)

    def test_the_cubic_basis_beats_cubic_convolution_on_a_cubic(self):
        # The same four-cell support spent two ways: approximation order 4
        # against 3.
        cubic = lambda t: 0.3 * t ** 3 - 0.7 * t ** 2 + 0.4 * t + 1.1
        probe = linspace(20.0, 40.0, 200)
        spline = np.abs(self._read(cubic, probe, interp=('spline', 3))
                        - cubic(probe)).max()
        convolution = np.abs(self._read(cubic, probe,
                                        interp=('catmull-rom', 3))
                             - cubic(probe)).max()
        self.assertLess(spline, 1e-6)
        self.assertGreater(convolution, 1e4 * spline)

    def test_the_interpolant_passes_through_the_samples(self):
        rng = default_rng(5)
        data = rng.normal(size=self.COUNT)
        carried = Grid(eye(2), GridTopology((self.COUNT,))).withprop('v', data)
        at = carried.topo.Loc(sx=arange(self.COUNT, dtype=float))
        for order in (2, 3):
            with self.subTest(order=order):
                got = ravel(asarray(carried.prop(
                    'v', at=at, interp=('spline', order))))
                self.assertLess(np.abs(got - data).max(), 1e-9)

    def test_a_two_dimensional_grid_is_filtered_along_each_axis(self):
        values = add.outer(arange(12.), arange(9.))
        grid = Grid(eye(3), GridTopology((12, 9))).withprop('v', values)
        (xs, ys) = (repeat(arange(12.), 9), tile(arange(9.), 12))
        at = grid.topo.Loc(sx=xs, sy=ys)
        for order in (2, 3):
            with self.subTest(order=order):
                got = ravel(asarray(grid.prop('v', at=at,
                                              interp=('spline', order))))
                self.assertLess(np.abs(got - (xs + ys)).max(), 1e-10)

    def test_the_boundary_rule_reaches_the_prefilter(self):
        # The coefficients of a ramp depend on how the data was extended, so the
        # three rules give three different answers near an edge -- and the same
        # ones away from it.
        ramp = lambda t: 1.0 + 3.0 * t
        near = [float(self._read(ramp, array([0.1]), interp=('spline', 3),
                                 border=one)[0])
                for one in ('constant', 'half-symmetric', 'whole-symmetric')]
        self.assertGreater(max(near) - min(near), 1e-6)
        middle = [float(self._read(ramp, array([30.0]), interp=('spline', 3),
                                  border=one)[0])
                  for one in ('constant', 'half-symmetric', 'whole-symmetric')]
        self.assertLess(max(middle) - min(middle), 1e-9)

    def test_a_read_may_ask_for_a_spline_the_property_does_not_carry(self):
        # The prefilter a property caches is the one for its *own* method, so a
        # read that overrides the method cannot use it and must filter again.
        cubic = lambda t: 0.3 * t ** 3 - 0.7 * t ** 2 + 0.4 * t + 1.1
        carried = self._carried(cubic, interp=('polynomial', 1))
        probe = linspace(20.0, 40.0, 50)
        at = carried.topo.Loc(sx=probe)
        got = ravel(asarray(carried.prop('v', at=at, interp=('spline', 3))))
        self.assertLess(np.abs(got - cubic(probe)).max(), 1e-6)


class TestThePrefilter(TestCase):
    '''The prefilter's rate and scale, against a direct solve.'''

    def test_the_rate_and_the_scale_are_the_ones_that_invert_p(self):
        # The module derives both from the basis's values rather than quoting
        # them, so this is the check that the derivation is right: the filter's
        # result, summed against p, must be the data back.
        from euclib.types import _grid
        for (degree, expected_rate, expected_scale) in ((2, -3.0 + sqrt(8.0), 8.0),
                                                        (3, sqrt(3.0) - 2.0, 6.0)):
            with self.subTest(degree=degree):
                (rate, scale) = _grid._rate(degree)
                self.assertAlmostEqual(rate, expected_rate, places=12)
                self.assertAlmostEqual(scale, expected_scale, places=12)

    def test_the_filter_inverts_the_basis_against_a_direct_solve(self):
        from euclib.types import _grid
        count = 60
        f = sin(arange(count, dtype=float) / 5.0)
        (coefficients, first) = _grid.prefilter(f.reshape(1, -1), (count,),
                                                'constant', 3)
        basis = _grid.BASES[3][0]
        # The same problem solved directly: solve p * c = f as a tridiagonal
        # system, on the data extended far enough past both ends that the free
        # ends there have no influence on the part being compared. That is the
        # constant extension the prefilter used, so the two must agree.
        margin = 60
        wide = concatenate([full(margin, f[0]), f, full(margin, f[-1])])
        rows = wide.shape[0]
        matrix = zeros((rows, rows))
        for k in range(rows):
            matrix[k, k] = float(basis(0))
            if k > 0:
                matrix[k, k - 1] = float(basis(1))
            if k < rows - 1:
                matrix[k, k + 1] = float(basis(1))
        want = linalg.solve(matrix, wide)[margin:margin + count]
        got = ravel(asarray(coefficients))[_grid.MARGIN:_grid.MARGIN + count]
        self.assertLess(np.abs(got - want).max(), 1e-9)


class TestGridLanczos(TestCase):
    '''Lanczos, whose weights must be normalized and whose polynomials do not
    come back.'''

    COUNT = 90

    def _read(self, field, positions, /, **kw):
        carried = Grid(eye(2), GridTopology((self.COUNT,))).withprop(
            'v', field(arange(self.COUNT, dtype=float)))
        at = carried.topo.Loc(sx=positions)
        return ravel(asarray(carried.prop('v', at=at, **kw)))

    def test_a_constant_field_comes_back_constant(self):
        # The whole reason the method normalizes: its weights do not sum to one
        # on their own, so without the division a constant does not come back.
        flat = lambda t: 0.0 * t + 3.0
        probe = linspace(20.0, 40.0, 200)
        for order in (2, 3):
            with self.subTest(order=order):
                got = self._read(flat, probe, interp=('lanczos', order))
                self.assertLess(np.abs(got - 3.0).max(), 1e-12)

    def test_it_interpolates_the_samples(self):
        rng = default_rng(7)
        data = rng.normal(size=self.COUNT)
        carried = Grid(eye(2), GridTopology((self.COUNT,))).withprop('v', data)
        at = carried.topo.Loc(sx=arange(self.COUNT, dtype=float))
        for order in (2, 3):
            with self.subTest(order=order):
                got = ravel(asarray(carried.prop(
                    'v', at=at, interp=('lanczos', order))))
                self.assertLess(np.abs(got - data).max(), 1e-12)

    def test_it_does_not_reproduce_polynomials(self):
        # What tells it apart from every method built before it. A polynomial
        # kernel satisfies the moment conditions identically and reproduces its
        # polynomials *exactly*; an approximation to the sinc satisfies them
        # only in the limit, so at a fixed sampling there is an error -- and
        # that is true even though it agrees on the constant above.
        affine = lambda t: 0.4 * t + 1.1
        probe = linspace(30.0, 60.0, 300)
        for order in (2, 3):
            with self.subTest(order=order):
                got = self._read(affine, probe, interp=('lanczos', order))
                self.assertGreater(np.abs(got - affine(probe)).max(), 1e-3)
        # ...where the polynomial kernel of the same support is exact.
        exact = self._read(affine, probe, interp=('catmull-rom', 3))
        self.assertLess(np.abs(exact - affine(probe)).max(), 1e-12)

    def test_a_constant_field_is_what_the_normalization_is_for(self):
        # Stated as the contrast it is: the same data, the same kernel, with the
        # weights summed as they come out.
        from euclib.types import _grid
        count = 60
        worst = 0.0
        for t in linspace(20.0, 40.0, 200):
            base = int(floor(t))
            ks = arange(base - 1, base + 3)
            w = array([float(_grid.lanczos2(t - k)) for k in ks])
            worst = max(worst, abs(float(w.sum()) * 3.0 - 3.0))
        self.assertGreater(worst, 1e-3)

    def test_a_triangle_reports_it_is_not_implemented(self):
        from euclib.abc import supported_interp
        mesh = TriMesh(array([[0., 1., 0.], [0., 0., 1.]]), TriTopology(
            array([[0], [1], [2]])))
        for order in (2, 3):
            with self.subTest(order=order):
                self.assertNotIn(('lanczos', order), supported_interp(mesh.topo))


class TestGridPolynomial(TestCase):
    '''The monomial least-squares fit, which is a fit and not an interpolation.'''

    COUNT = 40

    def _carried(self, field, /):
        (ix, iy) = meshgrid(arange(self.COUNT, dtype=float),
                            arange(self.COUNT, dtype=float), indexing='ij')
        return Grid(eye(3), GridTopology((self.COUNT, self.COUNT))).withprop(
            'v', field(ix, iy))

    def _read(self, field, points, /, **kw):
        carried = self._carried(field)
        (xs, ys) = (array([p[0] for p in points]), array([p[1] for p in points]))
        at = carried.topo.Loc(sx=xs, sy=ys)
        return ravel(asarray(carried.prop('v', at=at, **kw)))

    def test_it_reproduces_its_own_degree(self):
        # A polynomial of degree k through a degree-k fit is in the span, so the
        # fit is exact -- away from the edges, where the block is folded.
        quadratic = lambda x, y: 0.4 * x ** 2 - 0.3 * x * y + 0.25 * y ** 2
        cubic = lambda x, y: 0.3 * x ** 3 - 0.2 * x ** 2 * y + 0.1 * y ** 3
        points = [(20.0, 20.0), (20.5, 20.3), (19.7, 21.2)]
        for (degree, field) in ((2, quadratic), (3, cubic)):
            with self.subTest(degree=degree):
                got = self._read(field, points, interp=('polynomial', degree))
                want = array([field(*p) for p in points])
                self.assertLess(np.abs(got - want).max(), 1e-9)

    def test_it_does_not_interpolate_a_field_it_cannot_hold(self):
        # Where every other grid method returns the datum, a fit returns its
        # polynomial's value there -- which is the point of a fit rather than a
        # defect, and is what this checks so the difference is on the record.
        smooth = lambda x, y: sin(x / 6.0) * cos(y / 7.0)
        got = float(self._read(smooth, [(20.0, 20.0)],
                               interp=('polynomial', 2))[0])
        datum = float(smooth(20.0, 20.0))
        self.assertGreater(abs(got - datum), 1e-6)

    def test_it_is_not_continuous_across_a_cell_boundary(self):
        # The block of cells is a different block either side of the boundary,
        # so the field jumps. No other grid method does.
        smooth = lambda x, y: sin(x / 6.0) * cos(y / 7.0)
        # A step small enough that a continuous field's two values differ only
        # by the slope across it -- so what the first pair shows is a jump and
        # not a steep slope.
        step = 1e-6
        (left, right) = (
            float(self._read(smooth, [(20.0 - step, 20.0)],
                             interp=('polynomial', 2))[0]),
            float(self._read(smooth, [(20.0 + step, 20.0)],
                             interp=('polynomial', 2))[0]))
        self.assertGreater(abs(left - right), 1e-5)
        # ...where the linear blend, asked at the same two positions, does not.
        (blend, other) = (
            float(self._read(smooth, [(20.0 - step, 20.0)],
                             interp=('polynomial', 1))[0]),
            float(self._read(smooth, [(20.0 + step, 20.0)],
                             interp=('polynomial', 1))[0]))
        self.assertLess(abs(blend - other), 1e-5)

    def test_the_block_width_comes_from_the_degree(self):
        from euclib.types._interp import _fit_powers, _half_width
        for (degree, d, width) in ((2, 2, 1), (3, 2, 2), (2, 3, 1), (3, 3, 1)):
            with self.subTest(degree=degree, dimensions=d):
                self.assertEqual(_half_width(degree, d), width)
                self.assertGreaterEqual((2 * width + 1) ** d,
                                        len(_fit_powers(degree, d)))

    def test_a_field_of_higher_degree_is_not_reproduced(self):
        cubic = lambda x, y: 0.3 * x ** 3 - 0.2 * x ** 2 * y + 0.1 * y ** 3
        points = [(20.0, 20.0), (20.5, 20.3)]
        got = self._read(cubic, points, interp=('polynomial', 2))
        want = array([cubic(*p) for p in points])
        self.assertGreater(np.abs(got - want).max(), 1e-3)


class TestSegmentCatmullRom(TestCase):
    '''The cardinal cubic along a path, whose slopes come from the neighbours.'''

    COUNT = 12

    def _path(self, field, /, **kw):
        count = self.COUNT
        path = SegPath(stack([arange(count, dtype=float), zeros(count)]),
                       SegTopology([list(range(count - 1)),
                                    list(range(1, count))]))
        return path.withprop('f', array([[field(t) for t in range(count)]]), **kw)

    def _read(self, field, positions, /, **kw):
        carried = self._path(field)
        out = []
        for t in positions:
            i = int(floor(t))
            at = carried.topo.Loc(index=array([i]),
                                  weight=array([[1.0 - (t - i)]]))
            out.append(float(ravel(asarray(carried.prop('f', at=at, **kw)))[0]))
        return array(out)

    def _probe(self):
        return linspace(2.0, self.COUNT - 3.0, 40)

    def test_it_reproduces_the_affine_and_the_quadratic(self):
        # The central difference is exact for a quadratic and for everything
        # below it, which is what the rule is for.
        probe = self._probe()
        for (name, field) in (('affine', lambda t: 0.4 * t + 1.1),
                              ('quadratic', lambda t: -0.7 * t ** 2 + 0.4 * t + 1.1)):
            with self.subTest(field=name):
                got = self._read(field, probe, interp=('catmull-rom', 3))
                self.assertLess(np.abs(got - field(probe)).max(), 1e-12)

    def test_it_does_not_reproduce_a_cubic(self):
        # Where the estimated slopes do, which is the trade between the local
        # rule and the global fit.
        cubic = lambda t: 0.3 * t ** 3 - 0.7 * t ** 2 + 0.4 * t + 1.1
        probe = self._probe()
        got = self._read(cubic, probe, interp=('catmull-rom', 3))
        self.assertGreater(np.abs(got - cubic(probe)).max(), 1e-3)
        fitted = self._read(cubic, probe, interp=('bezier', 3))
        self.assertLess(np.abs(fitted - cubic(probe)).max(), 1e-10)

    def test_it_ignores_a_supplied_gradient(self):
        # The rule is a function of the values, as the grid method is; giving
        # the curve a gradient as well would make it neither rule nor fit.
        cubic = lambda t: 0.3 * t ** 3 - 0.7 * t ** 2 + 0.4 * t + 1.1
        probe = self._probe()
        supplied = stack([array([3.0, 0.0]) for _ in range(self.COUNT)],
                         axis=-1)[None]
        without = self._read(cubic, probe, interp=('catmull-rom', 3))
        with_gradient = self._read(cubic, probe, interp=('catmull-rom', 3),
                                   gradient=supplied)
        self.assertTrue(allclose(without, with_gradient))

    def test_it_is_the_grid_kernel_on_an_evenly_spaced_path(self):
        # The claim the method's documentation page makes: the cardinal cubic
        # and the grid's cubic convolution are one method in two settings.
        rng = default_rng(2)
        values = rng.normal(size=self.COUNT)
        probe = self._probe()
        by_path = self._read(lambda t: values[int(t)], probe,
                             interp=('catmull-rom', 3))
        grid = Grid(eye(2), GridTopology((self.COUNT,))).withprop(
            'f', values)
        at = grid.topo.Loc(sx=probe)
        by_grid = ravel(asarray(grid.prop('f', at=at,
                                          interp=('catmull-rom', 3))))
        self.assertLess(np.abs(by_path - by_grid).max(), 1e-12)

    def test_a_segment_offers_it_at_order_three_and_no_other(self):
        from euclib.abc import supported_interp
        path = self._path(lambda t: t)
        support = supported_interp(path.topo)
        self.assertIn(('catmull-rom', 3), support)
        for order in (0, 1, 2):
            with self.subTest(order=order):
                self.assertNotIn(('catmull-rom', order), support)


class TestATensorValuedProperty(TestCase):
    '''A property whose values are a torch tensor that requires a gradient.

    The reason the interpolation path was rewritten against ``immlib.math``:
    the values, the positions and a geometry's coordinates may each be a tensor,
    and if the weights are built with the inputs' arithmetic then the gradient
    survives. This checks the first of the three -- the values -- for the methods
    that convolve a kernel, which is where the translation has reached.

    The splines are included, and they are the interesting case: theirs is the one
    method that transforms the *data* before convolving it, so their prefilter is
    part of the differentiable path rather than beside it. The prefilter's two
    recursions are Python loops over the array, which torch tracks like any other
    in-place update, so they needed no rewriting --- only an allocation that
    follows the values' backend rather than numpy's.
    '''

    def _torch(self):
        try:
            import torch
        except ImportError:                                  # pragma: no cover
            self.skipTest("torch is not installed")
        return torch

    def test_a_tensor_valued_grid_interpolates_and_carries_its_gradient(self):
        torch = self._torch()
        count = 12
        values = torch.linspace(0.0, 3.0, count).requires_grad_(True)
        grid = Grid(eye(2), GridTopology((count,))).withprop('v', values)
        at = grid.topo.Loc(sx=array([2.5]))
        # Every grid method, the two fits included: they take a different path
        # from the kernels, through a least-squares solve rather than a
        # convolution, and the first version of this test listed only the
        # kernels and so missed them.
        for method in (('nearest', 0), ('polynomial', 1), ('catmull-rom', 3),
                       ('spline', 2), ('spline', 3), ('lanczos', 2),
                       ('lanczos', 3), ('polynomial', 2), ('polynomial', 3)):
            with self.subTest(method=method):
                got = grid.prop('v', at=at, interp=method)
                got = got.m if hasattr(got, 'm') else got
                self.assertIsInstance(got, torch.Tensor)
                self.assertTrue(got.requires_grad)
                got.sum().backward()
                self.assertIsNotNone(values.grad)
                values.grad = None

    def test_the_answer_is_the_same_as_for_a_numpy_property(self):
        # The backend must not change the answer, only who can differentiate it.
        torch = self._torch()
        count = 12
        plain = sin(arange(count, dtype=float) / 3.0)
        tensor = torch.tensor(plain, requires_grad=True)
        at = Grid(eye(2), GridTopology((count,))).topo.Loc(
            sx=linspace(1.0, count - 2.0, 25))
        by_numpy = ravel(asarray(Grid(eye(2), GridTopology((count,))).withprop(
            'v', plain).prop('v', at=at, interp=('catmull-rom', 3))))
        got = Grid(eye(2), GridTopology((count,))).withprop(
            'v', tensor).prop('v', at=at, interp=('catmull-rom', 3))
        got = got.m if hasattr(got, 'm') else got
        self.assertLess(np.abs(np.asarray(got.detach()) - by_numpy).max(), 1e-12)


class TestATensorPosition(TestCase):
    '''A query position given as a torch tensor that requires a gradient.

    The second of the three cases: the *positions* a property is read at may
    require a gradient, and the weights an interpolation builds are functions of
    them --- so `∂I/∂x` is the derivative of the weights, and it comes from
    building them with the positions' arithmetic rather than from any analytic
    formula.

    A position reaches the interpolation through two places that are *not*
    differentiable and must be detached: which cell or simplex it falls in, and
    which stencil a kernel's weights are taken over. Those are choices, and a
    gradient with respect to them would be meaningless; the arithmetic between
    them is what carries one.
    '''

    COUNT = 20

    def _torch(self):
        try:
            import torch
        except ImportError:                                  # pragma: no cover
            self.skipTest("torch is not installed")
        return torch

    def _grid(self, values):
        return Grid(eye(2), GridTopology((self.COUNT,))).withprop('v', values)

    def test_a_tensor_position_carries_its_gradient_through_every_kernel(self):
        # The gradient is the derivative of the *weights* with respect to the
        # position, and for every method here the weights are smooth in it, so
        # the answer is the interpolant's own slope. That is near the field's
        # slope and not equal to it -- an interpolant is not the field -- so the
        # check is against a finite difference of the same interpolation, taken
        # inside a cell: order 1 kinks at a cell boundary, and a centred
        # difference straddling a kink is the derivative on neither side.
        torch = self._torch()
        values = sin(arange(self.COUNT, dtype=float) / 3.0)
        # The fits included: their weights come from a design matrix that is a
        # function of the position, so the gradient reaches the position through
        # the solve rather than through a kernel's distance.
        for method in (('polynomial', 1), ('catmull-rom', 3), ('spline', 2),
                       ('spline', 3), ('lanczos', 2), ('lanczos', 3),
                       ('polynomial', 2), ('polynomial', 3)):
            with self.subTest(method=method):
                at = torch.tensor([5.3], requires_grad=True)
                got = self._grid(values).prop(
                    'v', at=self._grid(values).topo.Loc(sx=at), interp=method)
                got = got.m if hasattr(got, 'm') else got
                self.assertTrue(got.requires_grad)
                got.sum().backward()
                self.assertIsNotNone(at.grad)

                def summed(shift, /):
                    moved = at.detach() + shift
                    out = self._grid(values).prop(
                        'v', at=self._grid(values).topo.Loc(sx=moved),
                        interp=method)
                    out = out.m if hasattr(out, 'm') else out
                    return float(ravel(asarray(out)).sum())

                slope = float(ravel(asarray(at.grad))[0])
                for step in (1e-3, 1e-2):
                    want = (summed(step) - summed(-step)) / (2.0 * step)
                    self.assertAlmostEqual(slope, want, places=3)
                # Not the *field's* slope: an interpolant is not the field, and
                # the difference is the approximation error the method's order
                # describes. Comparing against it here would have tested the
                # data's smoothness rather than the gradient.

    def test_the_value_is_the_same_as_for_a_plain_array_position(self):
        torch = self._torch()
        values = sin(arange(self.COUNT, dtype=float) / 3.0)
        probe = linspace(1.0, self.COUNT - 2.0, 30)
        by_numpy = ravel(asarray(self._grid(values).prop(
            'v', at=self._grid(values).topo.Loc(sx=probe),
            interp=('catmull-rom', 3))))
        got = self._grid(values).prop(
            'v', at=self._grid(values).topo.Loc(sx=torch.tensor(probe)),
            interp=('catmull-rom', 3))
        got = got.m if hasattr(got, 'm') else got
        self.assertLess(np.abs(np.asarray(got.detach()) - by_numpy).max(), 1e-12)

    def test_a_tensor_value_and_a_tensor_position_together(self):
        # Both at once, which is what a resampling step that learns both is.
        torch = self._torch()
        values = torch.tensor(sin(arange(self.COUNT, dtype=float) / 3.0),
                              requires_grad=True)
        at = torch.tensor([5.5], requires_grad=True)
        got = self._grid(values).prop(
            'v', at=self._grid(values).topo.Loc(sx=at),
            interp=('catmull-rom', 3))
        got = got.m if hasattr(got, 'm') else got
        got.sum().backward()
        self.assertIsNotNone(values.grad)
        self.assertIsNotNone(at.grad)

    def test_nearest_detaches_because_a_cell_is_a_choice(self):
        # Order 0 reads the nearest cell and does no arithmetic, so there is
        # nothing for a gradient to flow through and the answer is a plain
        # array. That is the design rather than an omission: which cell a
        # position falls in is discrete.
        torch = self._torch()
        values = sin(arange(self.COUNT, dtype=float) / 3.0)
        at = torch.tensor([5.5], requires_grad=True)
        got = self._grid(values).prop(
            'v', at=self._grid(values).topo.Loc(sx=at), interp=('nearest', 0))
        got = got.m if hasattr(got, 'm') else got
        self.assertFalse(isinstance(got, torch.Tensor) and got.requires_grad)


class TestASimplexTensorProperty(TestCase):
    '''A tensor-valued property on a simplex element, as far as the pass has got.

    The grid is done; the simplex elements are being translated. What works here
    is the entry, which hands the values back rather than converting them, and
    the linear combination of a position's corner weights --- orders 0 and 1.
    The fits above linear are the next piece, and their being listed separately
    here is the report of where the boundary currently is.
    '''

    def _torch(self):
        try:
            import torch
        except ImportError:                                  # pragma: no cover
            self.skipTest("torch is not installed")
        return torch

    def _path(self, count=5):
        return SegPath(stack([arange(count, dtype=float), zeros(count)]),
                       SegTopology([list(range(count - 1)),
                                    list(range(1, count))]))

    def test_a_tensor_property_carries_its_gradient_at_orders_zero_and_one(self):
        torch = self._torch()
        count = 5
        at = self._path(count).topo.Loc(index=array([1]),
                                        weight=array([[0.4]]))
        for method in (('nearest', 0), ('polynomial', 1), ('bezier', 1)):
            with self.subTest(method=method):
                values = torch.tensor([1., 2., 3., 4., 5.],
                                      requires_grad=True)
                got = self._path(count).withprop('v', values).prop(
                    'v', at=at, interp=method)
                got = got.m if hasattr(got, 'm') else got
                self.assertTrue(got.requires_grad)
                got.sum().backward()
                self.assertIsNotNone(values.grad)

    def test_the_bezier_orders_carry_it_too(self):
        # The Bezier fits go through `segment_fit`, and the gradient they need
        # comes from `estimate_gradient`, which applies a sparse operator to the
        # values. That operator was the one piece of this pass needing a new
        # backend rather than a translation: a SciPy matrix times a tensor
        # converts the tensor, so the operator is now carried into the values'
        # backend and applied there.
        torch = self._torch()
        count = 6
        at = self._path(count).topo.Loc(index=array([1, 3]),
                                        weight=array([[0.4, 0.7]]))
        for method in (('bezier', 2), ('bezier', 3), ('catmull-rom', 3),
                       ('polynomial', 2), ('polynomial', 3)):
            with self.subTest(method=method):
                values = torch.tensor(sin(arange(count, dtype=float)),
                                      requires_grad=True)
                got = self._path(count).withprop('v', values).prop(
                    'v', at=at, interp=method)
                got = got.m if hasattr(got, 'm') else got
                self.assertTrue(got.requires_grad)
                got.sum().backward()
                self.assertIsNotNone(values.grad)

    def test_every_simplex_method_carries_it(self):
        # Every method of every element above a segment. The fits assign the
        # values into a control array, so that array is built in the backend of
        # the values and the weights promoted together; the operator a
        # Clough-Tocher edge datum comes from is applied in that backend too; and
        # each assignment of a computed value takes the magnitude first, since
        # immlib hands back a quantity and a tensor will not hold one.
        #
        # Clough-Tocher and Powell-Sabin have no tetrahedral form --- the first
        # splits a triangle into three and the second into six --- so a
        # tetrahedron reports them as unimplemented, which is the contract rather
        # than a gap here.
        torch = self._torch()
        elements = (
            ('triangle', TriMesh(array([[0., 1., 0.], [0., 0., 1.]]),
                                 TriTopology(array([[0], [1], [2]]),
                                             coord_count=3)),
             (('nearest', 0), ('polynomial', 1), ('polynomial', 2),
              ('polynomial', 3), ('bezier', 2), ('bezier', 3),
              ('clough-tocher', 3), ('powell-sabin', 2)),
             zeros((2, 1))),
            ('tetrahedron', TetMesh(
                array([[0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]),
                TetTopology(array([[0], [1], [2], [3]]), coord_count=4)),
             (('nearest', 0), ('polynomial', 2), ('polynomial', 3),
              ('bezier', 2), ('bezier', 3)),
             zeros((3, 1))))
        for (label, geom, methods, weights) in elements:
            count = geom.topo.coord_count
            for method in methods:
                with self.subTest(element=label, method=method):
                    values = torch.arange(1.0, count + 1.0,
                                          requires_grad=True)
                    loc = geom.topo.Loc(index=array([0]), weight=weights)
                    got = geom.withprop('v', values).prop(
                        'v', at=loc, interp=method)
                    got = got.m if hasattr(got, 'm') else got
                    got.sum().backward()
                    self.assertIsNotNone(values.grad)

    def test_a_supplied_tensor_gradient_carries_its_own_graph(self):
        # The third case, and the last of the three: a property's derivative may
        # be a tensor of its own. It is passed to the fits as it stands rather
        # than converted, so it reaches their arithmetic --- and a gradient that
        # a caller computed from something carries the graph of whatever that
        # was, which a conversion to numpy would cut.
        torch = self._torch()
        tri = TriMesh(array([[0., 1., 0.], [0., 0., 1.]]),
                      TriTopology(array([[0], [1], [2]]), coord_count=3))
        values = torch.tensor([1., 2., 3.], requires_grad=True)
        supplied = torch.tensor([[1., 1., 1.], [2., 2., 2.]],
                                requires_grad=True)
        carried = tri.withprop('v', values, gradient=supplied)
        loc = tri.topo.Loc(index=array([0]), weight=zeros((2, 1)))
        for order in (2, 3):
            with self.subTest(order=order):
                got = carried.prop('v', at=loc, interp=('bezier', order))
                got = got.m if hasattr(got, 'm') else got
                got.sum().backward()
                self.assertIsNotNone(values.grad)
                self.assertIsNotNone(supplied.grad)

    def test_a_masked_read_of_a_tensor_still_carries_a_graph(self):
        # The mask path reached the result through numpy's `copy`, which a tensor
        # does not have. It is copied through immlib now --- and copied rather
        # than viewed, because the array being marked belongs to the fit and must
        # not be mutated under it.
        torch = self._torch()
        tri = TriMesh(array([[0., 1., 0.], [0., 0., 1.]]),
                      TriTopology(array([[0], [1], [2]]), coord_count=3))
        values = torch.tensor([1., 2., 3.], requires_grad=True)
        mask = array([[False], [True], [False]])
        loc = tri.topo.Loc(index=array([0]), weight=zeros((2, 1)))
        got = tri.withprop('v', values, mask=mask).prop(
            'v', at=loc, interp=('bezier', 3), null=float('nan'))
        got = got.m if hasattr(got, 'm') else got
        self.assertTrue(isnan(float(ravel(asarray(got.detach()))[0])))
        got.sum().backward()
        self.assertIsNotNone(values.grad)

    def test_the_answer_matches_the_numpy_path(self):
        torch = self._torch()
        count = 5
        plain = sin(arange(count, dtype=float))
        at = self._path(count).topo.Loc(index=array([1, 2, 3]),
                                        weight=array([[0.4, 0.1, 0.7]]))
        by_numpy = ravel(asarray(self._path(count).withprop('v', plain).prop(
            'v', at=at, interp=('polynomial', 1))))
        got = self._path(count).withprop(
            'v', torch.tensor(plain)).prop('v', at=at,
                                           interp=('polynomial', 1))
        got = got.m if hasattr(got, 'm') else got
        self.assertLess(np.abs(np.asarray(got.detach()) - by_numpy).max(), 1e-12)


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


class TestTheBlockDesign(TestCase):
    '''The design matrix a block of stencils is fitted through, held to itself.

    `_block_design` writes the polynomial in each stencil's own frame, and its
    docstring gives the expression: the monomials are raised over the block at
    once, with the block axis beside the corner axis rather than among the
    powers. It says so because the shape is easy to get wrong --- an extra axis
    in the wrong place gives a design of the wrong width, and the widths *are*
    the monomials.

    **This test exists because nothing caught it.** Translating the function
    through immlib, I put two permutations in that the expression does not need,
    which left the design `(B, M, room)` where it must be `(B, M, W)` --- and all
    727 tests passed, the estimate's tests against an affine and a quadratic
    field among them. Stating the fault is not the same as explaining it: I have
    not worked out which route through `_gradient_blocks` left those meshes
    answering correctly, and the honest report is that a wrong design survived a
    full suite. What this holds is the expression the docstring gives, which is
    the thing that was wrong.
    '''

    def test_it_is_the_expression_its_docstring_gives(self):
        from euclib.types._interp import _block_design
        rng = default_rng(41)
        for (bounds, room, degree) in (((4, 5, 3), 2, 2), ((4, 5, 3), 3, 2),
                                       ((3, 6, 3), 1, 3), ((4, 5, 3), 2, 1)):
            (b, m, d) = bounds
            steps = rng.normal(size=(b, m, d))
            frames = np.stack([np.linalg.qr(rng.normal(size=(d, d)))[0]
                               for _ in range(b)])
            (basis, _, design) = _block_design(steps, frames, room, degree)
            exponents = np.asarray(basis)
            local = steps @ frames[:, :room, :].transpose(0, 2, 1)
            want = (local[:, :, None, :]
                    ** exponents[None, None, :, :]).prod(axis=-1)
            with self.subTest(room=room, degree=degree):
                self.assertEqual(np.asarray(design).shape, want.shape,
                                 "the design is not one column per monomial")
                self.assertLess(np.abs(np.asarray(design) - want).max(), 1e-12)


class TestTheEstimateAgainstAField(TestCase):
    '''The gradient estimate, against a field rather than against its operator.

    `_gradient_operator` and `_estimate_gradient` are two consumers of one
    description of the stencils --- the docstring for `_gradient_blocks` says so
    --- which means they agree with each other whether or not either is right.
    So this holds the estimate to something outside both: an affine field's
    slope is its own gradient, exactly, wherever the stencil is wide enough to
    see it.
    '''

    def _mesh(self, side, /):
        (ix, iy) = np.meshgrid(np.arange(side, dtype=float),
                               np.arange(side, dtype=float), indexing='ij')
        coords = np.vstack([ix.ravel(), iy.ravel()])
        quads = []
        for i in range(side - 1):
            for j in range(side - 1):
                k = i * side + j
                quads.append((k, k + 1, k + side))
                quads.append((k + 1, k + side + 1, k + side))
        return TriMesh(coords, TriTopology(np.array(quads).T))

    def test_it_gives_an_affine_field_s_slope(self):
        from euclib.types._interp import estimate_gradient
        mesh = self._mesh(6)
        coords = np.asarray(mesh.coords)
        slope = np.array([2.0, -1.5])
        field = (slope[0] * coords[0] + slope[1] * coords[1] + 3.0)[None]
        geom = mesh.withprop('v', field)
        prop = geom._prop_for('v', None)
        for order in (2, 3):
            got = np.asarray(estimate_gradient(geom, prop, order))
            want = slope[:, None] * np.ones((1, coords.shape[1]))
            with self.subTest(order=order):
                self.assertLess(np.abs(got - want).max(), 1e-9)
