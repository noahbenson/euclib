# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/test_interp.py
'''Tests for the interpolation engine in ``euclib.types._interp``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import allclose, arange, array, eye, isnan, nan, zeros

from euclib.types import (
    Grid, GridTopology, SegPath, SegTopology, TriMesh, TriTopology, VertexSet,
    VertexTopology)


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

    def test_a_higher_order_raises(self):
        for order in (2, 3):
            with self.assertRaises(NotImplementedError):
                _path().prop('t', at=array([[0.5], [0.]]), interp=order)


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
