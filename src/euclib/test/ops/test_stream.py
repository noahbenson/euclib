# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/test_stream.py
'''Tests for vector-field integration in ``_stream``.

A streamline is checked against fields whose curves can be written down: a
constant field gives a straight line whose length is the step count times the
step, a rotation gives a circle whose radius does not change, and a field that
vanishes stops the walk where it vanishes. The seeding, the direction, the step
cap, and the refusals are checked too, because those are the parts a caller
controls and the parts a wrong answer would be hardest to see in a drawing.
'''

# Dependencies ###############################################################

from __future__ import annotations

import numpy as np
from numpy import allclose, array, asarray, eye, linspace, meshgrid, zeros
import unittest
from unittest import TestCase

from immlib import math as imath

from euclib.ops import streamlines
from euclib.types import Grid, SegPath, grid


# Fixtures ###################################################################

def _grid(shape=(3, 3), spacing=1.0, origin=(0.0, 0.0)):
    '''An axis-aligned grid with a given cell size and first cell center.'''
    affine = eye(len(shape) + 1)
    for axis in range(len(shape)):
        affine[axis, axis] = spacing
        affine[axis, -1] = origin[axis]
    return grid(shape, affine=affine)


def _constant(shape=(3, 3), vector=(1.0, 0.0), **kw):
    '''A grid carrying a constant velocity field.'''
    values = zeros((len(vector),) + tuple(shape))
    values[:] = asarray(vector)[:, None, None]
    return _grid(shape, **kw).withprop('v', values)


def _rotation(count=21, /):
    '''A grid on ``[-2, 2]^2`` carrying the field ``(-y, x)``.'''
    coordinates = linspace(-2.0, 2.0, count)
    (xs, ys) = meshgrid(coordinates, coordinates, indexing='ij')
    base = _grid((count, count), spacing=4.0 / (count - 1),
                 origin=(-2.0, -2.0))
    return base.withprop('v', np.stack([-ys, xs], axis=0))


# Tests ######################################################################

class TestTheStreamlines(TestCase):
    '''The curves a velocity field carries a particle along.'''

    def test_a_constant_field_makes_a_straight_line(self):
        flow = _constant()
        path = streamlines(flow, 'v', start=array([0.0, 0.0]), step=0.25,
                           max_steps=100)
        coords = asarray(path.coords)
        self.assertTrue(allclose(coords[1], 0.0))
        self.assertTrue((np.diff(coords[0]) > 0.0).all())
        self.assertAlmostEqual(float(np.sum(path.measures)), 2.5, places=12)

    def test_the_step_is_a_distance_along_the_curve(self):
        flow = _constant()
        path = streamlines(flow, 'v', start=array([0.0, 1.0]), step=0.5,
                           max_steps=3)
        self.assertEqual(path.coord_count, 4)
        self.assertEqual(path.topo.simplex_count[1], 3)
        self.assertAlmostEqual(float(np.sum(path.measures)), 1.5, places=12)

    def test_the_default_step_is_half_the_shortest_cell(self):
        flow = _constant(spacing=2.0)
        path = streamlines(flow, 'v', start=array([0.0, 0.0]), max_steps=1)
        self.assertAlmostEqual(float(np.sum(path.measures)), 1.0, places=12)

    def test_the_direction_can_be_reversed(self):
        flow = _constant()
        path = streamlines(flow, 'v', start=array([2.0, 1.0]), step=0.25,
                           direction=-1, max_steps=100)
        coords = asarray(path.coords)
        self.assertTrue((np.diff(coords[0]) < 0.0).all())
        self.assertAlmostEqual(float(np.sum(path.measures)), 2.5, places=12)

    def test_the_midpoint_rule_agrees_on_a_constant_field(self):
        flow = _constant()
        path = streamlines(flow, 'v', start=array([0.0, 0.0]), step=0.25,
                           order=2, max_steps=4)
        self.assertAlmostEqual(float(np.sum(path.measures)), 1.0, places=12)
        self.assertTrue(allclose(asarray(path.coords)[1], 0.0))

    def test_a_rotational_field_keeps_its_radius(self):
        flow = _rotation()
        path = streamlines(flow, 'v', start=array([1.0, 0.0]), step=0.1,
                           max_steps=100)
        coords = asarray(path.coords)
        radius = np.sqrt((coords * coords).sum(axis=0))
        self.assertTrue(allclose(radius, 1.0, atol=1e-6))
        self.assertEqual(path.coord_count, 101)

    def test_the_streamline_stays_inside_the_grid(self):
        flow = _constant()
        path = streamlines(flow, 'v', start=array([0.0, 0.0]), step=0.25,
                           max_steps=100)
        coords = asarray(path.coords)
        self.assertLessEqual(coords[0].max(), 2.5 + 1e-12)

    def test_several_seeds_make_several_streamlines(self):
        flow = _constant()
        seeds = array([[0.0, 0.0, 0.0], [0.0, 1.0, 2.0]])
        path = streamlines(flow, 'v', start=seeds, step=0.25, max_steps=100)
        indices = asarray(path.topo.indices)
        self.assertEqual(path.coord_count, 33)
        self.assertEqual(indices.shape[1], 30)
        # Each seed's run is connected and does not touch the next one's.
        for (low, high, base) in ((0, 10, 0), (10, 20, 11), (20, 30, 22)):
            self.assertEqual(set(indices[:, low:high].reshape(-1)),
                             set(range(base, base + 11)))

    def test_a_zero_field_makes_no_segments(self):
        flow = _constant(vector=(0.0, 0.0))
        path = streamlines(flow, 'v', start=array([1.0, 1.0]))
        self.assertEqual(path.coord_count, 0)
        self.assertEqual(path.topo.simplex_count[1], 0)

    def test_a_field_that_vanishes_stops_the_walk(self):
        values = zeros((2, 3, 3))
        values[0, 0, :] = 1.0
        flow = _grid().withprop('v', values)
        path = streamlines(flow, 'v', start=array([0.0, 0.0]), step=0.25,
                           max_steps=100)
        self.assertGreater(path.topo.simplex_count[1], 0)
        self.assertLess(float(np.sum(path.measures)), 2.0)

    def test_max_steps_is_a_cap(self):
        flow = _constant()
        path = streamlines(flow, 'v', start=array([0.0, 0.0]), step=0.25,
                           max_steps=7)
        self.assertEqual(path.topo.simplex_count[1], 7)

    def test_the_default_start_is_the_center(self):
        flow = _constant()
        path = streamlines(flow, 'v', step=0.25, max_steps=100)
        self.assertTrue(allclose(asarray(path.coords)[:, 0], [1.0, 1.0]))
        self.assertAlmostEqual(float(np.sum(path.measures)), 1.5, places=12)

    def test_a_seed_on_the_edge_is_inside(self):
        flow = _constant()
        path = streamlines(flow, 'v', start=array([-0.5, 1.0]), step=0.25,
                           max_steps=100)
        self.assertGreater(path.topo.simplex_count[1], 0)

    def test_a_result_is_a_path(self):
        path = streamlines(_constant(), 'v', start=array([0.0, 0.0]))
        self.assertIsInstance(path, SegPath)
        self.assertEqual(path.dim, 2)

    def test_the_grid_is_not_changed(self):
        flow = _constant()
        before = asarray(imath.mag(flow.coords)).copy()
        streamlines(flow, 'v', start=array([0.0, 0.0]), step=0.25,
                    max_steps=3)
        self.assertTrue(np.array_equal(asarray(imath.mag(flow.coords)), before))


class TestTheStreamlineInputs(TestCase):
    '''What the operation accepts and what it refuses.'''

    def test_a_quantity_keeps_its_units(self):
        import pint
        ureg = pint.UnitRegistry()
        flow = _grid()
        values = zeros((2, 3, 3))
        values[0] = 1.0
        metric = Grid(asarray(imath.mag(flow.coords)) * ureg.meter,
                      flow.topo)
        metric = metric.withprop('v', values)
        path = streamlines(metric, 'v', start=array([0.0, 0.0]), step=0.5,
                           max_steps=1)
        self.assertEqual(str(path.coords.units), 'meter')
        self.assertAlmostEqual(float(imath.mag(np.sum(path.measures))), 0.5,
                               places=12)

    def test_a_tensor_field_becomes_an_array(self):
        try:
            import torch
        except ImportError:                     # pragma: no cover
            self.skipTest("torch is not installed")
        values = torch.zeros((2, 3, 3), dtype=torch.float64)
        values[0] = 1.0
        flow = _grid().withprop('v', values)
        path = streamlines(flow, 'v', start=array([0.0, 0.0]), step=0.25,
                           max_steps=2)
        self.assertIsInstance(path.coords, np.ndarray)
        self.assertAlmostEqual(float(np.sum(path.measures)), 0.5, places=12)

    def test_a_non_grid_is_refused(self):
        with self.assertRaises(TypeError):
            streamlines(42, 'v', start=array([0.0, 0.0]))

    def test_a_one_dimensional_grid_is_refused(self):
        with self.assertRaises(ValueError):
            streamlines(grid((4,)).withprop('v', zeros((1, 4))), 'v',
                        start=array([0.0]))

    def test_a_field_of_the_wrong_channel_shape_is_refused(self):
        flow = _grid().withprop('s', zeros((3, 3)))
        with self.assertRaises(ValueError):
            streamlines(flow, 's', start=array([0.0, 0.0]))

    def test_an_unknown_property_is_refused(self):
        with self.assertRaises(Exception):
            streamlines(_constant(), 'nope', start=array([0.0, 0.0]))

    def test_a_seed_outside_the_grid_is_refused(self):
        with self.assertRaises(ValueError):
            streamlines(_constant(), 'v', start=array([5.0, 0.0]))

    def test_a_bad_start_shape_is_refused(self):
        for start in (zeros((3, 1)), zeros((2, 2, 2)), zeros(3)):
            with self.subTest(start=start.shape):
                with self.assertRaises(ValueError):
                    streamlines(_constant(), 'v', start=start)

    def test_a_nonpositive_step_is_refused(self):
        for step in (0.0, -0.5):
            with self.subTest(step=step):
                with self.assertRaises(ValueError):
                    streamlines(_constant(), 'v', start=array([0.0, 0.0]),
                                step=step)

    def test_a_bad_order_is_refused(self):
        for order in (0, 1, 3, 5):
            with self.subTest(order=order):
                with self.assertRaises(ValueError):
                    streamlines(_constant(), 'v', start=array([0.0, 0.0]),
                                order=order)

    def test_a_bad_direction_is_refused(self):
        for direction in (0, 2, -2):
            with self.subTest(direction=direction):
                with self.assertRaises(ValueError):
                    streamlines(_constant(), 'v', start=array([0.0, 0.0]),
                                direction=direction)

    def test_a_negative_step_cap_is_refused(self):
        with self.assertRaises(ValueError):
            streamlines(_constant(), 'v', start=array([0.0, 0.0]),
                        max_steps=-1)


# Exports ####################################################################

if __name__ == '__main__':                      # pragma: no cover
    unittest.main()
