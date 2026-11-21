# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/test_raster.py
'''Tests for path rasterization in ``_raster``.

A raster is checked against shapes whose covered area can be written down: a
square of a known size covers a known number of cells, a square inside a square
leaves a hole, and a circle covers an area that approaches pi as the cells get
smaller. The outline is checked against the fill it surrounds, and the refusals
against the arguments that cannot describe a lattice.
'''

# Dependencies ###############################################################

from __future__ import annotations

import numpy as np
from numpy import array, asarray, cos, linspace, pi, sin, zeros
import unittest
from unittest import TestCase

from immlib import math as imath

from euclib.ops import rasterize
from euclib.types import Grid, SegPath, SegTopology, segpath


# Fixtures ###################################################################

def _square(size=4.0, low=(0.0, 0.0), /):
    '''A closed square path, with the first corner repeated as the last.'''
    (x, y) = low
    return segpath(array([[x, x + size, x + size, x, x],
                          [y, y, y + size, y + size, y]]))


def _ring(outer=4.0, inner=1.0, /):
    '''A square with a square hole, as two disjoint closed loops.'''
    (o, i) = (outer, inner)
    coords = array([[0.0, o, o, 0.0, i, o - i, o - i, i],
                    [0.0, 0.0, o, o, i, i, o - i, o - i]])
    segments = array([[0, 1, 2, 3, 4, 5, 6, 7],
                      [1, 2, 3, 0, 5, 6, 7, 4]])
    return SegPath(coords, SegTopology(segments, coord_count=8))


def _circle(radius=1.0, count=200, /):
    '''A closed circle sampled at ``count`` angles.'''
    angle = linspace(0.0, 2.0 * pi, count)
    return segpath(array([radius * cos(angle), radius * sin(angle)]))


def _filled(image, /):
    '''The number of covered cells in a raster grid.'''
    return int(asarray(image['raster']).sum())


# Tests ######################################################################

class TestRasterize(TestCase):
    '''Marking the cells a two-dimensional path covers.'''

    def test_a_square_fills_its_interior(self):
        image = rasterize(_square(), pitch=0.5, origin=array([0.0, 0.0]),
                          resolution=(8, 8))
        self.assertIsInstance(image, Grid)
        self.assertEqual(image.dim, 2)
        self.assertEqual(image.shape, (8, 8))
        self.assertEqual(_filled(image), 64)

    def test_the_cells_map_to_their_centers(self):
        image = rasterize(_square(), pitch=0.5, origin=array([0.0, 0.0]),
                          resolution=(8, 8))
        self.assertTrue(np.allclose(asarray(image.origin), [0.0, 0.0]))
        self.assertTrue(np.allclose(asarray(image.spacing), [0.5, 0.5]))
        self.assertTrue(np.allclose(
            asarray(image.affine.apply(array([[0.0], [0.0]])))[:, 0],
            [0.25, 0.25]))

    def test_a_repeated_closing_corner_still_closes(self):
        # `segpath` makes a chain, so this square's first and last corners are
        # two coordinate indices at one position. The loop test welds them.
        image = rasterize(_square(), pitch=0.5, origin=array([0.0, 0.0]),
                          resolution=(8, 8))
        self.assertEqual(_filled(image), 64)

    def test_a_loop_inside_a_loop_is_a_hole(self):
        image = rasterize(_ring(), pitch=0.5, origin=array([0.0, 0.0]),
                          resolution=(8, 8))
        # 8 by 8 outside, 4 by 4 inside: 64 - 16.
        self.assertEqual(_filled(image), 48)

    def test_the_outline_and_the_fill_add(self):
        def count(**kw):
            return _filled(rasterize(_ring(), pitch=0.5,
                                     origin=array([0.0, 0.0]),
                                     resolution=(8, 8), **kw))
        filled = count(fill=True, width=None)
        outline = count(fill=False, width=1.0)
        both = count(fill=True, width=1.0)
        self.assertGreater(filled, 0)
        self.assertGreater(outline, 0)
        self.assertGreater(both, filled)
        self.assertGreater(both, outline)

    def test_an_open_path_encloses_nothing(self):
        line = segpath(array([[0.0, 1.0], [0.0, 1.0]]))
        image = rasterize(line, pitch=0.1, origin=array([0.0, 0.0]),
                          resolution=(10, 10))
        self.assertEqual(_filled(image), 0)

    def test_an_open_path_can_be_drawn_as_an_outline(self):
        line = segpath(array([[0.0, 1.0], [0.0, 1.0]]))
        image = rasterize(line, pitch=0.1, origin=array([0.0, 0.0]),
                          resolution=(10, 10), fill=False, width=1.0)
        self.assertGreater(_filled(image), 0)

    def test_fill_and_width_can_both_be_off(self):
        image = rasterize(_square(), pitch=0.5, origin=array([0.0, 0.0]),
                          resolution=(8, 8), fill=False, width=None)
        self.assertEqual(_filled(image), 0)

    def test_a_circle_s_area_approaches_pi(self):
        for (pitch, tolerance) in ((0.05, 0.02), (0.01, 0.002)):
            with self.subTest(pitch=pitch):
                image = rasterize(_circle(), pitch=pitch,
                                  origin=array([-1.0, -1.0]),
                                  resolution=(int(2.0 / pitch),
                                              int(2.0 / pitch)))
                area = _filled(image) * pitch * pitch
                self.assertLess(abs(area - pi) / pi, tolerance)

    def test_a_rectangular_cell_is_accepted(self):
        image = rasterize(_square(), pitch=(0.5, 1.0),
                          origin=array([0.0, 0.0]), resolution=(8, 4))
        self.assertEqual(image.shape, (8, 4))
        self.assertTrue(np.allclose(asarray(image.spacing), [0.5, 1.0]))

    def test_the_defaults_cover_the_path(self):
        image = rasterize(_square())
        self.assertGreater(_filled(image), 0)
        bounds = asarray(imath.mag(image.bbox))
        self.assertLessEqual(bounds[0, 0], 0.0)
        self.assertLessEqual(bounds[1, 0], 0.0)
        self.assertGreaterEqual(bounds[0, 1], 4.0)
        self.assertGreaterEqual(bounds[1, 1], 4.0)

    def test_the_property_can_be_named(self):
        image = rasterize(_square(), pitch=1.0, origin=array([0.0, 0.0]),
                          resolution=(4, 4), name='covered')
        self.assertEqual(image['covered'].shape, (4, 4))
        self.assertNotIn('raster', image.properties)

    def test_a_quantity_keeps_its_units(self):
        import pint
        ureg = pint.UnitRegistry()
        square = segpath(array([[0.0, 2.0, 2.0, 0.0, 0.0],
                                [0.0, 0.0, 2.0, 2.0, 0.0]]) * ureg.meter)
        image = rasterize(square, pitch=1.0 * ureg.meter,
                          origin=array([0.0, 0.0]) * ureg.meter,
                          resolution=(2, 2))
        self.assertEqual(str(image.coords.units), 'meter')
        self.assertEqual(_filled(image), 4)

    def test_a_tensor_path_becomes_an_array(self):
        try:
            import torch
        except ImportError:                     # pragma: no cover
            self.skipTest("torch is not installed")
        coords = array([[0.0, 2.0, 2.0, 0.0], [0.0, 0.0, 2.0, 2.0]])
        path = SegPath(torch.tensor(coords),
                       SegTopology(array([[0, 1, 2, 3], [1, 2, 3, 0]]),
                                   coord_count=4))
        image = rasterize(path, pitch=1.0, origin=array([0.0, 0.0]),
                          resolution=(2, 2))
        self.assertIsInstance(image['raster'], np.ndarray)
        self.assertEqual(_filled(image), 4)

    def test_the_path_is_not_changed(self):
        square = _square()
        before = asarray(square.coords).copy()
        rasterize(square, pitch=1.0)
        self.assertTrue(np.array_equal(asarray(square.coords), before))


class TestRasterizeInputs(TestCase):
    '''What a raster refuses.'''

    def test_a_non_path_is_refused(self):
        with self.assertRaises(TypeError):
            rasterize(42)

    def test_a_three_dimensional_path_is_refused(self):
        with self.assertRaises(ValueError):
            rasterize(segpath(zeros((3, 2))))

    def test_an_empty_path_is_refused(self):
        empty = SegPath(zeros((2, 0)),
                        SegTopology(zeros((2, 0), dtype=int), coord_count=0))
        with self.assertRaises(ValueError):
            rasterize(empty)

    def test_a_bad_pitch_is_refused(self):
        for pitch in (0.0, -0.5, array([1.0, 2.0, 3.0])):
            with self.subTest(pitch=pitch):
                with self.assertRaises(ValueError):
                    rasterize(_square(), pitch=pitch)

    def test_a_bad_resolution_is_refused(self):
        for resolution in ((0, 2), (-1, 2), (2, 2, 2)):
            with self.subTest(resolution=resolution):
                with self.assertRaises(ValueError):
                    rasterize(_square(), pitch=1.0, resolution=resolution)

    def test_a_bad_origin_is_refused(self):
        with self.assertRaises(ValueError):
            rasterize(_square(), pitch=1.0, origin=array([0.0, 0.0, 0.0]))

    def test_a_negative_width_is_refused(self):
        with self.assertRaises(ValueError):
            rasterize(_square(), pitch=1.0, width=-1.0)


# Exports ####################################################################

if __name__ == '__main__':                      # pragma: no cover
    unittest.main()
