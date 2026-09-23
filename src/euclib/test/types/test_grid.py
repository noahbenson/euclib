# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/test_grid.py
'''Tests for grids: ``GridTopology`` and ``Grid``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import allclose, arange, array, eye, eye as _eye, zeros

from euclib.abc import is_geometry, is_topology
from euclib.types import (
    Grid, GridTopology, GridLoc1, GridLoc2, GridLoc3,
    affine_translation, affine_scaling)


# Fixtures ###################################################################

def _grid(shape=(4, 3)):
    '''A grid whose affine places the first cell's centre at (0, 0).

    The y axis is flipped, so the grid runs upward in index space and downward
    in global coordinates.
    '''
    affine = array([[1., 0., 0.],
                    [0., -1., 3.],
                    [0., 0., 1.]])
    return Grid(affine, GridTopology(shape))


# Tests ######################################################################

class TestGridTopology(TestCase):
    '''Tests for the grid topology.'''

    def test_extent_and_dimensions(self):
        topo = GridTopology((4, 3))
        self.assertTrue(is_topology(topo))
        self.assertEqual(topo.shape, (4, 3))
        self.assertEqual(topo.dim, 2)
        self.assertEqual(topo.local_dim, 2)
        self.assertEqual(topo.coord_count, 12)

    def test_a_three_dimensional_grid(self):
        topo = GridTopology((10, 12, 15))
        self.assertEqual(topo.dim, 3)
        self.assertEqual(topo.coord_count, 1800)

    def test_the_local_coordinate_type_follows_the_dimensions(self):
        self.assertIs(GridTopology((2, 2)).Loc, GridLoc2)
        self.assertIs(GridTopology((2, 2, 2)).Loc, GridLoc3)
        self.assertIs(GridTopology((2,)).Loc, GridLoc1)
        self.assertEqual(GridLoc3._fields, ('sx', 'sy', 'sz'))

    def test_the_extent_is_validated(self):
        for bad in ((), (1, 2, 3, 4), (0, 4), (4, -1)):
            with self.assertRaises(Exception):
                GridTopology(bad)

    def test_check_loc_accepts_mappings_and_sequences(self):
        topo = GridTopology((4, 3))
        by_map = topo.check_loc({'sx': array([0.5]), 'sy': array([1.5])})
        by_seq = topo.check_loc((array([0.5]), array([1.5])))
        self.assertTrue(allclose(by_map.sx, [0.5]))
        self.assertTrue(allclose(by_seq.sy, [1.5]))

    def test_check_loc_rejects_mismatched_components(self):
        topo = GridTopology((4, 3))
        with self.assertRaises(ValueError):
            topo.check_loc({'sx': array([0.5, 1.5]), 'sy': array([1.5])})


class TestGrid(TestCase):
    '''Tests for the grid geometry.'''

    def test_basics(self):
        g = _grid()
        self.assertTrue(is_geometry(g))
        self.assertEqual(g.dim, 2)
        self.assertEqual(g.shape, (4, 3))
        self.assertEqual(g.coord_count, 12)
        self.assertEqual(g.property_shape, (4, 3))

    def test_origin_and_spacing(self):
        g = _grid()
        # The affine carries an index to a cell's *centre*, so the grid's
        # corner --- half a step back along each index axis --- is a half step
        # away from the first centre. The y axis points down, so the corner is
        # above the first centre rather than below it.
        self.assertTrue(allclose(g.origin, [-0.5, 3.5]))
        self.assertTrue(allclose(g.spacing, [1., 1.]))

    def test_an_index_names_a_cell_centre(self):
        # The convention the whole library reads a grid through, stated as a
        # test: a cell's index is where its data lives, and the cell occupies
        # half a step on either side of that position.
        g = _grid()
        self.assertTrue(allclose(g.affine.apply(array([[0.], [0.]])),
                                 [[0.], [3.]]))
        self.assertTrue(allclose(g.to_local(array([[0.], [3.]])).sx, [0.]))
        # Half a step past the last centre is still within the grid; a step
        # past it is not.
        self.assertTrue(allclose(g.bbox,
                                 [[-0.5, 3.5], [0.5, 3.5]]))

    def test_a_grid_stores_an_affine_matrix_as_its_coords(self):
        g = _grid()
        self.assertEqual(tuple(g.coords.shape), (3, 3))
        # ...and it must be affine.
        with self.assertRaises(Exception):
            Grid(eye(3) * 2., GridTopology((4, 3)))

    def test_the_affine_must_match_the_dimensions(self):
        with self.assertRaises(Exception):
            Grid(eye(3), GridTopology((4, 3, 5)))
        with self.assertRaises(Exception):
            Grid(eye(4), GridTopology((4, 3)))

    def test_to_local_and_to_global_round_trip(self):
        g = _grid()
        query = array([[0., 2.], [0., 0.]])
        loc = g.to_local(query)
        self.assertTrue(allclose(loc.sx, [0., 2.]))
        self.assertTrue(allclose(loc.sy, [3., 3.]))
        self.assertTrue(allclose(g.to_global(loc), query))

    def test_to_global_at_arbitrary_index_positions(self):
        g = _grid()
        loc = g.topo.Loc(sx=array([0., 1., 2.]), sy=array([0., 0.5, 1.]))
        self.assertTrue(allclose(g.to_global(loc),
                                 [[0., 1., 2.], [3., 2.5, 2.]]))

    def test_bbox_covers_the_cells(self):
        # The box encloses the cells, not the centres: half a step before the
        # first centre along each axis to half a step past the last.
        g = _grid()
        self.assertTrue(allclose(g.bbox, [[-0.5, 3.5], [0.5, 3.5]]))

    def test_transforming_a_grid_composes_onto_its_affine(self):
        g = _grid()
        moved = g.transformed(affine_translation([10., 0.]))
        self.assertEqual(moved.shape, g.shape)
        self.assertTrue(allclose(moved.origin, [9.5, 3.5]))
        self.assertTrue(allclose(moved.bbox, [[9.5, 13.5], [0.5, 3.5]]))
        # The original is unchanged.
        self.assertTrue(allclose(g.origin, [-0.5, 3.5]))

    def test_scaling_a_grid_changes_its_spacing(self):
        scaled = _grid().transformed(affine_scaling([2., 3.]))
        self.assertTrue(allclose(scaled.spacing, [2., 3.]))

    def test_a_grid_property_has_the_grid_shape(self):
        g = _grid()
        p = g.withprop('v', zeros((4, 3)))
        self.assertEqual(p['v'].shape, (4, 3))
        with self.assertRaises(Exception):
            g.withprop('bad', zeros(12))

    def test_the_readme_jacobian_example(self):
        # A 3-D grid whose every voxel holds a 3x2 Jacobian.
        g = Grid(eye(4), GridTopology((10, 12, 15)))
        jac = zeros((3, 2, 10, 12, 15))
        for i in range(3):
            for j in range(2):
                jac[i, j] = 100 * i + j
        g2 = g.withprop('jacobian', jac)
        self.assertEqual(g2['jacobian'].shape, (3, 2, 10, 12, 15))
        res = g2['jacobian', 0, 0, 0]
        ((a, b), (c, d), (e, f)) = res
        self.assertEqual((a, b), (0., 1.))
        self.assertEqual((c, d), (100., 101.))
        self.assertEqual((e, f), (200., 201.))

    def test_a_grid_has_no_simplex_properties(self):
        with self.assertRaises(IndexError):
            _grid()[0, 'anything']
