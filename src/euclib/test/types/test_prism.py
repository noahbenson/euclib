# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/test_prism.py
'''Tests for prism meshes: ``PrismTopology`` and ``PrismMesh``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import allclose, array, stack, zeros

from euclib.abc import is_geometry, is_simplex_topology
from euclib.types import (
    PrismMesh, PrismTopology, PrismLoc, TetMesh, TriMesh)


# Fixtures ###################################################################

def _surfaces():
    '''The unit square, once at z = 0 and once at z = 2.'''
    lower = array([[0., 1., 0., 1.],
                   [0., 0., 1., 1.],
                   [0., 0., 0., 0.]])
    return (lower, lower + array([[0.], [0.], [2.]]))


def _prism():
    '''A sheet of two prisms, each one unit of area and two units thick.'''
    (lower, upper) = _surfaces()
    return PrismMesh(stack([lower, upper]), _topo())


def _topo():
    '''The shared triangle topology: the square split along its diagonal.'''
    return PrismTopology([[0, 0], [1, 3], [3, 2]])


# Tests ######################################################################

class TestPrismTopology(TestCase):
    '''Tests for the prism topology.'''

    def test_it_is_a_triangle_topology(self):
        topo = _topo()
        self.assertTrue(is_simplex_topology(topo))
        self.assertEqual(topo.order, 2)
        self.assertEqual(topo.dim, 2)

    def test_a_local_coordinate_has_three_components(self):
        topo = _topo()
        self.assertEqual(topo.local_dim, 3)
        self.assertEqual(topo.Loc._fields, ('index', 'weight', 'height'))
        self.assertEqual(topo.n_sides, 2)

    def test_it_decomposes_each_prism_into_three_tetrahedra(self):
        topo = _topo()
        self.assertEqual(topo.tetrahedra.shape, (4, 6))

    def test_check_loc_accepts_well_formed_coordinates(self):
        topo = _topo()
        loc = topo.check_loc(PrismLoc(array([0]), array([[0.25], [0.25]]),
                                      array([[0.5]])))
        self.assertEqual(loc.index.tolist(), [0])
        by_map = topo.check_loc({'index': array([0]),
                                 'weight': array([[0.25], [0.25]]),
                                 'height': array([[0.5]])})
        self.assertEqual(by_map.index.tolist(), [0])

    def test_check_loc_rejects_bad_shapes(self):
        topo = _topo()
        # A weight of the wrong depth.
        with self.assertRaises(ValueError):
            topo.check_loc(PrismLoc(array([0]), array([[0.25]]),
                                    array([[0.5]])))
        # A height of the wrong depth.
        with self.assertRaises(ValueError):
            topo.check_loc(PrismLoc(array([0]), array([[0.25], [0.25]]),
                                    array([0.5])))
        # Components that disagree about how many positions there are.
        with self.assertRaises(ValueError):
            topo.check_loc(PrismLoc(array([0, 1]),
                                    array([[0.25], [0.25]]),
                                    array([[0.5]])))


class TestPrismMesh(TestCase):
    '''Tests for the prism mesh.'''

    def test_basics(self):
        pm = _prism()
        self.assertTrue(is_geometry(pm))
        self.assertEqual(pm.dim, 3)
        self.assertEqual(pm.coord_count, 4)
        self.assertEqual(pm.order, 2)
        self.assertEqual(pm.property_shape, (4,))

    def test_the_two_surfaces_are_available_separately(self):
        pm = _prism()
        (lower, upper) = _surfaces()
        self.assertTrue(allclose(pm.coords0, lower))
        self.assertTrue(allclose(pm.coords1, upper))
        self.assertEqual(pm.coords.shape, (2, 3, 4))

    def test_coords_must_be_a_pair_of_matrices(self):
        with self.assertRaises(Exception):
            PrismMesh(zeros((3, 4)), _topo())

    def test_prisms_require_three_dimensions(self):
        with self.assertRaises(Exception):
            PrismMesh(zeros((2, 2, 4)), _topo())

    def test_to_global_blends_the_surfaces_by_elevation(self):
        pm = _prism()
        # A quarter of the way along each of the first two sides of triangle 0,
        # whose corners are the origin, (1, 0), and (1, 1).
        weight = array([[0.25], [0.25]])
        at0 = pm.to_global(pm.topo.Loc(array([0]), weight, array([[0.]])))
        at1 = pm.to_global(pm.topo.Loc(array([0]), weight, array([[1.]])))
        at_half = pm.to_global(pm.topo.Loc(array([0]), weight, array([[0.5]])))
        self.assertTrue(allclose(at0.ravel(), [0.75, 0.5, 0.0]))
        self.assertTrue(allclose(at1.ravel(), [0.75, 0.5, 2.0]))
        self.assertTrue(allclose(at_half.ravel(), [0.75, 0.5, 1.0]))

    def test_elevation_returns_a_triangle_mesh(self):
        pm = _prism()
        (lower, upper) = _surfaces()
        self.assertIsInstance(pm.elevation(0.), TriMesh)
        self.assertTrue(allclose(pm.elevation(0.).coords, lower))
        self.assertTrue(allclose(pm.elevation(1.).coords, upper))
        self.assertTrue(allclose(pm.elevation(0.5).coords,
                                 (lower + upper) / 2.))
        # The mesh shares the prism's triangle topology.
        self.assertEqual(pm.elevation(0.5).topo.order, 2)

    def test_measures_are_prism_volumes(self):
        # Each prism is half the unit square, two units thick.
        self.assertTrue(allclose(_prism().measures, [1., 1.]))

    def test_to_tetmesh_decomposes_the_prisms(self):
        pm = _prism()
        tm = pm.to_tetmesh()
        self.assertIsInstance(tm, TetMesh)
        self.assertEqual(tm.coord_count, 8)
        self.assertEqual(tm.topo.simplex_count[3], 6)
        # The tetrahedra fill the prisms, so their volumes agree.
        self.assertAlmostEqual(float(sum(tm.measures)),
                               float(sum(pm.measures)))

    def test_a_tetrahedral_decomposition_matches_a_known_volume(self):
        # One prism, one unit of area and three units thick.
        lower = array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        upper = lower + array([[0.], [0.], [3.]])
        pm = PrismMesh(stack([lower, upper]), PrismTopology([[0], [1], [2]]))
        self.assertAlmostEqual(float(sum(pm.measures)), 1.5)
        self.assertAlmostEqual(float(sum(pm.to_tetmesh().measures)), 1.5)

    def test_elevation_bearing_properties(self):
        pm = _prism()
        elevs = array([0., 1.])
        values = array([[10., 20., 30., 40.], [60., 70., 80., 90.]])
        pm2 = pm.withprop('temperature', (elevs, values))
        self.assertEqual(pm2['temperature'].shape, (2, 4))
        self.assertTrue(allclose(pm2['temperature'][0], values[0]))
        self.assertTrue(allclose(pm2['temperature'][1], values[1]))
        self.assertTrue(allclose(pm2.elevations['temperature'], elevs))
        # The original has no elevations.
        self.assertEqual(len(pm.elevations), 0)

    def test_a_property_without_elevations_needs_none(self):
        pm = _prism().withprop('thickness', array([2., 2., 2., 2.]))
        self.assertEqual(len(pm.elevations), 0)
        self.assertEqual(pm['thickness'].tolist(), [2., 2., 2., 2.])

    def test_to_local_round_trips_a_parallel_sided_prism(self):
        pm = _prism()
        for (index, u, v, e) in ((0, 0.25, 0.25, 0.0), (0, 0.25, 0.25, 0.5),
                                 (1, 0.1, 0.2, 0.8), (1, 0.3, 0.3, 1.0)):
            with self.subTest(index=index, u=u, v=v, e=e):
                loc = pm.topo.Loc(array([index]), array([[u], [v]]),
                                  array([[e]]))
                back = pm.to_local(pm.to_global(loc))
                self.assertEqual(int(back.index[0]), index)
                self.assertTrue(allclose(back.weight.ravel(), [u, v],
                                         atol=1e-9))
                self.assertTrue(allclose(back.height.ravel(), [e], atol=1e-9))

    def test_to_local_round_trips_a_skewed_prism(self):
        # The two surfaces are not parallel, so a position's local coordinates
        # are a nonlinear function of it; the solve must still recover them.
        lower = array([[0., 1., 0., 1.],
                       [0., 0., 1., 1.],
                       [0., 0., 0., 0.]])
        upper = array([[0., 1., 0., 1.],
                       [0., 0., 1., 1.],
                       [0., 0.5, 1., 3.]])
        pm = PrismMesh(stack([lower, upper]), _topo())
        for (index, u, v, e) in ((0, 0.25, 0.25, 0.5), (1, 0.1, 0.2, 0.8)):
            with self.subTest(index=index):
                loc = pm.topo.Loc(array([index]), array([[u], [v]]),
                                  array([[e]]))
                back = pm.to_local(pm.to_global(loc))
                self.assertEqual(int(back.index[0]), index)
                self.assertTrue(allclose(back.weight.ravel(), [u, v],
                                         atol=1e-9))
                self.assertTrue(allclose(back.height.ravel(), [e], atol=1e-9))

    def test_to_local_handles_many_positions_at_once(self):
        pm = _prism()
        tri = array([0, 0, 1, 1])
        (u, v) = (array([0.2, 0.3, 0.1, 0.4]), array([0.3, 0.2, 0.4, 0.1]))
        e = array([0.1, 0.9, 0.5, 0.0])
        back = pm.to_local(pm.to_global(pm.topo.Loc(tri, stack([u, v]), e[None])))
        self.assertTrue(allclose(back.index, tri))
        self.assertTrue(allclose(back.weight, stack([u, v]), atol=1e-9))
        self.assertTrue(allclose(back.height[0], e, atol=1e-9))

    def test_a_position_off_the_sheet_is_answered_by_its_nearest_surface(self):
        # A prism's parameterization can be inverted for a position beyond its
        # surfaces as readily as for one between them, so the lookup must clamp
        # to the object rather than extrapolate --- as it does for every other
        # geometry. The sheet spans z from 0 to 2.
        pm = _prism()
        above = pm.to_local(array([[0.2], [0.4], [10.0]]))
        self.assertAlmostEqual(float(above.height[0, 0]), 1.0)
        self.assertAlmostEqual(float(pm.to_global(above)[2, 0]), 2.0)
        below = pm.to_local(array([[0.2], [0.4], [-3.0]]))
        self.assertAlmostEqual(float(below.height[0, 0]), 0.0)
        self.assertAlmostEqual(float(pm.to_global(below)[2, 0]), 0.0)
