# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/test_topo.py
'''Tests for the concrete topologies in ``euclib.types._topo``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import array

from euclib.abc import is_loc, is_simplex_topology
from euclib.types import (
    VertexTopology, SegTopology, TriTopology, TetTopology,
    VertexLoc, SegLoc, TriLoc, TetLoc)


# Tests ######################################################################

class TestConcreteTopologies(TestCase):
    '''Tests for each concrete topology's order and validation.'''

    def test_orders(self):
        self.assertEqual(VertexTopology([[0, 1, 2]]).order, 0)
        self.assertEqual(SegTopology([[0, 1], [1, 2]]).order, 1)
        self.assertEqual(TriTopology([[0, 0], [1, 2], [2, 3]]).order, 2)
        self.assertEqual(TetTopology([[0], [1], [2], [3]]).order, 3)

    def test_local_dim_matches_order(self):
        self.assertEqual(VertexTopology([[0, 1]]).local_dim, 0)
        self.assertEqual(SegTopology([[0], [1]]).local_dim, 1)
        self.assertEqual(TriTopology([[0], [1], [2]]).local_dim, 2)
        self.assertEqual(TetTopology([[0], [1], [2], [3]]).local_dim, 3)

    def test_each_topology_rejects_the_wrong_number_of_rows(self):
        cases = ((VertexTopology, 1), (SegTopology, 2), (TriTopology, 3),
                 (TetTopology, 4))
        for (cls, rows) in cases:
            # The right number of rows is accepted...
            cls([[0]] * rows)
            # ...and any other number is not.
            for bad in (rows - 1, rows + 1):
                if bad < 1:
                    continue
                with self.assertRaises(Exception):
                    cls([[0]] * bad)

    def test_local_shape(self):
        self.assertEqual(SegTopology([[0], [1]]).local_shape(7), (1, 7))
        self.assertEqual(TriTopology([[0], [1], [2]]).local_shape(3), (2, 3))

    def test_all_are_simplex_topologies(self):
        topo = TriTopology([[0], [1], [2]])
        self.assertTrue(is_simplex_topology(topo))


class TestConcreteLocs(TestCase):
    '''Tests for each topology's local coordinate type and its validation.'''

    def test_loc_types(self):
        self.assertEqual(VertexLoc._fields, ('index',))
        self.assertEqual(SegLoc._fields, ('index', 'weight'))
        self.assertEqual(TriLoc._fields, ('index', 'weight'))
        self.assertEqual(TetLoc._fields, ('index', 'weight'))

    def test_loc_types_are_locs(self):
        self.assertTrue(is_loc(SegLoc(0, array([[0.5]]))))
        self.assertTrue(is_loc(TriLoc(0, array([[0.2], [0.3]]))))

    def test_check_loc_accepts_well_formed_coordinates(self):
        topo = TriTopology([[0], [1], [2]])
        loc = topo.check_loc(TriLoc(array([0, 1]),
                                    array([[0.2, 0.3], [0.3, 0.2]])))
        self.assertEqual(loc.index.tolist(), [0, 1])

    def test_check_loc_accepts_mappings_and_sequences(self):
        topo = SegTopology([[0], [1]])
        by_map = topo.check_loc({'index': array([0]), 'weight': [[0.5]]})
        by_seq = topo.check_loc((array([0]), [[0.5]]))
        self.assertEqual(by_map.index.tolist(), [0])
        self.assertEqual(by_seq.index.tolist(), [0])

    def test_check_loc_rejects_bad_shapes(self):
        topo = TriTopology([[0], [1], [2]])
        # The weight must be (2, N) for a triangle topology.
        with self.assertRaises(ValueError):
            topo.check_loc(TriLoc(array([0]), array([[0.5]])))
        # The index must be a vector.
        with self.assertRaises(ValueError):
            topo.check_loc(TriLoc(array([[0]]), array([[0.5], [0.5]])))
        # The index and the weight must describe the same positions.
        with self.assertRaises(ValueError):
            topo.check_loc(TriLoc(array([0, 1]), array([[0.5], [0.5]])))

    def test_vertex_loc_has_no_weights(self):
        topo = VertexTopology([[0, 1, 2]])
        self.assertEqual(topo.check_loc(VertexLoc(array([0, 2]))).index.tolist(),
                         [0, 2])
        with self.assertRaises(ValueError):
            topo.check_loc(VertexLoc(array([[0, 2]])))
