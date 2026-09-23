# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/abc/test_topo.py
'''Tests for the ``euclib.abc._topo`` module.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import array

from euclib.abc import (
    Topology, SimplexTopology, make_loc, is_loc, is_topology,
    is_simplex_topology, normalize_metadata, check_simplex_loc)


# A concrete topology for testing ############################################

TriLoc = make_loc('TriLoc', ('index', 'weight'))


class ExampleTopology(SimplexTopology):
    '''A minimal concrete simplex topology used to exercise the base types.'''

    Loc = TriLoc

    def check_loc(self, locs, /):
        return check_simplex_loc(self.Loc, self.local_dim, locs)


# Tests ######################################################################

class TestLoc(TestCase):
    '''Tests for the ``Loc`` machinery.'''

    def test_from_value_accepts_sequence(self):
        loc = TriLoc.from_value([2, 0.25])
        self.assertEqual(loc.index, 2)
        self.assertEqual(loc.weight, 0.25)

    def test_from_value_accepts_mapping(self):
        loc = TriLoc.from_value({'index': 3, 'weight': 0.5})
        self.assertEqual(loc.index, 3)
        self.assertEqual(loc.weight, 0.5)

    def test_from_value_passes_through(self):
        loc = TriLoc(1, 0.1)
        self.assertIs(TriLoc.from_value(loc), loc)

    def test_is_loc(self):
        self.assertTrue(is_loc(TriLoc(0, 0.0)))
        self.assertFalse(is_loc((0, 0.0)))
        self.assertFalse(is_loc(None))

    def test_fields_are_named(self):
        self.assertEqual(TriLoc._fields, ('index', 'weight'))


class TestTopologyAbstractness(TestCase):
    '''The abstract topology types may not be instantiated.'''

    def test_topology_is_abstract(self):
        with self.assertRaises(TypeError):
            Topology()

    def test_simplex_topology_is_abstract(self):
        with self.assertRaises(TypeError):
            SimplexTopology([[0], [1], [2]])

    def test_concrete_subclass_is_not_abstract(self):
        topo = ExampleTopology([[0, 0], [1, 2], [2, 3]])
        self.assertTrue(is_topology(topo))
        self.assertTrue(is_simplex_topology(topo))


class TestSimplexTopology(TestCase):
    '''Tests for the simplex topology's derived structure.'''

    def setUp(self):
        # Two triangles sharing an edge: 4 vertices, 5 edges, 2 triangles.
        self.topo = ExampleTopology([[0, 0], [1, 2], [2, 3]])

    def test_order_and_dims(self):
        self.assertEqual(self.topo.order, 2)
        self.assertEqual(self.topo.dim, 2)
        self.assertEqual(self.topo.local_dim, 2)

    def test_coord_count_defaults_to_largest_index_plus_one(self):
        self.assertEqual(self.topo.coord_count, 4)

    def test_coord_count_may_exceed_used_coordinates(self):
        topo = ExampleTopology([[0, 0], [1, 2], [2, 3]], coord_count=10)
        self.assertEqual(topo.coord_count, 10)
        self.assertEqual(topo.vertex_count, 4)
        self.assertEqual(int(topo.vertex_mask.sum()), 4)

    def test_coord_count_may_not_be_too_small(self):
        with self.assertRaises(Exception):
            ExampleTopology([[0, 0], [1, 2], [2, 3]], coord_count=2)

    def test_simplices_has_one_entry_per_order(self):
        simps = self.topo.simplices
        self.assertEqual(len(simps), self.topo.order + 1)
        # Entry k is a (k+1) x M_k matrix.
        for (k, s) in enumerate(simps):
            self.assertEqual(s.shape[0], k + 1)

    def test_primary_simplices_are_preserved(self):
        # Entry `order` is the caller's matrix, orientation intact.
        self.assertEqual(self.topo.simplices[self.topo.order].tolist(),
                         [[0, 0], [1, 2], [2, 3]])

    def test_lower_order_simplices_are_canonical(self):
        # 4 unique vertices...
        self.assertEqual(self.topo.simplices[0].tolist(), [[0, 1, 2, 3]])
        # ...and 5 unique edges, each with its corners sorted.
        self.assertEqual(self.topo.simplices[1].tolist(),
                         [[0, 0, 0, 1, 2],
                          [1, 2, 3, 2, 3]])

    def test_simplex_count(self):
        self.assertEqual(list(self.topo.simplex_count), [4, 5, 2])
        self.assertEqual(self.topo.vertex_count, 4)

    def test_lower_order_simplices_do_not_depend_on_orientation(self):
        # The same two triangles with differently ordered corners.
        other = ExampleTopology([[2, 0], [0, 3], [1, 2]])
        self.assertEqual(other.simplices[1].tolist(),
                         self.topo.simplices[1].tolist())
        self.assertEqual(other.simplices[0].tolist(),
                         self.topo.simplices[0].tolist())

    def test_vertex_mask_and_index_of(self):
        self.assertEqual(self.topo.vertex_mask.tolist(),
                         [True, True, True, True])
        self.assertEqual(self.topo.index_of.tolist(), [0, 1, 2, 3])
        # With unused coordinates, they are marked.
        topo = ExampleTopology([[0], [2], [3]], coord_count=6)
        self.assertEqual(topo.vertex_mask.tolist(),
                         [True, False, True, True, False, False])
        self.assertEqual(topo.index_of.tolist(), [0, -1, 1, 2, -1, -1])

    def test_point_cloud_topology(self):
        # A vertex topology has a single simplex order.
        topo = ExampleTopology([[0, 1, 2]])
        self.assertEqual(topo.order, 0)
        self.assertEqual(len(topo.simplices), 1)
        self.assertEqual(topo.simplices[0].tolist(), [[0, 1, 2]])
        self.assertEqual(list(topo.simplex_count), [3])

    def test_non_matrix_indices_are_rejected(self):
        with self.assertRaises(Exception):
            ExampleTopology([0, 1, 2])

    def test_float_indices_are_rejected(self):
        with self.assertRaises(Exception):
            ExampleTopology([[0.5], [1.5], [2.5]])


class TestTopologyMetadata(TestCase):
    '''Tests for topology backend and metadata handling.'''

    def test_backend_validation(self):
        self.assertIsNone(ExampleTopology([[0], [1]], backend=None).backend)
        self.assertEqual(
            ExampleTopology([[0], [1]], backend='numpy').backend, 'numpy')
        with self.assertRaises(Exception):
            ExampleTopology([[0], [1]], backend='jax')

    def test_metadata_is_made_lazy(self):
        from pcollections import ldict
        meta = normalize_metadata({'name': 'test'})
        self.assertIsInstance(meta, ldict)
        self.assertEqual(dict(meta), {'name': 'test'})
        topo = ExampleTopology([[0], [1]], metadata={'name': 'test'})
        self.assertIsInstance(topo.metadata, ldict)
        self.assertEqual(dict(topo.metadata), {'name': 'test'})
        # No metadata at all is empty rather than absent.
        self.assertIsInstance(ExampleTopology([[0], [1]]).metadata, ldict)
        self.assertEqual(len(ExampleTopology([[0], [1]]).metadata), 0)
        with self.assertRaises(Exception):
            ExampleTopology([[0], [1]], metadata='not-a-mapping')

    def test_equality_and_hash(self):
        a = ExampleTopology([[0, 0], [1, 2], [2, 3]])
        b = ExampleTopology([[0, 0], [1, 2], [2, 3]])
        c = ExampleTopology([[0, 0], [1, 2], [2, 4]])
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, c)
        self.assertEqual(len({a, b}), 1)
        self.assertNotEqual(a, ExampleTopology([[0, 0], [1, 2], [2, 3]],
                                               backend='numpy'))

    def test_coord_count_does_not_affect_equality(self):
        # coord_count says how large a coordinate matrix the topology is
        # valid for; it is not part of the connectivity, so two topologies that
        # connect the same simplices are equal regardless of it.
        a = ExampleTopology([[0, 0], [1, 2], [2, 3]])
        b = ExampleTopology([[0, 0], [1, 2], [2, 3]], coord_count=7)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        # Metadata is a label rather than part of the object, so it does not
        # distinguish two topologies either.
        self.assertEqual(a, ExampleTopology([[0, 0], [1, 2], [2, 3]],
                                            metadata={'name': 'x'}))
