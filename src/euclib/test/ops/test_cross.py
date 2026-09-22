# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/test_cross.py
'''Tests for the representation-crossing operations in ``euclib.ops._cross``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import allclose, arange, array, add as npadd, eye, isnan

from euclib.ops import positions_of, sample, transfer
from euclib.types import (
    Grid, GridTopology, PrismMesh, PrismTopology, TetMesh, TetTopology, TriMesh,
    TriTopology, VertexSet, VertexTopology)


# Fixtures ###################################################################

def _mesh():
    '''The unit square split along its diagonal, holding f = x + y.'''
    mesh = TriMesh(array([[0., 1., 0., 1.], [0., 0., 1., 1.]]),
                   TriTopology([[0, 0], [1, 3], [3, 2]]))
    return mesh.withprop('f', array([0., 1., 1., 2.]), interp=1)


def _grid():
    '''A 3x3 grid of cells at 0, 0.5, and 1 along each axis.'''
    return Grid(array([[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 1.]]),
                GridTopology((3, 3)))


def _cloud(*pairs):
    '''A point cloud from a sequence of (x, y) pairs.'''
    return VertexSet(array([[p[0] for p in pairs], [p[1] for p in pairs]]),
                     VertexTopology([list(range(len(pairs)))]))


# Tests ######################################################################

class TestPositions(TestCase):
    '''Where each kind of geometry's own data lives.'''

    def test_a_simplex_geometry_s_data_lives_at_its_coordinates(self):
        self.assertEqual(positions_of(_mesh()).shape, (2, 4))

    def test_a_grid_s_data_lives_at_its_cells(self):
        self.assertEqual(positions_of(_grid()).shape, (2, 9))

    def test_a_grid_s_cell_positions_follow_its_affine(self):
        # The cells of the 3x3 grid sit at 0, 0.5, and 1 along each axis.
        pos = positions_of(_grid())
        self.assertTrue(allclose(sorted(set(pos[0].tolist())), [0., 0.5, 1.]))

    def test_a_prism_mesh_s_data_lives_at_both_surfaces(self):
        lower = array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        pm = PrismMesh(array([lower, lower + array([[0.], [0.], [1.]])]),
                       PrismTopology([[0], [1], [2]]))
        self.assertEqual(positions_of(pm).shape, (3, 6))

    def test_positions_of_checks_its_argument(self):
        with self.assertRaises(TypeError):
            positions_of('not-a-geometry')


class TestTransfer(TestCase):
    '''Moving a property from one representation to another.'''

    def test_a_mesh_property_lands_on_a_grid(self):
        g = transfer(_mesh(), _grid(), 'f')
        self.assertEqual(g['f'].shape, (3, 3))
        # The property is f = x + y and the cells are half a unit apart.
        self.assertTrue(allclose(g['f'],
                                 0.5 * npadd.outer(arange(3), arange(3))))

    def test_a_grid_property_lands_on_a_mesh(self):
        g = transfer(_mesh(), _grid(), 'f')
        m = transfer(g, _mesh(), 'f')
        self.assertTrue(allclose(m['f'], _mesh()['f'], atol=1e-12))

    def test_a_grid_property_lands_on_a_point_cloud(self):
        g = transfer(_mesh(), _grid(), 'f')
        c = transfer(g, _cloud((0.25, 0.25), (0.75, 0.75)), 'f')
        self.assertTrue(allclose(c['f'], [0.5, 1.5], atol=1e-12))

    def test_the_property_may_be_renamed(self):
        g = transfer(_mesh(), _grid(), 'f', name='temperature')
        self.assertIn('temperature', g.properties)
        self.assertNotIn('f', g.properties)

    def test_metadata_can_be_set_on_the_result(self):
        g = transfer(_mesh(), _grid(), 'f', meta={'interp': 0})
        self.assertEqual(g.propinfo('f').interp, ('nearest', 0))

    def test_the_original_is_unchanged(self):
        mesh = _mesh()
        transfer(mesh, _grid(), 'f')
        self.assertNotIn('f', _grid().properties)
        self.assertEqual(len(mesh.properties), 1)

    def test_a_position_off_the_source_has_no_answer(self):
        # The mesh covers the unit square; a point cloud beyond it is answered
        # with the null value unless extrapolation is asked for.
        far = _cloud((9., 0.5), (0., 0.5))
        self.assertTrue(isnan(transfer(_mesh(), far, 'f')['f'][0]))
        # (9, 0.5) is nearest the mesh's right edge at (1, 0.5), where f = 1.5;
        # (0, 0.5) is already on the mesh's left edge, where f = 0.5.
        self.assertTrue(allclose(transfer(_mesh(), far, 'f', extrap=0)['f'],
                                 [1.5, 0.5], atol=1e-12))

    def test_transfer_between_meshes_of_different_resolution(self):
        # A coarse mesh whose property is linear, read onto a finer one: the
        # linear field must come through exactly.
        fine = TriMesh(array([[0., 1., 0., 1.], [0., 0., 1., 1.]]),
                       TriTopology([[0, 0], [1, 3], [3, 2]]))
        attended = transfer(_mesh(), fine, 'f')
        self.assertTrue(allclose(attended['f'], fine.withprop(
            'x', array([0., 1., 1., 2.]))['x'], atol=1e-12))

    def test_transfer_between_a_mesh_and_a_tetrahedral_mesh(self):
        # The mesh lies in the z = 0 plane over the square the tetrahedral
        # mesh's base shares.
        mesh = TriMesh(array([[0., 1., 0., 1.],
                              [0., 0., 1., 1.],
                              [0., 0., 0., 0.]]),
                       TriTopology([[0, 0], [1, 3], [3, 2]]))
        mesh = mesh.withprop('f', array([0., 1., 1., 2.]), interp=1)
        tet = TetMesh(array([[0., 1., 0., 0.], [0., 0., 1., 0.],
                             [0., 0., 0., 1.]]),
                      TetTopology([[0], [1], [2], [3]]))
        # Three of the tetrahedron's corners lie on the mesh; the fourth is a
        # unit above it, and is answered by the mesh's nearest point when
        # extrapolation is asked for.
        got = transfer(mesh, tet, 'f', extrap=0)
        self.assertTrue(allclose(got['f'], [0., 1., 1., 0.], atol=1e-12))

    def test_transfer_refuses_positions_of_the_wrong_dimension(self):
        # A 2-D grid's cells cannot be located in a 3-D mesh.
        tet = TetMesh(array([[0., 1., 0., 0.], [0., 0., 1., 0.],
                             [0., 0., 0., 1.]]),
                      TetTopology([[0], [1], [2], [3]]))
        g = transfer(_mesh(), _grid(), 'f')
        with self.assertRaises(ValueError):
            transfer(g, tet, 'f')

    def test_sample_returns_the_values_without_attaching_them(self):
        values = sample(_mesh(), _grid(), 'f')
        self.assertEqual(values.shape, (9,))
        self.assertTrue(allclose(values.reshape(3, 3),
                                 0.5 * npadd.outer(arange(3), arange(3))))

    def test_the_operations_check_their_arguments(self):
        with self.assertRaises(TypeError):
            sample(_mesh(), 'not-a-geometry', 'f')
        with self.assertRaises(KeyError):
            transfer(_mesh(), _grid(), 'no-such-property')
