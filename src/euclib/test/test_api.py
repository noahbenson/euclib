# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/test_api.py
'''Tests for the library's public surface: its namespace and its constructors.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import allclose, array, eye, stack
from pcollections import ldict

import euclib


# Tests ######################################################################

class TestNamespace(TestCase):
    '''What ``euclib`` exposes and how it names it.'''

    def test_the_types_are_imported_here(self):
        for name in ('Geometry', 'SimplexGeometry', 'Topology',
                     'SimplexTopology', 'Property',
                     'VertexSet', 'SegPath', 'TriMesh', 'TetMesh', 'PrismMesh',
                     'Grid', 'VertexTopology', 'SegTopology', 'TriTopology',
                     'TetTopology', 'PrismTopology', 'GridTopology',
                     'VertexLoc', 'SegLoc', 'TriLoc', 'TetLoc', 'PrismLoc'):
            with self.subTest(name=name):
                self.assertTrue(hasattr(euclib, name))

    def test_the_operations_are_imported_here(self):
        for name in ('distance', 'separation', 'nearest', 'geodesic',
                     'contains', 'path_crossings', 'path_intersections',
                     'mesh_intersections', 'voxel_intersections'):
            with self.subTest(name=name):
                self.assertTrue(hasattr(euclib, name))

    def test_the_operations_tools_stay_in_ops(self):
        # These are the tools the operations are built from, or names that would
        # not stand on their own unqualified.
        for name in ('positions_of', 'tolerance_of', 'sample', 'transfer'):
            with self.subTest(name=name):
                self.assertTrue(hasattr(euclib.ops, name))
                self.assertFalse(hasattr(euclib, name))
                self.assertNotIn(name, euclib.__all__)

    def test_a_type_is_the_same_object_under_both_paths(self):
        self.assertIs(euclib.Grid, euclib.types.Grid)
        self.assertIs(euclib.Geometry, euclib.abc.Geometry)
        self.assertIs(euclib.distance, euclib.ops.distance)
        self.assertIs(euclib.mesh_intersections, euclib.ops.mesh_intersections)

    def test_a_type_resolves_to_euclib(self):
        # A repr, a piece of documentation, and a new pickle should all name
        # euclib.Grid rather than the module it happens to be defined in.
        self.assertEqual(euclib.Grid.__module__, 'euclib')
        self.assertEqual(euclib.TriMesh.__module__, 'euclib')
        self.assertEqual(euclib.VertexLoc.__module__, 'euclib')
        self.assertTrue(repr(euclib.grid((2, 2))).startswith('euclib.Grid('))

    #: The public modules, shallowest first, as the library itself lists them.
    _PUBLIC = ('euclib', 'euclib.abc', 'euclib.types', 'euclib.utils',
               'euclib.ops')

    def _public_objects(self):
        '''Returns each public object with the modules that export it.'''
        from importlib import import_module
        from types import ModuleType
        found = {}
        for module_name in self._PUBLIC:
            module = import_module(module_name)
            for exported in module.__all__:
                obj = getattr(module, exported)
                if isinstance(obj, ModuleType):
                    continue
                found.setdefault(id(obj), (obj, []))[1].append(module_name)
        return list(found.values())

    def test_no_public_object_lives_in_a_private_module(self):
        # euclib.types._geom.TriMesh is not a name anyone should see: the
        # private module is an implementation detail.
        for (obj, _) in self._public_objects():
            source = getattr(obj, '__module__', None)
            if not isinstance(source, str) or not source.startswith('euclib.'):
                continue
            with self.subTest(obj=repr(obj)[:60]):
                self.assertFalse(
                    source.rsplit('.', 1)[-1].startswith('_'),
                    f"{obj!r} names the private module {source}")

    def test_the_shallowest_public_module_wins(self):
        # A name that euclib.types and euclib both hold belongs to euclib: the
        # shallowest public module that imports it is the one it names. An
        # object euclib merely re-exports keeps its own module.
        for (obj, places) in self._public_objects():
            source = getattr(obj, '__module__', None)
            if not isinstance(source, str) or not source.startswith('euclib.'):
                continue
            with self.subTest(obj=repr(obj)[:60]):
                self.assertEqual(source, places[0])

    def test_incidental_names_are_not_exposed(self):
        # The names that importing and initializing the module bound are not
        # part of the library.
        for name in ('annotations', 'reload', 'modules'):
            with self.subTest(name=name):
                self.assertFalse(hasattr(euclib, name))

    def test_every_exported_name_exists(self):
        for name in euclib.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(euclib, name))

    def test_the_version_is_a_submodule(self):
        # reload_euclib reloads what submodules names; a module left out of it
        # is never reloaded.
        self.assertIn('euclib._version', euclib.submodules)
        for name in euclib.submodules:
            with self.subTest(name=name):
                self.assertIn(name, __import__('sys').modules)


class TestConstructors(TestCase):
    '''The constructors, which build an object's topology for it.'''

    def test_points(self):
        cloud = euclib.points(array([[0., 1., 2.], [0., 0., 0.]]))
        self.assertIsInstance(cloud, euclib.VertexSet)
        self.assertEqual(cloud.coord_count, 3)
        self.assertEqual(cloud.topo.simplex_count[0], 3)
        # A subset of the coordinates may be named.
        part = euclib.points(array([[0., 1., 2.], [0., 0., 0.]]),
                             vertices=[0, 2])
        self.assertEqual(part.coord_count, 3)
        self.assertEqual(part.topo.simplex_count[0], 2)

    def test_segpath(self):
        path = euclib.segpath(array([[0., 1., 2.], [0., 0., 0.]]))
        self.assertIsInstance(path, euclib.SegPath)
        # The coordinates in order make up the path: two segments.
        self.assertEqual(path.topo.simplex_count[1], 2)
        # A subset of the coordinates may be named.
        part = euclib.segpath(array([[0., 1., 2.], [0., 0., 0.]]),
                              vertices=[0, 2])
        self.assertEqual(part.topo.simplex_count[1], 1)

    def test_trimesh(self):
        mesh = euclib.trimesh(array([[0., 1., 0.], [0., 0., 1.]]),
                              [[0], [1], [2]])
        self.assertIsInstance(mesh, euclib.TriMesh)
        self.assertEqual(mesh.topo.simplex_count[2], 1)

    def test_tetmesh(self):
        mesh = euclib.tetmesh(array([[0., 1., 0., 0.], [0., 0., 1., 0.],
                                     [0., 0., 0., 1.]]),
                              [[0], [1], [2], [3]])
        self.assertIsInstance(mesh, euclib.TetMesh)
        self.assertEqual(mesh.topo.simplex_count[3], 1)

    def test_prismmesh(self):
        lower = array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        mesh = euclib.prismmesh(stack([lower, lower + 1.]), [[0], [1], [2]])
        self.assertIsInstance(mesh, euclib.PrismMesh)
        self.assertEqual(mesh.topo.simplex_count[2], 1)

    def test_grid(self):
        grid = euclib.grid((4, 5))
        self.assertIsInstance(grid, euclib.Grid)
        self.assertEqual(grid.shape, (4, 5))
        # An affine of None is the identity: a cell's indices are its
        # coordinates.
        self.assertTrue(allclose(grid.coords, eye(3)))
        # A dtype is passed to the affine, whether given or defaulted.
        self.assertEqual(euclib.grid((4, 5), dtype='f4').coords.dtype.str,
                         '<f4')
        tilted = euclib.grid((2, 2), affine=array([[2., 0., 1.],
                                                   [0., 2., 0.],
                                                   [0., 0., 1.]]),
                             dtype='f8')
        self.assertEqual(tilted.origin.tolist(), [1., 0.])

    def test_the_constructors_take_properties_and_metadata(self):
        prop = euclib.Property(array([1., 2., 3.]), (3,))
        cloud = euclib.points(array([[0., 1., 2.], [0., 0., 0.]]),
                              properties={'a': prop},
                              metadata={'name': 'a cloud'})
        self.assertEqual(cloud['a'].tolist(), [1., 2., 3.])
        self.assertEqual(dict(cloud.metadata), {'name': 'a cloud'})

    def test_the_constructors_are_also_in_euclib_types(self):
        self.assertIs(euclib.grid, euclib.types.grid)
        self.assertIs(euclib.trimesh, euclib.types.trimesh)


class TestMetadata(TestCase):
    '''Metadata on the abstract types.'''

    def setUp(self):
        self.mesh = euclib.trimesh(array([[0., 1., 0.], [0., 0., 1.]]),
                                   [[0], [1], [2]])
        self.topo = self.mesh.topo

    def test_metadata_is_empty_lazy_by_default(self):
        for obj in (self.mesh, self.topo):
            with self.subTest(obj=type(obj).__name__):
                self.assertIsInstance(obj.metadata, ldict)
                self.assertEqual(len(obj.metadata), 0)

    def test_withmeta_takes_a_mapping_and_keywords(self):
        # The keywords are merged over the mapping.
        both = self.mesh.withmeta({'a': 1, 'b': 2}, b=3, c=4)
        self.assertEqual(dict(both.metadata), {'a': 1, 'b': 3, 'c': 4})
        # ...and a bare keyword, or a bare mapping, works too.
        self.assertEqual(dict(self.mesh.withmeta(a=1).metadata), {'a': 1})
        self.assertEqual(dict(self.mesh.withmeta({'a': 1}).metadata), {'a': 1})

    def test_withmeta_returns_a_copy(self):
        marked = self.mesh.withmeta(a=1)
        self.assertEqual(len(self.mesh.metadata), 0)
        self.assertIsNot(marked, self.mesh)
        # Merging over existing metadata rather than replacing it.
        self.assertEqual(dict(marked.withmeta(b=2).metadata), {'a': 1, 'b': 2})
        # Changing nothing returns the object itself.
        self.assertIs(self.mesh.withmeta(), self.mesh)

    def test_dropmeta(self):
        marked = self.mesh.withmeta(a=1, b=2, c=3)
        self.assertEqual(dict(marked.dropmeta('b').metadata), {'a': 1, 'c': 3})
        # Several keys at once, and a key that is not there.
        self.assertEqual(dict(marked.dropmeta('a', 'c', 'nope').metadata),
                         {'b': 2})
        self.assertIs(marked.dropmeta('nope'), marked)

    def test_metadata_is_not_part_of_equality(self):
        self.assertEqual(self.mesh.withmeta(a=1), self.mesh)
        self.assertEqual(hash(self.mesh.withmeta(a=1)), hash(self.mesh))
        # ...and not of a topology's either.
        self.assertEqual(self.topo.withmeta(a=1), self.topo)
        self.assertEqual(hash(self.topo.withmeta(a=1)), hash(self.topo))

    def test_a_bad_metadata_value_is_refused(self):
        with self.assertRaises(Exception):
            euclib.trimesh(array([[0., 1., 0.], [0., 0., 1.]]),
                           [[0], [1], [2]], metadata='not-a-mapping')
