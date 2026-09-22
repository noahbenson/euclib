# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/abc/test_geom.py
'''Tests for the ``euclib.abc._geom`` module.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import array, ones, zeros

from euclib.abc import (
    SimplexTopology, SimplexGeometry, Geometry,
    Property, split_property_name, make_loc, check_simplex_loc,
    is_geometry, is_simplex_geometry)


# Concrete types for testing #################################################

PointLoc = make_loc('PointLoc', ('index',))
TriLoc = make_loc('TriLoc', ('index', 'weight'))


class CloudTopology(SimplexTopology):
    '''A point-cloud topology: simplices of order 0.'''

    Loc = PointLoc

    def check_loc(self, locs, /):
        return check_simplex_loc(self.Loc, self.local_dim, locs)


class MeshTopology(SimplexTopology):
    '''A triangle-mesh topology.'''

    Loc = TriLoc

    def check_loc(self, locs, /):
        return check_simplex_loc(self.Loc, self.local_dim, locs)


class Cloud(SimplexGeometry):
    '''A point cloud over a 2- or 3-dimensional space.'''

    def to_local(self, coords, /):
        return None

    def to_global(self, locs, /):
        return None


def _cloud(coords=None, topo=None, **kw):
    '''Builds a four-point cloud, with unused-coordinate variants.'''
    if coords is None:
        coords = array([[0., 1., 2., 3.],
                        [0., 0., 0., 0.]])
    if topo is None:
        topo = CloudTopology([[0, 1, 2, 3]])
    return Cloud(coords, topo, **kw)


def _mesh():
    '''Builds a two-triangle mesh: 4 coordinates, 5 edges, 2 triangles.'''
    coords = array([[0., 1., 0., 1.],
                    [0., 0., 1., 1.],
                    [0., 0., 0., 0.]])
    topo = MeshTopology([[0, 0], [1, 2], [2, 3]])
    return Cloud(coords, topo)


# Tests ######################################################################

class TestGeometryBasics(TestCase):
    '''Tests for construction and the basic geometry fields.'''

    def test_construction(self):
        cloud = _cloud()
        self.assertTrue(is_geometry(cloud))
        self.assertTrue(is_simplex_geometry(cloud))
        self.assertEqual(cloud.dim, 2)
        self.assertEqual(cloud.coord_count, 4)
        self.assertEqual(cloud.order, 0)
        self.assertEqual(cloud.property_shape, (4,))
        self.assertEqual(cloud.vertex_count, 4)
        self.assertEqual(cloud.simplex_count[0], 4)

    def test_coords_is_validated_and_normalized(self):
        # A list of lists becomes an array, so that geom.coords is always the
        # validated payload rather than whatever was passed.
        cloud = _cloud(coords=[[0., 1., 2., 3.], [0., 0., 0., 0.]])
        self.assertEqual(tuple(cloud.coords.shape), (2, 4))

    def test_coords_must_be_a_matrix(self):
        with self.assertRaises(Exception):
            _cloud(coords=zeros(4))

    def test_coords_must_be_two_or_three_dimensional(self):
        with self.assertRaises(Exception):
            Cloud(zeros((4, 4)), CloudTopology([[0, 1, 2, 3]]))

    def test_coord_count_must_match_the_topology(self):
        # A topology declaring 4 coordinates cannot take a 5-column matrix.
        with self.assertRaises(Exception):
            Cloud(zeros((2, 5)), CloudTopology([[0, 1, 2, 3]]))

    def test_extra_coordinates_require_a_larger_coord_count(self):
        coords = zeros((2, 6))
        topo = CloudTopology([[0, 1, 2, 3]], coord_count=6)
        cloud = Cloud(coords, topo)
        self.assertEqual(cloud.coord_count, 6)
        self.assertEqual(cloud.vertex_count, 4)
        self.assertEqual(cloud.vertex_mask.tolist(),
                         [True, True, True, True, False, False])

    def test_topology_is_validated(self):
        with self.assertRaises(Exception):
            Cloud(zeros((2, 4)), 'not-a-topology')

    def test_abstract_types(self):
        with self.assertRaises(TypeError):
            Geometry(None, None)
        with self.assertRaises(TypeError):
            SimplexGeometry(zeros((2, 4)), CloudTopology([[0, 1, 2, 3]]))

    def test_backend(self):
        self.assertIsNone(_cloud().backend)
        self.assertEqual(_cloud(backend='numpy').backend, 'numpy')
        with self.assertRaises(Exception):
            _cloud(backend='jax')


class TestProperties(TestCase):
    '''Tests for coordinate and simplex properties.'''

    def setUp(self):
        self.cloud = _cloud()

    def test_no_properties_by_default(self):
        self.assertEqual(len(self.cloud.properties), 0)

    def test_withprop_creates_a_coordinate_property(self):
        c = self.cloud.withprop('temperature', zeros(4))
        self.assertIn('temperature', c.properties)
        # The original is unchanged.
        self.assertNotIn('temperature', self.cloud.properties)

    def test_lookup_returns_raw_values(self):
        c = self.cloud.withprop('temperature', array([1., 2., 3., 4.]))
        self.assertEqual(c['temperature'].tolist(), [1., 2., 3., 4.])

    def test_lookup_with_index(self):
        c = self.cloud.withprop('temperature', array([1., 2., 3., 4.]))
        self.assertEqual(c['temperature', 1].tolist(), 2.)
        self.assertEqual(c['temperature', [0, 2]].tolist(), [1., 3.])
        self.assertEqual(
            c['temperature', array([True, False, True, False])].tolist(),
            [1., 3.])

    def test_lookup_of_several_properties(self):
        c = (self.cloud
             .withprop('a', array([1., 2., 3., 4.]))
             .withprop('b', array([5., 6., 7., 8.])))
        (a, b) = c[['a', 'b']]
        self.assertEqual(a.tolist(), [1., 2., 3., 4.])
        self.assertEqual(b.tolist(), [5., 6., 7., 8.])

    def test_channel_dimensions_are_preserved_by_indexing(self):
        # A 3x4 property: three channels, one value per coordinate.
        vals = array([[1., 2., 3., 4.],
                      [5., 6., 7., 8.],
                      [9., 10., 11., 12.]])
        c = self.cloud.withprop('vec', vals)
        self.assertEqual(c['vec', 0].tolist(), [1., 5., 9.])
        self.assertEqual(c['vec', 0, 2].tolist(), 3.)

    def test_property_shape_is_validated(self):
        # A coordinate property must have exactly one value per coordinate.
        with self.assertRaises(Exception):
            self.cloud.withprop('bad', zeros(3))
        with self.assertRaises(Exception):
            self.cloud.withprop('bad', zeros(5))

    def test_missing_property_raises(self):
        with self.assertRaises(KeyError):
            self.cloud['nope']
        with self.assertRaises(KeyError):
            self.cloud.dropprop('nope')

    def test_dropprop(self):
        c = self.cloud.withprop('a', zeros(4))
        self.assertNotIn('a', c.dropprop('a').properties)

    def test_propinfo_returns_the_property_object(self):
        c = self.cloud.withprop('a', zeros(4), interp='linear')
        info = c.propinfo('a')
        self.assertIsInstance(info, Property)
        self.assertEqual(info.interp, 1)
        # ...while a lookup returns just the values.
        self.assertEqual(c['a'].tolist(), [0., 0., 0., 0.])

    def test_withmeta_updates_metadata_only(self):
        c = self.cloud.withprop('a', zeros(4), interp=1)
        d = c.withprop('a', interp=2)
        self.assertEqual(d.propinfo('a').interp, 2)
        self.assertEqual(d['a'].tolist(), c['a'].tolist())
        # Metadata cannot be set for a property that does not exist.
        with self.assertRaises(KeyError):
            self.cloud.withprop('nope', interp=1)

    def test_immutability(self):
        with self.assertRaises(TypeError):
            self.cloud['a'] = zeros(4)

    def test_properties_may_be_passed_at_construction(self):
        prop = Property(zeros(4), (4,))
        c = _cloud(properties={'a': prop})
        self.assertEqual(c['a'].tolist(), [0., 0., 0., 0.])

    def test_raw_values_are_rejected_at_construction(self):
        # Construction takes Property objects; withprop builds them.
        with self.assertRaises(Exception):
            _cloud(properties={'a': zeros(4)})

    def test_prop_returns_values(self):
        c = self.cloud.withprop('a', array([1., 2., 3., 4.]))
        self.assertEqual(c.prop('a').tolist(), [1., 2., 3., 4.])
        self.assertEqual(c.prop('a', at=[1, 2]).tolist(), [2., 3.])
        # Interpolation is not implemented yet, and says so.
        with self.assertRaises(NotImplementedError):
            c.prop('a', at=zeros((2, 2)))


class TestSimplexProperties(TestCase):
    '''Tests for properties attached to simplices rather than coordinates.'''

    def setUp(self):
        self.mesh = _mesh()

    def test_geometry_shape(self):
        self.assertEqual(self.mesh.order, 2)
        self.assertEqual(self.mesh.coord_count, 4)
        self.assertEqual(list(self.mesh.simplex_count), [4, 5, 2])

    def test_default_simplex_properties_are_empty(self):
        self.assertEqual(len(self.mesh.simplex_properties), 3)
        for props in self.mesh.simplex_properties:
            self.assertEqual(len(props), 0)

    def test_simplex_property_by_order(self):
        # One value per triangle (order 2), of which there are two.
        m = self.mesh.withprop((2, 'area'), array([0.5, 0.5]))
        self.assertEqual(m[2, 'area'].tolist(), [0.5, 0.5])
        self.assertIn('area', m.simplex_properties[2])
        # The original is unchanged.
        self.assertNotIn('area', self.mesh.simplex_properties[2])

    def test_dropprop_of_a_simplex_property(self):
        # simplex_properties is a sequence with one entry per order, so
        # removing a property must rebuild the sequence rather than install a
        # bare mapping in its place.
        m = (self.mesh
             .withprop((2, 'area'), array([0.5, 0.25]))
             .withprop((1, 'weight'), array([1., 2., 3., 4., 5.])))
        dropped = m.dropprop((2, 'area'))
        self.assertNotIn('area', dropped.simplex_properties[2])
        # The other orders survive, with their own properties intact.
        self.assertIn('weight', dropped.simplex_properties[1])
        self.assertEqual(dropped[1, 'weight'].tolist(), [1., 2., 3., 4., 5.])
        self.assertEqual(len(dropped.simplex_properties), 3)

    def test_simplex_property_shape_is_validated(self):
        with self.assertRaises(Exception):
            self.mesh.withprop((2, 'area'), array([0.5, 0.5, 0.5]))
        # An order-1 property needs one value per edge (five of them).
        with self.assertRaises(Exception):
            self.mesh.withprop((1, 'weight'), array([0.5, 0.5]))

    def test_order_out_of_range(self):
        with self.assertRaises(IndexError):
            self.mesh[3, 'area']
        with self.assertRaises(IndexError):
            self.mesh.withprop((3, 'area'), array([1.]))

    def test_vertex_property_falls_back_to_coordinate_property(self):
        # Without a vertex property, mesh[0, prop] is the coordinate property
        # at the coordinates the topology actually uses.
        m = self.mesh.withprop('a', array([1., 2., 3., 4.]))
        self.assertEqual(m[0, 'a'].tolist(), [1., 2., 3., 4.])
        # mesh[prop] remains the coordinate property.
        self.assertEqual(m['a'].tolist(), [1., 2., 3., 4.])

    def test_vertex_property_wins_over_coordinate_property(self):
        m = (self.mesh
             .withprop('a', array([1., 2., 3., 4.]))
             .withprop((0, 'a'), array([9., 9., 9., 9.])))
        self.assertEqual(m[0, 'a'].tolist(), [9., 9., 9., 9.])
        self.assertEqual(m['a'].tolist(), [1., 2., 3., 4.])

    def test_fallback_restricts_to_used_coordinates(self):
        # A topology that uses only three of six coordinates.
        coords = array([[0., 1., 2., 3., 4., 5.],
                        [0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 0.]])
        topo = MeshTopology([[0, 0], [1, 2], [2, 3]], coord_count=6)
        m = Cloud(coords, topo).withprop('a', array([1., 2., 3., 4., 5., 6.]))
        # Coordinates 4 and 5 are unused, so the vertex property omits them.
        self.assertEqual(m[0, 'a'].tolist(), [1., 2., 3., 4.])
        self.assertEqual(m['a'].tolist(), [1., 2., 3., 4., 5., 6.])


class TestEquality(TestCase):
    '''Tests for geometry equality and hashing.'''

    def test_identical_geometries_are_equal(self):
        self.assertEqual(_cloud(), _cloud())
        self.assertEqual(hash(_cloud()), hash(_cloud()))

    def test_different_coordinates_are_not_equal(self):
        a = _cloud()
        b = _cloud(coords=array([[0., 1., 2., 9.],
                                 [0., 0., 0., 0.]]))
        self.assertNotEqual(a, b)

    def test_properties_do_not_affect_equality(self):
        a = _cloud()
        b = a.withprop('a', zeros(4))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_unused_coordinates_do_not_affect_equality(self):
        # The README requires that extra, unreferenced coordinates are not
        # compared when objects are examined for equality.
        c1 = _cloud(coords=array([[0., 1., 2., 3., 100.],
                                  [0., 0., 0., 0., 100.]]),
                    topo=CloudTopology([[0, 1, 2, 3]], coord_count=5))
        c2 = _cloud(coords=array([[0., 1., 2., 3., -7.],
                                  [0., 0., 0., 0., -7.]]),
                    topo=CloudTopology([[0, 1, 2, 3]], coord_count=5))
        self.assertEqual(c1, c2)
        self.assertEqual(hash(c1), hash(c2))

    def test_declared_coordinate_count_does_not_affect_equality(self):
        # The same connectivity and the same used coordinates identify the same
        # geometry, even when one topology declares room for extra coordinates.
        small = _cloud()
        big = _cloud(coords=array([[0., 1., 2., 3., 99.],
                                   [0., 0., 0., 0., 99.]]),
                     topo=CloudTopology([[0, 1, 2, 3]], coord_count=5))
        self.assertEqual(small, big)
        self.assertEqual(hash(small), hash(big))

    def test_different_topologies_are_not_equal(self):
        a = _cloud()
        b = _cloud(topo=CloudTopology([[0, 1, 2], [3, 3, 3]]))
        self.assertNotEqual(a, b)

    def test_compared_against_another_type(self):
        self.assertNotEqual(_cloud(), 'not-a-geometry')


class TestNameSplitting(TestCase):
    '''Tests for the ``(order, name)`` property-name convention.'''

    def test_bare_name(self):
        self.assertEqual(split_property_name('flux'), (None, 'flux'))

    def test_order_and_name(self):
        self.assertEqual(split_property_name((2, 'flux')), (2, 'flux'))

    def test_none_or_ellipsis_order(self):
        self.assertEqual(split_property_name((None, 'flux')), (None, 'flux'))
        self.assertEqual(split_property_name((Ellipsis, 'flux')),
                         (None, 'flux'))

    def test_two_element_name_that_is_not_a_pair(self):
        # A 2-tuple whose first element is not an order is a bare name.
        self.assertEqual(split_property_name(('a', 'b')), (None, ('a', 'b')))
