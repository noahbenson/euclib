# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/test_readme.py
'''Tests that ``euclib`` behaves the way ``README.md`` says it does.

The README is the specification: it describes the shapes, the conventions, and
the property interface that the library is answerable for. This module encodes
its examples and its checkable claims as literal tests, so that the text and the
behavior cannot drift apart without one of them failing. Each test names the
README line it comes from, so a failure points at the paragraph that needs
correcting as well as at the code.

Where the README describes something the library does not do yet, the test
*skips* with the reason naming what is missing, rather than asserting the
current behavior as though it were right. A skip is visible in the run and says
which promise is outstanding; an assertion of the wrong thing would hide it.

A note on the README's "API and Usage Examples" section: it holds the import
and nothing else, so the worked examples live in the documentation gallery
under ``docs/`` rather than here. This module covers the *specification*, which
is the prose and the inline examples above that section.
'''

# Dependencies ###############################################################

from __future__ import annotations

import inspect
from unittest import TestCase

import numpy as np

import euclib as el
from euclib import Property
from euclib.abc import SimplexGeometry, SimplexTopology


# Fixtures ###################################################################

def _mesh(*, extra=False):
    '''The two-triangle square, holding a coordinate property ``'x'``.

    With ``extra``, a fifth coordinate is added that no triangle uses; the
    README says such coordinates are allowed and are not compared when
    objects are tested for equality.
    '''
    coords = np.array([[0., 1., 0., 1.], [0., 0., 1., 1.]])
    values = np.array([10., 20., 30., 40.])
    if extra:
        coords = np.concatenate([coords, np.array([[9.], [9.]])], axis=1)
        values = np.concatenate([values, np.array([50.])])
    mesh = el.trimesh(coords, [[0, 0], [1, 3], [3, 2]])
    return mesh.withprop('x', values)


def _innermost(call, /):
    '''Returns the innermost exception a call raises, or ``None``.

    A property's fields are validated by ``calc``s, and ``immlib`` reports a
    failure in one of those as a ``PlanError`` that wraps the error the caller
    caused. The README documents the errors themselves --- a ``ValueError`` for
    an invalid order, a ``NotImplementedError`` for an unimplemented one --- so
    the wrapper is unwrapped and the cause is what these tests check.
    '''
    try:
        call()
    except Exception as exc:
        while exc.__cause__ is not None:
            exc = exc.__cause__
        return exc
    return None


# Getting started ############################################################

class TestGettingStarted(TestCase):
    '''README 455-462.'''

    def test_the_library_imports_under_its_documented_name(self):
        # The README's one worked example.
        self.assertTrue(hasattr(el, 'types'))
        self.assertEqual(el.__name__, 'euclib')


# Coordinates and shapes #####################################################

class TestCoordinatesAndShapes(TestCase):
    '''README 89-99 and 44-59.'''

    def test_a_2d_collection_is_a_2_by_n_matrix(self):
        cloud = el.points(np.array([[0., 1., 2.], [0., 0., 0.]]))
        self.assertEqual(tuple(cloud.coords.shape), (2, 3))

    def test_a_3d_collection_is_a_3_by_n_matrix(self):
        cloud = el.points(np.zeros((3, 4)))
        self.assertEqual(tuple(cloud.coords.shape), (3, 4))

    def test_a_property_ends_in_the_coordinate_count(self):
        # README 54-56: a property encoding a shape ``A...`` per vertex has a
        # shape of ``A..., N``.
        mesh = _mesh().withprop('pair', np.arange(8.).reshape(2, 4))
        self.assertEqual(tuple(np.asarray(mesh['pair']).shape), (2, 4))

    def test_a_grid_property_puts_its_channels_first(self):
        # README 56-59: a 2x3 Jacobian per voxel of an (R, C, S) grid has the
        # shape (2, 3, R, C, S).
        grid = el.grid((10, 12, 15))
        grid = grid.withprop('jacobian', np.zeros((2, 3, 10, 12, 15)))
        self.assertEqual(tuple(np.asarray(grid['jacobian']).shape),
                         (2, 3, 10, 12, 15))

    def test_one_voxel_of_such_a_property_unpacks_as_documented(self):
        # README 236-240, which unpacks the Jacobian of the first voxel into a
        # nested tuple of six names. That example's Jacobian is a 3 by 2 --- the
        # derivative of three outputs with respect to two inputs --- while the
        # shape example above uses a 2 by 3; the two demonstrate the same
        # channels-first ordering with different numbers.
        values = np.zeros((3, 2, 10, 12, 15))
        values[:, :, 0, 0, 0] = [[1., 2.], [3., 4.], [5., 6.]]
        grid = el.grid((10, 12, 15)).withprop('jacobian', values)
        ((dfx_dx, dfx_dy), (dfy_dx, dfy_dy), (dfz_dx, dfz_dy)) = \
            grid['jacobian', 0, 0, 0]
        self.assertEqual([float(v) for v in (dfx_dx, dfy_dx, dfz_dx)],
                         [1., 3., 5.])
        self.assertEqual([float(v) for v in (dfx_dy, dfy_dy, dfz_dy)],
                         [2., 4., 6.])

    def test_coordinates_may_be_given_as_a_mapping(self):
        # README 94-99: individual points and collections of points can also be
        # represented as a mapping with keys 'x' and 'y' (and 'z').
        cloud = el.points({'x': [0., 1.], 'y': [0., 0.]})
        self.assertTrue(np.allclose(np.asarray(cloud.coords),
                                    [[0., 1.], [0., 0.]]))
        # A mapping of scalars is a single point, which the library reads as a
        # one-column matrix.
        point = el.points({'x': 2., 'y': 3.})
        self.assertEqual(tuple(point.coords.shape), (2, 1))
        # Every simplex constructor reads them, and so does a topology that
        # needs the coordinate count.
        mesh = el.trimesh({'x': [0., 1., 0.], 'y': [0., 0., 1.]},
                          [[0], [1], [2]])
        self.assertEqual(tuple(mesh.coords.shape), (2, 3))
        self.assertEqual(mesh.coord_count, 3)
        solid = el.tetmesh({'x': [0., 1., 0., 0.], 'y': [0., 0., 1., 0.],
                            'z': [0., 0., 0., 1.]}, [[0], [1], [2], [3]])
        self.assertEqual(tuple(solid.coords.shape), (3, 4))
        path = el.segpath({'x': [0., 1., 2.], 'y': [0., 0., 0.]})
        self.assertEqual(path.topo.simplex_count[1], 2)
        # A key that is not an axis is refused rather than ignored.
        with self.assertRaises(ValueError):
            el.points({'w': [0., 1.], 'y': [0., 0.]})
        # ...as is a mapping that names no axis at all.
        with self.assertRaises(ValueError):
            el.points({'w': [0., 1.]})

    def test_the_mapping_conversion_is_available_to_callers(self):
        # The conversion is one function, so a caller who wants the matrix a
        # mapping stands for does not have to construct a geometry to get it.
        coords = el.abc.as_coords({'x': np.array([1., 2.]),
                                   'z': np.array([3., 4.])})
        self.assertTrue(np.allclose(coords, [[1., 2.], [3., 4.]]))
        # A value that is not a mapping passes through untouched.
        matrix = np.zeros((2, 3))
        self.assertIs(el.abc.as_coords(matrix), matrix)


# Copies, not mutation #######################################################

class TestImmutability(TestCase):
    '''README 20-23 and 76-84.'''

    def test_changing_coordinates_produces_a_copy(self):
        mesh = _mesh()
        moved = mesh.copy(coords=mesh.coords + 1.0)
        self.assertTrue(np.allclose(np.asarray(mesh.coords) + 1.0,
                                    np.asarray(moved.coords)))
        self.assertFalse(np.allclose(np.asarray(mesh.coords),
                                     np.asarray(moved.coords)))

    def test_the_original_is_never_changed(self):
        mesh = _mesh()
        marked = mesh.withprop('y', np.ones(4))
        self.assertNotIn('y', mesh.properties)
        self.assertIn('y', marked.properties)


# Local coordinates ##########################################################

class TestLocalCoordinates(TestCase):
    '''README 101-118.'''

    def test_a_triangle_mesh_locates_a_point_by_triangle_and_two_weights(self):
        # README 110-113: "the index of the triangle and the first two
        # barycentric coordinates within it".
        loc = _mesh().to_local(np.array([[0.2], [0.2]]))
        self.assertEqual(loc._fields, ('index', 'weight'))
        self.assertEqual(np.shape(loc.weight), (2, 1))

    def test_global_and_local_arguments_are_named_as_documented(self):
        # README 115-118: functions expecting local coordinates use `locs` and
        # those expecting global coordinates use `coords`.
        for method in (SimplexGeometry.to_local, SimplexGeometry.to_global):
            with self.subTest(method=method.__name__):
                params = [p for p in inspect.signature(method).parameters
                          if p != 'self']
                self.assertEqual(params,
                                 ['coords' if method.__name__ == 'to_local'
                                  else 'locs'])


# Geometric objects and topologies ###########################################

class TestObjectsAndTopologies(TestCase):
    '''README 121-134 and 385-426.'''

    def test_every_geometry_has_a_topology(self):
        for geom in (_mesh(), el.points(np.zeros((2, 3))),
                     el.grid((2, 2)),
                     el.prismmesh(np.zeros((2, 3, 3)), [[0], [1], [2]])):
            with self.subTest(geom=type(geom).__name__):
                self.assertTrue(hasattr(geom, 'topo'))

    def test_the_abstract_bases_are_as_documented(self):
        self.assertTrue(issubclass(el.TriMesh, SimplexGeometry))
        self.assertTrue(issubclass(el.TriTopology, SimplexTopology))
        self.assertTrue(issubclass(el.TriMesh, el.abc.Geometry))

    def test_a_point_clouds_indices_name_every_coordinate(self):
        # README 396-398: "a 1 x N matrix ... typically just np.arange(N)".
        cloud = el.points(np.zeros((2, 5)))
        self.assertEqual(np.asarray(cloud.topo.indices).shape, (1, 5))
        self.assertEqual(np.asarray(cloud.topo.indices).ravel().tolist(),
                         list(range(5)))

    def test_simplices_hold_one_entry_per_order(self):
        # README 409-415: an llist whose length is the order plus one, index 2
        # being the triangles themselves, identical to `indices`.
        mesh = _mesh()
        self.assertEqual(len(mesh.topo.simplices), mesh.topo.order + 1)
        self.assertTrue(np.array_equal(np.asarray(mesh.topo.simplices[2]),
                                       np.asarray(mesh.topo.indices)))

    def test_simplex_count_counts_the_simplices(self):
        # README 425-426.
        mesh = _mesh()
        self.assertEqual(
            np.asarray(mesh.topo.simplex_count).tolist(),
            [int(np.shape(s)[-1]) for s in mesh.topo.simplices])

    def test_extra_coordinates_are_allowed_and_ignored(self):
        # README 417-420.
        with_extra = _mesh(extra=True)
        self.assertEqual(with_extra.topo.coord_count, 5)
        self.assertEqual(with_extra.topo.vertex_count, 4)
        self.assertEqual(with_extra, _mesh())

    def test_simplex_properties_are_one_dictionary_per_order(self):
        # README 435-437 says a tuple of ldicts, one per simplex type. The
        # container is an llist rather than a tuple --- the same lazy sequence
        # that `simplices` uses --- so what is checked is the structure the
        # README describes rather than the Python type it names.
        mesh = _mesh().withprop((2, 'flux'), np.array([1.5, 2.5]))
        props = mesh.simplex_properties
        self.assertEqual(len(props), mesh.topo.order + 1)
        self.assertEqual(np.asarray(props[2]['flux'].value).tolist(), [1.5, 2.5])


# Properties #################################################################

class TestProperties(TestCase):
    '''README 194-254 and 306-380.'''

    def test_a_property_may_be_looked_up_by_name(self):
        self.assertEqual(np.asarray(_mesh()['x']).tolist(),
                         [10., 20., 30., 40.])

    def test_several_properties_may_be_looked_up_at_once(self):
        # README 205-208.
        mesh = _mesh().withprop('y', np.array([1., 2., 3., 4.]))
        (x, y) = mesh[['x', 'y']]
        self.assertEqual(np.asarray(x).tolist(), [10., 20., 30., 40.])
        self.assertEqual(np.asarray(y).tolist(), [1., 2., 3., 4.])

    def test_a_mask_indexes_a_property_and_keeps_its_channels(self):
        # README ~228: "mesh['flux', ii] for a boolean mask ii ... returns the
        # property for the indices given by ii ... still ... with its typical
        # shape", meaning the channel dimensions stay put.
        mesh = _mesh().withprop('pair', np.arange(8.).reshape(2, 4))
        keep = np.array([True, False, True, False])
        self.assertEqual(tuple(np.asarray(mesh['pair', keep]).shape), (2, 2))
        self.assertEqual(np.asarray(mesh['pair', keep]).tolist(),
                         [[0., 2.], [4., 6.]])

    def test_the_property_extraction_method_has_its_documented_signature(self):
        # README 276-280.
        self.assertEqual(str(inspect.signature(el.Geometry.prop)),
                         '(self, property, /, at=Ellipsis, **kw)')

    def test_a_mask_and_a_null_value_poison_an_interpolation(self):
        # README 355-366 and 367-372: a value interpolated from an invalid
        # vertex returns the null value instead.
        mesh = _mesh().withprop('param', np.array([0., 1., 2., 3.]),
                                mask=np.array([True, False, True, True]))
        at = np.array([[0.3], [0.3]])
        self.assertTrue(np.isnan(float(mesh.prop('param', at=at)[0])))
        unmasked = _mesh().withprop('param', np.array([0., 1., 2., 3.]))
        self.assertFalse(np.isnan(float(unmasked.prop('param', at=at)[0])))

    def test_metadata_may_be_updated_without_new_values(self):
        # README 374-380.
        mesh = _mesh().withprop('x', interp=1)
        self.assertEqual(np.asarray(mesh['x']).tolist(), [10., 20., 30., 40.])
        self.assertEqual(mesh.properties['x'].interp, ('polynomial', 1))

    def test_a_simplex_property_is_named_by_an_order_and_a_name(self):
        # README 321-326.
        mesh = _mesh().withprop((2, 'surface_area'), np.array([0.5, 0.5]))
        self.assertEqual(np.asarray(mesh[2, 'surface_area']).tolist(),
                         [0.5, 0.5])
        self.assertNotIn('surface_area', mesh.properties)

    def test_adding_and_removing_return_copies(self):
        # README 321-324.
        mesh = _mesh()
        marked = mesh.withprop('y', np.ones(4))
        self.assertNotIn('y', mesh.properties)
        self.assertNotIn('y', marked.dropprop('y').properties)


class TestPropertyMetadata(TestCase):
    '''README 331-372.'''

    def test_interpolation_orders_are_recognized(self):
        # README 334-338: 'None, 0, 1, 2, and 3' for no interpolation through
        # cubic. Every one of them is *recognized* here. What is built is not a
        # property's business --- it does not know what it will be attached to,
        # and the same combination may be built for one element and not another
        # --- so a geometry is what refuses one it cannot honour, and it does so
        # when the property is attached.
        self.assertEqual(Property(np.ones(4), (4,), interp=0).interp,
                         ('nearest', 0))
        self.assertEqual(Property(np.ones(4), (4,), interp=1).interp,
                         ('polynomial', 1))
        for order in (2, 3):
            with self.subTest(order=order):
                self.assertEqual(
                    Property(np.ones(4), (4,), interp=order).interp,
                    ('polynomial', order))
        # An order outside the range, or a method euclib does not define, is a
        # mistake rather than a gap, and is a `ValueError`.
        for bad in (5, 'cubic-spline'):
            with self.subTest(bad=bad):
                self.assertIsInstance(
                    _innermost(lambda: Property(np.ones(4), (4,), interp=bad)),
                    ValueError)

    def test_the_default_interpolation_is_the_one_the_readme_promises(self):
        # README 339-341 says Ellipsis means nearest-neighbor for a
        # non-continuous value "and otherwise cubic (3) interpolation". The
        # library's default for continuous data is linear, which is a
        # placeholder: the higher orders are not implemented yet, and the default
        # returns to a higher one when they are (see euclib._init, where
        # default_quantitative_interp says so, and the plan's note on the
        # quadratic/cubic basis). This test says so rather than encoding either
        # answer as the truth.
        self.assertEqual(Property(np.arange(4), (4,)).interp, ('nearest', 0))
        self.assertEqual(Property(np.ones(4), (4,)).interp, ('polynomial', 1))
        self.skipTest(
            "README 339-341 promises cubic (3) as the default for continuous "
            "data; ('polynomial', 1) stands in until the higher orders are "
            "implemented, at which point the default moves and this skip goes "
            "away.")

    def test_extrapolation_may_be_none_or_zero(self):
        # README 345-352.
        self.assertIsNone(Property(np.ones(4), (4,)).extrap)
        self.assertEqual(Property(np.ones(4), (4,), extrap=0).extrap, 0)
        self.assertIsInstance(
            _innermost(lambda: Property(np.ones(4), (4,), extrap=1)),
            ValueError)

    def test_the_null_value_follows_the_dtype(self):
        # README 367-370: NaN for floating point, dtype.type() otherwise.
        self.assertTrue(np.isnan(Property(np.ones(4), (4,)).null))
        self.assertEqual(Property(np.arange(4), (4,)).null, 0)

    def test_the_dtype_defaults_to_the_values_own(self):
        # README 353-354.
        prop = Property(np.ones(4, dtype='float32'), (4,))
        self.assertEqual(np.dtype(prop.value.dtype), np.dtype('float32'))


# Element access and simplex properties ######################################

class TestItemAccess(TestCase):
    '''README 439-452.'''

    def test_a_bare_name_is_the_coordinate_property(self):
        # README 442-443.
        self.assertEqual(np.asarray(_mesh()['x']).shape, (4,))

    def test_an_order_and_a_name_is_the_simplex_property(self):
        # README 444-446: mesh[2, 'x'] is mesh.simplex_properties[2]['x'].
        # README 444 says the two are equivalent. They are, in that both name
        # the same property, but `simplex_properties` holds Property objects and
        # `[]` returns values, so the comparison is against the Property's value.
        mesh = _mesh().withprop((2, 'flux'), np.array([1.5, 2.5]))
        self.assertEqual(
            np.asarray(mesh[2, 'flux']).tolist(),
            np.asarray(mesh.simplex_properties[2]['flux'].value).tolist())

    def test_order_zero_restricts_an_unused_coordinate_away(self):
        # README 446-450: mesh[0, prop] extracts the property for the points
        # the mesh uses, where mesh[prop] extracts it for every coordinate.
        mesh = _mesh(extra=True)
        self.assertEqual(np.asarray(mesh['x']).shape, (5,))
        self.assertEqual(np.asarray(mesh[0, 'x']).tolist(),
                         [10., 20., 30., 40.])

    def test_order_zero_falls_back_to_a_coordinate_property(self):
        # README 450-452: if a property exists for the coordinates but not for
        # the vertices, mesh[0, prop] extracts the coordinate property at the
        # active vertices.
        mesh = _mesh(extra=True)
        self.assertEqual(np.asarray(mesh[0, 'x']).tolist(),
                         [10., 20., 30., 40.])

    def test_ellipsis_is_ignored_where_an_order_may_be_given(self):
        # README 291-293.
        mesh = _mesh()
        self.assertEqual(np.asarray(mesh[Ellipsis, 'x']).tolist(),
                         np.asarray(mesh['x']).tolist())

    def test_an_order_of_none_names_the_coordinate_property(self):
        # README 286-290: the order may be 0, 1, 2, 3, or None, the last
        # meaning the property belongs to the object's coordinates.
        mesh = _mesh()
        self.assertEqual(np.asarray(mesh[None, 'x']).tolist(),
                         np.asarray(mesh['x']).tolist())


# Prisms #####################################################################

class TestPrismProperties(TestCase):
    '''README 255-272.'''

    def _prism(self):
        lower = np.array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        return el.prismmesh(np.stack([lower, lower + 1.]), [[0], [1], [2]])

    def test_a_prism_property_may_carry_an_elevation_dimension(self):
        # README 265-266: a matrix of values with shape (100, N), the 100 being
        # the elevation dimension.
        values = np.tile(np.arange(100, dtype=float)[:, None], (1, 3))
        prism = self._prism().withprop('temperature', values)
        self.assertEqual(tuple(np.asarray(prism['temperature']).shape), (100, 3))

    def test_elevations_may_be_given_explicitly(self):
        # README 266-272: a property may be a tuple (elevs, values).
        prism = self._prism().withprop(
            'temperature', (np.array([0., 1.]), np.zeros((2, 3))))
        self.assertEqual(tuple(np.asarray(prism['temperature']).shape), (2, 3))


# Grids ######################################################################

class TestGrids(TestCase):
    '''README 166-192.'''

    def test_the_affine_carries_an_index_to_a_cell_center(self):
        # README 182-192: the affine describes how image indices are
        # translated into global coordinates, one index to the center of the
        # cell it numbers.
        grid = el.grid((2, 2, 2))
        centers = np.asarray(el.ops.positions_of(grid))
        self.assertTrue(np.allclose(centers[:, 0], [0., 0., 0.]))

    def test_the_grid_is_specified_by_its_corner(self):
        # README 168-171: a grid is specified by the global coordinates of the
        # corner of the grid, its spacing, its orientation, and its dimensions.
        grid = el.grid((2, 2, 2))
        self.assertTrue(np.allclose(grid.origin, [-0.5, -0.5, -0.5]))
        self.assertTrue(np.allclose(grid.spacing, [1., 1., 1.]))
        self.assertEqual(grid.shape, (2, 2, 2))
