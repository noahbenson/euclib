# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/test_prism.py
'''Tests for prism meshes: ``PrismTopology`` and ``PrismMesh``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import (allclose, array, asarray, concatenate, linalg,
                   mean as np_mean, ones, stack, zeros)

from euclib.abc import is_geometry, is_simplex_topology
from euclib.types import (
    PrismMesh, PrismTopology, PrismLoc, TetMesh, TriMesh)
from euclib.types._geom import prism_layer_values


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

    def test_the_decomposition_fills_a_stack_of_layers(self):
        '''The same fan, between every pair of adjacent layers.

        A property may carry more elevations than the geometry has surfaces, and
        the way those are interpolated is to fill the *stack* of layers with the
        same three tetrahedra per pair --- so that a position's height falls
        between the two layers its own tetrahedron has for corners, and the
        blend is one the tetrahedral methods already give. The check is that
        the pieces fill the stack: their volumes must add to its.

        With uneven spacing, because real elevations are uneven, and that is the
        case a fixed layer size would get wrong.
        '''
        from euclib.types._topo import prism_tetrahedra
        lower = array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        volume = 0.5                       # a unit right triangle, one tall

        def tet_volume(corners, /):
            (a, b, c, d) = corners
            return abs(linalg.det(stack([b - a, c - a, d - a]))) / 6.0

        for elevations in (array([0.0, 1.0]), array([0.0, 0.5, 1.0]),
                           array([0.0, 0.2, 0.9, 1.0]),
                           array([0.0, 0.1, 0.2, 0.3, 1.0])):
            coords = concatenate([lower + array([[0.], [0.], [t]])
                                  for t in elevations], axis=1)
            tets = prism_tetrahedra(array([[0], [1], [2]]), len(elevations), 3)
            total = sum(tet_volume([coords[:, i] for i in column])
                        for column in asarray(tets).T.tolist())
            want = volume * (elevations[-1] - elevations[0])
            with self.subTest(layers=len(elevations)):
                self.assertEqual(asarray(tets).shape[1], 3 * (len(elevations) - 1))
                self.assertLess(abs(total - want), 1e-12,
                                "the tetrahedra do not fill the stack")

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


class TestTheLayerCache(TestCase):
    '''The stack of layers a property's elevations ask for.

    A property may name more elevations than the geometry has surfaces, and
    interpolating one means filling the *stack* with tetrahedra --- one layer per
    elevation, and `prism_tetrahedra`\'s fan between each pair. Then a position\'s
    height falls between two layers, and those are the two layers the tetrahedron
    it lies in has for corners, so the blend the property wants comes out of the
    tetrahedral methods rather than from anything written for the purpose.

    It is worth building once: two properties naming the same elevations want the
    same stack, and a mesh may be read repeatedly. The cache is keyed by the
    elevation *vector* as a tuple, and a property with *matrix* elevations --- one
    per position, which is rare --- is left out, having no single stack to build.
    '''

    def _prism(self, /):
        lower = array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        return (PrismMesh(stack([lower, lower + array([[0.], [0.], [1.]])]),
                          PrismTopology([[0], [1], [2]])),
                lower)

    #: A field that bends in z, so no fit of two surfaces can hold it: the
    #: *middle* layer is the point. A lambda in a class body is bound as a
    #: method, so it is declared static rather than taking `self`.
    FIELD = staticmethod(lambda p, t: 2.0 * p[0] + 3.0 * p[1] + t ** 2)

    def test_it_builds_one_stack_per_elevation_vector(self):
        (mesh, lower) = self._prism()
        ev = array([0.0, 0.5, 1.0])
        vals = array([[[self.FIELD(lower[:, i], t) for i in range(3)]
                       for t in ev]])
        carried = mesh.withprop('v', (ev, vals))
        self.assertEqual(sorted(carried._tetlayer_cache), [(0.0, 0.5, 1.0)])
        self.assertEqual([n for (n, _) in carried._tetlayer_cache.values()],
                         [('v',)])
        # Two properties naming the same elevations share the one stack.
        two = carried.withprop('w', (ev, vals * 2.0))
        self.assertEqual(len(two._tetlayer_cache), 1)
        self.assertEqual([n for (n, _) in two._tetlayer_cache.values()],
                         [('v', 'w')])
        # And one with its own elevations gets its own.
        three = two.withprop('u', (array([0.0, 1.0]), zeros((2, 3))))
        self.assertEqual(len(three._tetlayer_cache), 2)

    def test_the_stack_is_the_layers_and_the_fan_between_them(self):
        (mesh, lower) = self._prism()
        ev = array([0.0, 0.5, 1.0])
        carried = mesh.withprop('v', (ev, zeros((1, 3, 3))))
        layer = carried.tetlayer(ev)
        self.assertIsInstance(layer, TetMesh)
        self.assertEqual(layer.coord_count, 9)          # three layers of three
        self.assertEqual(layer.simplex_count[layer.order], 6)   # three per pair
        # Each layer is the two surfaces blended, which is what `elevation` does.
        coords = asarray(layer.coords)
        for (k, t) in enumerate(ev):
            self.assertTrue(allclose(coords[:, 3 * k:3 * (k + 1)],
                                     lower * (1.0 - t)
                                     + (lower + array([[0.], [0.], [1.]])) * t))
        # And the same stack comes back, however often it is asked for.
        self.assertIs(carried.tetlayer(ev), layer)

    def test_it_interpolates_the_elevations_it_was_given(self):
        '''The point of the exercise, and checked against the field.

        At an elevation the answer must be that elevation's own values, exactly;
        between two the tetrahedron blends them linearly, because that is what a
        linear fit does between its own corners --- and the blend's value is known
        independently, so it is not merely self-consistent.
        '''
        (mesh, lower) = self._prism()
        ev = array([0.0, 0.5, 1.0])
        vals = array([[[self.FIELD(lower[:, i], t) for i in range(3)]
                       for t in ev]])
        carried = mesh.withprop('v', (ev, vals))
        # The property's values are (elevation, corner); the stack's coordinates
        # are the layers end to end, so the two orders agree under a reshape.
        layer = carried.tetlayer(ev).withprop(
            'v', vals.reshape(1, -1), gradient=zeros((1, 3, vals.size)))

        for (x, y) in ((0.33, 0.33), (0.2, 0.5)):
            for z in (0.0, 0.5, 1.0):
                at = array([[x], [y], [z]])
                got = layer.prop('v', at=at, interp=('polynomial', 1))
                got = float(mag(got).ravel()[0])
                with self.subTest(point=(x, y, z)):
                    self.assertLess(abs(got - self.FIELD(array([x, y]), z)),
                                    1e-12,
                                    "an elevation's own values did not come back")
            # Between layers the blend is the mean of the two nearest, which for
            # this field is known: (f(0) + f(0.5)) / 2 at the midpoint.
            at = array([[x], [y], [0.25]])
            got = float(mag(layer.prop('v', at=at,
                                       interp=('polynomial', 1))).ravel()[0])
            want = 0.5 * (self.FIELD(array([x, y]), 0.0)
                          + self.FIELD(array([x, y]), 0.5))
            with self.subTest(point=(x, y, 0.25)):
                self.assertLess(abs(got - want), 1e-12,
                                "the blend between layers is not linear")


def mag(one, /):
    '''The magnitude, whether the argument is a quantity or an
    array.'''
    return asarray(one.m) if hasattr(one, 'm') else asarray(one)


class TestPrismInterpolation(TestCase):
    '''What a prism will and will not interpolate.

    A prism takes the first order and refuses the rest. It is not a simplex, so
    there is no element to fit the higher orders on; and a property of a prism
    may carry more elevations than the geometry has surfaces, so its values have
    an axis that a fit of the geometry's two surfaces cannot see. Both are
    *deferred* rather than refused by design --- see the roadmap --- and until
    they are built, the refusal has to say so rather than fail somewhere deep:
    it used to accept the request and raise `LazyError` from the gradient
    estimate, which names neither the method nor the reason.
    '''

    def _mesh(self, /):
        lower = array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        return PrismMesh(stack([lower, lower + array([[0.], [0.], [1.]])]),
                         PrismTopology([[0], [1], [2]]))

    def test_it_declares_what_it_can_honour(self):
        from euclib.abc import supported_interp
        mesh = self._mesh()
        self.assertEqual(supported_interp(mesh.topo),
                         (('nearest', 0), ('polynomial', 1), ('bezier', 1)),
                         "a prism claims an interpolation it cannot honour")

    def test_the_first_order_answers_and_the_rest_refuses(self):
        mesh = self._mesh()
        at = np_mean(concatenate([mesh.coords0, mesh.coords1], axis=1),
                     axis=1)[:, None]
        carried = mesh.withprop('v', ones(3))
        for method in (('nearest', 0), ('polynomial', 1), ('bezier', 1)):
            with self.subTest(method=method):
                carried.prop('v', at=at, interp=method)
        for method in (('polynomial', 2), ('polynomial', 3), ('bezier', 2),
                       ('bezier', 3)):
            with self.subTest(method=method):
                with self.assertRaises(NotImplementedError) as caught:
                    carried.prop('v', at=at, interp=method)
                self.assertIn('not implemented', str(caught.exception).lower())


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


class TestTheValueDistribution(TestCase):
    '''Giving a stack of layers the values its vertices have.

    A prism property's values are per *triangle*, per elevation, per corner ---
    ``(C..., K, M, 3)`` --- and a stack's tetrahedra are built from the prism
    mesh's own *coordinates*. A coordinate is shared by every triangle that has
    it for a corner, so the value a vertex takes has to be read off the
    triangles that name it.

    The reading is the **mean**, and the interesting case is the one where the
    triangles disagree: for a continuous property they agree and the mean is the
    value, but an inconsistent input has no "first" triangle to take it from
    that is not an arbitrary choice, and averaging is the symmetric answer. That
    is what these check, since it is the choice a reader would want to see
    argued rather than buried.
    '''

    #: Two triangles sharing coordinate 2, over five coordinates.
    INDICES = array([[0, 2], [1, 3], [2, 4]])

    def _values(self, /):
        # (C=1, K=1, M=2, 3): triangle 0 says 1, 2, 3 and triangle 1 says 3, 4, 5.
        return array([[[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]]])

    def test_each_coordinate_takes_what_its_triangles_say(self):
        got = asarray(prism_layer_values(self._values(), self.INDICES, 5))
        self.assertEqual(got.shape, (1, 1, 5))
        # Coordinate 2 is named by both, with 3 either time.
        self.assertTrue(allclose(got.ravel(), [1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_a_coordinate_two_triangles_disagree_about_is_averaged(self):
        values = self._values()
        values[0, 0, 1, 0] = 30.0            # now triangle 1 says 30 and 3
        got = asarray(prism_layer_values(values, self.INDICES, 5))
        self.assertTrue(allclose(got.ravel(), [1.0, 2.0, 16.5, 4.0, 5.0]),
                        "the shared coordinate did not take the mean")

    def test_the_elevation_axis_is_carried_through(self):
        two = concatenate([self._values(), self._values() + 10.0], axis=1)
        got = asarray(prism_layer_values(two, self.INDICES, 5))
        self.assertEqual(got.shape, (1, 2, 5))
        self.assertTrue(allclose(got[0, 0], [1.0, 2.0, 3.0, 4.0, 5.0]))
        self.assertTrue(allclose(got[0, 1], [11.0, 12.0, 13.0, 14.0, 15.0]))

    def test_the_channels_lead(self):
        chan = concatenate([self._values(), self._values() * 2.0], axis=0)
        got = asarray(prism_layer_values(chan, self.INDICES, 5))
        self.assertEqual(got.shape, (2, 1, 5))
        self.assertTrue(allclose(got[1, 0], [2.0, 4.0, 6.0, 8.0, 10.0]))


class TestReadingThroughTheStack(TestCase):
    '''A prism whose property names more than two elevations.

    A prism property may carry a *vector* of elevations --- more layers than the
    geometry's two surfaces --- and there is no fit of the surfaces that can hold
    the values on the layers between. What can hold them is the stack of
    tetrahedra between the layers, which is a `TetMesh` like any other: so the
    answer is the stack's own interpolation, with the property's values put onto
    its vertices and the position carried out to the stack's coordinates.

    The checks here are the two the design turns on --- that a value *on* a layer
    comes back exactly, since that layer is a corner of some tetrahedron, and
    that a value *between* two layers comes back as their linear blend. A field
    that bends in z makes both visible: nothing that fits only the two surfaces
    can reproduce the middle of it.
    '''

    #: The elevations the property below is given.
    ELEVATIONS = array([0.0, 0.5, 1.0])

    def _prism(self, /):
        '''A unit prism whose property bends in z.'''
        lower = array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        mesh = PrismMesh(stack([lower, lower + array([[0.], [0.], [1.]])]),
                         PrismTopology([[0], [1], [2]]))
        # (C..., K, M, 3): no channels, a row per elevation, one triangle.
        values = array([[self._field(lower[:, i], t) for i in range(3)]
                        for t in self.ELEVATIONS]).reshape(3, 1, 3)
        return mesh.withprop('v', (self.ELEVATIONS, values))

    @staticmethod
    def _field(point, height, /):
        '''A field that bends in z: ``2x + 3y + z^2``.'''
        return 2.0 * point[0] + 3.0 * point[1] + height ** 2

    @staticmethod
    def _at(mesh, height, /):
        '''The centre of the prism's only triangle, at a height.'''
        return mesh.topo.Loc(index=array([0]),
                             weight=array([[1.0 / 3.0], [1.0 / 3.0]]),
                             height=array([[height]]))

    def _read(self, mesh, height, interp, /):
        res = mesh.prop('v', at=self._at(mesh, height), interp=interp)
        return float(asarray(getattr(res, 'm', res)).ravel()[0])

    def test_a_value_on_a_layer_comes_back_exactly(self):
        mesh = self._prism()
        # The centre of the triangle is the mean of its three corners, and the
        # field is affine in x and y, so what belongs there is the field at the
        # centre --- once the z term is known, which the layer fixes.
        for height in self.ELEVATIONS:
            got = self._read(mesh, height, ('polynomial', 1))
            want = self._field(array([1. / 3., 1. / 3.]), height)
            self.assertAlmostEqual(got, want, places=12)

    def test_a_value_between_two_layers_is_their_linear_blend(self):
        mesh = self._prism()
        for (height, below, above) in ((0.25, 0.0, 0.5), (0.75, 0.5, 1.0)):
            got = self._read(mesh, height, ('polynomial', 1))
            at = array([1. / 3., 1. / 3.])
            want = 0.5 * (self._field(at, below) + self._field(at, above))
            self.assertAlmostEqual(got, want, places=12)
            # And the deviation from the field's own value is the second
            # difference of `z^2` over the half interval --- which is the error
            # of a linear interpolation of a quadratic, not a defect.
            self.assertAlmostEqual(want - self._field(at, height), 0.0625,
                                   places=12)

    def test_the_orders_the_surface_refuses_are_answered_by_the_stack(self):
        mesh = self._prism()
        # A quadratic in z, on three layers: the cubic Bezier fit reproduces it.
        for interp in (('bezier', 2), ('bezier', 3)):
            got = self._read(mesh, 0.5, interp)
            want = self._field(array([1. / 3., 1. / 3.]), 0.5)
            self.assertAlmostEqual(got, want, places=6,
                                   msg=f"{interp} did not reproduce the field")

    def test_a_scalar_elevation_is_left_to_the_surface_methods(self):
        # One elevation is not a stack: nothing is between the layers, so the
        # prism's own shorter list applies and the higher orders still refuse.
        lower = array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        mesh = PrismMesh(stack([lower, lower + array([[0.], [0.], [1.]])]),
                         PrismTopology([[0], [1], [2]]))
        mesh = mesh.withprop('v', (array([0.0]), ones((1, 3))))
        self._read(mesh, 0.0, ('polynomial', 1))
        with self.assertRaises(NotImplementedError):
            self._read(mesh, 0.5, ('bezier', 2))

    def test_a_supplied_gradient_is_refused_rather_than_ignored(self):
        mesh = self._prism()
        at = self._at(mesh, 0.5)
        with self.assertRaises(ValueError):
            mesh.prop('v', at=at, interp=('bezier', 2),
                      gradient=zeros((1, 3, 3, 1, 3)))
