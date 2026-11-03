# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/test_smooth.py
'''Tests for the discrete Laplace operator in ``euclib.ops._smooth``.'''

# Dependencies ###############################################################

from __future__ import annotations

import inspect

from numpy import (allclose, asarray, eye, meshgrid, ones, stack, zeros)
import unittest
from unittest import TestCase

from euclib.ops._smooth import (boundary_vertices, cotangents, mass,
                                smooth, smooth_values, smoothing_kernel,
                                stiffness)
from euclib.types import TriMesh, TriTopology
from euclib.utils import simplex_measures


# Fixtures ###################################################################

def _square(n=3):
    '''An n by n grid of unit squares, each split into two triangles.'''
    g = asarray([float(i) for i in range(n + 1)])
    (X, Y) = meshgrid(g, g, indexing='ij')
    coords = stack([X.ravel(), Y.ravel()])
    def vid(i, j):
        return i * (n + 1) + j
    tris = []
    for i in range(n):
        for j in range(n):
            tris.append([vid(i, j), vid(i+1, j), vid(i, j+1)])
            tris.append([vid(i+1, j), vid(i+1, j+1), vid(i, j+1)])
    return (coords, asarray(tris).T)


def _right_triangle():
    '''The unit right triangle, right-angled at the origin.'''
    return (asarray([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]),
            asarray([[0], [1], [2]]))


# Tests ######################################################################

class TestTheCotangents(TestCase):
    '''The cotangent of each corner's angle.'''

    def test_a_right_triangle_has_cotangents_zero_one_one(self):
        # Right-angled at the origin, so that corner's cotangent is zero and the
        # two at forty-five degrees are one.
        (coords, indices) = _right_triangle()
        self.assertTrue(allclose(cotangents(coords, indices).ravel(), [0, 1, 1]))

    def test_a_flat_triangle_does_not_divide_by_zero(self):
        # Three collinear points: the sine is zero and the cotangent is not
        # defined, but nothing should be a NaN.
        coords = asarray([[0., 1., 2.], [0., 0., 0.]])
        indices = asarray([[0], [1], [2]])
        self.assertFalse(bool((cotangents(coords, indices) != cotangents(
            coords, indices)).any()))


class TestTheOperators(TestCase):
    '''The stiffness and the mass.'''

    def test_the_stiffness_annihilates_constants(self):
        # Every row sums to zero, which is what a Laplacian does to a constant
        # field.
        (coords, indices) = _square()
        self.assertTrue(allclose(stiffness(coords, indices).sum(axis=1), 0.0))

    def test_the_stiffness_is_symmetric(self):
        (coords, indices) = _square()
        L = stiffness(coords, indices)
        self.assertEqual(int((L - L.T).nnz), 0)

    def test_the_mass_is_the_area_the_mesh_covers(self):
        # The shares must *partition* each triangle, and a vertex is a corner of
        # several, so this is the test that catches an accumulation which keeps
        # only the last of a repeated index.
        (coords, indices) = _square()
        M = mass(coords, indices)
        self.assertAlmostEqual(float(M.sum()),
                               float(asarray(simplex_measures(coords,
                                                              indices)).sum()),
                               places=12)

    def test_the_mass_is_positive(self):
        # Meyer's mixed area exists so that this holds on an obtuse triangle,
        # where the plain Voronoi area goes negative.
        (coords, indices) = _square()
        self.assertTrue(bool((mass(coords, indices).diagonal() > 0).all()))

    def test_the_mass_of_one_triangle_is_its_area(self):
        (coords, indices) = _right_triangle()
        self.assertAlmostEqual(float(mass(coords, indices).sum()), 0.5,
                               places=12)

class TestTheSmoothingStep(TestCase):
    '''One step of the diffusion equation.'''

    def test_a_constant_is_preserved(self):
        # A Laplacian annihilates constants, so a constant field is unchanged
        # whatever the step --- including at the fixed boundary, whose
        # contribution to the free vertices' equations has to be moved to the
        # other side for this to hold.
        (coords, indices) = _square()
        for fix in (True, False):
            with self.subTest(fixboundary=fix):
                self.assertTrue(allclose(
                    smooth_values(asarray([5.0] * coords.shape[1]), coords, indices,
                           fixboundary=fix), 5.0))

    def test_a_linear_field_is_preserved(self):
        # The cotangent Laplacian is linear-precise: a plane through the mesh is
        # already harmonic, so smoothing leaves it exactly alone.
        (coords, indices) = _square()
        self.assertTrue(allclose(smooth_values(coords[0], coords, indices),
                                 coords[0]))
        self.assertTrue(allclose(smooth_values(coords[1], coords, indices),
                                 coords[1]))

    def test_a_bump_moves_toward_its_neighbours(self):
        (coords, indices) = _square()
        bump = (coords[0] - 1.5) ** 2 + (coords[1] - 1.5) ** 2
        inside = ~boundary_vertices(indices)
        # Its minimum is at the middle; smoothing pulls it up toward the rim.
        self.assertTrue(float(smooth_values(bump, coords, indices)[inside].min())
                        > float(bump[inside].min()))

    def test_a_larger_step_smooths_more(self):
        (coords, indices) = _square()
        bump = (coords[0] - 1.5) ** 2 + (coords[1] - 1.5) ** 2
        inside = ~boundary_vertices(indices)
        amounts = [float(smooth_values(bump, coords, indices, loglambda=lam
                                )[inside].mean()) for lam in (-6, -2, 0, 2)]
        self.assertEqual(amounts, sorted(amounts))

    def test_the_explicit_step_agrees_with_the_implicit_for_a_small_step(self):
        # They are two approximations of the same step, so they meet as the step
        # goes to zero.
        (coords, indices) = _square()
        bump = (coords[0] - 1.5) ** 2 + (coords[1] - 1.5) ** 2
        implicit = smooth_values(bump, coords, indices, loglambda=-8)
        explicit = smooth_values(bump, coords, indices, loglambda=-8,
                          method='explicit')
        self.assertTrue(allclose(implicit, explicit, atol=1e-5))

    def test_the_boundary_keeps_its_values_by_default(self):
        (coords, indices) = _square()
        rim = boundary_vertices(indices)
        # A bump, not a plane: the Laplacian preserves planes exactly, so a
        # plane would leave every vertex unchanged and the second assertion
        # below would prove nothing.
        before = (coords[0] - 1.5) ** 2 + (coords[1] - 1.5) ** 2
        after = smooth_values(before, coords, indices)
        self.assertTrue(allclose(after[rim], before[rim]))
        self.assertFalse(allclose(after[~rim], before[~rim]))

    def test_an_unknown_method_is_refused(self):
        (coords, indices) = _square()
        with self.assertRaises(ValueError):
            smooth_values(coords[0], coords, indices, method='spectral')

class TestTheSmoothOperation(TestCase):
    '''The entry point: a name or an array in, a `Property` out.'''

    def _mesh(self):
        (coords, indices) = _square()
        return TriMesh(coords, TriTopology(indices))

    def test_a_name_and_an_array_give_the_same_answer(self):
        (coords, _) = _square()
        bump = (coords[0] - 1.5) ** 2 + (coords[1] - 1.5) ** 2
        mesh = self._mesh().withprop('b', bump)
        self.assertTrue(allclose(asarray(smooth(mesh, 'b').value),
                                 asarray(smooth(mesh, bump).value)))

    def test_it_returns_a_property_and_leaves_the_geometry_alone(self):
        (coords, _) = _square()
        bump = (coords[0] - 1.5) ** 2 + (coords[1] - 1.5) ** 2
        mesh = self._mesh().withprop('b', bump)
        out = smooth(mesh, 'b')
        self.assertEqual(type(out).__name__, 'Property')
        self.assertEqual(out.spatial_shape, mesh.property_shape)
        # ...and the property it was given still holds what it held.
        self.assertTrue(allclose(asarray(mesh.propinfo('b').value), bump))

    def test_a_grid_smooths_with_a_kernel(self):
        # A grid's cells are boxes, so its smoothing is a kernel along each axis
        # rather than a Laplacian over elements.
        from euclib.types import Grid, GridTopology
        grid = Grid(eye(4), GridTopology((5, 5, 5)))
        step = zeros((5, 5, 5)); step[2:] = 10.0
        blurred = asarray(smooth(grid.withprop('s', step), 's',
                                 kernel='gaussian', sigma=1.0).value)
        self.assertAlmostEqual(float(blurred[2, 2, 2]), 7.0, delta=0.05)
        # The bilateral filter does not smooth across the step: a neighbour ten
        # away contributes almost nothing, however near it is.
        kept = asarray(smooth(grid.withprop('s', step), 's',
                              kernel='bilateral', sigma=1.0,
                              rangesigma=0.5).value)
        self.assertAlmostEqual(float(kept[2, 2, 2]), 10.0, places=6)
        self.assertAlmostEqual(float(kept[1, 2, 2]), 0.0, places=6)

def _cube():
    '''The unit cube as six tetrahedra, all eight vertices on its skin.'''
    coords = asarray([[0., 1., 0., 0., 1., 1., 0., 1.],
                      [0., 0., 1., 0., 1., 0., 1., 1.],
                      [0., 0., 0., 1., 0., 1., 1., 1.]])
    def vid(i, j, k):
        return (i * 2 + j) * 2 + k
    tets = [
        [vid(0,0,0), vid(1,0,0), vid(0,1,0), vid(0,0,1)],
        [vid(1,0,0), vid(1,1,0), vid(0,1,0), vid(1,1,1)],
        [vid(0,1,0), vid(1,1,0), vid(0,1,1), vid(1,1,1)],
        [vid(0,0,0), vid(0,1,0), vid(0,0,1), vid(0,1,1)],
        [vid(0,0,0), vid(1,0,0), vid(0,0,1), vid(1,0,1)],
        [vid(1,0,0), vid(1,1,0), vid(1,0,1), vid(1,1,1)],
    ]
    return (coords, asarray(tets).T)


class TestTheBoundary(TestCase):
    '''Which vertices are on a mesh's boundary, for any order.'''

    def test_a_triangle_mesh_s_boundary_is_its_edges(self):
        (_, indices) = _square()
        b = boundary_vertices(indices)
        # The rim of a four by four grid of points, and not the interior four.
        self.assertEqual(int(b.sum()), 12)

    def test_a_solid_s_boundary_is_its_faces(self):
        # Counting edges would find none: in a tetrahedral mesh every edge is
        # shared by at least three tetrahedra. The boundary is the faces used
        # once, and every vertex of this cube touches one.
        (_, indices) = _cube()
        self.assertEqual(int(boundary_vertices(indices).sum()), 8)

    def test_a_single_tetrahedron_has_all_its_vertices_on_its_boundary(self):
        self.assertEqual(int(boundary_vertices(asarray([[0], [1], [2], [3]]))
                             .sum()), 4)

    def test_a_point_cloud_is_all_boundary(self):
        self.assertEqual(int(boundary_vertices(asarray([[0, 1, 2]])).sum()), 3)

class TestTheImageKernels(TestCase):
    '''The kernels a grid smooths with.'''

    def test_the_gaussian_reaches_three_sigma(self):
        for sigma in (0.5, 1.0, 2.0, 3.0):
            with self.subTest(sigma=sigma):
                (weights, radius) = smoothing_kernel('gaussian', sigma)
                self.assertEqual(radius, max(1, int(round(3.0 * sigma))))
                self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)

    def test_the_box_is_a_half_width_of_sigma(self):
        (weights, radius) = smoothing_kernel('box', 2.0)
        self.assertEqual(radius, 2)
        self.assertTrue(allclose(weights, 0.2))

    def test_a_tiny_sigma_is_a_delta_not_a_wrong_kernel(self):
        # The radius floors at one, but the sampled gaussian is already a delta
        # at that width, so a tiny sigma smooths by nothing rather than by a
        # three-point average.
        (weights, _) = smoothing_kernel('gaussian', 0.1)
        self.assertTrue(allclose(weights, [0.0, 1.0, 0.0], atol=1e-9))

    def test_every_kernel_reaches_the_same_distance(self):
        # `sigma` must mean one thing: the bilateral used to truncate where the
        # separable kernels round, so sigma 1.5 reached four cells in one and
        # six in the other.
        from euclib.ops._smooth import smooth_bilateral
        (_, separable) = smoothing_kernel('gaussian', 1.5)
        source = inspect.getsource(smooth_bilateral)
        self.assertIn('around(3.0 * float(sigma))', source)

    def test_an_image_constant_survives_every_kernel(self):
        from euclib.types import Grid, GridTopology
        grid = Grid(eye(4), GridTopology((5, 5, 5)))
        held = grid.withprop('c', ones((5, 5, 5)) * 3.0)
        for kernel in ('gaussian', 'box', 'bilateral'):
            with self.subTest(kernel=kernel):
                from euclib.ops import smooth as smooth_op
                self.assertTrue(allclose(
                    asarray(smooth_op(held, 'c', kernel=kernel).value), 3.0))

