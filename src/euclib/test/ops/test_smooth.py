# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/ops/test_smooth.py
'''Tests for the discrete Laplace operator in ``euclib.ops._smooth``.'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import allclose, asarray, meshgrid, stack
import unittest
from unittest import TestCase

from euclib.ops._smooth import (boundary_vertices, cotangents, mass,
                                smooth, stiffness)
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
                    smooth(asarray([5.0] * coords.shape[1]), coords, indices,
                           fixboundary=fix), 5.0))

    def test_a_linear_field_is_preserved(self):
        # The cotangent Laplacian is linear-precise: a plane through the mesh is
        # already harmonic, so smoothing leaves it exactly alone.
        (coords, indices) = _square()
        self.assertTrue(allclose(smooth(coords[0], coords, indices),
                                 coords[0]))
        self.assertTrue(allclose(smooth(coords[1], coords, indices),
                                 coords[1]))

    def test_a_bump_moves_toward_its_neighbours(self):
        (coords, indices) = _square()
        bump = (coords[0] - 1.5) ** 2 + (coords[1] - 1.5) ** 2
        inside = ~boundary_vertices(indices)
        # Its minimum is at the middle; smoothing pulls it up toward the rim.
        self.assertTrue(float(smooth(bump, coords, indices)[inside].min())
                        > float(bump[inside].min()))

    def test_a_larger_step_smooths_more(self):
        (coords, indices) = _square()
        bump = (coords[0] - 1.5) ** 2 + (coords[1] - 1.5) ** 2
        inside = ~boundary_vertices(indices)
        amounts = [float(smooth(bump, coords, indices, loglambda=lam
                                )[inside].mean()) for lam in (-6, -2, 0, 2)]
        self.assertEqual(amounts, sorted(amounts))

    def test_the_explicit_step_agrees_with_the_implicit_for_a_small_step(self):
        # They are two approximations of the same step, so they meet as the step
        # goes to zero.
        (coords, indices) = _square()
        bump = (coords[0] - 1.5) ** 2 + (coords[1] - 1.5) ** 2
        implicit = smooth(bump, coords, indices, loglambda=-8)
        explicit = smooth(bump, coords, indices, loglambda=-8,
                          method='explicit')
        self.assertTrue(allclose(implicit, explicit, atol=1e-5))

    def test_the_boundary_keeps_its_values_by_default(self):
        (coords, indices) = _square()
        rim = boundary_vertices(indices)
        # A bump, not a plane: the Laplacian preserves planes exactly, so a
        # plane would leave every vertex unchanged and the second assertion
        # below would prove nothing.
        before = (coords[0] - 1.5) ** 2 + (coords[1] - 1.5) ** 2
        after = smooth(before, coords, indices)
        self.assertTrue(allclose(after[rim], before[rim]))
        self.assertFalse(allclose(after[~rim], before[~rim]))

    def test_an_unknown_method_is_refused(self):
        (coords, indices) = _square()
        with self.assertRaises(ValueError):
            smooth(coords[0], coords, indices, method='spectral')

