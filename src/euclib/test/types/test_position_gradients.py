# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/test_position_gradients.py
'''Tests that a tensor position, and a tensor value, carry their gradients.

An interpolation here is ``sum_k w_k v_k``: the weights follow the geometry and
the position, and only the values enter linearly. So a tensor *position* must
have a derivative through the weights and a tensor *value* must have the
weights themselves as its derivative, and both are checkable against something
that is not the code's own answer --- a central difference, or the weight vector
read off by hand.

Two traps are worth writing down, because both were fallen into while this was
being converted and both make a broken test look like a passing one.

**A partition of unity has no derivative.** An element's basis sums to one, and
so do the weights a position gets, so differentiating a *sum* of either is zero
however the code is written --- and it came out zero when the path was entirely
broken. It is a *nonsymmetric* combination that has to be differentiated.

**A piece is a comparison.** Which piece of a split element holds a position is
decided by comparing the triangle's own weights, so two positions a hair apart
either side of a boundary between pieces lie in different pieces, and the
difference between them measures the jump rather than a derivative. An entry is
therefore compared only where the piece is the same on both sides of the step.

A third is about the geometry and not the arithmetic: a triangle *is its plane*,
so a position off that plane is outside the triangle and answers with the null
value. That is why these tests query positions that lie on the element.
'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

from numpy import allclose, append, arange, array, asarray, float64, linspace
from numpy import abs as nabs
from numpy import zeros
from numpy.linalg import norm

import euclib as el
from euclib.types import _ct, _ps


# Helpers ####################################################################

#: The step the central differences are taken with. Small enough that the
#: truncation error is far below the tolerance, and large enough that the
#: round-off in the difference is not.
STEP = 1e-6


def mag(one, /):
    '''The magnitude, whether the argument is a quantity or an array.'''
    return asarray(one.m) if hasattr(one, 'm') else asarray(one)


def against_difference(weights, scalar, pieces_of, analytic, /):
    '''The worst disagreement between a gradient and a central difference.

    Returns it along with the number of entries it was taken over, so that a
    test can say how much of the array it actually compared. The entries whose
    step crosses a piece boundary are skipped: the piece changes there, the
    function is discontinuous, and there is no derivative to compare against.
    '''
    want = zeros(weights.shape)
    done = zeros(weights.shape, dtype=bool)
    base = pieces_of(weights)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            (plus, minus) = (zeros(weights.shape), zeros(weights.shape))
            plus[i, j] = STEP
            minus[i, j] = -STEP
            # `minus` carries its own sign, so both are *added*: subtracting it
            # here makes the two evaluations the same point and the difference
            # identically zero, which is how this first read as agreeing.
            if not (pieces_of(weights + plus)[j] == base[j]
                    and pieces_of(weights + minus)[j] == base[j]):
                continue
            want[i, j] = ((scalar(weights + plus) - scalar(weights + minus))
                          / (2 * STEP))
            done[i, j] = True
    return (nabs(analytic - want)[done].max(), int(done.sum()))


# Fixtures ###################################################################

#: A triangle in the plane z = 0, so that a position can be moved in two
#: directions while staying on it.
PLANAR = array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

#: The same triangle turned out of every coordinate plane, at the centroid ---
#: which lies on it, where a position just off it would be outside.
TILTED = array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.4, 0.3, 0.2]])

#: A tetrahedron, whose interior a position has all three directions in.
TETRAHEDRON = array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])

#: An affine field over three dimensions and its exact gradient. Affine,
#: because every method here reproduces the affine exactly, which is what makes
#: the gradient a known number rather than a comparison with another method.
AFFINE = lambda q: 2.0 * q[0] + 3.0 * q[1] - 1.0 * q[2] + 7.0
AFFINE_SLOPE = array([2.0, 3.0, -1.0])


class TestTheElementBasis(TestCase):
    '''The Bernstein basis, which is where a tensor weight enters an element.'''

    def _torch(self):
        try:
            import torch
        except ImportError:                                  # pragma: no cover
            self.skipTest("torch is not installed")
        return torch

    def test_a_tensor_weight_gives_the_same_basis_the_arrays_do(self):
        '''The conversion must not change the numbers.

        A basis is a function of the weights, and making it take a tensor is
        supposed to be about *how* it computes them, not what they are. Both
        shapes are checked, since the two-dimensional one takes a different
        branch in the cubic element.
        '''
        torch = self._torch()
        rng = asarray([0.3, 0.25, 0.15])
        for (name, mod) in (('the cubic', _ct), ('the quadratic', _ps)):
            for weights in (rng, rng.reshape(3, 1)):
                want = mag(mod.bernstein(weights))
                got = mag(mod.bernstein(torch.tensor(weights, dtype=torch.float64)))
                with self.subTest(element=name, shape=weights.shape):
                    self.assertEqual(float(nabs(got - want).max()), 0.0,
                                     "a tensor weight changed the basis")

    def test_the_basis_of_barycentric_weights_sums_to_one(self):
        '''What says the basis is the element's own and not some other.

        The weights are normalized before the sum is taken, because the
        partition of unity is a property of *barycentric* weights and three
        arbitrary numbers sum to whatever they sum to.
        '''
        for (name, mod) in (('the cubic', _ct), ('the quadratic', _ps)):
            weights = array([0.4, 0.35])
            whole = append(weights, 1.0 - weights.sum())
            with self.subTest(element=name):
                self.assertLess(abs(mag(mod.bernstein(whole)).sum() - 1.0),
                                1e-14)

    def test_a_tensor_weights_basis_has_the_derivative_it_should(self):
        '''Against a central difference of the array path, not against itself.

        The basis is a partition of unity, so its *sum* has derivative zero
        however wrong the code is. A nonsymmetric combination is what is
        differentiated, and a fixed set of coefficients makes the two
        evaluations the same function.
        '''
        torch = self._torch()
        rng = arange(1.0, 16.0)
        for (name, mod) in (('the cubic', _ct), ('the quadratic', _ps)):
            size = 10 if name == 'the cubic' else 6
            coef = (rng[:size] / 7.0) - 1.0
            whole = array([0.42, 0.31, 0.27])

            def scalar(two, _mod=mod, _coef=coef):
                full = append(two, 1.0 - two.sum())
                return float((mag(_mod.bernstein(full)) * _coef).sum())

            want = zeros((2,))
            for i in range(2):
                plus = zeros((2,))
                plus[i] = STEP
                want[i] = (scalar(whole[:2] + plus)
                           - scalar(whole[:2] - plus)) / (2 * STEP)

            two = torch.tensor(whole[:2].copy(), requires_grad=True,
                               dtype=torch.float64)
            full = torch.cat([two, (1.0 - two.sum())[None]])
            (mod.bernstein(full) * torch.tensor(coef)).sum().backward()
            with self.subTest(element=name):
                self.assertLess(nabs(two.grad.numpy() - want).max(), 1e-6)


class TestThePieceWeights(TestCase):
    '''The weights within the piece that holds a position.

    This is the last conversion the position path needed: the search for the
    nearest element *selects*, and a selection has no derivative, so the weights
    are computed again afterwards on the undetached position.
    '''

    def _torch(self):
        try:
            import torch
        except ImportError:                                  # pragma: no cover
            self.skipTest("torch is not installed")
        return torch

    #: The incenter's weights, which the Powell-Sabin split is built from. It is
    #: side *lengths* the function wants, since the incenter's barycentric
    #: weights are the lengths of the sides opposite each corner.
    CENTRE = mag(_ps.centre_weights(array([1.3, 1.1, 1.5])))

    def pieces_of(self, mod, /):
        if mod is _ct:
            return lambda two: asarray(mod.sub_weights(two)[0])
        return lambda two: asarray(mod.sub_weights(two, self.CENTRE)[0])

    def scalar_of(self, mod, coef, /):
        if mod is _ct:
            return lambda two: float((mag(mod.sub_weights(two)[1]) * coef).sum())
        return lambda two: float(
            (mag(mod.sub_weights(two, self.CENTRE)[1]) * coef).sum())

    def test_a_tensor_gives_the_same_weights_the_arrays_do(self):
        '''Including outside the triangle, which the extrapolation asks for.'''
        torch = self._torch()
        rng = asarray([-0.4, -0.3, -0.2, 0.1, 0.3, 0.45])
        weights = zeros((2, rng.size))
        for (i, value) in enumerate(rng):
            weights[0, i] = value
            weights[1, i] = rng[rng.size - 1 - i] / 2.0
        for (name, mod) in (('the cubic', _ct), ('the quadratic', _ps)):
            (want_pieces, want) = ((mod.sub_weights(weights) if mod is _ct
                                    else mod.sub_weights(weights, self.CENTRE)))
            (got_pieces, got) = ((mod.sub_weights(
                torch.tensor(weights, dtype=torch.float64)) if mod is _ct
                else mod.sub_weights(torch.tensor(weights, dtype=torch.float64),
                                     self.CENTRE)))
            with self.subTest(element=name):
                self.assertEqual(float(nabs(mag(got) - mag(want)).max()), 0.0,
                                 "a tensor weight changed the piece weights")
                self.assertTrue((asarray(got_pieces)
                                 == asarray(want_pieces)).all())

    def test_the_piece_weights_are_a_partition_of_unity(self):
        '''The three weights within a piece name a position in that piece.

        Which is why they sum to one, and why summing them to differentiate
        them proves nothing --- the reason the check below is nonsymmetric.
        '''
        weights = array([[0.42], [0.31]])
        for (name, mod) in (('the cubic', _ct), ('the quadratic', _ps)):
            inside = (mod.sub_weights(weights)[1] if mod is _ct
                      else mod.sub_weights(weights, self.CENTRE)[1])
            with self.subTest(element=name):
                self.assertLess(abs(float(mag(inside).sum()) - 1.0), 1e-14)

    def test_a_tensor_weights_piece_has_the_derivative_it_should(self):
        '''Against a central difference, entry by entry.

        The entries whose step crosses a piece boundary are skipped and
        counted: the piece is a comparison, so there is a jump there and no
        derivative to compare against. The count is asserted too, so that a
        version that skipped everything could not pass.
        '''
        torch = self._torch()
        # Weights inside the reference triangle, where no entry is near a
        # boundary, plus one outside it, which the extrapolation asks for.
        weights = array([[0.42, -0.35, 0.20],
                         [0.31, -0.28, 0.25]])
        coef = array([1.0, -0.5, 0.25])
        for (name, mod) in (('the cubic', _ct), ('the quadratic', _ps)):
            two = torch.tensor(weights.copy(), requires_grad=True,
                               dtype=torch.float64)
            inside = (mod.sub_weights(two)[1] if mod is _ct
                      else mod.sub_weights(two, self.CENTRE)[1])
            (inside * torch.tensor(coef)).sum().backward()
            (worst, checked) = against_difference(
                weights, self.scalar_of(mod, coef), self.pieces_of(mod),
                two.grad.numpy())
            with self.subTest(element=name):
                self.assertLess(worst, 1e-6,
                                "the derivative a tensor weight carries is"
                                " not the one the numbers have")
                self.assertEqual(checked, weights.size,
                                 "no entry was compared, so this proves"
                                 " nothing")


class TestTheInterpolation(TestCase):
    '''The whole path: a position, or a property's values, as a tensor.'''

    def _torch(self):
        try:
            import torch
        except ImportError:                                  # pragma: no cover
            self.skipTest("torch is not installed")
        return torch

    def carried(self, geom, /):
        return geom.withprop('v', AFFINE(geom.coords)[None])

    def test_a_tensor_position_carries_the_fields_gradient(self):
        '''The value is the field's at that position, and so is the gradient.

        Two independent claims from one call, which is what makes this worth
        more than checking that a gradient exists: a path that returned the
        right value for the wrong reason, or a gradient that was not the
        field's, would fail one of them.
        '''
        torch = self._torch()
        cases = [
            ('a triangle in the plane z = 0',
             el.trimesh(PLANAR, array([[0], [1], [2]])),
             array([[0.2], [0.3], [0.0]]), array([2.0, 3.0, 0.0])),
            ('a triangle turned out of the coordinate planes',
             el.trimesh(TILTED, array([[0], [1], [2]])),
             None, None),
            ('a tetrahedron',
             el.tetmesh(TETRAHEDRON, array([[0], [1], [2], [3]])),
             array([[0.2], [0.3], [0.25]]), array([2.0, 3.0, -1.0])),
        ]
        methods = (('polynomial', 1), ('polynomial', 2), ('bezier', 3),
                   ('clough-tocher', 3), ('powell-sabin', 2))
        for (label, geom, at, slope) in cases:
            carried = self.carried(geom)
            if at is None:
                # The centroid of the turned triangle, which lies on it: a
                # position off its plane is outside the triangle and answers
                # with the null value instead.
                at = geom.coords.mean(axis=1)[:, None]
                slope = None
            for method in methods:
                pos = torch.tensor(at, requires_grad=True, dtype=torch.float64)
                try:
                    got = carried.prop('v', at=pos, interp=method)
                except NotImplementedError:
                    continue
                got = got.m if hasattr(got, 'm') else got
                with self.subTest(where=label, method=method):
                    self.assertLess(
                        abs(float(asarray(got.detach()).ravel()[0])
                            - float(AFFINE(at)[0])), 1e-9,
                        "the value is not the field's at the position")
                    self.assertTrue(got.requires_grad,
                                    "a tensor position must carry a gradient")
                    got.sum().backward()
                    if slope is None:
                        continue
                    self.assertLess(
                        nabs(pos.grad.numpy().ravel() - slope).max(), 1e-9,
                        "the position's gradient is not the field's")

    def test_the_order_zero_method_answers_without_a_gradient(self):
        '''Which piece holds a position is a comparison, and selects a value.

        Nothing continuous is involved, so there is no derivative to have ---
        and the answer is still the nearest sample's, which is what says the
        array that comes back is the right one and not an empty result.
        '''
        torch = self._torch()
        geom = el.trimesh(PLANAR, array([[0], [1], [2]]))
        carried = self.carried(geom)
        pos = torch.tensor(array([[0.2], [0.3], [0.0]]), requires_grad=True,
                           dtype=torch.float64)
        got = carried.prop('v', at=pos, interp=('nearest', 0))
        self.assertFalse(hasattr(got, 'requires_grad'),
                         "an order-0 answer is a selection and has no graph")
        corner = AFFINE(geom.coords)
        self.assertIn(float(asarray(got).ravel()[0]), corner,
                      "the answer is not one of the element's own values")

    def test_a_tensor_value_carries_the_weights(self):
        '''The derivative of an interpolation with respect to its values *is*
        its weights.

        Which is the definition of the interpolation being linear in them, and
        it is a number to check rather than an identity to assert: the weights
        sum to one, as a partition of unity, and they are what the position's
        barycentric coordinates are.
        '''
        torch = self._torch()
        geom = el.trimesh(PLANAR, array([[0], [1], [2]]))
        at = array([[0.2], [0.3], [0.0]])
        # The position's weights in the triangle: the corners are the origin,
        # (1, 0, 0) and (0, 1, 0), so a position is 0.5 of the first and its own
        # coordinates for the others.
        want = array([0.5, 0.2, 0.3])
        for method in (('polynomial', 1), ('bezier', 3), ('clough-tocher', 3),
                       ('powell-sabin', 2)):
            values = torch.tensor(AFFINE(geom.coords)[None], requires_grad=True,
                                  dtype=torch.float64)
            carried = geom.withprop('v', values)
            got = carried.prop('v', at=at, interp=method)
            got = got.m if hasattr(got, 'm') else got
            got.sum().backward()
            with self.subTest(method=method):
                self.assertLess(nabs(values.grad.numpy().ravel() - want).max(),
                                1e-9)

    def test_the_array_path_still_answers_with_arrays(self):
        '''The backend follows the arguments, and nothing here changed that.

        A conversion that made every answer a tensor would pass every gradient
        test above and be wrong: an all-array call must return an array.
        '''
        geom = el.trimesh(PLANAR, array([[0], [1], [2]]))
        carried = self.carried(geom)
        at = array([[0.2], [0.3], [0.0]])
        # Order 0 is a selection and answers with the nearest corner's value
        # rather than the affine's, so it is checked against the corners.
        got = carried.prop('v', at=at, interp=('nearest', 0))
        self.assertFalse(hasattr(got, 'requires_grad'))
        self.assertIn(float(asarray(got).ravel()[0]), AFFINE(geom.coords),
                      "an order-0 answer is one of the element's own values")
        for method in (('polynomial', 1), ('bezier', 3),
                       ('clough-tocher', 3), ('powell-sabin', 2)):
            got = carried.prop('v', at=at, interp=method)
            with self.subTest(method=method):
                self.assertFalse(hasattr(got, 'requires_grad'))
                self.assertLess(abs(float(asarray(got).ravel()[0])
                                    - float(AFFINE(at)[0])), 1e-9)

    def test_a_position_off_a_triangles_plane_is_outside_it(self):
        '''A surface has no thickness, so a position off it is not in it.

        This is what made a first version of these tests read as a failure: the
        query position was off the triangle's plane, so the answer was the null
        value and the gradient was zero --- correctly, and for a reason that had
        nothing to do with tensors.
        '''
        geom = el.trimesh(TILTED, array([[0], [1], [2]]))
        carried = self.carried(geom)
        centroid = geom.coords.mean(axis=1)[:, None]
        # A direction the plane does not contain: the cross product of two of
        # the triangle's edges.
        edge = geom.coords[:, 1] - geom.coords[:, 0]
        other = geom.coords[:, 2] - geom.coords[:, 0]
        normal = array([edge[1] * other[2] - edge[2] * other[1],
                        edge[2] * other[0] - edge[0] * other[2],
                        edge[0] * other[1] - edge[1] * other[0]])
        normal = normal / norm(normal)
        off = centroid + 0.1 * normal[:, None]
        got = asarray(carried.prop('v', at=off, interp=('polynomial', 1)))
        self.assertTrue(asarray(got).ravel()[0] != asarray(got).ravel()[0],
                        "a position off the plane is outside the triangle and"
                        " has no answer")
