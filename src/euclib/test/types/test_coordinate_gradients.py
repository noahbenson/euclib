# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/types/test_coordinate_gradients.py
'''Tests for a geometry whose *coordinates* are a tensor.

The third of the three things a caller may want a derivative through, after the
property's values and the query's positions, and the one this library had not
allowed at all: the constructors converted the coordinates with ``asarray``, and
numpy's ``asarray`` on a tensor that requires a gradient raises rather than
sharing its memory. So a mesh could not be built on one.

Two things are being separated here, and the separation is the substance of the
change rather than an implementation detail.

**A selection detaches, and should.** ``nearest_vertices`` and
``closest_simplex`` answer with an index. Which simplex is nearest is a
comparison, so there is no derivative to carry and dropping one costs nothing.
That is what ``_as_numpy`` does and why its docstring gives that reason.

**A function of the coordinates must not.** A measure, a corner's weight, an
evaluated polynomial --- these vary continuously with where the corners are, and
detaching them would not fail, it would quietly return a derivative missing a
term. That is the failure worth guarding against, so the cases where the
derivative cannot be had are *refused* rather than answered.
'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

import numpy as np
from numpy import array, asarray, zeros
from numpy import abs as nabs
from numpy import allclose

import euclib as el
from euclib.abc import is_geometry
from euclib.types import TriMesh, TriTopology


# Fixtures ###################################################################

#: A triangle in the plane z = 0. In the plane because a position off it is
#: outside a surface --- which is correct, and would answer with the null rather
#: than a number to differentiate.
BASE = array([[0., 1., 0.],
              [0., 0., 1.],
              [0., 0., 0.]])

TOPOLOGY = TriTopology([[0], [1], [2]])

#: A position inside it.
AT = array([[0.15], [0.2], [0.0]])

#: An affine field and its exact gradient. Affine because every method here
#: reproduces it exactly, so a wrong answer is a wrong *interpolation* and not a
#: disagreement about what the method should give.
FIELD = (2.0 * BASE[0] + 3.0 * BASE[1] + 7.0)[None]
SLOPE = array([2.0, 3.0, 0.0])


def mag(one, /):
    '''The magnitude, whether the argument is a quantity or an array.

    Detached, because a tensor geometry's answers carry a gradient and
    ``numpy.asarray`` refuses one that does --- which is the whole subject of
    this file.
    '''
    out = one.m if hasattr(one, 'm') else one
    return asarray(out.detach() if hasattr(out, 'detach') else out)


class TestBuildingOnTensors(TestCase):
    '''A geometry may be built on coordinates that carry a derivative.'''

    def _torch(self):
        try:
            import torch
        except ImportError:                                  # pragma: no cover
            self.skipTest("torch is not installed")
        return torch

    def test_the_class_and_the_one_line_constructor_both_accept_one(self):
        '''Both entry points, since they convert differently.

        The class never converted; the one-line constructor wrapped the
        coordinates in ``asarray`` after ``as_coords`` had already left an
        array-like alone, and that outer call was the whole of the obstacle.
        '''
        torch = self._torch()
        coords = torch.tensor(BASE, dtype=torch.float64, requires_grad=True)
        for (label, geom) in (
                ('the class', TriMesh(coords, TOPOLOGY)),
                ('the constructor', el.trimesh(coords, [[0], [1], [2]]))):
            with self.subTest(entry=label):
                self.assertTrue(is_geometry(geom))
                self.assertTrue(getattr(geom.coords, 'requires_grad', False),
                                "the coordinates did not keep their backend")
                self.assertTrue(allclose(mag(geom.measures), [0.5]))
                self.assertEqual(tuple(geom.bbox.shape), (3, 2))

    def test_an_array_geometry_is_unchanged(self):
        '''The backend follows the argument, so an array stays an array.

        A conversion that made every geometry a tensor would pass every test
        above and break every caller that wants an array.
        '''
        geom = el.trimesh(BASE, [[0], [1], [2]])
        self.assertFalse(hasattr(geom.coords, 'requires_grad'))
        self.assertIsInstance(asarray(geom.measures), type(asarray(BASE)))


class TestTheCoordinateDerivative(TestCase):
    '''``d(value)/d(coordinates)``, checked against a central difference.

    The difference is what makes this worth more than checking that a gradient
    exists: a path that returned the right value for the wrong reason, or a
    derivative with a term missing, would fail it. It is also how the one real
    defect here was found --- an estimate-fed order 3 was out by 0.32 while
    looking perfectly well-formed.
    '''

    def _torch(self):
        try:
            import torch
        except ImportError:                                  # pragma: no cover
            self.skipTest("torch is not installed")
        return torch

    def value(self, coords, method, /):
        '''The interpolation, with the gradient supplied.

        Supplied because an *estimated* one cannot carry the coordinates'
        derivative --- that is what the test below is about --- and these are
        checking the derivative that can be had.
        '''
        geom = TriMesh(asarray(coords), TOPOLOGY)
        out = geom.withprop('v', FIELD, gradient=self._gradient()).prop(
            'v', at=AT, interp=method)
        return float(mag(out).ravel()[0])

    def _gradient(self):
        '''The field's exact gradient, shaped as a property's is.

        ``(C..., D, N)``: one channel, three dimensions, one column per
        coordinate --- and the field is affine, so every coordinate carries the
        same slope.
        '''
        return array([[[2.0], [3.0], [0.0]]]).repeat(3, axis=2)

    def test_it_is_the_derivative_of_the_whole_interpolation(self):
        torch = self._torch()
        # Only the first two rows are perturbed. A nudge in z tips the triangle
        # out of the query's plane, the query then falls off the surface, and
        # the answer becomes the null --- which is correct, and useless as a
        # difference.
        step = 1e-6
        for method in (('polynomial', 1), ('bezier', 3), ('polynomial', 2),
                       ('powell-sabin', 2), ('clough-tocher', 3)):
            coords = torch.tensor(BASE, dtype=torch.float64, requires_grad=True)
            geom = TriMesh(coords, TOPOLOGY)
            out = geom.withprop('v', FIELD, gradient=self._gradient()).prop(
                'v', at=AT, interp=method)
            out = out.m if hasattr(out, 'm') else out
            self.assertTrue(out.requires_grad,
                            "a tensor geometry's answer must carry a gradient")
            out.sum().backward()

            want = zeros(BASE.shape)
            for i in range(2):
                for j in range(BASE.shape[1]):
                    (up, down) = (zeros(BASE.shape), zeros(BASE.shape))
                    up[i, j], down[i, j] = step, -step
                    # Both are *added*: `down` carries its own sign, and
                    # subtracting it here makes the two evaluations the same
                    # point, so the difference is identically zero and the test
                    # reports the gradient's magnitude as the error. Which is
                    # exactly what it did.
                    want[i, j] = (self.value(BASE + up, method)
                                  - self.value(BASE + down, method)) / (2 * step)
            with self.subTest(method=method):
                self.assertLess(nabs(coords.grad.numpy()[:2]
                                     - want[:2]).max(), 1e-6)
                # The third row cannot matter: a triangle *is* its plane, so a
                # position's z does not reach the interpolation at all.
                self.assertEqual(float(nabs(coords.grad.numpy()[2]).max()), 0.0)

    def test_an_estimated_gradient_carries_the_derivative_too(self):
        '''No gradient supplied, so the fit reads one the library estimates.

        The estimate was the last thing that could not carry the derivative: it
        is one sparse operator over the whole mesh, a constant built once per
        mesh and cast to floats. `_gradient_blocks` and `_estimate_gradient`
        now build it from the coordinates on the call instead, which is the same
        construction the operator is collected from --- so the two cannot drift
        apart, and the estimate is held to a *field* rather than to its operator
        by `TestTheEstimateAgainstAField`.

        On a mesh whose stencils are not symmetric, and that is a real
        condition rather than a convenience: the frames the fit is written in
        come from a singular-value decomposition, whose backward is undefined
        when a stencil's singular values are equal. A single triangle's stencil
        has them at `[1, 1, 0]`, and the gradient there comes back `nan` ---
        loudly, at the backward pass, rather than quietly. See the roadmap.
        '''
        torch = self._torch()
        side = 6
        (ix, iy) = np.meshgrid(np.arange(side, dtype=float),
                               np.arange(side, dtype=float), indexing='ij')
        coords = np.vstack([ix.ravel(), iy.ravel()])
        quads = []
        for i in range(side - 1):
            for j in range(side - 1):
                k = i * side + j
                quads.append((k, k + 1, k + side))
                quads.append((k + 1, k + side + 1, k + side))
        quads = np.array(quads).T
        at = array([[2.3], [2.4]])
        field = (2.0 * coords[0] + 3.0 * coords[1] + 7.0)[None]

        def value(one, method, /):
            geom = TriMesh(asarray(one), TriTopology(quads))
            out = geom.withprop('v', field).prop('v', at=at, interp=method)
            return float(mag(out).ravel()[0])

        step = 1e-6
        for method in (('polynomial', 2), ('bezier', 3), ('powell-sabin', 2),
                       ('clough-tocher', 3)):
            tensor = torch.tensor(coords, dtype=torch.float64,
                                  requires_grad=True)
            geom = TriMesh(tensor, TriTopology(quads))
            out = geom.withprop('v', field).prop('v', at=at, interp=method)
            out = out.m if hasattr(out, 'm') else out
            out.sum().backward()
            got = tensor.grad.numpy()
            want = zeros(coords.shape)
            for i in range(2):
                for k in range(coords.shape[1]):
                    (up, down) = (zeros(coords.shape), zeros(coords.shape))
                    up[i, k], down[i, k] = step, -step
                    # Both are *added*: `down` carries its own sign.
                    want[i, k] = (value(coords + up, method)
                                  - value(coords + down, method)) / (2 * step)
            with self.subTest(method=method):
                self.assertFalse(np.isnan(got).any(),
                                 "the gradient of an estimated fit is nan")
                self.assertLess(nabs(got - want).max(), 1e-6)

    def test_the_same_fit_answers_once_the_gradient_is_supplied(self):
        '''Which is what says the refusal above is about the estimate and not
        about tensor coordinates as such.'''
        torch = self._torch()
        coords = torch.tensor(BASE, dtype=torch.float64, requires_grad=True)
        geom = TriMesh(coords, TOPOLOGY)
        carried = geom.withprop('v', FIELD, gradient=self._gradient())
        for method in (('bezier', 3), ('polynomial', 2)):
            out = carried.prop('v', at=AT, interp=method)
            out = out.m if hasattr(out, 'm') else out
            with self.subTest(method=method):
                self.assertLess(abs(float(mag(out.detach()).ravel()[0]) - 7.9),
                                1e-9)

    def test_the_same_elements_answer_for_an_array_geometry(self):
        '''The array path still answers, which is the common case.'''
        geom = TriMesh(BASE, TOPOLOGY)
        carried = geom.withprop('v', FIELD, gradient=self._gradient())
        for method in (('clough-tocher', 3), ('powell-sabin', 2)):
            with self.subTest(method=method):
                out = carried.prop('v', at=AT, interp=method)
                self.assertLess(abs(float(mag(out).ravel()[0]) - 7.9), 1e-9)


class TestTheLeftoverConversions(TestCase):
    '''The two places that converted a local coordinate's parts with `asarray`.

    A tensor that requires a gradient has no numpy array to convert to, so
    `asarray` *raises* rather than reading past it --- which is why these are
    bugs and not merely losses of a derivative. Both are checks rather than
    arithmetic: one compares the shapes of a prism local coordinate's
    components, the other asks whether a grid position lies within the grid's
    extent. Neither wants a derivative, so `to_array(detach=True)` is the right
    reading of both --- and it is what makes them answer for a tensor at all.
    '''

    def test_a_prism_local_coordinate_may_hold_tensors(self):
        try:
            import torch
        except ImportError:                                  # pragma: no cover
            self.skipTest("torch is not installed")
        from euclib.types import PrismTopology
        topo = PrismTopology(array([[0], [1], [2]]), coord_count=4)
        weight = torch.tensor([[0.2], [0.3]], dtype=torch.float64,
                              requires_grad=True)
        height = torch.tensor([[0.5]], dtype=torch.float64,
                              requires_grad=True)
        loc = topo.Loc(index=array([0]), weight=weight, height=height)
        checked = topo.check_loc(loc)
        # It is a check: the coordinate comes back as it went in.
        self.assertTrue(checked.weight is weight,
                        "the check did not hand the coordinate back untouched")
        self.assertTrue(checked.weight.requires_grad)

    def test_a_grid_containment_question_may_be_asked_of_tensors(self):
        try:
            import torch
        except ImportError:                                  # pragma: no cover
            self.skipTest("torch is not installed")
        grid = el.grid((6, 6))
        inside = torch.tensor([[2.5], [3.5]], dtype=torch.float64,
                              requires_grad=True)
        outside = torch.tensor([[9.5], [3.5]], dtype=torch.float64)
        self.assertTrue(bool(el.contains(grid, inside)[0]))
        self.assertFalse(bool(el.contains(grid, outside)[0]))
