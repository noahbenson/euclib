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
        for method in (('polynomial', 1), ('bezier', 3), ('polynomial', 2)):
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

    def test_an_estimated_gradient_is_refused_rather_than_answered(self):
        '''The one case where the derivative cannot be had, and why refusing beats answering.

        The estimate is one sparse operator applied to the values, and that
        operator is a constant --- a function of the mesh, cached per mesh, and
        cast to floats --- so it cannot carry a graph. An estimate-fed fit would
        then return a derivative with a term missing rather than without one,
        which nothing downstream could detect. Measured: an estimate-fed order 3
        is out by 0.32 against a central difference, where every path that has
        its gradient supplied agrees to 1e-9.
        '''
        torch = self._torch()
        coords = torch.tensor(BASE, dtype=torch.float64, requires_grad=True)
        geom = TriMesh(coords, TOPOLOGY)
        carried = geom.withprop('v', FIELD)
        for method in (('bezier', 3), ('polynomial', 2)):
            with self.subTest(method=method):
                with self.assertRaises(ValueError) as caught:
                    carried.prop('v', at=AT, interp=method)
                self.assertIn("supply the gradient",
                              str(caught.exception).lower().replace('`', ''))

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

    def test_the_split_elements_are_refused_and_say_why(self):
        '''Clough-Tocher and Powell-Sabin, which cannot have it yet.

        Clough-Tocher's twelfth number is per *edge* and is estimated through
        one sparse operator over the whole mesh --- a scipy matrix built once
        per mesh and cast to floats, so it cannot carry a graph. That estimate
        is a function of where the corners are, so answering anyway loses a
        term: measured, Clough-Tocher comes out 0.19 from a central difference
        where every method with all its terms agrees to 1e-9. Powell-Sabin's
        incenter and basis are still numpy. Both are refused rather than
        answered, which is what the assertion here is really checking --- that
        the failure is a sentence and not a derivative that looks fine.
        '''
        torch = self._torch()
        coords = torch.tensor(BASE, dtype=torch.float64, requires_grad=True)
        geom = TriMesh(coords, TOPOLOGY)
        carried = geom.withprop('v', FIELD, gradient=self._gradient())
        for method in (('clough-tocher', 3), ('powell-sabin', 2)):
            with self.subTest(method=method):
                with self.assertRaises(ValueError) as caught:
                    carried.prop('v', at=AT, interp=method)
                self.assertIn("tensor coordinates", str(caught.exception))

    def test_the_same_elements_answer_for_an_array_geometry(self):
        '''Which is what says the refusal is about the coordinates' *backend*
        and not about the methods, which are supported and tested elsewhere.'''
        geom = TriMesh(BASE, TOPOLOGY)
        carried = geom.withprop('v', FIELD, gradient=self._gradient())
        for method in (('clough-tocher', 3), ('powell-sabin', 2)):
            with self.subTest(method=method):
                out = carried.prop('v', at=AT, interp=method)
                self.assertLess(abs(float(mag(out).ravel()[0]) - 7.9), 1e-9)
