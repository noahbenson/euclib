# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/utils/test_dispatch.py
'''Tests for the kernel registry and its gate in ``euclib.utils._dispatch``.

These tests do not measure the C kernels against the pure-Python ones; that is
``test_parity``. What is tested here is the machinery that chooses between them:
that the right implementation is picked for a given set of arguments, and that a
call the C kernels cannot take still reaches the one that can.
'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase, skipUnless

import numpy as np

import euclib
from euclib.utils import split_cells, tetrahedron_box_vertices
from euclib.utils import _dispatch
from euclib.utils._core import using_c_extension
from euclib.utils._dispatch import Kernel, is_dispatchable, is_native_float64


# A pair of kernels to exercise the registry with ############################

def _double(x, /):
    '''Doubles a number.'''
    return 2 * x


def _double_c(x, /):
    '''Doubles a number, as the C extension would.'''
    return 2 * x


# Tests ######################################################################

class TestEligibility(TestCase):
    '''What the C kernels will and will not read.'''

    def test_float64_contiguous_is_native(self):
        self.assertTrue(is_native_float64(np.ones((2, 2))))
        self.assertTrue(is_native_float64(np.ones(3)))

    def test_other_dtypes_are_not(self):
        for dtype in ('float32', 'int64', 'float16', 'complex128'):
            with self.subTest(dtype=dtype):
                self.assertFalse(is_native_float64(np.ones(2, dtype=dtype)))

    def test_non_contiguous_is_not(self):
        # A transposed array has the right dtype and the wrong layout; reading
        # it in C would mean copying it first, which is not what the kernel is
        # there to do.
        self.assertFalse(is_native_float64(np.ones((2, 3)).T))
        self.assertFalse(is_native_float64(np.ones((3, 4))[:, ::2]))

    def test_non_arrays_are_not(self):
        # An object with a shape and a dtype that is not a NumPy array --- a
        # PyTorch tensor, most of all --- cannot be read by the C kernels at
        # all, and its shape, its dtype, and where its elements live are
        # whatever the tensor says rather than what a C kernel assumes.
        class FakeArray:
            shape = (2, 2)
            dtype = np.dtype('float64')

        self.assertFalse(is_native_float64(FakeArray()))
        self.assertFalse(is_native_float64([[1.0, 2.0]]))
        self.assertFalse(is_native_float64(1.0))

    def test_scalars_have_no_bearing(self):
        arrays = np.ones((2, 2))
        self.assertTrue(is_dispatchable((arrays, arrays), {}))
        self.assertTrue(is_dispatchable((arrays, arrays, 0.5), {}))
        self.assertTrue(is_dispatchable((), {'x': 1, 'tolerance': 0.0}))

    def test_one_bad_argument_is_enough(self):
        good = np.ones((2, 2))
        bad = np.ones((2, 2), dtype='float32')
        self.assertFalse(is_dispatchable((good, bad), {}))
        self.assertFalse(is_dispatchable((good,), {'bounds': bad}))


class TestKernel(TestCase):
    '''The callable the registry builds.'''

    def test_it_calls_the_accelerated_one_when_it_applies(self):
        calls = []
        kernel = Kernel(_double, lambda x, /: calls.append(x) or 2 * x)
        self.assertEqual(kernel(3), 6)
        self.assertEqual(calls, [3])

    def test_it_stays_native_without_an_accelerated_one(self):
        kernel = Kernel(_double)
        self.assertEqual(kernel(3), 6)
        self.assertFalse(kernel.accelerated_here)

    def test_a_plain_number_is_eligible(self):
        # A tolerance, a count, or a flag has no dtype or layout to preserve,
        # so it does not keep a call off the accelerated path.
        kernel = Kernel(_double, lambda x, /: 'accelerated')
        self.assertEqual(kernel(3), 'accelerated')

    def test_it_falls_back_for_an_ineligible_call(self):
        kernel = Kernel(_double, lambda x, /: 'accelerated')
        self.assertEqual(kernel(np.ones(2, dtype='float32')).tolist(), [2., 2.])

    def test_a_nested_list_is_not_recognized_as_an_array(self):
        # A list of lists has neither a dtype nor a backend to preserve, so
        # nothing is lost by letting the C kernel convert it --- and the C
        # kernels do convert their arguments, as the pure-Python ones do.
        kernel = Kernel(_double, lambda x, /: 'accelerated')
        self.assertEqual(kernel([[1.0, 2.0]]), 'accelerated')

    def test_it_keeps_the_native_interface(self):
        # A kernel is what the library documents; its name and docstring are the
        # pure-Python kernel's, whether or not the C one is in use.
        kernel = Kernel(_double, _double_c)
        self.assertEqual(kernel.__name__, '_double')
        self.assertEqual(kernel.__doc__, 'Doubles a number.')
        self.assertEqual(kernel.__module__, __name__)
        self.assertIn('_double', repr(kernel))

    def test_it_reports_the_native_signature(self):
        # A kernel is a wrapper, and without being told so, introspection
        # answers with the signature of ``__call__`` --- ``(*args, **kwargs)``
        # --- to ``help``, to an editor, and to whatever holds the library's
        # docstrings to their signatures.
        from inspect import Parameter, signature

        def takes(a, b, c=1):
            return (a, b, c)

        kernel = Kernel(takes)
        self.assertEqual(list(signature(kernel).parameters), ['a', 'b', 'c'])
        found = signature(kernel).parameters['c']
        self.assertEqual(found.default, 1)
        self.assertEqual(found.kind, Parameter.POSITIONAL_OR_KEYWORD)
        self.assertIs(kernel.__wrapped__, takes)

    def test_the_registered_kernels_report_their_signatures(self):
        from inspect import signature
        self.assertEqual(list(signature(split_cells).parameters),
                         ['centers', 'bounds'])
        self.assertEqual(list(signature(tetrahedron_box_vertices).parameters),
                         ['tet', 'bounds', 'tolerance'])


class TestRegistry(TestCase):
    '''The kernels the library actually registers.'''

    def test_the_kernels_are_registered(self):
        self.assertIsInstance(split_cells, Kernel)
        self.assertIsInstance(tetrahedron_box_vertices, Kernel)

    def test_the_extension_is_used_when_it_is_there(self):
        self.assertEqual(split_cells.accelerated_here, using_c_extension)
        self.assertEqual(tetrahedron_box_vertices.accelerated_here,
                         using_c_extension)

    def test_the_native_implementation_is_the_pure_python_one(self):
        from euclib.utils import _pycore
        self.assertIs(split_cells.native, _pycore.split_cells)
        self.assertIs(tetrahedron_box_vertices.native,
                      _pycore.tetrahedron_box_vertices)

    def test_the_kernels_match_on_a_plain_call(self):
        # Whatever the path, the answer is the same: this is the whole point of
        # having two implementations.
        bounds = np.array([[0., 1.], [0., 1.], [0., 1.]])
        centers = np.array([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]])
        self.assertEqual(split_cells(centers, bounds).tolist(),
                         split_cells.native(centers, bounds).tolist())
        tet = np.array([[0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        self.assertEqual(tetrahedron_box_vertices(tet, bounds).shape,
                         tetrahedron_box_vertices.native(tet, bounds).shape)

    def test_an_ineligible_call_still_answers(self):
        # A float32 call cannot go to C, and must still be right --- in float32.
        bounds = np.array([[0., 1.], [0., 1.], [0., 1.]], dtype='float32')
        centers = np.array([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]],
                           dtype='float32')
        found = split_cells(centers, bounds)
        self.assertEqual(found.tolist(), [0, 7])
        # The pure-Python kernel builds the indices by arithmetic on the input
        # rather than by casting to an integer, so it reports the dtype its
        # caller would have gotten on a machine with no compiler.
        self.assertEqual(found.dtype, np.dtype('int64'))

    def test_the_intersection_uses_the_dispatching_kernel(self):
        # The corner search is looked up through the seam in the pure-Python
        # module, so that an intersection takes the fast path too.
        from euclib.utils import _pycore
        self.assertIsNotNone(getattr(_pycore, '_vertices_kernel', None))
        seen = []
        original = _pycore._vertices_kernel

        def spy(tet, bounds, tolerance=0.0):
            seen.append(1)
            return original(tet, bounds, tolerance)

        tet = np.array([[0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        bounds = np.array([[0., 1.], [0., 1.], [0., 1.]])
        try:
            _pycore._vertices_kernel = spy
            _pycore.tetrahedron_box_intersection(tet, bounds)
        finally:
            _pycore._vertices_kernel = original
        self.assertEqual(len(seen), 1)

    def test_the_public_flag_is_a_bool(self):
        self.assertIsInstance(euclib.using_c_extension, bool)


@skipUnless(using_c_extension, "the C extension is not built")
class TestAgainstTheExtension(TestCase):
    '''Tests that only mean something when the extension is loaded.'''

    def test_the_c_kernels_are_the_ones_registered(self):
        from euclib._c import _core
        self.assertIs(split_cells.accelerated, _core.split_cells)
        self.assertIs(tetrahedron_box_vertices.accelerated,
                      _core.tetrahedron_box_vertices)

    def test_the_signatures_match(self):
        # A kernel is called the same way whichever implementation answers.
        from inspect import signature
        from euclib.utils import _pycore
        for (kernel, native) in ((split_cells, _pycore.split_cells),
                                 (tetrahedron_box_vertices,
                                  _pycore.tetrahedron_box_vertices)):
            with self.subTest(kernel=kernel.__name__):
                self.assertEqual([p.name for p in
                                  signature(kernel.native).parameters.values()],
                                 [p.name for p in
                                  signature(native).parameters.values()])

    def test_the_c_kernels_take_the_same_keywords(self):
        from euclib._c import _core
        bounds = np.array([[0., 1.], [0., 1.], [0., 1.]])
        centers = np.array([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]])
        self.assertEqual(
            _core.split_cells(centers=centers, bounds=bounds).tolist(),
            _core.split_cells(centers, bounds).tolist())
        tet = np.array([[0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        self.assertEqual(
            _core.tetrahedron_box_vertices(tet=tet, bounds=bounds,
                                           tolerance=0.0).shape,
            _core.tetrahedron_box_vertices(tet, bounds).shape)

    def test_the_c_kernel_reports_a_dimension_mismatch_as_the_python_one_does(
            self):
        from euclib.utils import _pycore
        centers = np.ones((2, 3))
        bounds = np.ones((3, 2))
        with self.assertRaises(ValueError) as native:
            _pycore.split_cells(centers, bounds)
        with self.assertRaises(ValueError) as accelerated:
            split_cells.accelerated(centers, bounds)
        self.assertEqual(str(native.exception), str(accelerated.exception))


class TestTheModuleItself(TestCase):
    '''The registry's own helpers.'''

    def test_kernel_returns_a_kernel(self):
        self.assertIsInstance(_dispatch.kernel(_double), Kernel)

    def test_is_arraylike(self):
        self.assertTrue(_dispatch._is_arraylike(np.ones(2)))
        self.assertFalse(_dispatch._is_arraylike(1.0))


class TestArgumentsTheGateDeclines(TestCase):
    '''What the gate declines, and what happens to those calls.

    Worth being exact about: for the kernels that exist today, declining a call
    does not change the answer. The C kernels convert what they are given, and a
    ``float32`` array, a transposed view, and even a PyTorch tensor all come back
    the same way either way. What the gate avoids is the conversion, which is a
    copy that can cost more than the kernel saves.

    These tests pin both halves of that: the gate's rule, and the promise that a
    declined call still answers --- with the NumPy answer the kernels document,
    since they are the ones that convert.
    '''

    def setUp(self):
        from euclib._init import checktorch
        if checktorch() is None:
            self.skipTest('torch is not installed')
        import torch
        self.torch = torch

    def test_a_tensor_argument_stays_off_the_c_path(self):
        torch = self.torch
        centers = torch.tensor([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]])
        bounds = torch.tensor([[0., 1.], [0., 1.], [0., 1.]])
        self.assertFalse(_dispatch.is_dispatchable((centers, bounds), {}))
        # The answer is the same either way, and, as the kernels document it,
        # it is a NumPy array.
        self.assertEqual(split_cells(centers, bounds).tolist(), [0, 7])

    def test_a_tensor_with_a_gradient_can_be_passed(self):
        torch = self.torch
        centers = torch.tensor([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]],
                               requires_grad=True)
        bounds = torch.tensor([[0., 1.], [0., 1.], [0., 1.]])
        # Which sub-cell a point falls in is not a differentiable question, and
        # the kernel detaches the gradient rather than failing on it.
        self.assertEqual(split_cells(centers, bounds).tolist(), [0, 7])

    def test_a_tensor_tetrahedron_answers_the_same_as_a_numpy_one(self):
        torch = self.torch
        corners = [[0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
        bounds = [[0., 1.], [0., 1.], [0., 1.]]
        from_tensor = tetrahedron_box_vertices(torch.tensor(corners),
                                               torch.tensor(bounds))
        from_numpy = tetrahedron_box_vertices(np.array(corners),
                                              np.array(bounds))
        self.assertEqual(from_tensor.shape, from_numpy.shape)
        self.assertTrue(np.array_equal(from_tensor, from_numpy))

    def test_a_float32_array_is_declined_and_still_answered(self):
        corners = np.array([[0., 1., 0., 0.], [0., 0., 1., 0.],
                            [0., 0., 0., 1.]], dtype='float32')
        bounds = np.array([[0., 1.], [0., 1.], [0., 1.]], dtype='float32')
        self.assertFalse(_dispatch.is_dispatchable((corners, bounds), {}))
        found = tetrahedron_box_vertices(corners, bounds)
        # The pure-Python kernel widens to float64 as well: its planes are
        # built as Python lists, so float64 is what the kernel returns on any
        # backend. The dtype is not what the gate is keeping.
        self.assertEqual(found.dtype, np.dtype('float64'))
        self.assertEqual(found.shape, (3, 4))

    def test_a_transposed_view_is_declined_and_still_answered(self):
        bounds = np.array([[0., 1.], [0., 1.], [0., 1.]])
        strided = np.zeros((3, 8))
        strided[:, ::2] = np.array([[0., 1., 0., 0.], [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
        view = strided[:, ::2]
        self.assertFalse(view.flags.c_contiguous)
        self.assertFalse(_dispatch.is_dispatchable((view, bounds), {}))
        self.assertEqual(tetrahedron_box_vertices(view, bounds).shape, (3, 4))
