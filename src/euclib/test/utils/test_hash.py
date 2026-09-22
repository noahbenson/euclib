# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/utils/test_hash.py
'''Tests for the ``euclib.utils._hash`` module.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase, skipUnless

from numpy import array, zeros

from euclib.utils import content_hash
from euclib._init import checktorch


# Tests ######################################################################

class TestContentHash(TestCase):
    '''Tests for ``content_hash``.'''

    def test_equal_arrays_hash_equal(self):
        a = array([[0., 1.], [2., 3.]])
        b = array([[0., 1.], [2., 3.]])
        self.assertNotEqual(id(a), id(b))
        self.assertEqual(content_hash(a), content_hash(b))

    def test_unequal_arrays_hash_differently(self):
        a = array([[0., 1.], [2., 3.]])
        b = array([[0., 1.], [2., 4.]])
        self.assertNotEqual(content_hash(a), content_hash(b))

    def test_shape_and_dtype_matter(self):
        a = zeros((2, 3))
        b = zeros((3, 2))
        self.assertNotEqual(content_hash(a), content_hash(b))
        self.assertNotEqual(content_hash(zeros(3, dtype='i8')),
                            content_hash(zeros(3, dtype='f8')))

    def test_scalars_and_sequences(self):
        self.assertEqual(content_hash(7), content_hash(7))
        self.assertEqual(content_hash((1, 2, 3)), content_hash((1, 2, 3)))
        self.assertNotEqual(content_hash((1, 2, 3)), content_hash([1, 2, 3]))

    def test_mappings_are_order_independent(self):
        a = {'x': array([1., 2.]), 'y': array([3., 4.])}
        b = {'y': array([3., 4.]), 'x': array([1., 2.])}
        self.assertEqual(content_hash(a), content_hash(b))

    def test_nested_structures(self):
        a = {'p': [array([[0., 1.]]), (1, 2)]}
        b = {'p': [array([[0., 1.]]), (1, 2)]}
        self.assertEqual(content_hash(a), content_hash(b))
        c = {'p': [array([[0., 2.]]), (1, 2)]}
        self.assertNotEqual(content_hash(a), content_hash(c))

    def test_quantities(self):
        pint = __import__('pint')
        ureg = pint.UnitRegistry()
        a = 5 * ureg.meter
        b = 5 * ureg.meter
        c = 5 * ureg.second
        self.assertEqual(content_hash(a), content_hash(b))
        self.assertNotEqual(content_hash(a), content_hash(c))

    @skipUnless(checktorch() is not None, 'torch is not installed')
    def test_tensors(self):
        torch = checktorch()
        a = torch.zeros((2, 2))
        b = torch.zeros((2, 2))
        c = torch.ones((2, 2))
        self.assertEqual(content_hash(a), content_hash(b))
        self.assertNotEqual(content_hash(a), content_hash(c))

    @skipUnless(checktorch() is not None, 'torch is not installed')
    def test_tensors_in_mappings(self):
        torch = checktorch()
        a = {'p': torch.zeros((2, 2))}
        b = {'p': torch.zeros((2, 2))}
        self.assertEqual(content_hash(a), content_hash(b))
