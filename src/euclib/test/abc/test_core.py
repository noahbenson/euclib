# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/abc/test_core.py
'''Tests for the ``euclib.abc._core`` module.

These tests pin the ``immlib.workflow.plantype`` behaviors on which the entire
``euclib`` class hierarchy depends. If ``immlib`` changes one of them, these
tests are where the failure should surface.
'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase, skip


# Tests ######################################################################

class TestPlantypeABC(TestCase):
    '''Tests for the ``plantypeABC`` metaclass and the ``plantype`` semantics.'''

    def test_metaclass_composition(self):
        '''``plantypeABC`` must be a subclass of both ``plantype`` and
        ``ABCMeta``.
        '''
        from abc import ABCMeta
        from immlib.workflow import plantype
        from euclib.abc import plantypeABC
        self.assertTrue(issubclass(plantypeABC, plantype))
        self.assertTrue(issubclass(plantypeABC, ABCMeta))

    def test_abstract_methods_are_enforced(self):
        '''An abstract ``planobject`` subclass may not be instantiated until
        its abstract methods are implemented.
        '''
        from euclib.abc import planobject, plantypeABC, abstractmethod, calc

        class AbstractShape(planobject, metaclass=plantypeABC):
            def __init__(self, size):
                self.size = size
            @calc('doubled')
            def calc_doubled(size):
                return size * 2
            @abstractmethod
            def describe(self):
                '''Describes the shape.'''

        self.assertEqual(AbstractShape.__abstractmethods__, frozenset(
            {'describe'}))
        with self.assertRaises(TypeError):
            AbstractShape(4)

        class ConcreteShape(AbstractShape):
            def __init__(self, size):
                self.size = size
            def describe(self):
                return f"size={self.size}"

        self.assertEqual(ConcreteShape.__abstractmethods__, frozenset())
        shape = ConcreteShape(4)
        self.assertEqual(shape.doubled, 8)
        self.assertEqual(shape.describe(), 'size=4')

    def test_calc_is_overridden_by_method_name(self):
        '''Redefining a calc method in a subclass replaces the inherited calc,
        including the value names that only the base calc produced.
        '''
        from euclib.abc import planobject, calc

        class Base(planobject):
            def __init__(self, coords):
                self.coords = coords
            @calc('dim', 'coord_count')
            def proc_coords(coords):
                return ('matrix', len(coords))

        class Derived(Base):
            def __init__(self, coords, topo):
                self.coords = coords
                self.topo = topo
            @calc('dim', 'coord_count')
            def proc_coords(coords, topo):
                return ('affine', 1)

        # The derived calc consumes topo, a value the base calc did not.
        self.assertEqual(tuple(Base.plan.inputs), ('coords',))
        self.assertEqual(tuple(Derived.plan.inputs), ('coords', 'topo'))
        base = Base([[0, 1], [0, 1]])
        derived = Derived([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'topo')
        self.assertEqual((base.dim, base.coord_count), ('matrix', 2))
        self.assertEqual((derived.dim, derived.coord_count), ('affine', 1))

    def test_subclass_inherits_parent_init(self):
        '''A ``planobject`` subclass that does not define ``__init__`` inherits
        its nearest base's initializer.

        This was not always the case: ``plantype`` used to fall back to the
        generic ``merge``-based initializer whenever a class omitted
        ``__init__``, silently discarding the parent's initialization and
        breaking positional construction. The behavior is pinned here so that a
        regression is caught.
        '''
        from euclib.abc import planobject, calc

        class Parent(planobject):
            def __init__(self, coords, topo):
                self.coords = coords
                self.topo = topo
            @calc('dim')
            def proc_coords(coords):
                return 'matrix'

        class Child(Parent):
            # Deliberately no __init__.
            @calc('dim')
            def proc_coords(coords, topo):
                return 'affine'

        self.assertEqual(tuple(Child.plan.inputs), ('coords', 'topo'))
        # Positional construction works, so the parent's initializer ran.
        child = Child([[1, 0], [0, 1]], 't')
        self.assertEqual(child.dim, 'affine')
        self.assertEqual(child.coords, [[1, 0], [0, 1]])
        # Keyword construction works as well.
        self.assertEqual(Child(coords=None, topo='t').dim, 'affine')

    def test_filters_feed_downstream_calcs(self):
        '''A filter's output is what downstream calcs consume.

        ``euclib`` relies on this: every stored field is normalized by a filter,
        and other calcs read the normalized value rather than the raw one. The
        plan must therefore order a filter ahead of every calc that consumes
        the value it filters --- including calcs that are themselves filters,
        which is the case ``immlib`` got wrong until it was fixed. This test
        pins the guarantee so that a regression surfaces here rather than as a
        mysterious type error deep inside a geometry calculation.
        '''
        from euclib.abc import planobject, calc

        seen = {}

        class Filtered(planobject):
            def __init__(self, x):
                self.x = x
            @calc('x', lazy=False)
            def proc_x(x):
                return x * 10
            # A calc that is itself a filter (y is both input and output).
            @calc('y', lazy=False)
            def proc_y(x, y=None):
                seen['filter'] = x
                return y
            # A calc that is not a filter.
            @calc('z', lazy=False)
            def proc_z(x):
                seen['plain'] = x
                return x

        obj = Filtered(1)
        # Both consumers saw the filtered value (10), not the raw one (1).
        self.assertEqual(seen['filter'], 10)
        self.assertEqual(seen['plain'], 10)
        self.assertEqual(obj.x, 10)
        self.assertEqual(obj.z, 10)

    def test_duplicate_defining_calc_raises(self):
        '''A plan value produced by two calcs in the same class is an error.'''
        from euclib.abc import planobject, calc

        with self.assertRaises(Exception):
            class Duplicate(planobject):
                def __init__(self, x):
                    self.x = x
                @calc('value')
                def calc_a(x):
                    return x
                @calc('value')
                def calc_b(x):
                    return x

    def test_plan_inputs_match_constructor(self):
        '''The plan's inputs must be exactly the attributes set by
        ``__init__``.
        '''
        from euclib.abc import planobject, calc

        class Simple(planobject):
            def __init__(self, a, b):
                self.a = a
                self.b = b
            @calc('ab')
            def calc_ab(a, b):
                return a + b

        self.assertEqual(tuple(Simple.plan.inputs), ('a', 'b'))
        self.assertEqual(Simple(1, 2).ab, 3)
        # A constructor that fails to supply an input is an error.
        with self.assertRaises(TypeError):
            Simple(1)
