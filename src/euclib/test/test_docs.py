# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/test_docs.py
'''Tests that ``euclib``'s documentation is valid.

Every public object's docstring must parse as NumPy-style documentation, its
summary must exist, it must document the parameters its signature takes, and its
sections must appear in the order Sphinx and numpydoc expect.

This is a port of ``immlib``'s test of the same name, which is why the walk is
over every module and every public member rather than over the names a package
exports: a docstring that nothing imports is still documentation, and euclib's
own defects of this kind were in places a namespace-only walk does not reach.

Two differences from immlib's version are worth naming. euclib's docstrings use
only the standard NumPy sections --- a ``@calc`` documents its inputs as
parameters and its outputs in ``Returns`` --- so the ``CALC_DOC_SECTIONS`` hook
immlib needs does not apply. And two of euclib's public kernels are not
functions but instances of a registry class that wraps one, so the walk follows
``callable`` rather than ``inspect.isfunction``; the instances answer for their
signatures because ``Kernel`` carries ``__signature__``.

Two contracts here are additions rather than ports, and a reader comparing the
two files should know which: immlib's walk yields an object only if it already
has a docstring, so its test cannot report a *missing* one. Added:
``test_every_exported_callable_is_documented`` and
``test_the_walk_reaches_what_it_should``, the second of which is what keeps the
first from passing because the walk found nothing.
'''

# Dependencies ###############################################################

from __future__ import annotations

import inspect
from importlib import import_module
from types import ModuleType
from unittest import TestCase

from docshare import DocShareError, docparse, docwrap


# Collecting the public surface ##############################################

def euclib_modules():
    '''Yields every module of the ``euclib`` package except the tests.'''
    import pkgutil
    import euclib
    yield euclib
    for found in pkgutil.walk_packages(euclib.__path__, 'euclib.'):
        if '.test' in found.name:
            continue
        try:
            yield import_module(found.name)
        except ImportError:                  # pragma: no cover - optional
            continue


def _documented_value(value, /):
    '''Returns the object to read a docstring from, or ``None``.

    A property's docstring belongs to its getter; a classmethod's or
    staticmethod's to the function inside; and a kernel's to the instance
    itself, which carries the docstring of the implementation it dispatches to.
    '''
    if isinstance(value, property):
        return value.fget
    if isinstance(value, (staticmethod, classmethod)):
        return value.__func__
    return value


def _is_public_member(value, /):
    '''Determines whether a module member is one this test is answerable for.

    Functions and classes, and the callable instances that euclib's kernel
    registry produces. A value that is not callable --- a constant, a module
    that was imported for convenience --- is not documentation and is skipped.
    '''
    if isinstance(value, ModuleType) or inspect.isbuiltin(value):
        return False
    if inspect.isfunction(value) or inspect.isclass(value):
        return True
    return callable(value)


def public_objects():
    '''Yields ``(name, object)`` for every public object that has a docstring.

    Classes are walked as well as modules, so a method's docstring is checked
    along with its class's. An object is yielded only for a docstring of its
    own: ``euclib`` is answerable for the documentation it writes, not for what
    it inherits.
    '''
    seen = set()

    def members(obj, qualname):
        for (name, value) in vars(obj).items():
            if name.startswith('_'):
                continue
            value = _documented_value(value)
            if not _is_public_member(value):
                continue
            if getattr(value, '__module__', '') and not str(
                    value.__module__).startswith('euclib'):
                continue
            if id(value) in seen:
                continue
            seen.add(id(value))
            if value.__doc__ and value.__doc__.strip():
                yield (f'{qualname}.{name}', value)
            if inspect.isclass(value):
                yield from members(value, f'{qualname}.{name}')

    for module in euclib_modules():
        yield from members(module, module.__name__)


# Tests ######################################################################

class TestDocs(TestCase):
    '''The validity of euclib's own documentation.'''

    def test_docstrings_parse(self):
        '''Every public docstring parses as NumPy-style documentation.'''
        count = 0
        for (name, obj) in public_objects():
            with self.subTest(name=name):
                try:
                    doc = docparse(obj, format='numpy')
                except DocShareError as e:
                    self.fail(f"{name}: {type(e).__name__}: {e}")
                self.assertTrue(doc.summary, f"{name} has no summary line")
                count += 1
        # Guard against the collection silently finding nothing.
        self.assertGreater(count, 100)

    def test_docstrings_match_signatures(self):
        '''Every public docstring documents parameters that exist.

        A parameter a function takes from ``**kwargs`` is documented
        deliberately although the signature does not name it, so for a function
        that has a ``**kwargs`` the documented extras are declared with
        ``extraparam`` rather than reported. A function without one has nowhere
        for an extra name to come from, so it is reported.
        '''
        for (name, obj) in public_objects():
            if not (inspect.isfunction(obj) or callable(obj)):
                continue
            if inspect.isclass(obj):
                continue
            with self.subTest(name=name):
                try:
                    docwrap(format='numpy', extraparam=self._kwargs_params(obj))(obj)
                except (DocShareError, TypeError, ValueError) as e:
                    # A builtin or a class without an inspectable signature has
                    # no contract to hold it to.
                    if isinstance(e, (TypeError, ValueError)):
                        continue
                    self.fail(f"{name}: {type(e).__name__}: {e}")

    @staticmethod
    def _kwargs_params(obj):
        '''Returns the parameters ``obj`` documents that its signature does not
        name, when it has a ``**kwargs`` for them to come from.'''
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):       # pragma: no cover
            return ()
        params = sig.parameters
        if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            return ()
        doc = docparse(obj, format='numpy')
        section = doc.section('parameters')
        if section is None:
            return ()
        return tuple(nm for item in section.items for nm in item.names
                     if nm not in params)

    def test_sections_are_in_numpy_order(self):
        '''Parameters precedes Returns in every public docstring.

        Sphinx and numpydoc both expect the standard order, and a section
        written in the wrong place reads as though the object had no such
        section at all.
        '''
        order = ('Parameters', 'Returns', 'Raises')
        for (name, obj) in public_objects():
            doc = inspect.getdoc(obj)
            positions = [(doc.find(f'{s}\n{"-" * len(s)}'), s) for s in order]
            positions = [(i, s) for (i, s) in positions if i >= 0]
            with self.subTest(name=name):
                self.assertEqual(
                    positions, sorted(positions),
                    f"{name}'s sections are out of order:"
                    f" {[s for (_, s) in positions]}")

    def test_every_exported_callable_is_documented(self):
        '''Every callable the library exports has a docstring.

        This is the one contract the walk above cannot state, because it yields
        only the objects that already have a docstring.
        '''
        modules = ('euclib', 'euclib.abc', 'euclib.types', 'euclib.utils',
                   'euclib.ops')
        for module_name in modules:
            module = import_module(module_name)
            for name in module.__all__:
                obj = getattr(module, name)
                if isinstance(obj, ModuleType) or not callable(obj):
                    continue
                with self.subTest(name=f'{module_name}.{name}'):
                    self.assertTrue(
                        obj.__doc__ and obj.__doc__.strip(),
                        f"{module_name}.{name} has no docstring")

    def test_the_examples_in_the_docstrings_run(self):
        '''Every example a docstring gives produces what it says.

        An example that is not run is an example that is wrong eventually: the
        behavior it demonstrates changes and the text does not. This is the
        only place the docstrings' examples are run, and there are few of them,
        which is why the guard below is that at least one was found rather than
        that many were.
        '''
        import contextlib
        import doctest
        import io
        attempted = 0
        for module in euclib_modules():
            # doctest reports to stdout, including that a module had no
            # examples at all; only what a failure says is worth keeping.
            report = io.StringIO()
            with contextlib.redirect_stdout(report):
                result = doctest.testmod(module)
            attempted += result.attempted
            if result.failed:
                self.fail(f"{module.__name__}: {result.failed} of"
                          f" {result.attempted} examples failed:\n"
                          f"{report.getvalue()}")
        self.assertGreater(attempted, 0, "no docstring examples were found")

    def test_the_walk_reaches_what_it_should(self):
        '''The collection finds the objects it is meant to check.

        A test that silently examines nothing passes, so the walk is held to
        the kinds of object it is meant to reach: a method, a calc, a loc type,
        and the kernels, which are callable instances rather than functions and
        are the reason the walk follows ``callable``. Names are matched by
        their ending, because a name is qualified by the module the walk found
        the object in, and several modules hold the same object.
        '''
        found = dict(public_objects())

        def endings(suffix):
            return [name for name in found if name.endswith(suffix)]

        for suffix in ('.withprop', '.proc_coords', '.kernel', '.positions_of',
                       '.GridLoc3'):
            with self.subTest(suffix=suffix):
                self.assertTrue(endings(suffix),
                                f"no docstring was collected for {suffix!r}")
        self.assertTrue(any('proc_' in name for name in found),
                        "no calc docstring was collected")
