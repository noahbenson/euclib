# -*- coding: utf-8 -*-
###############################################################################
# euclib/abc/_core.py
'''Core metaclass and shared machinery for ``euclib``'s abstract types.

Every type in ``euclib`` is an ``immlib`` ``planobject``: its constructor
arguments are the inputs of a calculation plan and its fields are the plan's
lazily-computed outputs. Abstract ``euclib`` types (``Geometry``,
``SimplexGeometry``, ``Topology``, and ``SimplexTopology``) additionally need
to enforce that their abstract methods are implemented, which requires
combining ``immlib``'s ``plantype`` with ``abc.ABCMeta``.

Two behaviors of ``plantype`` shape how every ``euclib`` class must be written.
Both were verified against ``immlib`` before this module was written and are
pinned by tests in ``euclib.test.abc``.

**Calculations are overridden by method name.** When a subclass defines a
``@calc`` method with the same name as one inherited from a base class, the
subclass's version replaces the base's, and the value names that only the base
produced are dropped from the plan. This is how each concrete geometry supplies
its own validation for its own ``coords`` payload.

**A subclass inherits its parent's initializer.** ``plantype`` used to rebuild
the initializer at every level from that class's own attributes, so a subclass
that omitted ``__init__`` silently received the default ``merge``-based one and
lost its parent's validation. That is fixed in ``immlib``: a subclass that
omits ``__init__`` now inherits its nearest base's, so ``euclib`` classes define
``__init__`` only when their inputs genuinely differ. The regression is pinned
by ``euclib.test.abc.test_subclass_inherits_parent_init``.
'''

# Dependencies ###############################################################

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping

from immlib.workflow import calc, plantype, planobject

from pcollections import ldict

from ..utils import content_hash, values_equal
from .._init import backend_names, checktorch


# Metaclass ##################################################################

class plantypeABC(plantype, ABCMeta):
    '''The metaclass for ``euclib``'s abstract plan-object types.

    ``plantypeABC`` combines ``immlib.workflow.plantype``, which builds a lazy
    calculation plan from a class's ``@calc`` methods, with ``abc.ABCMeta``,
    which enforces that a class's abstract methods are implemented before the
    class can be instantiated. Classes that use it inherit from ``planobject``
    as usual and may mark methods with ``@abstractmethod``.

    See the module documentation for the two ``plantype`` behaviors that
    constrain how subclasses must be written.
    '''
    __slots__ = ()


# Normalization ##############################################################

def normalize_backend(backend, /):
    '''Normalizes a ``backend`` value.

    Parameters
    ----------
    backend : str or None
        ``'numpy'``, ``'torch'``, or ``None``. ``None`` means that values keep
        whatever backend they already use.

    Returns
    -------
    str or None
        The validated backend.

    Raises
    ------
    ValueError
        If the value is not a valid backend, or if ``'torch'`` is requested but
        PyTorch is not installed.
    '''
    if backend is None:
        return None
    if backend not in backend_names:
        raise ValueError(
            f"invalid backend: {backend!r}; expected one of"
            f" {backend_names} or None")
    if backend == 'torch' and checktorch() is None:
        raise ValueError(
            "the 'torch' backend was requested, but PyTorch is not installed")
    return backend


# Metadata ###################################################################

def normalize_metadata(metadata, /):
    '''Normalizes a ``metadata`` value to a lazy dictionary.

    Parameters
    ----------
    metadata : mapping or None
        Arbitrary metadata to attach to an object. A mapping is converted to a
        lazy ``pcollections.ldict``, so that it is hashable and cannot be
        modified in place.

    Returns
    -------
    pcollections.ldict
        The metadata; empty when none was given.
    '''
    if metadata is None:
        return ldict.empty
    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"metadata must be a mapping or None; found {type(metadata)}")
    return ldict(metadata)


class MetaObject(planobject):
    '''A ``planobject`` that carries arbitrary metadata.

    An object's metadata is a lazy dictionary of anything the user wishes to
    record about it, and is not otherwise interpreted: ``euclib`` never reads
    it. It is attached with ``withmeta`` and removed with ``dropmeta``, both of
    which return copies, and it takes no part in equality --- two objects whose
    structure agrees are equal whatever labels have been hung on them.

    Parameters
    ----------
    metadata : mapping or None, optional
        The metadata to attach. The default, ``None``, attaches none.

    Attributes
    ----------
    metadata : pcollections.ldict
        The object's metadata, empty when none was attached.
    '''

    def __init__(self, metadata=None):
        self.metadata = metadata

    @calc('metadata', lazy=False)
    def proc_metadata(metadata):
        '''Normalizes the object's metadata to a lazy dictionary.

        Returns
        -------
        metadata : pcollections.ldict
            The metadata, empty when none was given.
        '''
        return normalize_metadata(metadata)

    def withmeta(self, mapping=None, /, **kw):
        '''Returns a copy of the object with metadata added or replaced.

        Parameters
        ----------
        mapping : mapping or None, optional
            Metadata to add. The default, ``None``, adds none.
        **kw
            Metadata to add, merged over ``mapping``.

        Returns
        -------
        MetaObject
            A copy with the merged metadata; ``self`` when nothing changes.
        '''
        updates = dict(mapping) if mapping is not None else {}
        updates.update(kw)
        if not updates:
            return self
        merged = dict(self.metadata)
        merged.update(updates)
        return self.copy(metadata=ldict(merged))

    def dropmeta(self, *keys):
        '''Returns a copy of the object with metadata removed.

        Parameters
        ----------
        *keys
            The names of the metadata to remove. A name the object does not
            have is ignored.

        Returns
        -------
        MetaObject
            A copy without the named metadata; ``self`` when nothing changes.
        '''
        dropped = set(keys)
        kept = {k: v for (k, v) in self.metadata.items() if k not in dropped}
        if len(kept) == len(self.metadata):
            return self
        return self.copy(metadata=ldict(kept))


# Content Comparison #########################################################

def plan_inputs(obj, /):
    '''Returns the plan inputs of a ``planobject`` as a persistent dictionary.

    Parameters
    ----------
    obj : planobject
        The object whose inputs are wanted.

    Returns
    -------
    pcollections.pdict
        The object's plan inputs, keyed by input name.
    '''
    return object.__getattribute__(obj, '_plandict_').inputs


def planobject_eq(a, b, /, exclude=()):
    '''Compares two ``planobject`` objects by the content of their inputs.

    ``immlib`` defines equality in terms of the inputs dictionary, but that
    comparison is unusable when an input is an array, because comparing
    dictionaries compares their values with ``==`` and the result of an array
    comparison is itself an array. This function compares the same inputs, but
    value by value, using ``euclib.utils.values_equal``.

    Parameters
    ----------
    a, b : planobject
        The objects to compare.
    exclude : sequence of str, optional
        Names of inputs that should not take part in the comparison. The
        default, ``()``, compares every input.

    Returns
    -------
    bool
        ``True`` if the objects have the same type and the same inputs, apart
        from those excluded.
    '''
    if type(a) is not type(b):
        return False
    ax = plan_inputs(a)
    bx = plan_inputs(b)
    if ax.keys() != bx.keys():
        return False
    skip = set(exclude)
    return all(values_equal(ax[k], bx[k]) for k in ax if k not in skip)


def planobject_hash(obj, /, exclude=()):
    '''Returns a content hash of a ``planobject``'s inputs.

    The hash is consistent with ``planobject_eq``: objects that compare equal
    under it produce the same hash.

    Parameters
    ----------
    obj : planobject
        The object to hash.
    exclude : sequence of str, optional
        Names of inputs that should not take part in the hash. The default,
        ``()``, hashes every input.

    Returns
    -------
    int
        The hash.
    '''
    inputs = plan_inputs(obj)
    skip = set(exclude)
    return hash((type(obj), tuple(
        (k, content_hash(inputs[k])) for k in sorted(inputs) if k not in skip)))


# Exports ####################################################################

__all__ = ('plantypeABC', 'planobject', 'plantype', 'calc', 'abstractmethod',
           'normalize_backend', 'MetaObject', 'normalize_metadata',
           'plan_inputs', 'planobject_eq', 'planobject_hash')
