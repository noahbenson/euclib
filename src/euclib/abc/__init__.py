# -*- coding: utf-8 -*-
###############################################################################
# euclib/abc/__init__.py
'''Abstract base types for the ``euclib`` library.

The types in this subpackage define the interfaces that the concrete geometric
types in ``euclib.types`` implement: ``Geometry`` and ``SimplexGeometry`` for
objects that occupy space, and ``Topology`` and ``SimplexTopology`` for the
connectivity that those objects realize. ``Property`` describes a value
attached to a geometric object along with the metadata that governs how it is
interpolated.

This subpackage must not import ``euclib.types`` at module scope; a circular
import would result. Where an abstract method needs a concrete type, it imports
it inside the method body.
'''

# Dependencies ###############################################################

from __future__ import annotations

from ._core import (
    MetaObject, normalize_metadata,
    plantypeABC, planobject, plantype, calc, abstractmethod,
    plan_inputs, planobject_eq, planobject_hash)
from ._topo import (
    Topology, SimplexTopology,
    LocMixin, make_loc, is_loc, check_simplex_loc, normalize_indices,
    is_topology, is_simplex_topology)
from ._geom import (
    Geometry, SimplexGeometry,
    is_geometry, is_simplex_geometry,
    normalize_properties, split_property_name, as_query, check_coordinfo)
from ._property import (
    Property,
    is_property,
    QUANTITATIVE, QUALITATIVE, VARTYPES,
    INTERP_METHODS, INTERP_ORDERS, INTERP_QUALITATIVE, INTERP_SUPPORTED,
    EXTRAP_ORDERS, UNSET, default_interp,
    normalize_backend, normalize_vartype, normalize_interp, normalize_extrap,
    normalize_dtype, normalize_null, normalize_mask, normalize_unit,
    convert_value)


# Exports ####################################################################

__all__ = (
    'plantypeABC', 'planobject', 'plantype', 'calc', 'abstractmethod',
    'MetaObject',
    'plan_inputs', 'planobject_eq', 'planobject_hash',
    'Property', 'is_property',
    'QUANTITATIVE', 'QUALITATIVE', 'VARTYPES',
    'INTERP_METHODS', 'INTERP_ORDERS', 'INTERP_QUALITATIVE',
    'INTERP_SUPPORTED', 'EXTRAP_ORDERS', 'UNSET', 'default_interp',
    'normalize_backend', 'normalize_vartype', 'normalize_interp',
    'normalize_extrap', 'normalize_dtype', 'normalize_null', 'normalize_mask',
    'normalize_unit', 'convert_value',
    'Topology', 'SimplexTopology',
    'LocMixin', 'make_loc', 'is_loc', 'check_simplex_loc',
    'normalize_indices',
    'is_topology', 'is_simplex_topology',
    'normalize_backend', 'normalize_metadata',
    'Geometry', 'SimplexGeometry',
    'is_geometry', 'is_simplex_geometry',
    'normalize_properties', 'split_property_name',
    'as_query', 'check_coordinfo')
