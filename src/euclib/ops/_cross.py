# -*- coding: utf-8 -*-
###############################################################################
# euclib/ops/_cross.py
'''Interpolation between representations.

The README's motivating promise is that a property can be moved between the
representations of the same underlying object: a temperature measured at a
mesh's vertices can be read at a grid's voxels, and an intensity sampled on a
grid can be read at a mesh's vertices. Nothing about that needs a new
mechanism, because every geometry can already say where a position lies in its
own terms and every property can already be interpolated at a position. What is
needed is a definition of *where* one representation's data lives, so that it
can be asked for at another's positions.

That definition is :func:`positions_of`: a simplex geometry's data lives at its
coordinates, a grid's data lives at its cells, and a prism mesh's data lives at
both of its surfaces. Reading a property at those positions is
:func:`sample`; attaching the result to the other geometry is :func:`transfer`.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import arange, asarray, meshgrid, stack

from ..abc import Geometry
from ..types import Grid, PrismMesh


# Positions ##################################################################

def positions_of(geom, /):
    '''Returns the positions at which a geometry's own data lives.

    These are the positions one geometry's data is read at to be carried onto
    another, so which they are decides what a transfer means. A simplex
    geometry's data lives at its coordinates. A grid's lives at the centre of
    each cell, which is the position its affine carries the cell's index to ---
    not the cell's corner, and not an anchor anywhere else in the cell. A prism
    mesh's lives at both of its surfaces, one value per surface per coordinate.

    Parameters
    ----------
    geom : Geometry
        The geometry.

    Returns
    -------
    array-like
        A ``(D, Q)`` matrix of positions: a simplex geometry's coordinates, a
        grid's cell centres, or a prism mesh's two surfaces.

    Raises
    ------
    TypeError
        If the argument is not a geometry.
    '''
    if not isinstance(geom, Geometry):
        raise TypeError(f"expected a Geometry; found {type(geom)}")
    if isinstance(geom, Grid):
        # A grid has no coordinates of its own; its cells are where its data
        # lives, at the centres the affine takes the integer indices to.
        shape = tuple(geom.shape)
        axes = [a.reshape(-1)
                for a in meshgrid(*(arange(s) for s in shape), indexing='ij')]
        return geom.affine.apply(stack(axes, axis=0))
    if isinstance(geom, PrismMesh):
        # A prism's data lives on both of its surfaces.
        return asarray(prism_surfaces(geom))
    return geom.coords


def prism_surfaces(geom, /):
    '''Returns a prism mesh's two surfaces side by side.

    Parameters
    ----------
    geom : PrismMesh
        The prism mesh.

    Returns
    -------
    array-like
        A ``(D, 2N)`` matrix whose first ``N`` columns are the first surface
        and whose second ``N`` are the second.
    '''
    from numpy import concatenate
    return concatenate([asarray(geom.coords0), asarray(geom.coords1)], axis=1)


# Operations #################################################################

def sample(source, target, property, /, **kw):
    '''Reads a property of one geometry at the positions of another.

    Parameters
    ----------
    source : Geometry
        The geometry that the property belongs to.
    target : Geometry
        The geometry whose positions are read at.
    property : hashable
        The property's name.
    **kw
        Metadata that overrides the property's own: ``interp``, ``extrap``,
        ``null``, and ``mask``.

    Returns
    -------
    array-like
        The values, with the property's channel dimensions and one value per
        position of ``target``.
    '''
    if not isinstance(target, Geometry):
        raise TypeError(f"expected a Geometry; found {type(target)}")
    return source.prop(property, at=positions_of(target), **kw)


def transfer(source, target, property, /, name=None, meta=None, **kw):
    '''Returns a copy of ``target`` carrying a property sampled from ``source``.

    Parameters
    ----------
    source : Geometry
        The geometry that the property belongs to.
    target : Geometry
        The geometry to attach the sampled property to.
    property : hashable
        The property's name.
    name : hashable or None, optional
        The name to attach the property under. The default, ``None``, uses the
        property's own name.
    meta : mapping or None, optional
        Metadata for the new property, named as in ``Property``.
    **kw
        Metadata that overrides the source property's own while sampling.

    Returns
    -------
    Geometry
        A copy of ``target`` with the sampled property attached.

    Examples
    --------
    >>> import numpy as np
    >>> from euclib.ops import transfer
    >>> from euclib.types import Grid, GridTopology, TriMesh, TriTopology
    >>> mesh = TriMesh(np.array([[0., 1., 0.], [0., 0., 1.]]),
    ...                TriTopology([[0], [1], [2]]))
    >>> mesh = mesh.withprop('v', np.array([0., 1., 1.]), interp=1)
    >>> grid = Grid(np.eye(3), GridTopology((2, 2)))
    >>> transfer(mesh, grid, 'v')['v'].shape
    (2, 2)
    '''
    values = sample(source, target, property, **kw)
    shape = tuple(target.property_shape)
    # A geometry's property is shaped like its own components, so a value that
    # was sampled at one position per component is reshaped to suit the target:
    # a grid's data, for instance, is an image rather than a list of cells.
    values = asarray(values).reshape(tuple(values.shape[:-1]) + shape)
    return target.withprop(name if name is not None else property, values,
                           **(dict(meta) if meta else {}))


# Exports ####################################################################

__all__ = ('positions_of', 'sample', 'transfer')
