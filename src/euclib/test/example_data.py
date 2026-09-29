# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/example_data.py
'''The example subject `bert` of FreeSurfer, as prism meshes.

The `example_data` directory beside this package holds four FreeSurfer surface
files: the white-matter surface and the pial surface of the left and right
hemispheres. Each pair shares its triangle topology and differs only in its
coordinates --- that is what a cortical surface *is*, a sheet of triangles with
a thickness --- so a pair is a prism mesh, and between them they are 533,836
prisms over 266,922 coordinates. That is the scale the library is for, and a
four-cell cube is not.

Reading them needs `nibabel`. It is not a dependency of the library and the
library never reads these files; this module is for the tests and the
benchmarks, and it says so rather than quietly requiring something that is only
in this environment.
'''
# Dependencies ###############################################################

from __future__ import annotations

import pathlib

from numpy import arange, asarray, stack, unique, zeros

# Constants ##################################################################

#: Where the data lives: `example_data` beside `src`, at the repository's root.
DATA = pathlib.Path(__file__).parents[3] / 'example_data'

#: The hemispheres, and the two surfaces each has.
HEMISPHERES = ('lh', 'rh')
SURFACES = ('white', 'pial')

#: The thickness of the prism mesh a *patch* keeps, in millimetres. The whole
#: brain is 266,202 prisms on one side; a test that wants the geometry rather
#: than the scale wants a piece small enough to run in a moment, and a box is
#: the piece that stays connected where a slice of the face list would not.
PATCH_EXTENT = 30.0


# Reading ####################################################################

def _freesurfer():
    '''The `nibabel` reader, or a message saying what is missing.'''
    try:
        from nibabel.freesurfer import read_geometry
    except ImportError as e:                     # pragma: no cover
        raise ImportError(
            f"reading the example surfaces needs nibabel, which is not"
            f" installed; it is not a dependency of euclib itself") from e
    return read_geometry


def surface(hemi='lh', kind='white', /):
    '''One FreeSurfer surface's coordinates and triangle corners.

    Parameters
    ----------
    hemi : str, optional
        ``'lh'`` or ``'rh'``. The default is ``'lh'``.
    kind : str, optional
        ``'white'`` or ``'pial'``. The default is ``'white'``.

    Returns
    -------
    coords : numpy.ndarray
        A ``(3, N)`` matrix of the surface's coordinates, in millimetres.
    indices : numpy.ndarray
        A ``(3, M)`` integer matrix of its triangle corners.
    '''
    if hemi not in HEMISPHERES:
        raise ValueError(f"a hemisphere is one of {HEMISPHERES}; found {hemi!r}")
    if kind not in SURFACES:
        raise ValueError(f"a surface is one of {SURFACES}; found {kind!r}")
    path = DATA / f'{hemi}.{kind}'
    if not path.exists():
        raise FileNotFoundError(
            f"the example data is not where it should be; expected {path}")
    (coords, faces) = _freesurfer()(str(path))
    # A `(3, N)` matrix and a `(3, M)` matrix, which is the way round euclib
    # takes them, rather than the file's one point per row.
    return (asarray(coords, dtype='float64').T,
            asarray(faces, dtype='int64').T)


def prism_mesh(hemi='lh', /):
    '''The prism mesh one hemisphere's white and pial surfaces make.

    Parameters
    ----------
    hemi : str, optional
        ``'lh'`` or ``'rh'``. The default is ``'lh'``.

    Returns
    -------
    PrismMesh
        266,202 prisms on the left hemisphere and 267,634 on the right. These
        are the whole brain; `patch` gives a piece of one.
    '''
    from ..types import PrismMesh, PrismTopology
    (white, indices) = surface(hemi, 'white')
    (pial, other) = surface(hemi, 'pial')
    if indices.shape != other.shape or not (indices == other).all():
        # The pair has to share its topology, or the two surfaces are two meshes
        # rather than one prism mesh with a thickness.
        raise ValueError(
            f"the {hemi} white and pial surfaces do not share a topology, so"
            f" they do not make a prism mesh")
    # The corner count is stated rather than inferred. A FreeSurfer surface
    # declares more vertices than its triangles reference --- the left white
    # surface declares 133,103 and names 132,674 --- so inferring the count from
    # the largest corner leaves the mesh short of the coordinates it was given.
    return PrismMesh(stack([white, pial]),
                     PrismTopology(indices, coord_count=white.shape[1]))


def patch(hemi='lh', extent=PATCH_EXTENT, /):
    '''A connected piece of one hemisphere's surfaces, as a prism mesh.

    Every triangle whose corners all fall inside a box of `extent` millimetres
    around the surfaces' centre. A slice of the face list would be the same size
    and scattered over the whole brain, which is not the kind of mesh anything
    is being tested on; a box is a piece of the cortex.

    Parameters
    ----------
    hemi : str, optional
        ``'lh'`` or ``'rh'``. The default is ``'lh'``.
    extent : float, optional
        The width of the box, in millimetres. The default is 30.

    Returns
    -------
    PrismMesh
        A prism mesh of the triangles the box holds.
    '''
    from ..types import PrismMesh, PrismTopology
    (white, indices) = surface(hemi, 'white')
    (pial, _) = surface(hemi, 'pial')
    middle = 0.5 * (white.min(axis=1) + white.max(axis=1))
    within = (abs(white - middle[:, None]) <= 0.5 * extent).all(axis=0)
    kept = within[indices].all(axis=0)
    if not kept.any():
        raise ValueError(
            f"a {extent} millimetre box holds no triangle of the {hemi}"
            f" hemisphere; it spans"
            f" {[round(float(v), 1) for v in white.max(axis=1) - white.min(axis=1)]}")
    # Reindexed onto the corners the kept triangles use, so that the patch is a
    # mesh of its own rather than the whole brain's coordinates with a mask over
    # them. This also settles the count the file declares: every coordinate the
    # patch has is one some triangle of it names.
    used = unique(indices[:, kept])
    remap = zeros(white.shape[1], dtype=indices.dtype)
    remap[used] = arange(used.size, dtype=indices.dtype)
    return PrismMesh(stack([white[:, used], pial[:, used]]),
                     PrismTopology(remap[indices[:, kept]],
                                   coord_count=used.size))
