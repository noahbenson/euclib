# -*- coding: utf-8 -*-
###############################################################################
# euclib/ops/_smooth.py
'''Smoothing a property over a mesh, or an image over a grid.

The mesh side is the discrete Laplace operator: the *stiffness* matrix, whose
off-diagonal entries say how strongly two vertices are coupled, and the *mass*
matrix, whose diagonal says how much of the mesh each vertex owns. The pair is
what a diffusion equation is written with, and the smoothing step is one step of
it --- implicit by default, which is stable at any step size.

The stiffness is the cotangent one: an edge's weight is half the sum of the
cotangents of the angles opposite it, which is what the linear finite-element
discretization gives on a triangle and what it gives on a tetrahedron too. One
implementation serves both, the difference being only what each simplex's
measure is.

The mass is Meyer's *mixed* Voronoi area: the Voronoi area of a vertex within
each triangle, except that a triangle obtuse at a corner gives that corner half
its area and the other two a quarter each. The plain Voronoi area goes negative
on an obtuse triangle, and the mixed form is what keeps the mass positive.

The Laplacian is normalized by the mass before it is used, so that the step size
is dimensionless: `M^-1 L` has eigenvalues in [0, 2] whatever the mesh's size or
shape, and a step of one means the same thing on every mesh.
'''

# Dependencies ###############################################################

from __future__ import annotations

from numpy import (add, arange, asarray, concatenate, isin, maximum,
                   minimum, ones, sort, sqrt, unique, where, zeros)
from scipy.sparse import coo_matrix

from ..abc import SimplexGeometry
from ..utils import simplex_measures


# The operator ###############################################################

def _cotangent(u, v, /):
    '''The cotangent of the angle between two vectors.

    `(u . v) / |u x v|` rather than `1 / tan`, because the cross product's
    magnitude is the sine and is never zero for two of a triangle's edges, where
    a tangent of a right angle would be infinite. The magnitude comes from the
    Gram determinant, `|u|^2 |v|^2 - (u . v)^2`, which is the squared cross
    product in any dimension --- numpy defines a cross only in three.
    '''
    dot = (u * v).sum(axis=0)
    sine = sqrt(maximum((u * u).sum(axis=0) * (v * v).sum(axis=0) - dot * dot,
                        0.0))
    return dot / where(sine > 0.0, sine, 1.0)


def cotangents(coords, indices, /):
    '''The cotangent of each corner's angle, for each simplex.

    A corner's angle is between the two edges leaving it, so corner ``k``'s
    cotangent is that of the angle between the vectors to the other two corners.
    '''
    coords = asarray(coords)
    indices = asarray(indices)
    (count, number) = indices.shape
    out = zeros((count, number))
    for k in range(count):
        # The *positions* of the other two corners, not the difference of their
        # indices: the angle at corner k is between the edges leaving it.
        u = coords[:, indices[(k + 1) % count]] - coords[:, indices[k]]
        v = coords[:, indices[(k + 2) % count]] - coords[:, indices[k]]
        out[k] = _cotangent(u, v)
    return out


def stiffness(coords, indices, /):
    '''The linear finite-element stiffness matrix, as a sparse COO.

    Each entry is ``V (grad phi_i . grad phi_j)`` summed over the simplices that
    share the two vertices, the gradient of a barycentric coordinate being
    constant within one simplex. The sign is the Laplacian's rather than the
    stiffness matrix's: a diagonal entry is positive and the off-diagonal ones
    negative, which is what ``-(1/2) cot`` gives on a triangle and what makes
    ``mass + step * stiffness`` positive definite. That is the *general* form: on a triangle it is
    exactly the cotangent formula --- ``V (grad phi_i . grad phi_j)`` equals
    ``-(1/2) cot`` of the angle opposite the edge --- and on a tetrahedron it is
    something the cotangent formula does not give, which is why it is written
    this way rather than as cotangents.

    The diagonal is the negated row sum, so the matrix annihilates constants.

    The boundary is the *natural* one: a vertex's equation is its row of the
    weak Laplacian, so the condition implied at a boundary vertex is
    ``df/dn = 0``. Nothing here holds the boundary still --- that is the
    caller's choice, and `smooth_values` makes it with `fixboundary`.
    '''
    from numpy import linalg
    coords = asarray(coords)
    indices = asarray(indices)
    (count, number) = indices.shape
    k = count - 1
    size = int(indices.max()) + 1 if indices.size else 0
    ends = coords[:, indices]                          # (D, K+1, M)
    jac = ends[:, :k, :] - ends[:, k:k + 1, :]         # (D, K, M)
    measure = asarray(simplex_measures(coords, indices))
    # The barycentric gradient of each of the first K corners; the last is the
    # negative of their sum, since the weights add to one.
    grad = linalg.pinv(jac.transpose(2, 0, 1))         # (M, K, D)
    full = concatenate([grad, -grad.sum(axis=1, keepdims=True)], axis=1)
    rows = []
    cols = []
    data = []
    for m in range(number):
        here = indices[:, m]
        for i in range(count):
            for j in range(count):
                weight = float(measure[m]) * float(full[m, i] @ full[m, j])
                rows.append(int(here[i]))
                cols.append(int(here[j]))
                data.append(weight)
    rows = asarray(rows)
    cols = asarray(cols)
    data = asarray(data)
    total = zeros(size)
    add.at(total, rows, data)
    return coo_matrix((concatenate([data, -total]),
                       (concatenate([rows, arange(size)]),
                        concatenate([cols, arange(size)]))),
                      shape=(size, size)).tocsr()


def mass(coords, indices, /):
    '''Meyer's mixed Voronoi area, one entry per vertex, as a sparse diagonal.'''
    coords = asarray(coords)
    indices = asarray(indices)
    (count, number) = indices.shape
    area = asarray(simplex_measures(coords, indices))
    cot = cotangents(coords, indices)
    size = int(indices.max()) + 1 if indices.size else 0
    out = zeros(size)
    # Whether the triangle is obtuse at all, and if so at which corner: a
    # cotangent is negative exactly there, the angle being greater than a right
    # one.
    blunt = (number == 3) & (cot < 0).any(axis=0)
    where_blunt = (cot < 0).argmax(axis=0)
    for k in range(count):
        # A corner's Voronoi share of a triangle is an eighth of the sum over
        # its two leaving edges of `cot(opposite) * |edge|^2`.
        (one, two) = ((k + 1) % count, (k + 2) % count)
        u = coords[:, indices[one]] - coords[:, indices[k]]
        v = coords[:, indices[two]] - coords[:, indices[k]]
        share = ((u * u).sum(axis=0) * cot[two]
                 + (v * v).sum(axis=0) * cot[one]) / 8.0
        # Meyer's mixed area: a triangle obtuse at a corner gives that corner
        # half its area and the other two a quarter each, because the Voronoi
        # cells of those two fall outside it and the Voronoi share goes negative.
        mine = share
        blunt_here = blunt & (where_blunt == k)
        blunt_else = blunt & (where_blunt != k)
        mine = (1.0 - blunt) * asarray(mine) + blunt_here * (area / 2.0) \
            + blunt_else * (area / 4.0)
        # `add.at` rather than `out[...] += `: a vertex is a corner of several
        # triangles, and the fancy-indexed `+=` keeps only the last of a
        # repeated index instead of accumulating them --- the shares then total
        # less than the mesh's area, which is how this was found.
        add.at(out, indices[k], asarray(mine))
    return coo_matrix((out, (arange(size), arange(size))),
                      shape=(size, size)).tocsr()

def boundary_vertices(indices, /):
    '''Which vertices lie on a boundary, for a mesh of any order.

    The boundary of a mesh is its simplices' *faces* --- the corners left when
    one is dropped --- that belong to a single element. A triangle's faces are
    its edges, a tetrahedron's are its triangles, and one rule serves both:
    count the faces, and the ones counted once are on the boundary.

    Counting *edges* instead would be right for a triangle and wrong for a
    tetrahedron, where a closed mesh still has every edge shared by three or
    more tetrahedra and would report no boundary at all.
    '''
    indices = asarray(indices)
    (count, number) = indices.shape
    size = int(indices.max()) + 1 if indices.size else 0
    if count < 2:
        # A point has no faces, so every point of a cloud is on its boundary.
        out = zeros(size, dtype=bool)
        out[indices.ravel()] = True
        return out
    keys = []
    for dropped in range(count):
        corners = asarray([indices[k] for k in range(count) if k != dropped])
        keys.append(pack_corners(corners, size))
    flat = concatenate(keys) if keys else zeros(0, dtype='int64')
    (_, counts) = unique(flat, return_counts=True)
    on_face = unique(flat)[counts == 1]
    out = zeros(size, dtype=bool)
    for dropped in range(count):
        corners = asarray([indices[k] for k in range(count) if k != dropped])
        here = pack_corners(corners, size)
        hit = isin(here, on_face)
        out[corners[:, hit].ravel()] = True
    return out


def pack_corners(corners, size, /):
    '''A face's corners, sorted and packed into one integer.'''
    corners = sort(asarray(corners), axis=0)
    key = zeros(corners.shape[1], dtype='int64')
    for row in corners:
        key = key * size + row
    return key


def _is_tensor(x, /):
    '''Whether a value belongs to a PyTorch tensor's backend.'''
    return type(x).__module__.split('.')[0] == 'torch'


def _operator(matrix, like, /):
    '''A sparse operator, on the backend and in the dtype of `like`.'''
    if not _is_tensor(like):
        return matrix
    import torch
    return torch.as_tensor(asarray(matrix.todense()), dtype=like.dtype,
                           device=like.device)


def _clone(values, /):
    '''A copy, spelled the backend's way.'''
    return values.clone() if _is_tensor(values) else values.copy()


def _solve(matrix, right, /):
    '''Solves a sparse system with one right-hand side per channel.

    `right` is ``(N, C)`` --- one row per unknown, one column per channel ---
    and so is the answer. `spsolve` handles both shapes it can take, but a
    single channel comes back flat, so that is made two-dimensional.

    A tensor's right-hand side is solved by torch, and *densely*: torch 2.13
    has no sparse solve, and the dense one is differentiable where a conversion
    to numpy would silently drop the gradient. The cost is the matrix's size
    squared rather than its nonzeros --- the one place the two backends differ
    in more than a type.
    '''
    if _is_tensor(right):
        import torch
        dense = (matrix if _is_tensor(matrix)
                 else torch.as_tensor(asarray(matrix.todense()),
                                      dtype=right.dtype,
                                      device=right.device))
        return torch.linalg.solve(dense, right)
    from scipy.sparse.linalg import spsolve
    solved = spsolve(matrix.tocsc(), asarray(right))
    if solved.ndim == 1:
        solved = solved.reshape(-1, 1)
    return solved


def smooth_values(values, coords, indices, /, loglambda=0.0,
                  method='implicit', fixboundary=True):
    '''Smooths a value array over a simplex mesh.

    One step of the diffusion equation, with the cotangent Laplacian and Meyer's
    mixed mass. The operator is `M^-1 L`, which is dimensionless --- its
    eigenvalues lie in [0, 2] whatever the mesh's size or shape --- so a step of
    one means the same thing on every mesh, and `loglambda` is exponential in it
    so that linear changes in the parameter are multiplicative in the step.

    Parameters
    ----------
    values : array-like
        ``(C..., N)``: the value at each vertex, with the channel dimensions
        leading.
    coords, indices : numpy.ndarray
        The mesh's coordinates and its ``(K+1, M)`` matrix of corners.
    loglambda : float, optional
        The logarithm of the step. The default, ``0``, is a step of one.
    method : str, optional
        ``'implicit'``, the default, which is stable at any step, or
        ``'explicit'``, which is cheaper per step and stable only below a bound
        the mesh decides.
    fixboundary : bool, optional
        Whether boundary vertices keep their values. The default, ``True``,
        does; ``False`` smooths them along their interior edges.

    Returns
    -------
    numpy.ndarray
        The smoothed values, shaped as the input was.
    '''
    from math import exp
    backend = _is_tensor(values)
    if not backend:
        values = asarray(values, dtype='float64')
    # The channels lead and the vertices trail, so the arithmetic is done on
    # `(C, N)` and the shape is put back at the end.
    shape = values.shape[:-1]
    flat = values.reshape(-1, values.shape[-1])          # (C, N)
    L = _operator(stiffness(coords, indices), values)
    M = _operator(mass(coords, indices), values)
    step = exp(float(loglambda))
    free = ones(L.shape[0], dtype=bool)
    if fixboundary:
        free = ~boundary_vertices(indices)
    if method == 'explicit':
        # f - lambda M^-1 L f, which needs no solve.
        moves = (L @ flat.T) / M.diagonal()[:, None]
        out = _clone(flat)
        out[:, free] -= step * moves.T[:, free]
        return out.reshape(shape + (out.shape[-1],))
    if method != 'implicit':
        raise ValueError(
            f"unknown smoothing method {method!r}; expected 'implicit' or"
            f" 'explicit'")
    # (M + lambda L) f' = M f, restricted to the vertices whose values move.
    # The restricted system's right-hand side is not just its own rows: a free
    # vertex's equation involves its neighbours' values, and the fixed ones are
    # known --- so their contribution moves to the other side. Without that the
    # operator stops annihilating constants, and a constant field comes back
    # smoothed toward whatever the fixed vertices happen to be.
    full = M + L * step
    if not backend:
        full = full.tocsr()
    known = ~free
    right = M @ flat.T
    if known.any():
        right = right - full[:, known] @ flat.T[known, :]
    system = full[free][:, free]
    solved = _solve(system, right[free, :])
    out = _clone(flat)
    out[:, free] = solved.T
    return out.reshape(shape + (out.shape[-1],))

def smooth(geom, values, /, loglambda=0.0, method='implicit',
           fixboundary=True, sigma=1.0, kernel='gaussian', border='edge',
           rangesigma=1.0):
    '''Smooths a property of a geometry, or a value array, over that geometry.

    The values come from a property of ``geom`` when ``values`` is its name, and
    are used as given when it is an array, a tensor or a quantity. What comes
    back is a `Property` --- the geometry is not changed, and a caller attaches
    the result with ``withprop``, which is how every other operation here
    behaves.

    Parameters
    ----------
    geom : SimplexGeometry
        The geometry whose elements the smoothing runs over.
    values : hashable or array-like
        The property's name, or the values themselves.
    loglambda : float, optional
        The logarithm of the step; see `smooth_values`.
    method : str, optional
        ``'implicit'``, the default, or ``'explicit'``.
    fixboundary : bool, optional
        Whether boundary vertices keep their values. The default does.

    Returns
    -------
    Property
        The smoothed values, with the metadata the property carried when one was
        named.

    Notes
    -----
    The Laplacian's natural boundary condition is ``df/dn = 0``, so a field with
    a gradient across the boundary is *not* left alone there unless the boundary
    is held --- which is what `fixboundary` is for, and why it defaults to
    holding it. A plane is preserved by the default and moves without it, and
    that is the condition rather than a defect.

    Raises
    ------
    TypeError
        If the geometry is not a simplex geometry, whose elements the Laplacian
        is built from --- a grid's cells are boxes rather than simplices.
    '''
    from ..abc import Property
    from ..types import Grid
    if isinstance(geom, Grid):
        # A grid's cells are boxes, so its smoothing is a kernel along each
        # axis rather than a Laplacian over elements.
        array = (geom.propinfo(values).value if isinstance(values, str)
                 else values)
        if kernel == 'bilateral':
            out = smooth_bilateral(asarray(array), tuple(geom.shape),
                                   sigma=sigma, rangesigma=rangesigma,
                                   border=border)
        else:
            out = smooth_grid(asarray(array), tuple(geom.shape), sigma=sigma,
                              kernel=kernel, border=border)
        return Property(out, geom.property_shape)
    if not isinstance(geom, SimplexGeometry):
        raise TypeError(
            f"expected a SimplexGeometry or a Grid; found {type(geom).__name__}")
    if isinstance(values, str):
        held = geom.propinfo(values)
        array = held.value
        # A smoothed field is still the same kind of thing: its type, its unit
        # and its interpolation go with it.
        meta = {'vartype': held.vartype, 'interp': held.interp,
                'extrap': held.extrap, 'border': held.border,
                'mask': held.mask, 'null': held.null,
                'dtype': held.dtype, 'detach': held.detach}
        meta = {k: v for (k, v) in meta.items() if v is not None}
    else:
        array = values
        meta = {}
    out = smooth_values(asarray(array), asarray(geom.coords),
                        geom.topo.indices, loglambda=loglambda, method=method,
                        fixboundary=fixboundary)
    return Property(out, geom.property_shape, **meta)

# Images #####################################################################

#: The border policies a smoothing may pad with, named as numpy's `pad` names
#: them: an image is smoothed at its edge by pretending something lies beyond.
BORDERS = {'constant': 'constant', 'edge': 'edge', 'reflect': 'reflect',
           'wrap': 'wrap'}


def smoothing_kernel(name, sigma, /):
    '''A one-dimensional smoothing kernel and its radius.

    Parameters
    ----------
    name : str
        ``'gaussian'``, whose weights are exp(-x^2 / 2 sigma^2) normalised, or
        ``'box'``, whose weights are equal over the window.
    sigma : float
        The kernel's width, in cells. A gaussian is cut at three sigma; a box's
        half-width is sigma rounded.

    Returns
    -------
    weights : numpy.ndarray
        The kernel's weights, summing to one.
    radius : int
        How far it reaches either side.

    Raises
    ------
    ValueError
        If the kernel is not one of those two.
    '''
    from numpy import exp, full, round as around
    sigma = float(sigma)
    if sigma <= 0.0:
        return (asarray([1.0]), 0)
    if name == 'box':
        radius = max(1, int(around(sigma)))
        return (full(2 * radius + 1, 1.0 / (2 * radius + 1)), radius)
    if name == 'gaussian':
        radius = max(1, int(around(3.0 * sigma)))
        x = arange(-radius, radius + 1, dtype='float64')
        weights = exp(-(x * x) / (2.0 * sigma * sigma))
        return (weights / weights.sum(), radius)
    raise ValueError(
        f"unknown smoothing kernel {name!r}; expected 'gaussian' or 'box'")


def _blur_axis(values, axis, weights, radius, border, /):
    '''One separable pass of a kernel along one axis.'''
    from numpy import pad, take
    values = asarray(values)
    pad_width = [(0, 0)] * values.ndim
    pad_width[axis] = (radius, radius)
    padded = pad(values, pad_width, mode=BORDERS[border])
    out = zeros(values.shape)
    for (k, weight) in enumerate(weights):
        index = [slice(None)] * values.ndim
        index[axis] = slice(k, k + values.shape[axis])
        out += weight * padded[tuple(index)]
    return out


def smooth_grid(values, shape, /, sigma=1.0, kernel='gaussian',
                border='edge'):
    '''Smooths an image with a separable kernel.

    A grid's cells are the unit boxes of an index space, so its smoothing is a
    kernel applied along each axis in turn rather than a Laplacian over elements
    --- which is exactly what makes it separable, and cheap.

    Parameters
    ----------
    values : array-like
        ``(C..., X...)``: the image, with any channel dimensions leading and
        the grid's own shape trailing.
    shape : tuple of int
        The grid's shape, which says how many of the trailing axes are the
        grid's own --- four for a 2-D image with two channels, two for the
        image alone.
    sigma : float, optional
        The kernel's width in cells. The default is one.
    kernel : str, optional
        ``'gaussian'``, the default, or ``'box'``.
    border : str, optional
        How the image is continued past its edge: ``'edge'``, the default, or
        ``'constant'``, ``'reflect'`` or ``'wrap'``.

    Returns
    -------
    numpy.ndarray
        The smoothed image, shaped as the input was.
    '''
    values = asarray(values, dtype='float64')
    (weights, radius) = smoothing_kernel(kernel, sigma)
    if radius == 0:
        return values.copy()
    out = values
    # The channel dimensions lead, so the grid's own axes are the last ones ---
    # as many as the shape says.
    rank = len(tuple(shape))
    if rank > values.ndim:
        raise ValueError(
            f"a grid of {rank} dimensions cannot smooth a {values.ndim}-"
            f"dimensional array")
    for axis in range(values.ndim - rank, values.ndim):
        out = _blur_axis(out, axis, weights, radius, border)
    return out

def smooth_bilateral(values, shape, /, sigma=1.0, rangesigma=1.0,
                     border='edge'):
    '''Smooths an image with a bilateral filter.

    The weights are the product of a spatial gaussian and a gaussian in the
    *values*, so a step in the image is not smoothed across: a neighbour whose
    value is far from the centre's contributes little however near it is. That
    is why it is not separable --- the weights depend on the values, so the two
    axes cannot be taken one at a time.

    Parameters
    ----------
    values : array-like
        ``(C..., X...)``: the image, channels leading and the grid's shape
        trailing.
    shape : tuple of int
        The grid's shape, which says how many of the trailing axes are its own.
    sigma : float, optional
        The spatial width, in cells.
    rangesigma : float, optional
        The width in value. The default, one, smooths differences of about one
        and leaves larger steps alone.
    border : str, optional
        How the image is continued past its edge.

    Returns
    -------
    numpy.ndarray
        The smoothed image, shaped as the input was.
    '''
    from numpy import around, exp, ndindex, pad
    values = asarray(values, dtype='float64')
    rank = len(tuple(shape))
    if rank > values.ndim:
        raise ValueError(
            f"a grid of {rank} dimensions cannot smooth a {values.ndim}-"
            f"dimensional array")
    # Rounded as the separable kernels round theirs, so that `sigma` means the
    # same reach whatever kernel it is given to.
    radius = max(1, int(around(3.0 * float(sigma))))
    (spatial, _) = smoothing_kernel('gaussian', sigma)
    pad_width = [(0, 0)] * values.ndim
    for axis in range(values.ndim - rank, values.ndim):
        pad_width[axis] = (radius, radius)
    padded = pad(values, pad_width, mode=BORDERS[border])
    total = zeros(values.shape)
    weight = zeros(values.shape)
    for offset in ndindex(*([2 * radius + 1] * rank)):
        shift = tuple(o - radius for o in offset)
        index = [slice(None)] * values.ndim
        for (k, axis) in enumerate(range(values.ndim - rank, values.ndim)):
            index[axis] = slice(offset[k], offset[k] + values.shape[axis])
        near = padded[tuple(index)]
        spatial_weight = 1.0
        for (k, o) in enumerate(offset):
            # `o` is the window's index along its axis, which is the kernel's
            # own index: the kernel has one weight per window position.
            spatial_weight = spatial_weight * spatial[o]
        close = exp(-((near - values) ** 2) / (2.0 * float(rangesigma) ** 2))
        weight_here = spatial_weight * close
        total += weight_here * near
        weight += weight_here
    return total / weight
