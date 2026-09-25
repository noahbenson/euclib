# -*- coding: utf-8 -*-
###############################################################################
# euclib/utils/_spatial.py
'''Spatial subdivision: quadtrees and octrees.

A quadtree or an octree answers "which of these things is near this position?"
without examining all of them, by halving space repeatedly and remembering which
half each thing fell into. The subdivision itself is a kernel ---
``split_cells``, and its named wrappers ``quadtree_split`` and
``octree_split`` --- so that a C implementation can take it over; this module is
the structure that uses it.

Two details make the structure useful for the searches in this library, which
look for the *nearest simplex* to a position rather than for a point:

- Each item is stored as a **sphere**: a center and a radius that contains it.
  A simplex's bounding box gives a cheap such sphere (``simplex_boxes``), where
  the simplex itself would be expensive to test against.
- A query therefore asks for **candidates** --- the items whose sphere comes
  within a radius of the position --- rather than for a single answer. That is
  what makes it exact for a nearest-point search: the true nearest item is
  among the candidates whenever the radius is at least its distance, so a
  caller can search the candidates exactly and enlarge the radius until the
  answer it finds is provably the nearest one. ``euclib.utils.candidates_for``
  does that for a set of simplices.

.. note:: The subdivision is one-piece-per-item: an item is stored in the
    single node that contains its center, not in every node its sphere
    overlaps. The radii are accounted for when pruning, so the answers are
    still exact, but an item much larger than the search radius makes the
    pruning weaker.

    **What that means in practice, measured on a triangulated grid mesh with
    200 query positions at a radius of 0.02: 3,042 triangles answered in 0.019 s
    against 0.098 s by brute force, 12,482 in 0.029 s against 0.539 s, and
    28,322 in 0.041 s against 1.181 s** --- five times faster at the smallest
    size and twenty-nine at the largest, where the pruning leaves 58 candidates
    of 28,322. The tree is not, however, a universal win: given random
    triangles whose corners are random points, so that each item spans much of
    the domain, almost every item is a candidate and the per-query traversal in
    Python costs three to four times what the vectorized brute force does. The
    structure pays when the items are small relative to the region being
    searched, which is what a mesh of a surface or a volume is.
'''

# Dependencies ###############################################################

from __future__ import annotations

from heapq import heappop, heappush, heapreplace

from numpy import (
    arange, ascontiguousarray, asarray, concatenate, intp, maximum, sqrt,
    stack, where, zeros)

from ._core import c_spatial, split_cells


def _is_number(value, /):
    '''Determines whether a value is a single number rather than an array.

    A NumPy scalar answers to ``shape`` --- it is an array of no dimensions ---
    so asking whether the attribute is there is not enough; what separates it
    from an array is that its shape has nothing in it.
    '''
    ndim = getattr(value, 'ndim', None)
    if ndim is not None:
        return ndim == 0
    return not hasattr(value, '__len__')


# The tree ###################################################################

class SpatialTree:
    '''A quadtree or octree over items that each occupy a sphere.

    Parameters
    ----------
    centers : numpy.ndarray
        A ``(D, M)`` matrix of item centers.
    radii : numpy.ndarray
        A length-``M`` vector of item radii; every point of item ``i`` lies
        within ``radii[i]`` of ``centers[:, i]``.
    bounds : numpy.ndarray or None, optional
        The ``(D, 2)`` box to subdivide. The default, ``None``, uses the box
        that contains every item's sphere.
    max_items : int, optional
        The number of items a node may hold before it is subdivided. The
        default is 16.
    max_depth : int, optional
        The greatest depth the subdivision may reach. The default is 8, so a
        node may be as small as ``1 / 256`` of the root along each axis.

    Attributes
    ----------
    dim : int
        The number of dimensions the tree subdivides: 2 for a quadtree and 3
        for an octree.
    '''

    def __init__(self, centers, radii, bounds=None, max_items=16, max_depth=8):
        centers = asarray(centers)
        radii = asarray(radii)
        if centers.ndim != 2:
            raise ValueError(
                f"centers must be a (D, M) matrix; found {centers.ndim}"
                " dimensions")
        if radii.shape != (centers.shape[1],):
            raise ValueError(
                f"radii must have one entry per item: {centers.shape[1]}, not"
                f" {radii.shape}")
        (self.dim, count) = centers.shape
        self.centers = centers
        self.radii = radii
        if bounds is None:
            # An empty tree has nothing to bound, so it gets a unit box; the
            # queries it answers are all empty.
            bounds = (zeros((self.dim, 2)) if count == 0
                      else self._sphere_bounds(centers, radii))
        else:
            bounds = asarray(bounds)
        # A flat axis has no halves to split it into, so it is given one.
        span = bounds[:, 1] - bounds[:, 0]
        pad = where(span > 0, 0.0, 0.5)
        bounds = bounds + concatenate([-pad[:, None], pad[:, None]], axis=1)
        self.bounds = bounds
        (self._low, self._high, self._child, self._items, self._maxr) = (
            [], [], [], [], [])
        self._build(arange(count), bounds, 0, max_items, max_depth)

    @staticmethod
    def _sphere_bounds(centers, radii):
        '''The box containing every item's sphere.'''
        low = (centers - radii[None, :]).min(axis=1)
        high = (centers + radii[None, :]).max(axis=1)
        return concatenate([low[:, None], high[:, None]], axis=1)

    def _child_bounds(self, bounds, cell, /):
        '''The box of one sub-cell of a box.'''
        middle = (bounds[:, 0] + bounds[:, 1]) / 2.0
        low = bounds[:, 0].copy()
        high = bounds[:, 1].copy()
        for axis in range(self.dim):
            if (cell >> axis) & 1:
                low[axis] = middle[axis]
            else:
                high[axis] = middle[axis]
        return concatenate([low[:, None], high[:, None]], axis=1)

    def _build(self, items, bounds, depth, max_items, max_depth, /):
        '''Adds one node, and its descendants, and returns its index.'''
        index = len(self._low)
        self._low.append(bounds[:, 0].copy())
        self._high.append(bounds[:, 1].copy())
        self._child.append(None)
        self._items.append(None)
        self._maxr.append(
            float(self.radii[items].max()) if items.size else 0.0)
        if items.size <= max_items or depth >= max_depth:
            self._items[index] = items
            return index
        cells = split_cells(self.centers[:, items], bounds)
        child = [-1] * (2 ** self.dim)
        for cell in range(2 ** self.dim):
            selected = items[cells == cell]
            if selected.size == 0:
                continue
            child[cell] = self._build(selected, self._child_bounds(bounds, cell),
                                      depth + 1, max_items, max_depth)
        self._child[index] = child
        return index

    def _box_distance(self, node, point, /):
        '''The distance from a position to a node's box.'''
        below = self._low[node] - point
        above = point - self._high[node]
        excess = maximum(maximum(below, above), 0.0)
        return float(sqrt((excess * excess).sum()))

    # The flattened tree ####################################################

    def _flattened(self, /):
        '''The tree as the arrays the C queries read, built once.

        A node's items, and its children, become runs of a flat array, with one
        offset per node saying where each run begins; and the node indexes
        themselves, so that a run of children needs no further indirection.

        Returns
        -------
        tuple
            The nine arrays the compiled queries take, in their order.
        '''
        if getattr(self, '_flat', None) is None:
            nodes = len(self._low)
            (item_start, child_start) = (zeros(nodes + 1, dtype=intp),
                                         zeros(nodes + 1, dtype=intp))
            items = []
            children = []
            for i in range(nodes):
                item_start[i] = len(items)
                child_start[i] = len(children)
                if self._items[i] is not None:
                    items.extend(int(j) for j in self._items[i])
                else:
                    children.extend(int(c) for c in self._child[i] if c >= 0)
            item_start[nodes] = len(items)
            child_start[nodes] = len(children)
            # Each array is made contiguous here, once. The compiled queries
            # read their arguments through the buffer protocol and copy anything
            # that is not contiguous, and a mesh's centers are usually a strided
            # view into a larger array --- so a copy per call of what can be
            # copies once is the difference between a query taking microseconds
            # and taking half a millisecond.
            self._flat = (
                ascontiguousarray(self._low, dtype='float64'),
                ascontiguousarray(self._high, dtype='float64'),
                ascontiguousarray(self._maxr, dtype='float64'),
                item_start, asarray(items, dtype=intp),
                child_start, asarray(children, dtype=intp),
                ascontiguousarray(self.centers, dtype='float64'),
                ascontiguousarray(self.radii, dtype='float64'))
        return self._flat

    # Queries ###############################################################

    def candidate_runs(self, query, radius, /):
        '''Returns the candidates of each query as one flat run of indices.

        This is ``candidates`` in the form a caller that wants all of them at
        once can use without a Python loop per position: the answers are
        concatenated, and a length-``Q`` vector of counts says where each
        position's run begins. A batched search over thousands of positions
        reads them in this form, and building the list of separate arrays is
        then done once, by ``candidates``, on top of it.

        Parameters
        ----------
        query : numpy.ndarray
            A ``(D, Q)`` matrix of query positions.
        radius : float or numpy.ndarray
            The radius to search within: one for every position, or one per
            position. The items a position can reach differ from one position
            to the next --- a mesh's triangles are not all the same size --- so
            a caller asking about all of them at once gives one radius for
            each.

        Returns
        -------
        counts : numpy.ndarray
            A length-``Q`` vector of the number of candidates per position.
        found : numpy.ndarray
            The concatenated item indices, ``counts[q]`` of them for position
            ``q``, sorted within each run.
        '''
        query = asarray(query)
        if query.ndim == 1:
            query = query.reshape(-1, 1)
        if query.shape[0] != self.dim:
            raise ValueError(
                f"this tree subdivides {self.dim}-dimensional space, but the"
                f" query has dimension {query.shape[0]}")
        if query.shape[1] == 0:
            return (zeros(0, dtype=intp), zeros(0, dtype=intp))
        single = float(radius) if _is_number(radius) else None
        if single is None:
            radii = asarray(radius, dtype='float64')
            if radii.shape != (query.shape[1],):
                raise ValueError(
                    f"a radius per position takes one entry per position:"
                    f" {query.shape[1]}, not {radii.shape}")
        if c_spatial is not None:
            return c_spatial.candidates(
                *(self._flattened()), ascontiguousarray(query, dtype='float64'),
                (single if single is not None
                 else ascontiguousarray(radii, dtype='float64')))
        runs = [self._candidates_one(query[:, q],
                                     single if single is not None
                                     else float(radii[q]))
                for q in range(query.shape[1])]
        return (asarray([run.size for run in runs], dtype=intp),
                concatenate(runs))

    def candidates(self, query, radius, /):
        '''Returns the items whose spheres come within a radius of each query.

        The answer is *conservative in the safe direction*: an item is included
        whenever any point of its sphere is within ``radius`` of the position,
        so no item that could be the nearest is ever missed, though an item
        that is merely near may be included too.

        Parameters
        ----------
        query : numpy.ndarray
            A ``(D, Q)`` matrix of query positions.
        radius : float
            The radius to search within.

        Returns
        -------
        list of numpy.ndarray
            One sorted array of item indices per query position.
        '''
        (counts, found) = self.candidate_runs(query, radius)
        res = []
        at = 0
        for count in counts:
            res.append(found[at:at + count])
            at += count
        return res

    def _candidates_one(self, point, radius, /):
        '''The items within a radius of one position.'''
        found = []
        stack = [0]
        while stack:
            node = stack.pop()
            if self._box_distance(node, point) > radius + self._maxr[node]:
                continue
            if self._items[node] is not None:
                items = self._items[node]
                away = sqrt(((self.centers[:, items]
                              - point[:, None]) ** 2).sum(axis=0))
                near = items[away <= radius + self.radii[items]]
                found.extend(near.tolist())
            else:
                stack.extend(c for c in self._child[node] if c >= 0)
        found.sort()
        return asarray(found, dtype=int)

    def nearest(self, query, /, k=1):
        '''Returns the ``k`` items nearest each query position.

        An item is ranked by the distance from the position to its *sphere*,
        which is the lower bound of its true distance; the caller resolves the
        ranking exactly, as ``candidates`` allows.

        Parameters
        ----------
        query : numpy.ndarray
            A ``(D, Q)`` matrix of query positions.
        k : int, optional
            The number of items to return per position. The default is 1.

        Returns
        -------
        index : numpy.ndarray
            A ``(k, Q)`` matrix of item indices.
        distance : numpy.ndarray
            A ``(k, Q)`` matrix of the distances to those items' spheres.
        '''
        query = asarray(query)
        if query.ndim == 1:
            query = query.reshape(-1, 1)
        if c_spatial is not None:
            return c_spatial.nearest(
                *(self._flattened()), ascontiguousarray(query, dtype='float64'),
                int(k))
        indices = []
        distances = []
        for q in range(query.shape[1]):
            (idx, dist) = self._nearest_one(query[:, q], int(k))
            indices.append(idx)
            distances.append(dist)
        return (stack(indices, axis=1), stack(distances, axis=1))

    def _lower_bound(self, node, point, /):
        '''A distance no item of a node can be nearer than.

        A node's box bounds where its items *begin*, not where they end: an
        item's sphere may reach beyond the box that holds its center, so the
        bound is the distance to the box less the largest radius in the
        subtree, which is what ``_maxr`` stores.
        '''
        return max(self._box_distance(node, point) - self._maxr[node], 0.0)

    def _nearest_one(self, point, k, /):
        '''The ``k`` nearest items to one position.'''
        heap = [(self._lower_bound(0, point), 0)]
        # A max-heap of the best found so far, held as negative distances.
        best = []
        while heap:
            (bound, node) = heap[0]
            if len(best) >= k and bound > -best[0][0]:
                break
            (bound, node) = heappop(heap)
            if self._items[node] is not None:
                items = self._items[node]
                away = sqrt(((self.centers[:, items]
                              - point[:, None]) ** 2).sum(axis=0))
                away = maximum(away - self.radii[items], 0.0)
                for (distance, item) in zip(away.tolist(), items.tolist()):
                    if len(best) < k:
                        heappush(best, (-distance, item))
                    elif distance < -best[0][0]:
                        heapreplace(best, (-distance, item))
            else:
                for child in self._child[node]:
                    if child >= 0:
                        heappush(heap, (self._lower_bound(child, point),
                                        child))
        found = sorted((-distance, item) for (distance, item) in best)
        return (asarray([item for (_, item) in found], dtype=int),
                asarray([distance for (distance, _) in found]))


# Exports ####################################################################

__all__ = ('SpatialTree',)
