# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/bench_c_kernels.py
'''Measures the C kernels against the pure-Python ones.

This is a harness rather than a test: what it prints depends on the machine, so
nothing here asserts a speedup. It exists so that a claim of acceleration can be
checked rather than believed, and so that the operations built on the kernels
can be measured as a whole --- a kernel that is three times faster is of no
interest if the operation around it spends its time elsewhere.

Run it with::

    python3 -m euclib.test.bench_c_kernels

Each row reports the two implementations, the ratio between them, and the
dtype and shape of the answer, so that a silent disagreement between them would
be visible alongside the timings.
'''

# Dependencies ###############################################################

from __future__ import annotations

from time import perf_counter

import numpy as np

from euclib import grid, ops, tetmesh, trimesh
from euclib.utils import split_cells, tetrahedron_box_vertices
from euclib.utils._core import using_c_extension


# Timing #####################################################################

def timeit(call, *args, repeat=7, **kwargs):
    '''Returns the best of several timings of a call, in seconds.

    The best rather than the mean: what is wanted is the cost of the call, and
    a slow run reflects the machine rather than the kernel.
    '''
    best = float('inf')
    for _ in range(repeat):
        start = perf_counter()
        call(*args, **kwargs)
        best = min(best, perf_counter() - start)
    return best


def row(label, native, accelerated, *args, **kwargs):
    '''Times both implementations of a call and prints them side by side.'''
    answer = native(*args, **kwargs)
    shape = getattr(answer, 'shape', '-')
    dtype = getattr(getattr(answer, 'dtype', None), 'name', '-')
    slow = timeit(native, *args, **kwargs)
    fast = timeit(accelerated, *args, **kwargs)
    speedup = slow / fast if fast else float('inf')
    print(f"  {label:<34} {slow * 1e3:>9.3f} ms {fast * 1e3:>9.3f} ms"
          f" {speedup:>7.2f}x   {str(shape):<12} {dtype}")


# Cases ######################################################################

def bench_split_cells():
    '''Times the bisection at the sizes a spatial index meets.'''
    print("split_cells (the bisection a quadtree or octree repeats)")
    native = split_cells.native
    accelerated = split_cells.accelerated
    if accelerated is None:
        print("  the C extension is not built")
        return
    rng = np.random.default_rng(0)
    bounds = np.array([[-2., 2.], [-2., 2.], [-2., 2.]])
    for count in (1000, 100000, 1000000):
        centers = np.ascontiguousarray(rng.normal(size=(3, count)),
                                       dtype='float64')
        row(f"{count} points", native, accelerated, centers, bounds)
    centers = np.ascontiguousarray(rng.normal(size=(3, 1000000)),
                                   dtype='float64')
    row("1000000 points, 2 dimensions", native, accelerated,
        np.ascontiguousarray(centers[:2]), bounds[:2])
    print()


def bench_vertices():
    '''Times the corner search that a voxel intersection is built from.'''
    print("tetrahedron_box_vertices (the corner search)")
    native = tetrahedron_box_vertices.native
    accelerated = tetrahedron_box_vertices.accelerated
    if accelerated is None:
        print("  the C extension is not built")
        return
    rng = np.random.default_rng(1)
    for count in (1, 100, 10000):
        pairs = []
        for _ in range(count):
            tet = np.ascontiguousarray(rng.normal(size=(3, 4)), dtype='float64')
            center = rng.normal(size=3) * 0.6
            half = rng.uniform(0.2, 1.5, size=3)
            pairs.append((tet, np.ascontiguousarray(
                np.stack([center - half, center + half], axis=1))))
        (tet, bounds) = pairs[0]
        label = "one pair" if count == 1 else f"{count} pairs"
        slow = timeit(native, tet, bounds)
        fast = timeit(accelerated, tet, bounds)
        print(f"  {label:<34} {slow * 1e6:>9.3f} us {fast * 1e6:>9.3f} us"
              f" {slow / fast:>7.2f}x")
        if count > 1:
            def all_python():
                return [native(t, b) for (t, b) in pairs]

            def all_c():
                return [accelerated(t, b) for (t, b) in pairs]

            slow = timeit(all_python, repeat=3)
            fast = timeit(all_c, repeat=3)
            print(f"  {'(as a batch)':<34} {slow * 1e3:>9.3f} ms"
                  f" {fast * 1e3:>9.3f} ms {slow / fast:>7.2f}x")
    print()


def _tet_block(count, /):
    '''Builds a tetrahedral mesh of a ``count``-cubed block of unit cubes.

    Each cube is cut into six tetrahedra by the Kuhn decomposition: take the
    corner at one end, the corner at the other, and, for each of the six orders
    in which the three axes can be taken, the two corners those steps reach.
    Every cube is cut the same way, so the faces where two cubes meet are cut
    the same way on both sides and the pieces fit together.
    '''
    corners = []
    tets = []
    for i in range(count):
        for j in range(count):
            for k in range(count):
                base = len(corners)
                for x in range(2):
                    for y in range(2):
                        for z in range(2):
                            corners.append((i + x, j + y, k + z))
                low = base
                high = base + (1 + 2 + 4)      # index of the (1, 1, 1) corner
                for (a, b) in ((0, 1), (0, 2), (1, 0),
                               (1, 2), (2, 0), (2, 1)):
                    first = low + (1 << a)
                    second = low + (1 << a) + (1 << b)
                    tets.append([low, first, second, high])
    return (np.array(corners, dtype=float).T, np.array(tets, dtype=int).T)


def bench_voxel_intersections():
    '''Times the operation the corner search serves, end to end.'''
    print("ops.voxel_intersections (the operation built on it)")
    if not using_c_extension:
        print("  the C extension is not built")
        return
    (coords, indices) = _tet_block(6)
    mesh = tetmesh(coords, indices)
    cells = grid((12, 12, 12), affine=np.array([[0.5, 0., 0., 0.],
                                                [0., 0.5, 0., 0.],
                                                [0., 0., 0.5, 0.],
                                                [0., 0., 0., 1.]]))
    from euclib.utils import _pycore
    print(f"  mesh: {mesh.topo.simplex_count[3]} tetrahedra,"
          f" grid: {cells.shape}")
    # The operation takes the corner search through the dispatcher, so the two
    # routes are compared by pointing the dispatcher at each in turn.
    dispatching = _pycore._vertices_kernel
    try:
        for (label, kernel) in (("pure Python", dispatching.native),
                                ("C extension", dispatching.accelerated)):
            if kernel is None:
                continue
            _pycore._vertices_kernel = kernel
            start = perf_counter()
            (pieces, from_tet, from_voxel) = ops.voxel_intersections(
                mesh, cells)
            elapsed = perf_counter() - start
            print(f"  {label:<34} {elapsed * 1e3:>9.3f} ms"
                  f"   {from_voxel.shape[1]} pieces")
    finally:
        _pycore._vertices_kernel = dispatching
    print()


# Main #######################################################################

def _sheet(side):
    '''A triangulated sheet, as a mesh and its coordinates.'''
    axis = np.linspace(0.0, 1.0, side)
    (gridx, gridy) = np.meshgrid(axis, axis)
    coords = np.stack([gridx.ravel(), gridy.ravel()])
    faces = []
    for (i, j) in np.ndindex(side - 1, side - 1):
        (a, b) = (i * side + j, i * side + j + 1)
        (c, d) = ((i + 1) * side + j, (i + 1) * side + j + 1)
        faces += [[a, b, d], [a, d, c]]
    mesh = trimesh(coords, np.array(faces).T)
    values = np.sin(coords[0] * 12.0) * np.cos(coords[1] * 12.0)
    gradient = np.stack([12.0 * np.cos(coords[0] * 12.0)
                         * np.cos(coords[1] * 12.0),
                         -12.0 * np.sin(coords[0] * 12.0)
                         * np.sin(coords[1] * 12.0)])
    return (mesh, coords, values, gradient)


def bench_index_queries():
    '''Times the index queries one position at a time and as one array.

    The compiled queries answer a whole array of positions in a call, and they
    win at every size --- including one position, once the tree's arrays are
    contiguous when they reach them. The one-position row is what the search
    does today, once per position; the batched rows are what it would do if it
    advanced its queries together, which is the work the plan lists next.
    '''
    print("the spatial index's queries (per position, and batched)")
    from euclib.utils import _spatial
    (mesh, coords, values, gradient) = _sheet(40)
    mesh = mesh.withprop('f', values, gradient=gradient)
    tree = mesh.spatial_index
    if tree is None:
        print("  no index was built for this mesh")
        return
    rng = np.random.default_rng(0)
    where = coords[:, rng.choice(coords.shape[1], 200)]
    fast = _spatial.c_spatial
    # The compiled queries read the tree through arrays flattened the first time
    # they are asked for something. That is a cost of the tree, and charging it
    # to the first row would read as the kernels being slow on one position ---
    # which is the mistake the contiguous-array fix was about, so it is settled
    # here rather than in the numbers.
    flat = tree._flattened()
    for (label, kernel) in (("python tree", None), ("compiled", fast)):
        if kernel is None:
            continue
        _spatial.c_spatial = kernel
        tree.nearest(where[:, :1], k=1)
        tree.candidates(where[:, :1], 0.02)
    _spatial.c_spatial = fast
    for count in (1, 10, 100, 200):
        one = np.ascontiguousarray(where[:, :count])
        _spatial.c_spatial = None
        start = perf_counter()
        for i in range(count):
            tree.nearest(one[:, i:i + 1], k=1)
            tree.candidates(one[:, i:i + 1], 0.02)
        slow = perf_counter() - start
        _spatial.c_spatial = fast
        start = perf_counter()
        tree.nearest(one, k=1)
        tree.candidates(one, 0.02)
        quick = perf_counter() - start
        ratio = slow / quick if quick else float('inf')
        print(f"  {count:5d} positions  python {slow * 1e6:9.1f} us"
              f"   compiled {quick * 1e6:9.1f} us   {ratio:6.2f}x")
    _spatial.c_spatial = fast
    print()


def bench_interpolation():
    '''Times interpolation with a supplied gradient and with an estimated one.

    Nothing caches the estimate, so a property that carries no gradient pays for
    one on every call; these two rows are what that costs, and are the reason
    the plan records caching as an open item rather than a solved one.
    '''
    print("interpolating a property (the cost of not caching the estimate)")
    (mesh, coords, values, gradient) = _sheet(40)
    supplied = mesh.withprop('f', values, gradient=gradient)
    estimated = mesh.withprop('f', values)
    rng = np.random.default_rng(0)
    where = np.ascontiguousarray(coords[:, rng.choice(coords.shape[1], 200)])
    nodes = coords.shape[1]
    # The index is built on the first call that locates a position, which the
    # "supplied" row would otherwise pay and the "estimated" row would not ---
    # reading as an estimate that is *faster* than no estimate at all, which is
    # how this row first came out.
    supplied.prop('f', at=where[:, :1], interp=1)
    for order in (1, 2, 3):
        taken = {}
        for (label, geom) in (("supplied", supplied), ("estimated", estimated)):
            start = perf_counter()
            geom.prop('f', at=where, interp=order)
            taken[label] = perf_counter() - start
        extra = taken['estimated'] - taken['supplied']
        print(f"  order {order}: with a gradient {taken['supplied'] * 1e3:7.1f} ms"
              f"   estimated {taken['estimated'] * 1e3:7.1f} ms"
              f"   the estimate adds {extra * 1e3:7.1f} ms")
    print(f"  ({nodes} vertices in this mesh; the estimate is made once per"
          f" call, for every vertex, and cached nowhere)")
    print()


def bench_search():
    '''Times the nearest-simplex search as the number of positions grows.

    This is what an interpolation pays before it can fit anything, and it is the
    whole cost of locating positions. A row that is flat in the number of
    positions is the work being array-shaped; a row that grows with it is a
    Python loop per position wearing the mask of a batched call.
    '''
    print("locating positions in a mesh (the nearest-simplex search)")
    (mesh, coords, values, gradient) = _sheet(60)
    rng = np.random.default_rng(0)
    nodes = coords.shape[1]
    # The index is built, and flattened, the first time a position is located,
    # which is a cost of the geometry rather than of the search; one call takes
    # it out of the first row.
    mesh.to_local(coords[:, :1])
    for count in (10, 100, 1000, 10000):
        where = np.ascontiguousarray(
            coords[:, rng.choice(coords.shape[1], count)])
        start = perf_counter()
        mesh.to_local(where)
        took = perf_counter() - start
        print(f"  {count:6d} positions  {took * 1e3:9.2f} ms"
              f"   {took / count * 1e6:8.2f} us each")
    print(f"  ({nodes} vertices, {mesh.topo.simplex_count.sum()} simplices;"
          f" the index exists above {mesh.spatial_index_min_items})")
    print()


def main():
    '''Prints the timings.'''
    print(f"using the C extension: {using_c_extension}\n")
    bench_split_cells()
    bench_vertices()
    bench_index_queries()
    bench_interpolation()
    bench_search()
    bench_voxel_intersections()


if __name__ == '__main__':
    main()
