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

from euclib import grid, ops, tetmesh
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

def main():
    '''Prints the timings.'''
    print(f"using the C extension: {using_c_extension}\n")
    bench_split_cells()
    bench_vertices()
    bench_voxel_intersections()


if __name__ == '__main__':
    main()
