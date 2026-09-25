# -*- coding: utf-8 -*-
###############################################################################
# euclib/test/utils/test_spatial.py
'''Tests for the spatial subdivision in ``euclib.utils._spatial``.'''

# Dependencies ###############################################################

from __future__ import annotations

from unittest import TestCase

import numpy as np

from numpy import (
    allclose, arange, argsort, array, concatenate, flatnonzero, linspace,
    maximum, meshgrid, ones, sqrt, stack, zeros)

from euclib.utils import (
    SpatialTree, bounds_of, octree_split, quadtree_split, simplex_boxes,
    split_cells)
from euclib.utils import closest_simplex


# Fixtures ###################################################################

def grid_mesh(n):
    '''A triangulated ``n`` by ``n`` grid: small, well-shaped triangles.'''
    (xs, ys) = meshgrid(linspace(0, 1, n), linspace(0, 1, n))
    coords = stack([xs.reshape(-1), ys.reshape(-1)])
    triangles = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            triangles.append([a, b, d])
            triangles.append([a, d, c])
    return (coords, array(triangles).T)


def _near(points, query, radius):
    '''The indices of the points within a radius of each query, by brute force.'''
    return [flatnonzero(((points - query[:, [q]]) ** 2).sum(axis=0)
                        <= radius ** 2 + 1e-12)
            for q in range(query.shape[1])]


# Tests ######################################################################

class TestSubdivision(TestCase):
    '''Tests for the subdivision kernels.'''

    def test_bounds_of_points(self):
        coords = array([[0., 1., 0.5], [-2., 0., 3.]])
        self.assertTrue(allclose(bounds_of(coords), [[0., 1.], [-2., 3.]]))

    def test_bounds_of_simplices(self):
        coords = array([[0., 1., 0.5, 9.], [0., 0., 0.5, 9.]])
        # Only the corners of the listed simplices are counted, so the far
        # corner at (1, 0) is left out.
        self.assertTrue(allclose(bounds_of(coords, array([[0], [2], [3]])),
                                 [[0., 9.], [0., 9.]]))

    def test_simplex_boxes_contain_their_simplices(self):
        coords = array([[0., 1., 0.5], [0., 0., 0.5]])
        (centers, radii) = simplex_boxes(coords, array([[0], [1], [2]]))
        self.assertTrue(allclose(centers, [[0.5], [0.25]]))
        self.assertTrue(allclose(radii, [sqrt(0.5 ** 2 + 0.25 ** 2)]))

    def test_a_split_places_points_in_the_halves_that_hold_them(self):
        bounds = array([[0., 1.], [0., 1.]])
        points = array([[0.1, 0.9, 0.4, 0.6], [0.1, 0.9, 0.6, 0.4]])
        # Bit 0 is the upper half along x, bit 1 the upper half along y.
        self.assertEqual(split_cells(points, bounds).tolist(), [0, 3, 2, 1])

    def test_a_split_in_three_dimensions(self):
        bounds = zeros((3, 2)) + [[0., 1.]]
        points = array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1]])
        self.assertEqual(split_cells(points, bounds).tolist(), [4, 3])

    def test_the_named_splits_agree_with_the_general_one(self):
        bounds2 = zeros((2, 2)) + [[0., 1.]]
        bounds3 = zeros((3, 2)) + [[0., 1.]]
        points = array([[0.1, 0.9], [0.2, 0.8]])
        self.assertEqual(quadtree_split(points, bounds2).tolist(),
                         split_cells(points, bounds2).tolist())
        points3 = concatenate([points, [[0.3, 0.7]]], axis=0)
        self.assertEqual(octree_split(points3, bounds3).tolist(),
                         split_cells(points3, bounds3).tolist())

    def test_the_named_splits_check_their_dimensions(self):
        with self.assertRaises(ValueError):
            octree_split(zeros((2, 4)), zeros((2, 2)))
        with self.assertRaises(ValueError):
            quadtree_split(zeros((3, 4)), zeros((3, 2)))

    def test_a_split_checks_that_points_and_box_agree(self):
        with self.assertRaises(ValueError):
            split_cells(zeros((2, 4)), zeros((3, 2)))


class TestSpatialTree(TestCase):
    '''Tests for the quadtree and octree.'''

    def test_a_tree_reports_its_dimension(self):
        self.assertEqual(SpatialTree(zeros((2, 4)), zeros(4)).dim, 2)
        self.assertEqual(SpatialTree(zeros((3, 4)), zeros(4)).dim, 3)

    def test_it_checks_its_arguments(self):
        # Radii must have one entry per item.
        with self.assertRaises(ValueError):
            SpatialTree(zeros((2, 4)), zeros(3))
        # Centers must be a matrix.
        with self.assertRaises(ValueError):
            SpatialTree(zeros(4), zeros(4))
        # A query must match the tree's dimension.
        tree = SpatialTree(zeros((2, 4)), zeros(4))
        with self.assertRaises(ValueError):
            tree.candidates(zeros((3, 2)), 0.1)

    def test_an_empty_tree_answers_nothing(self):
        tree = SpatialTree(zeros((2, 0)), zeros(0))
        self.assertEqual([c.tolist() for c in tree.candidates(ones((2, 3)), 1.)],
                         [[], [], []])

    def test_a_tree_finds_its_own_items(self):
        tree = SpatialTree(zeros((2, 4)), zeros(4))
        found = tree.candidates(zeros((2, 1)), 0.5)[0]
        self.assertEqual(found.tolist(), [0, 1, 2, 3])

    def test_candidates_agree_with_brute_force_for_points(self):
        # A point has no extent, so the tree's answer must be exactly the set
        # of points within the radius.
        points = (arange(200).reshape(2, 100) % 10) / 9.0
        query = array([[0.05, 0.5, 0.95], [0.05, 0.5, 0.95]])
        tree = SpatialTree(points, zeros(points.shape[1]))
        for radius in (0.01, 0.15, 0.5):
            with self.subTest(radius=radius):
                got = tree.candidates(query, radius)
                want = _near(points, query, radius)
                self.assertTrue(all(allclose(a, b)
                                    for (a, b) in zip(got, want)))

    def test_nearest_agrees_with_brute_force(self):
        (coords, tri) = grid_mesh(12)
        (centers, radii) = simplex_boxes(coords, tri)
        tree = SpatialTree(centers, radii)
        query = array([[0.1, 0.37, 0.8], [0.2, 0.51, 0.65]])
        (index, distance) = tree.nearest(query, k=3)
        for q in range(query.shape[1]):
            away = maximum(sqrt(((centers - query[:, [q]]) ** 2).sum(axis=0))
                           - radii, 0.0)
            # The *distances* are what a nearest search promises. Which item it
            # names at a given distance is not fixed when several are equally
            # near, so the items it does name are checked against the distances
            # it reports rather than against a particular tie-breaking order.
            self.assertTrue(allclose(sorted(distance[:, q].tolist()),
                                     sorted(away)[:3]))
            self.assertTrue(allclose(away[index[:, q]], distance[:, q]))

    def test_candidates_always_contain_the_nearest_simplex(self):
        # The property the searches depend on: whatever radius is used, an item
        # whose nearest point is within it is among that query's candidates.
        (coords, tri) = grid_mesh(20)
        (centers, radii) = simplex_boxes(coords, tri)
        tree = SpatialTree(centers, radii)
        query = array([[0.11, 0.63, 0.42], [0.29, 0.07, 0.55]])
        (index, weight) = closest_simplex(coords, tri, query)
        for radius in (0.01, 0.05, 0.2):
            with self.subTest(radius=radius):
                cands = tree.candidates(query, radius)
                for q in range(query.shape[1]):
                    corners = tri[:, index[q]]
                    found = (coords[:, corners[0]] * weight[0, q]
                             + coords[:, corners[1]] * weight[1, q]
                             + coords[:, corners[2]] * (1.0 - weight[:, q].sum()))
                    away = sqrt(((query[:, q] - found) ** 2).sum())
                    if away <= radius:
                        self.assertIn(int(index[q]), cands[q].tolist())

    def test_a_tree_does_not_miss_an_item_that_sticks_out_of_its_cell(self):
        # An item whose center is left of the midpoint but whose sphere reaches
        # right of it must still be found from a query on the right, even
        # though the two are in different cells.
        centers = array([[0.1, 0.4], [0.5, 0.5]])
        radii = array([0.0, 0.3])
        tree = SpatialTree(centers, radii, bounds=zeros((2, 2)) + [[0., 1.]],
                           max_items=1, max_depth=4)
        found = tree.candidates(array([[0.75], [0.5]]), 0.1)[0]
        self.assertIn(1, found.tolist())
        # The nearest search must account for it too, which is what the lower
        # bound's subtraction of the subtree's largest radius is for.
        (index, _) = tree.nearest(array([[0.75], [0.5]]), k=1)
        self.assertEqual(int(index[0, 0]), 1)

    def test_a_tree_respects_its_depth_limit(self):
        # Every item in one spot: the subdivision cannot separate them, so it
        # must stop at the depth it was given rather than recursing forever.
        tree = SpatialTree(zeros((2, 100)), zeros(100), max_items=1,
                           max_depth=3)
        self.assertEqual((tree.candidates(zeros((2, 1)), 0.5)[0]).size, 100)


class TestCandidateRuns(TestCase):
    '''The flat form of the candidates, which the batched search reads.'''

    def test_the_runs_are_the_arrays(self):
        (coords, tri) = grid_mesh(12)
        (centers, radii) = simplex_boxes(coords, tri)
        tree = SpatialTree(centers, radii)
        query = array([[0.05, 0.31, 0.6, 0.99], [0.05, 0.72, 0.4, 0.01]])
        for radius in (0.01, 0.1, 0.5):
            with self.subTest(radius=radius):
                (counts, found) = tree.candidate_runs(query, radius)
                separate = tree.candidates(query, radius)
                self.assertEqual(counts.tolist(),
                                 [run.size for run in separate])
                at = 0
                for run in separate:
                    self.assertEqual(found[at:at + run.size].tolist(),
                                     run.tolist())
                    at += run.size

    def test_an_empty_query_has_no_runs(self):
        (coords, tri) = grid_mesh(8)
        tree = SpatialTree(*simplex_boxes(coords, tri))
        (counts, found) = tree.candidate_runs(zeros((2, 0)), 0.5)
        self.assertEqual(counts.size, 0)
        self.assertEqual(found.size, 0)
        self.assertEqual(tree.candidates(zeros((2, 0)), 0.5), [])


class TestBatchedClosestSimplex(TestCase):
    '''The indexed nearest-simplex search, which advances its positions together.

    The brute-force search over every simplex is the definition, and the indexed
    one must agree with it exactly: it is the same fit, over the candidates the
    index promises contain the answer. That promise is what makes the two the
    same search rather than two similar ones, so these tests run the index at
    several radii and several sizes of query.
    '''

    def test_it_agrees_with_examining_every_simplex(self):
        (coords, tri) = grid_mesh(20)
        (centers, radii) = simplex_boxes(coords, tri)
        tree = SpatialTree(centers, radii)
        rng = np.random.default_rng(0)
        # Inside the grid, on its edges, and well outside it, where the answer
        # is a corner of some boundary triangle.
        inside = rng.uniform(0.0, 1.0, size=(2, 60))
        outside = rng.uniform(-0.5, 1.5, size=(2, 40))
        query = concatenate([inside, outside], axis=1)
        (index, weight) = closest_simplex(coords, tri, query, tree=tree)
        (want_index, want_weight) = closest_simplex(coords, tri, query)
        self.assertTrue((index == want_index).all(),
                        f"{(index != want_index).sum()} of {query.shape[1]}"
                        " positions took a different simplex")
        self.assertTrue(allclose(weight, want_weight, atol=1e-12))

    def test_it_agrees_one_position_at_a_time(self):
        # The radius the search uses is shared by the positions still looking,
        # so a search of one position must agree with the same position inside a
        # larger batch: the shared radius must not change any answer.
        (coords, tri) = grid_mesh(16)
        tree = SpatialTree(*simplex_boxes(coords, tri))
        rng = np.random.default_rng(3)
        query = rng.uniform(-0.2, 1.2, size=(2, 30))
        (whole_index, whole_weight) = closest_simplex(coords, tri, query,
                                                      tree=tree)
        for q in range(query.shape[1]):
            with self.subTest(position=q):
                (index, weight) = closest_simplex(coords, tri, query[:, q:q + 1],
                                                  tree=tree)
                self.assertEqual(int(index[0]), int(whole_index[q]))
                self.assertTrue(allclose(weight[:, 0], whole_weight[:, q],
                                         atol=1e-12))

    def test_a_position_on_a_vertex_takes_that_vertex(self):
        (coords, tri) = grid_mesh(10)
        tree = SpatialTree(*simplex_boxes(coords, tri))
        # A grid node is shared by several triangles and is a corner of each, so
        # the answer is fixed: the position itself, with all of the weight on
        # the corner the search names.
        query = coords[:, [0, 33, 66]].copy()
        (index, weight) = closest_simplex(coords, tri, query, tree=tree)
        for q in range(query.shape[1]):
            corners = tri[:, index[q]]
            self.assertIn(0, (coords[:, corners] - query[:, [q]]).__abs__()
                          .sum(axis=0).round(9).tolist())
            self.assertTrue(allclose(
                coords[:, corners] @ concatenate([weight[:, q],
                                                  [1.0 - weight[:, q].sum()]]),
                query[:, q], atol=1e-9))

    def test_many_positions_at_once_agree(self):
        # A batch large enough to take the slots a candidate at a time rather
        # than all at once: the shapes of the fits must not depend on it.
        (coords, tri) = grid_mesh(40)
        tree = SpatialTree(*simplex_boxes(coords, tri))
        rng = np.random.default_rng(11)
        query = rng.uniform(-0.1, 1.1, size=(2, 1200))
        (index, weight) = closest_simplex(coords, tri, query, tree=tree)
        (want_index, want_weight) = closest_simplex(coords, tri, query)
        self.assertTrue((index == want_index).all())
        self.assertTrue(allclose(weight, want_weight, atol=1e-12))

    def test_an_empty_query_answers_nothing(self):
        (coords, tri) = grid_mesh(8)
        tree = SpatialTree(*simplex_boxes(coords, tri))
        (index, weight) = closest_simplex(coords, tri, zeros((2, 0)), tree=tree)
        self.assertEqual(index.size, 0)
        self.assertEqual(weight.shape, (2, 0))

    def test_the_escape_hatch_agrees(self):
        # A search that widens until nearly every simplex is a candidate is
        # answered from the whole mesh instead, which is correct rather than
        # merely equivalent --- the candidates are a subset of what the global
        # search examines. The threshold is lowered here to reach it.
        from euclib.utils import _pycore
        (coords, tri) = grid_mesh(20)
        tree = SpatialTree(*simplex_boxes(coords, tri))
        rng = np.random.default_rng(5)
        query = rng.uniform(-0.3, 1.3, size=(2, 200))
        (want_index, want_weight) = closest_simplex(coords, tri, query)
        saved = _pycore._MAX_SLOTS
        try:
            _pycore._MAX_SLOTS = 3
            (index, weight) = closest_simplex(coords, tri, query, tree=tree)
        finally:
            _pycore._MAX_SLOTS = saved
        self.assertTrue((index == want_index).all())
        self.assertTrue(allclose(weight, want_weight, atol=1e-12))
