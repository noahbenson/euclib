/* -*- coding: utf-8 -*-
 *******************************************************************************
 * euclib/_c/_spatial.c
 *
 * The C counterpart of the spatial index's queries.
 *
 * The pure-Python tree in euclib.utils._spatial answers one query position at a
 * time, with a Python-level descent of the tree for each. That is the shape of
 * code the library is not supposed to use: a caller interpolating a property at
 * as many positions as a mesh has vertices pays the interpreter's overhead for
 * every one of them. These kernels do the same descents over an *array* of
 * positions in one call, which is what makes them worth having in C rather than
 * as another Python loop.
 *
 * Two queries, matching the two the Python tree offers:
 *
 *   nearest     the k items whose spheres come nearest each position, ranked
 *               by the distance to the sphere, which is a lower bound on the
 *               distance to the item itself; the caller resolves the ranking
 *               exactly, as the index is built to allow.
 *   candidates  every item whose sphere comes within a radius of a position,
 *               conservative in the safe direction: no item that could be the
 *               nearest is ever missed, though an item that is merely near may
 *               be included.
 *
 * Both are defined by the pure-Python implementations, which the tests run them
 * against; this file is an acceleration of those, not a second definition of
 * what they mean. The tie-breaking is part of that: a heap of (distance,
 * index) pairs in Python breaks ties by index, and so do these.
 *
 * The tree arrives already flattened: one box per node, the largest radius in
 * each node's subtree, and the items and children of each node in a
 * compressed-sparse-row layout, so that a node's contents are a contiguous
 * range of an array.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* The greatest number of dimensions a tree may subdivide. */
#define MAX_DIM 3

/* The distance from a position to a node's box. */
static double
box_distance(const double *low, const double *high, int dim,
             const double *point)
{
    double total = 0.0;
    int i;
    for (i = 0; i < dim; ++i) {
        double below = low[i] - point[i];
        double above = point[i] - high[i];
        double excess = below > above ? below : above;
        if (excess > 0.0) {
            total += excess * excess;
        }
    }
    return sqrt(total);
}

/* The distance from a position to an item's sphere, at least zero: an item's
 * sphere is what the index bounds an item by, and a position inside the sphere
 * is at no distance from it. */
static double
sphere_distance(const double *centers, const double *radii, int dim,
                npy_intp count, npy_intp item, const double *point)
{
    double total = 0.0;
    int i;
    for (i = 0; i < dim; ++i) {
        /* A (D, M) matrix of centers is contiguous along the items, so one
         * item's coordinate is a stride of the whole item count away. */
        double away = centers[i * count + item] - point[i];
        total += away * away;
    }
    total = sqrt(total) - radii[item];
    return total > 0.0 ? total : 0.0;
}


/* A node stack that grows as the descent needs it *****************************/

typedef struct {
    double bound;
    npy_intp node;
} entry;

typedef struct {
    entry *data;
    npy_intp size;
    npy_intp capacity;
} heap;

static int
heap_reserve(heap *h, npy_intp wanted)
{
    if (wanted <= h->capacity) {
        return 0;
    }
    {
        npy_intp capacity = h->capacity ? h->capacity * 2 : 64;
        entry *grown = NULL;
        while (capacity < wanted) {
            capacity *= 2;
        }
        grown = (entry *)realloc(h->data, (size_t)capacity * sizeof(entry));
        if (grown == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        h->data = grown;
        h->capacity = capacity;
    }
    return 0;
}

/* Pushes an entry, ordering the heap by bound and then by node, which is the
 * order Python's heapq gives a sequence of tuples. A *minimum* heap, used for
 * the nodes still to examine. */
static int
heap_push(heap *h, double bound, npy_intp node)
{
    npy_intp i;
    if (heap_reserve(h, h->size + 1) != 0) {
        return -1;
    }
    i = h->size++;
    h->data[i].bound = bound;
    h->data[i].node = node;
    while (i > 0) {
        npy_intp parent = (i - 1) / 2;
        if (h->data[parent].bound < h->data[i].bound
                || (h->data[parent].bound == h->data[i].bound
                    && h->data[parent].node <= h->data[i].node)) {
            break;
        }
        {
            entry swap = h->data[parent];
            h->data[parent] = h->data[i];
            h->data[i] = swap;
        }
        i = parent;
    }
    return 0;
}

static entry
heap_pop(heap *h)
{
    entry top = h->data[0];
    npy_intp i = 0;
    --h->size;
    h->data[0] = h->data[h->size];
    for (;;) {
        npy_intp left = 2 * i + 1;
        npy_intp right = left + 1;
        npy_intp least = i;
        if (left < h->size
                && (h->data[left].bound < h->data[least].bound
                    || (h->data[left].bound == h->data[least].bound
                        && h->data[left].node < h->data[least].node))) {
            least = left;
        }
        if (right < h->size
                && (h->data[right].bound < h->data[least].bound
                    || (h->data[right].bound == h->data[least].bound
                        && h->data[right].node < h->data[least].node))) {
            least = right;
        }
        if (least == i) {
            break;
        }
        {
            entry swap = h->data[least];
            h->data[least] = h->data[i];
            h->data[i] = swap;
        }
        i = least;
    }
    return top;
}

/* A stack of node indices, for the traversals whose order does not matter.

 * The candidates query visits every node whose box comes within the radius and
 * collects the items it finds; the order it visits them in changes nothing, so
 * the pending nodes need no ordering. A heap here would cost a comparison and a
 * swap per node for an answer nobody reads, which on a tree of fifty thousand
 * nodes is the difference between a query taking microseconds and taking a
 * millisecond.
 */
typedef struct {
    npy_intp *data;
    npy_intp size;
    npy_intp capacity;
} nodelist;

static int
nodelist_push(nodelist *s, npy_intp node)
{
    if (s->size >= s->capacity) {
        npy_intp capacity = s->capacity ? s->capacity * 2 : 64;
        npy_intp *grown = (npy_intp *)realloc(s->data,
                                              (size_t)capacity
                                              * sizeof(npy_intp));
        if (grown == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        s->data = grown;
        s->capacity = capacity;
    }
    s->data[s->size++] = node;
    return 0;
}

static npy_intp
nodelist_pop(nodelist *s)
{
    return s->data[--s->size];
}


/* The greatest bound no item of a node can be nearer than. */
static double
lower_bound(const double *low, const double *high, const double *maxr,
            int dim, npy_intp node, const double *point)
{
    double distance = box_distance(low + node * dim, high + node * dim, dim,
                                   point);
    double bound = distance - maxr[node];
    return bound > 0.0 ? bound : 0.0;
}


/* The tree as the kernels read it ********************************************/

typedef struct {
    int dim;
    const double *low;
    const double *high;
    const double *maxr;
    const npy_intp *item_start;
    const npy_intp *items;
    const npy_intp *child_start;
    const npy_intp *children;
    const double *centers;
    const double *radii;
    npy_intp nodes;
    npy_intp item_count;
} tree;


/* Reading the arguments ******************************************************/

/* Takes a contiguous array of a given dtype and dimensionality. */
static PyArrayObject *
field(PyObject *obj, int typenum, int ndim, const char *name)
{
    PyArrayObject *found = (PyArrayObject *)PyArray_FROM_OTF(
        obj, typenum, NPY_ARRAY_IN_ARRAY);
    if (found == NULL) {
        return NULL;
    }
    if (PyArray_NDIM(found) != ndim) {
        PyErr_Format(PyExc_ValueError, "%s must have %d dimensions", name,
                     ndim);
        Py_DECREF(found);
        return NULL;
    }
    return found;
}

/* The tree's arrays, and the query, all read into a `tree`. The caller keeps
 * the arrays alive until the descent is done. */
#define TREE_FIELDS 9
static int
read_tree(PyObject *args, int *dim_out, PyArrayObject **keep,
          PyArrayObject **query_out, npy_intp *k_out, double *radius_out,
          tree *out, int want_radius)
{
    PyObject *low = NULL, *high = NULL, *maxr = NULL, *istart = NULL,
             *items = NULL, *cstart = NULL, *children = NULL, *centers = NULL,
             *radii = NULL, *query = NULL;
    double radius = 0.0;
    npy_intp k = 1;
    int dim = 0;

    if (want_radius) {
        if (!PyArg_ParseTuple(args, "OOOOOOOOOOd", &low, &high, &maxr,
                              &istart, &items, &cstart, &children, &centers,
                              &radii, &query, &radius)) {
            return -1;
        }
    } else {
        if (!PyArg_ParseTuple(args, "OOOOOOOOOOn", &low, &high, &maxr, &istart,
                              &items, &cstart, &children, &centers, &radii,
                              &query, &k)) {
            return -1;
        }
    }

    keep[0] = field(low, NPY_DOUBLE, 2, "low");
    keep[1] = field(high, NPY_DOUBLE, 2, "high");
    keep[2] = field(maxr, NPY_DOUBLE, 1, "maxr");
    keep[3] = field(istart, NPY_INTP, 1, "item_start");
    keep[4] = field(items, NPY_INTP, 1, "items");
    keep[5] = field(cstart, NPY_INTP, 1, "child_start");
    keep[6] = field(children, NPY_INTP, 1, "children");
    keep[7] = field(centers, NPY_DOUBLE, 2, "centers");
    keep[8] = field(radii, NPY_DOUBLE, 1, "radii");
    keep[9] = field(query, NPY_DOUBLE, 2, "query");
    {
        int i;
        for (i = 0; i < 10; ++i) {
            if (keep[i] == NULL) {
                return -1;
            }
        }
    }
    dim = (int)PyArray_DIM(keep[0], 1);
    if (dim < 1 || dim > MAX_DIM) {
        PyErr_Format(PyExc_ValueError,
                     "a tree subdivides 1, 2, or 3 dimensions; this one has"
                     " %d", dim);
        return -1;
    }
    if ((int)PyArray_DIM(keep[10 - 1], 0) != dim) {
        PyErr_SetString(PyExc_ValueError,
                        "the query has a different dimension than the tree");
        return -1;
    }
    if (PyArray_DIM(keep[7], 0) != dim || PyArray_DIM(keep[0], 0)
            != PyArray_DIM(keep[1], 0)) {
        PyErr_SetString(PyExc_ValueError,
                        "the tree's arrays disagree about their dimensions");
        return -1;
    }
    if (PyArray_DIM(keep[2], 0) != PyArray_DIM(keep[0], 0)
            || PyArray_DIM(keep[3], 0) != PyArray_DIM(keep[0], 0) + 1
            || PyArray_DIM(keep[5], 0) != PyArray_DIM(keep[0], 0) + 1) {
        PyErr_SetString(PyExc_ValueError,
                        "the tree's per-node arrays disagree about how many"
                        " nodes there are");
        return -1;
    }
    out->dim = dim;
    out->low = (const double *)PyArray_DATA(keep[0]);
    out->high = (const double *)PyArray_DATA(keep[1]);
    out->maxr = (const double *)PyArray_DATA(keep[2]);
    out->item_start = (const npy_intp *)PyArray_DATA(keep[3]);
    out->items = (const npy_intp *)PyArray_DATA(keep[4]);
    out->child_start = (const npy_intp *)PyArray_DATA(keep[5]);
    out->children = (const npy_intp *)PyArray_DATA(keep[6]);
    out->centers = (const double *)PyArray_DATA(keep[7]);
    out->radii = (const double *)PyArray_DATA(keep[8]);
    out->nodes = PyArray_DIM(keep[0], 0);
    out->item_count = PyArray_DIM(keep[7], 1);
    *dim_out = dim;
    *query_out = keep[9];
    *k_out = k;
    if (radius_out != NULL) {
        *radius_out = radius;
    }
    return 0;
}


/* nearest ********************************************************************/

PyDoc_STRVAR(nearest_doc,
"nearest(low, high, maxr, item_start, items, child_start, children, centers,\n\
        radii, query, k, /)\n\
--\n\
\n\
Finds the k items nearest each of a set of positions, in one call.\n\
\n\
The first nine arguments are a spatial tree flattened into arrays --- the box of\n\
each node (low, high), the largest radius in each node's subtree (maxr), the\n\
items of each node and the children of each node as compressed sparse rows, and\n\
the items' own centers and radii. The trees Python builds are described by\n\
euclib.utils._spatial, which answers the same query one position at a time, and\n\
is the definition this kernel is tested against.\n\
\n\
Items are ranked by the distance from the position to their *sphere*, which is a\n\
lower bound on their true distance; a caller that needs the exact nearest\n\
resolves the ranking among the candidates, as the index is built to allow.\n\
\n\
Parameters\n\
----------\n\
low, high : numpy.ndarray\n\
    The ``(N, D)`` boxes of the tree's nodes.\n\
maxr : numpy.ndarray\n\
    A length-``N`` vector of the largest item radius in each node's subtree.\n\
item_start, items : numpy.ndarray\n\
    A length-``N+1`` vector of offsets and the item indices they index.\n\
child_start, children : numpy.ndarray\n\
    The same layout for the child nodes of each node.\n\
centers : numpy.ndarray\n\
    The ``(D, M)`` centers of the items.\n\
radii : numpy.ndarray\n\
    A length-``M`` vector of the items' radii.\n\
query : numpy.ndarray\n\
    A ``(D, Q)`` matrix of positions.\n\
k : int\n\
    How many items to return per position.\n\
\n\
Returns\n\
-------\n\
index : numpy.ndarray\n\
    A ``(k, Q)`` matrix of item indices, nearest first.\n\
distance : numpy.ndarray\n\
    A ``(k, Q)`` matrix of the distances to those items' spheres.\n");

static PyObject *
core_nearest(PyObject *self, PyObject *args)
{
    PyArrayObject *keep[TREE_FIELDS + 1];
    PyArrayObject *query = NULL;
    PyArrayObject *index_out = NULL;
    PyArrayObject *distance_out = NULL;
    PyObject *result = NULL;
    tree tree_data;
    heap nodes = {NULL, 0, 0};
    heap best = {NULL, 0, 0};
    npy_intp k = 1;
    npy_intp q = 0;
    int dim = 0;
    int i;
    npy_intp position;

    for (i = 0; i <= TREE_FIELDS; ++i) {
        keep[i] = NULL;
    }
    if (read_tree(args, &dim, keep, &query, &k, NULL, &tree_data, 0) != 0) {
        goto done;
    }
    if (k < 1) {
        PyErr_SetString(PyExc_ValueError, "k must be at least 1");
        goto done;
    }
    q = PyArray_DIM(query, 1);
    {
        /* Python's tree stacks one array per position, so its answer has as
         * many rows as a position can have answers: k, or every item when the
         * tree holds fewer than k. */
        npy_intp dims[2];
        dims[0] = k < tree_data.item_count ? k : tree_data.item_count;
        dims[1] = q;
        index_out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_INTP);
        distance_out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (index_out == NULL || distance_out == NULL) {
            goto done;
        }
    }
    for (position = 0; position < q; ++position) {
        /* A (D, Q) query is contiguous along the positions, not along the
         * coordinates, so one position is gathered a coordinate at a time. */
        double point[MAX_DIM];
        npy_intp *found = NULL;
        double *away = NULL;
        for (i = 0; i < dim; ++i) {
            point[i] = ((const double *)PyArray_DATA(query))[i * q + position];
        }
        found = (npy_intp *)PyArray_DATA(index_out) + position;
        away = (double *)PyArray_DATA(distance_out) + position;
        npy_intp taken = 0;

        nodes.size = 0;
        best.size = 0;
        if (heap_push(&nodes, lower_bound(tree_data.low, tree_data.high,
                                          tree_data.maxr, dim, 0, point),
                      0) != 0) {
            goto done;
        }
        while (nodes.size > 0) {
            entry top = nodes.data[0];
            if (best.size >= k && top.bound > -best.data[0].bound) {
                break;
            }
            top = heap_pop(&nodes);
            if (tree_data.item_start[top.node + 1]
                    > tree_data.item_start[top.node]) {
                /* A leaf: the items are this node's own. */
                npy_intp at;
                for (at = tree_data.item_start[top.node];
                        at < tree_data.item_start[top.node + 1]; ++at) {
                    npy_intp item = tree_data.items[at];
                    double distance = sphere_distance(tree_data.centers,
                                                      tree_data.radii, dim,
                                                      tree_data.item_count,
                                                      item, point);
                    /* Held as a min-heap of (-distance, item), which is what
                     * Python's heapq holds: the top is the worst of the k, so
                     * replacing it is a pop and a push. */
                    if (best.size < k) {
                        if (heap_push(&best, -distance, item) != 0) {
                            goto done;
                        }
                    } else if (distance < -best.data[0].bound) {
                        (void)heap_pop(&best);
                        if (heap_push(&best, -distance, item) != 0) {
                            goto done;
                        }
                    }
                }
            } else {
                npy_intp at;
                for (at = tree_data.child_start[top.node];
                        at < tree_data.child_start[top.node + 1]; ++at) {
                    npy_intp child = tree_data.children[at];
                    if (heap_push(&nodes, lower_bound(tree_data.low,
                                                      tree_data.high,
                                                      tree_data.maxr, dim,
                                                      child, point),
                                  child) != 0) {
                        goto done;
                    }
                }
            }
        }
        /* The best were kept as a max-heap; the caller wants them nearest
         * first, and Python's ordering breaks ties by item index. */
        {
            entry sorted[1024];
            entry *spill = NULL;
            entry *held = sorted;
            npy_intp at;
            if (best.size > 1024) {
                spill = (entry *)malloc((size_t)best.size * sizeof(entry));
                if (spill == NULL) {
                    PyErr_NoMemory();
                    goto done;
                }
                held = spill;
            }
            for (at = 0; at < best.size; ++at) {
                held[at] = best.data[at];
            }
            /* An insertion sort: k is the number of items wanted, which is
             * small (one, for the radius a search starts from). */
            for (at = 1; at < best.size; ++at) {
                entry carry = held[at];
                npy_intp j = at;
                while (j > 0
                        && (-held[j - 1].bound > -carry.bound
                            || (-held[j - 1].bound == -carry.bound
                                && held[j - 1].node > carry.node))) {
                    held[j] = held[j - 1];
                    --j;
                }
                held[j] = carry;
            }
            for (at = 0; at < best.size; ++at) {
                found[at * q] = held[at].node;
                away[at * q] = -held[at].bound;
                ++taken;
            }
            if (spill != NULL) {
                free(spill);
            }
        }
        (void)taken;
    }
    result = Py_BuildValue("NN", (PyObject *)index_out,
                           (PyObject *)distance_out);
    index_out = NULL;
    distance_out = NULL;

done:
    for (i = 0; i <= TREE_FIELDS; ++i) {
        Py_XDECREF(keep[i]);
    }
    Py_XDECREF(index_out);
    Py_XDECREF(distance_out);
    free(nodes.data);
    free(best.data);
    return result;
}


/* candidates *****************************************************************/

/* Compares two item indices, for qsort. */
static int
by_item(const void *a, const void *b)
{
    npy_intp left = *(const npy_intp *)a;
    npy_intp right = *(const npy_intp *)b;
    if (left < right) {
        return -1;
    }
    return left > right ? 1 : 0;
}

PyDoc_STRVAR(candidates_doc,
"candidates(low, high, maxr, item_start, items, child_start, children,\n\
           centers, radii, query, radius, /)\n\
--\n\
\n\
Finds the items whose spheres come within a radius of each of a set of\n\
positions, in one call.\n\
\n\
The arguments are those of `nearest`, with a radius in place of ``k``; the tree\n\
is the same flattened spatial index, and euclib.utils._spatial answers the same\n\
query one position at a time and is the definition this kernel is tested\n\
against.\n\
\n\
The answer is conservative in the safe direction: an item is included whenever\n\
any point of its sphere comes within the radius, so no item that could be the\n\
nearest is missed, though an item that is merely near may be included as well.\n\
A caller that needs the exact nearest among them resolves that among the\n\
candidates.\n\
\n\
Parameters\n\
----------\n\
low, high, maxr, item_start, items, child_start, children, centers, radii : numpy.ndarray\n\
    The flattened tree, as `nearest` describes it.\n\
query : numpy.ndarray\n\
    A ``(D, Q)`` matrix of positions.\n\
radius : float\n\
    The radius to search within.\n\
\n\
Returns\n\
-------\n\
counts : numpy.ndarray\n\
    A length-``Q`` vector of how many items each position found.\n\
indices : numpy.ndarray\n\
    A length-``sum(counts)`` vector holding every position's items in turn,\n\
    each run sorted by item index.\n");

static PyObject *
core_candidates(PyObject *self, PyObject *args)
{
    PyArrayObject *keep[TREE_FIELDS + 1];
    PyArrayObject *query = NULL;
    PyArrayObject *counts_out = NULL;
    PyArrayObject *indices_out = NULL;
    PyObject *result = NULL;
    tree tree_data;
    nodelist nodes = {NULL, 0, 0};
    npy_intp *collected = NULL;
    npy_intp capacity = 0;
    npy_intp total = 0;
    double radius = 0.0;
    npy_intp k = 1;
    npy_intp q = 0;
    int dim = 0;
    int i;
    npy_intp position;

    for (i = 0; i <= TREE_FIELDS; ++i) {
        keep[i] = NULL;
    }
    if (read_tree(args, &dim, keep, &query, &k, &radius, &tree_data, 1) != 0) {
        goto done;
    }
    q = PyArray_DIM(query, 1);
    {
        npy_intp dims[1];
        dims[0] = q;
        counts_out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
        if (counts_out == NULL) {
            goto done;
        }
    }
    /* The items are collected per position and kept in one growing array; the
     * count of each position's run is written as it is finished. */
    for (position = 0; position < q; ++position) {
        double point[MAX_DIM];
        npy_intp start = total;
        npy_intp *count = (npy_intp *)PyArray_DATA(counts_out) + position;
        for (i = 0; i < dim; ++i) {
            point[i] = ((const double *)PyArray_DATA(query))[i * q + position];
        }

        nodes.size = 0;
        if (nodelist_push(&nodes, 0) != 0) {
            goto done;
        }
        /* The Python tree walks its nodes with a stack; the order items are
         * collected in does not matter, since the run is sorted below. */
        while (nodes.size > 0) {
            npy_intp node = nodelist_pop(&nodes);
            if (box_distance(tree_data.low + node * dim,
                             tree_data.high + node * dim, dim, point)
                    > radius + tree_data.maxr[node]) {
                continue;
            }
            if (tree_data.item_start[node + 1] > tree_data.item_start[node]) {
                npy_intp at;
                for (at = tree_data.item_start[node];
                        at < tree_data.item_start[node + 1]; ++at) {
                    npy_intp item = tree_data.items[at];
                    if (sphere_distance(tree_data.centers, tree_data.radii,
                                        dim, tree_data.item_count, item, point)
                            <= radius) {
                        if (total >= capacity) {
                            npy_intp grown = capacity ? capacity * 2 : 64;
                            npy_intp *bigger = (npy_intp *)realloc(
                                collected, (size_t)grown * sizeof(npy_intp));
                            if (bigger == NULL) {
                                PyErr_NoMemory();
                                goto done;
                            }
                            collected = bigger;
                            capacity = grown;
                        }
                        collected[total++] = item;
                    }
                }
            } else {
                npy_intp at;
                for (at = tree_data.child_start[node];
                        at < tree_data.child_start[node + 1]; ++at) {
                    if (nodelist_push(&nodes, tree_data.children[at]) != 0) {
                        goto done;
                    }
                }
            }
        }
        *count = total - start;
        qsort(collected + start, (size_t)(total - start), sizeof(npy_intp),
              by_item);
    }
    {
        npy_intp dims[1];
        dims[0] = total;
        indices_out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INTP);
        if (indices_out == NULL) {
            goto done;
        }
        if (total > 0) {
            memcpy(PyArray_DATA(indices_out), collected,
                   (size_t)total * sizeof(npy_intp));
        }
    }
    result = Py_BuildValue("NN", (PyObject *)counts_out,
                           (PyObject *)indices_out);
    counts_out = NULL;
    indices_out = NULL;

done:
    for (i = 0; i <= TREE_FIELDS; ++i) {
        Py_XDECREF(keep[i]);
    }
    Py_XDECREF(counts_out);
    Py_XDECREF(indices_out);
    free(collected);
    free(nodes.data);
    return result;
}


/* Module *********************************************************************/

static PyMethodDef spatial_methods[] = {
    {"nearest", core_nearest, METH_VARARGS, nearest_doc},
    {"candidates", core_candidates, METH_VARARGS, candidates_doc},
    {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(spatial_doc,
"The compiled queries of the euclib spatial index.\n\
\n\
Each answers a whole array of positions in one call, where the pure-Python tree\n\
in euclib.utils._spatial answers one position at a time. That tree is the\n\
definition of what these queries mean; the tests run each kernel against it\n\
and require them to agree.\n");

static struct PyModuleDef spatial_module = {
    PyModuleDef_HEAD_INIT,
    "euclib._c._spatial",
    spatial_doc,
    -1,
    spatial_methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__spatial(void)
{
    PyObject *module = NULL;
    import_array();
    module = PyModule_Create(&spatial_module);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
