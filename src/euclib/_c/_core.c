/* -*- coding: utf-8 -*-
 *******************************************************************************
 * euclib/_c/_core.c
 *
 * The optional C kernels of the euclib library.
 *
 * Every function here has a pure-Python counterpart in euclib.utils._pycore,
 * and that counterpart is the definition of correct behavior: the tests run
 * each kernel against the other and require them to agree. A kernel belongs
 * here when its work is a loop that Python cannot vectorize away.
 *
 *   split_cells                  the bisection a quadtree or an octree repeats
 *   tetrahedron_box_vertices     the corners of the region a tetrahedron and a
 *                                box share
 *   tetrahedron_box_region       that region, filled with tetrahedra
 *
 * The last two pay. The corner search solves a 3 by 3 system for each triple of
 * the ten planes that bound the two objects, of which there are 120; the region
 * is then filled by cutting the faces those planes carry, which is a great deal
 * of small arithmetic that Python spends most of a microsecond on each step of.
 * A voxelized mesh means one pair of these calls per overlapping tetrahedron and
 * voxel.
 *
 * The extension is built optionally, so a machine without a compiler gets the
 * pure-Python kernels rather than an installation that fails.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <math.h>

/* The planes that bound a tetrahedron and a box between them, and the number
 * of triples of those planes that there are. */
#define NPLANES 10
#define NTRIPLES 120

/* The determinant below which three planes count as having no common point.
 * This matches the tolerance euclib.utils._pycore uses. */
#define EPSILON 1e-12

/* The relative size below which a deduplication step is taken to be this
 * fraction of the region's extent. */
#define RELATIVE_STEP 1e-9

/* How far outside a plane, as a fraction of the sizes involved, a point may
 * fall and still count as lying on it. Solving for a corner and then asking
 * which side of each plane it is on does not answer the same way twice when
 * the corner lies on that plane, which it usually does; this is the width of
 * the band in which the answer is taken to be "on it". This matches
 * euclib.utils._pycore. */
#define FEASIBILITY_EPSILON 1e-12

/* The coordinate of one corner of a (3, 4) tetrahedron. The corner index is the
 * fast one, as in the C-contiguous array the pure-Python kernel receives. */
#define TET_AT(c, axis, corner) ((c)[(axis) * 4 + (corner)])


/* Helpers ********************************************************************/

static double
det3(const double *a, const double *b, const double *c)
{
    return a[0] * (b[1] * c[2] - b[2] * c[1])
         - a[1] * (b[0] * c[2] - b[2] * c[0])
         + a[2] * (b[0] * c[1] - b[1] * c[0]);
}

static double
dot3(const double *a, const double *b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/* Reads an argument as a C-contiguous float64 array of a given shape, or
 * reports why it cannot be read. Returns 0 on success and -1 on failure. */
static int
as_f64_matrix(PyObject *obj, PyArrayObject **out, npy_intp rows, npy_intp cols,
              const char *name)
{
    *out = (PyArrayObject *)PyArray_FROM_OTF(obj, NPY_DOUBLE,
                                             NPY_ARRAY_IN_ARRAY);
    if (*out == NULL) {
        return -1;
    }
    if (PyArray_NDIM(*out) != 2
            || PyArray_DIM(*out, 0) != rows || PyArray_DIM(*out, 1) != cols) {
        PyErr_Format(PyExc_ValueError,
                     "%s must be a (%ld, %ld) float64 array",
                     name, (long)rows, (long)cols);
        Py_DECREF(*out);
        *out = NULL;
        return -1;
    }
    return 0;
}


/* split_cells ****************************************************************/

PyDoc_STRVAR(split_cells_doc,
"split_cells(centers, bounds, /)\n\
--\n\
\n\
Places points in the sub-cells of a box's bisection.\n\
\n\
The box is halved along each axis, which makes ``2**D`` sub-cells --- four\n\
quadrants in two dimensions and eight octants in three --- and each point is\n\
placed in the one that contains it. This is the single step that a quadtree or\n\
an octree repeats as it descends.\n\
\n\
Parameters\n\
----------\n\
centers : numpy.ndarray\n\
    A ``(D, M)`` matrix of points.\n\
bounds : numpy.ndarray\n\
    The ``(D, 2)`` box being subdivided.\n\
\n\
Returns\n\
-------\n\
cells : numpy.ndarray\n\
    A length-``M`` vector of sub-cell indices, from 0 to ``2**D - 1``. The\n\
    cell whose index has bit *d* set is the upper half along axis *d*.\n");

static PyObject *
core_split_cells(PyObject *self, PyObject *args, PyObject *keywords)
{
    static char *names[] = {"centers", "bounds", NULL};
    PyObject *centers_obj = NULL;
    PyObject *bounds_obj = NULL;
    PyArrayObject *centers = NULL;
    PyArrayObject *bounds = NULL;
    PyArrayObject *out = NULL;
    PyObject *result = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, keywords, "OO", names,
                                     &centers_obj, &bounds_obj)) {
        return NULL;
    }
    centers = (PyArrayObject *)PyArray_FROM_OTF(centers_obj, NPY_DOUBLE,
                                                NPY_ARRAY_IN_ARRAY);
    if (centers == NULL) {
        return NULL;
    }
    bounds = (PyArrayObject *)PyArray_FROM_OTF(bounds_obj, NPY_DOUBLE,
                                               NPY_ARRAY_IN_ARRAY);
    if (bounds == NULL) {
        Py_DECREF(centers);
        return NULL;
    }
    if (PyArray_NDIM(centers) != 2 || PyArray_NDIM(bounds) != 2) {
        PyErr_SetString(PyExc_ValueError,
                        "centers and bounds must be 2-dimensional");
        goto done;
    }
    if (PyArray_DIM(centers, 0) != PyArray_DIM(bounds, 0)) {
        PyErr_Format(PyExc_ValueError,
                     "the points have dimension %ld, but the box has"
                     " dimension %ld",
                     (long)PyArray_DIM(centers, 0),
                     (long)PyArray_DIM(bounds, 0));
        goto done;
    }
    if (PyArray_DIM(bounds, 1) != 2) {
        PyErr_SetString(PyExc_ValueError, "bounds must be a (D, 2) array");
        goto done;
    }

    {
        npy_intp d = PyArray_DIM(centers, 0);
        npy_intp m = PyArray_DIM(centers, 1);
        const double *c = (const double *)PyArray_DATA(centers);
        const double *b = (const double *)PyArray_DATA(bounds);
        npy_intp dims[1];
        npy_int64 *w = NULL;
        npy_intp i, j;

        dims[0] = m;
        out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT64);
        if (out == NULL) {
            goto done;
        }
        w = (npy_int64 *)PyArray_DATA(out);
        for (j = 0; j < m; ++j) {
            npy_int64 cell = 0;
            for (i = 0; i < d; ++i) {
                double middle = (b[i * 2] + b[i * 2 + 1]) / 2.0;
                if (c[i * m + j] > middle) {
                    cell |= ((npy_int64)1 << i);
                }
            }
            w[j] = cell;
        }
    }
    result = (PyObject *)out;
    out = NULL;

done:
    Py_XDECREF(out);
    Py_DECREF(centers);
    Py_DECREF(bounds);
    return result;
}


/* tetrahedron_box_vertices ***************************************************/

PyDoc_STRVAR(tetrahedron_box_vertices_doc,
"tetrahedron_box_vertices(tet, bounds, tolerance=0.0, /)\n\
--\n\
\n\
Finds the corners of the region a tetrahedron and a box share.\n\
\n\
The region is the part of space that lies inside both, so its corners are the\n\
points where three of the ten bounding planes meet and no plane excludes them.\n\
That makes the search a matter of trying every triple, which is bounded and\n\
small: at most 120 points to test, of which the real corners are the ones that\n\
survive every constraint.\n\
\n\
Parameters\n\
----------\n\
tet : numpy.ndarray\n\
    A ``(3, 4)`` matrix of the tetrahedron's corners.\n\
bounds : numpy.ndarray\n\
    A ``(3, 2)`` box, such as one voxel of a grid.\n\
tolerance : float, optional\n\
    How far outside a plane a corner may lie and still be counted. The default\n\
    is ``0``.\n\
\n\
Returns\n\
-------\n\
vertices : numpy.ndarray\n\
    A ``(3, V)`` matrix of the region's corners, in no particular order. It is\n\
    empty when the two do not meet.\n");

/* The planes that bound a tetrahedron and a box between them, and the corners
 * where three of them meet. The planes are returned as well as the corners,
 * because the region the two share is filled from its faces, and its faces lie
 * in these planes. */
static void
region_corners(const double *c, const double *b, double tolerance,
               double kept[NTRIPLES][3], int *nkept_out,
               double normal[NPLANES][3], double offset[NPLANES])
{
    double found[NTRIPLES][3];
    int nfound = 0;
    int nkept = 0;
    double step = 0.0;
    double scale = 0.0;
    int i, j, k, p;

    /* The four faces of the tetrahedron, each oriented so that the corner
     * it omits is on the inside. */
    for (i = 0; i < 4; ++i) {
        int others[3];
        double u[3], v[3], w[3];
        int n = 0;
        for (j = 0; j < 4; ++j) {
            if (j != i) {
                others[n++] = j;
            }
        }
        for (p = 0; p < 3; ++p) {
            u[p] = TET_AT(c, p, others[1]) - TET_AT(c, p, others[0]);
            v[p] = TET_AT(c, p, others[2]) - TET_AT(c, p, others[0]);
            w[p] = TET_AT(c, p, i) - TET_AT(c, p, others[0]);
        }
        normal[i][0] = u[1] * v[2] - u[2] * v[1];
        normal[i][1] = u[2] * v[0] - u[0] * v[2];
        normal[i][2] = u[0] * v[1] - u[1] * v[0];
        if (dot3(normal[i], w) > 0.0) {
            normal[i][0] = -normal[i][0];
            normal[i][1] = -normal[i][1];
            normal[i][2] = -normal[i][2];
        }
        offset[i] = normal[i][0] * TET_AT(c, 0, others[0])
                  + normal[i][1] * TET_AT(c, 1, others[0])
                  + normal[i][2] * TET_AT(c, 2, others[0]);
    }
    /* The six sides of the box. */
    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            normal[4 + 2 * i][j] = (j == i) ? 1.0 : 0.0;
            normal[5 + 2 * i][j] = (j == i) ? -1.0 : 0.0;
        }
        offset[4 + 2 * i] = b[2 * i + 1];
        offset[5 + 2 * i] = -b[2 * i];
    }

    /* Every triple of planes that meets at a point no plane excludes. */
    for (i = 0; i < NPLANES; ++i) {
        for (j = i + 1; j < NPLANES; ++j) {
            for (k = j + 1; k < NPLANES; ++k) {
                double det = det3(normal[i], normal[j], normal[k]);
                double point[3];
                int ok = 1;
                if (fabs(det) <= EPSILON) {
                    continue;
                }
                /* Cramer's rule. The three planes are the rows of the
                 * system, so it is their columns that have to be passed
                 * here: a determinant is unchanged by the transpose, which
                 * makes the arguments to det3 equally readable as columns.
                 * The three offsets are the right-hand side, and they take
                 * the place of one column in turn. */
                {
                    double rows[3][3];
                    double ci[3], cj[3], ck[3], rhs[3];
                    for (p = 0; p < 3; ++p) {
                        rows[0][p] = normal[i][p];
                        rows[1][p] = normal[j][p];
                        rows[2][p] = normal[k][p];
                    }
                    for (p = 0; p < 3; ++p) {
                        ci[p] = rows[p][0];
                        cj[p] = rows[p][1];
                        ck[p] = rows[p][2];
                    }
                    rhs[0] = offset[i];
                    rhs[1] = offset[j];
                    rhs[2] = offset[k];
                    point[0] = det3(rhs, cj, ck) / det;
                    point[1] = det3(ci, rhs, ck) / det;
                    point[2] = det3(ci, cj, rhs) / det;
                }
                {
                    /* A corner usually lies exactly on some of the planes
                     * it is not built from, and whether a solved copy of it
                     * falls just inside or just outside one of them is
                     * decided by rounding. A slack proportional to the
                     * sizes involved keeps those corners, which is the
                     * difference between a mesh and a grid that share a
                     * face meeting there or missing each other. */
                    double extent = 1.0;
                    double slack;
                    for (p = 0; p < 3; ++p) {
                        if (fabs(point[p]) > extent) {
                            extent = fabs(point[p]);
                        }
                    }
                    for (p = 0; p < NPLANES; ++p) {
                        if (fabs(offset[p]) > extent) {
                            extent = fabs(offset[p]);
                        }
                    }
                    slack = FEASIBILITY_EPSILON * extent;
                    for (p = 0; p < NPLANES; ++p) {
                        if (dot3(normal[p], point)
                                > offset[p] + tolerance + slack) {
                            ok = 0;
                            break;
                        }
                    }
                }
                if (ok) {
                    for (p = 0; p < 3; ++p) {
                        found[nfound][p] = point[p];
                    }
                    ++nfound;
                }
            }
        }
    }

    /* Every feasible triple yields its own copy of a corner, and the copies
     * differ in the last bits of their coordinates; exact equality would keep
     * them all, leaving a set so nearly degenerate that no triangulation of it
     * can be taken. Two copies within `step` of one another are taken to be the
     * same corner, and the first of them is the one kept. Comparing them to one
     * another rather than to a rounding grid matters: a grid puts two copies of
     * one corner in different cells whenever they fall on either side of a
     * cell's edge, which is a decision rounding should not make. */
    for (i = 0; i < nfound; ++i) {
        for (p = 0; p < 3; ++p) {
            double v = fabs(found[i][p]);
            if (v > scale) {
                scale = v;
            }
        }
    }
    if (scale < 1.0) {
        scale = 1.0;
    }
    step = tolerance > RELATIVE_STEP * scale ? tolerance
                                             : RELATIVE_STEP * scale;
    for (i = 0; i < nfound; ++i) {
        int seen = 0;
        for (j = 0; j < nkept; ++j) {
            double distance = 0.0;
            for (p = 0; p < 3; ++p) {
                double difference = found[i][p] - kept[j][p];
                distance += difference * difference;
            }
            if (distance <= step * step) {
                seen = 1;
                break;
            }
        }
        if (!seen) {
            for (p = 0; p < 3; ++p) {
                kept[nkept][p] = found[i][p];
            }
            ++nkept;
        }
    }
    *nkept_out = nkept;
}

static PyObject *
core_tetrahedron_box_vertices(PyObject *self, PyObject *args,
                              PyObject *keywords)
{
    static char *names[] = {"tet", "bounds", "tolerance", NULL};
    PyObject *tet_obj = NULL;
    PyObject *bounds_obj = NULL;
    PyArrayObject *tet = NULL;
    PyArrayObject *bounds = NULL;
    PyArrayObject *out = NULL;
    PyObject *result = NULL;
    double tolerance = 0.0;
    double normal[NPLANES][3];
    double offset[NPLANES];
    double kept[NTRIPLES][3];
    int nkept = 0;
    int t, p;

    if (!PyArg_ParseTupleAndKeywords(args, keywords, "OO|d", names, &tet_obj,
                                     &bounds_obj, &tolerance)) {
        return NULL;
    }
    if (as_f64_matrix(tet_obj, &tet, 3, 4, "tet") != 0) {
        return NULL;
    }
    if (as_f64_matrix(bounds_obj, &bounds, 3, 2, "bounds") != 0) {
        Py_DECREF(tet);
        return NULL;
    }
    region_corners((const double *)PyArray_DATA(tet),
                   (const double *)PyArray_DATA(bounds), tolerance,
                   kept, &nkept, normal, offset);
    {
        npy_intp dims[2];
        double *w = NULL;
        dims[0] = 3;
        dims[1] = nkept;
        out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (out == NULL) {
            goto done;
        }
        w = (double *)PyArray_DATA(out);
        for (t = 0; t < nkept; ++t) {
            for (p = 0; p < 3; ++p) {
                w[p * nkept + t] = kept[t][p];
            }
        }
    }
    result = (PyObject *)out;
    out = NULL;

done:
    Py_XDECREF(out);
    Py_DECREF(tet);
    Py_DECREF(bounds);
    return result;
}


/* The region a tetrahedron and a box share ***********************************/

/* The most corners one face of the region can have. A face is given every
 * corner that lies on its plane, and the corners are bounded by the triples of
 * planes, so that is the bound --- a wide `tolerance` can put a great many of
 * them on one plane, and dropping such a face would leave the region unfilled. */
#define MAXFACE NTRIPLES

/* Whether one corner comes before another, which orders them the way the
 * Python does: by x, then by y, then by z. */
static int
corner_before(const double *a, const double *b)
{
    int p;
    for (p = 0; p < 3; ++p) {
        if (a[p] < b[p]) {
            return 1;
        }
        if (a[p] > b[p]) {
            return 0;
        }
    }
    return 0;
}

/* Two directions across a plane, at right angles to one another and to it. */
static void
plane_frame(const double *normal, double *across, double *along)
{
    double seed[3] = {0.0, 0.0, 0.0};
    double size;
    int best = 0;
    int i;

    for (i = 1; i < 3; ++i) {
        if (fabs(normal[i]) < fabs(normal[best])) {
            best = i;
        }
    }
    seed[best] = 1.0;
    across[0] = normal[1] * seed[2] - normal[2] * seed[1];
    across[1] = normal[2] * seed[0] - normal[0] * seed[2];
    across[2] = normal[0] * seed[1] - normal[1] * seed[0];
    size = sqrt(across[0] * across[0] + across[1] * across[1]
                + across[2] * across[2]);
    across[0] /= size;
    across[1] /= size;
    across[2] /= size;
    along[0] = normal[1] * across[2] - normal[2] * across[1];
    along[1] = normal[2] * across[0] - normal[0] * across[2];
    along[2] = normal[0] * across[1] - normal[1] * across[0];
}

/* Whether a run of corners spans an area rather than a line or a point. */
static int
face_is_flat(const double kept[][3], const int *on, int count,
             const double *normal)
{
    double across[3], along[3], origin[3], line[3], span = 0.0;
    int has_line = 0;
    int i, p;

    if (count < 3) {
        return 0;
    }
    plane_frame(normal, across, along);
    for (p = 0; p < 3; ++p) {
        origin[p] = kept[on[0]][p];
    }
    for (i = 1; i < count; ++i) {
        double step[3];
        double across_size, along_size;
        for (p = 0; p < 3; ++p) {
            step[p] = kept[on[i]][p] - origin[p];
        }
        across_size = fabs(dot3(step, across));
        along_size = fabs(dot3(step, along));
        if (across_size > span) {
            span = across_size;
        }
        if (along_size > span) {
            span = along_size;
        }
    }
    if (span == 0.0) {
        return 0;
    }
    /* A corner that is off the line the others lie on gives the face its
     * area. */
    for (i = 1; i < count; ++i) {
        double step[3];
        for (p = 0; p < 3; ++p) {
            step[p] = kept[on[i]][p] - origin[p];
        }
        if (fabs(dot3(step, across)) + fabs(dot3(step, along))
                > 1e-9 * span) {
            for (p = 0; p < 3; ++p) {
                line[p] = step[p];
            }
            has_line = 1;
            break;
        }
    }
    if (!has_line) {
        return 0;
    }
    for (i = 1; i < count; ++i) {
        double step[3];
        double sa, sl;
        for (p = 0; p < 3; ++p) {
            step[p] = kept[on[i]][p] - origin[p];
        }
        sa = dot3(step, across);
        sl = dot3(step, along);
        if (fabs(sa * line[2] - sl * line[1])
                + fabs(sa * line[0] - sl * line[2])
                + fabs(sl * line[0] - sa * line[1])
                > 1e-9 * span * span) {
            return 1;
        }
    }
    return 0;
}

/* The corners of one face, in the order they go round it.
 *
 * The order is taken from where the corners are, so that two pieces meeting
 * along a face --- which hold the same corners on it --- cut it into the same
 * triangles. Which corner the list starts at, and which way round it goes, are
 * settled from the corners too, because the fan of a face is a fan from that
 * corner. `order` comes back holding positions within `on`. */
static void
face_order(const double kept[][3], const int *on, int count,
           const double *normal, int *order)
{
    double across[3], along[3], middle[3], angle[MAXFACE];
    int i, j, first;

    plane_frame(normal, across, along);
    for (i = 0; i < 3; ++i) {
        double total = 0.0;
        for (j = 0; j < count; ++j) {
            total += kept[on[j]][i];
        }
        middle[i] = total / count;
    }
    for (i = 0; i < count; ++i) {
        double step[3];
        for (j = 0; j < 3; ++j) {
            step[j] = kept[on[i]][j] - middle[j];
        }
        angle[i] = atan2(dot3(step, along), dot3(step, across));
        order[i] = i;
    }
    /* By angle, and within one angle by the position in `on`, which is what
     * sorting pairs of them does. */
    for (i = 1; i < count; ++i) {
        int here = order[i];
        double key = angle[here];
        j = i - 1;
        while (j >= 0 && (angle[order[j]] > key
                          || (angle[order[j]] == key && order[j] > here))) {
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = here;
    }
    /* Start at the smallest corner, which is the one a fan is taken from. */
    first = 0;
    for (i = 1; i < count; ++i) {
        if (corner_before(kept[on[order[i]]], kept[on[order[first]]])) {
            first = i;
        }
    }
    if (first > 0) {
        int rotated[MAXFACE];
        for (i = 0; i < count; ++i) {
            rotated[i] = order[(first + i) % count];
        }
        for (i = 0; i < count; ++i) {
            order[i] = rotated[i];
        }
    }
    /* The two ways round a face are mirror images, so the one that goes toward
     * the smaller of the two neighbours of the first corner is the one taken. */
    if (count > 2
            && corner_before(kept[on[order[count - 1]]],
                             kept[on[order[1]]])) {
        int mirrored[MAXFACE];
        mirrored[0] = order[0];
        for (i = 1; i < count; ++i) {
            mirrored[i] = order[count - i];
        }
        for (i = 0; i < count; ++i) {
            order[i] = mirrored[i];
        }
    }
}

/* The region's fills, as groups of four corners of it. Returns the number of
 * them, or -1 if an array could not be made. */
static npy_intp
region_fill(const double kept[][3], int count,
            const double normal[NPLANES][3], const double offset[NPLANES],
            double tolerance, npy_intp *filled, npy_intp limit)
{
    int seen[NPLANES][MAXFACE];
    int seen_count[NPLANES];
    int nseen = 0;
    int smallest = 0;
    double extent = 1.0;
    double slack;
    npy_intp total = 0;
    int i, j, k, p;

    for (i = 0; i < count; ++i) {
        for (p = 0; p < 3; ++p) {
            double v = fabs(kept[i][p]);
            if (v > extent) {
                extent = v;
            }
        }
        if (corner_before(kept[i], kept[smallest])) {
            smallest = i;
        }
    }
    /* A corner the corner search put on a plane is on it to within rounding,
     * not exactly, so the test of which corners a plane carries allows for
     * that --- far below the smallest distance between two corners of one
     * region, so it can only join up corners that are the same point. */
    slack = tolerance + EPSILON * extent;
    for (p = 0; p < NPLANES; ++p) {
        int on[MAXFACE];
        int non = 0;
        int order[MAXFACE];
        int duplicate = 0;

        for (i = 0; i < count; ++i) {
            if (fabs(dot3(normal[p], kept[i]) - offset[p]) <= slack) {
                on[non++] = i;
            }
        }
        if (non < 3 || non > MAXFACE) {
            continue;
        }
        if (!face_is_flat(kept, on, non, normal[p])) {
            continue;
        }
        /* Two planes can carry the same face --- a tetrahedron's face against
         * a box's, say --- and it is cut once when they do. */
        for (i = 0; i < nseen && !duplicate; ++i) {
            if (seen_count[i] != non) {
                continue;
            }
            duplicate = 1;
            for (j = 0; j < non; ++j) {
                int here = 0;
                for (k = 0; k < non; ++k) {
                    if (seen[i][k] == on[j]) {
                        here = 1;
                        break;
                    }
                }
                if (!here) {
                    duplicate = 0;
                    break;
                }
            }
        }
        if (duplicate) {
            continue;
        }
        for (j = 0; j < non; ++j) {
            seen[nseen][j] = on[j];
        }
        seen_count[nseen] = non;
        ++nseen;
        face_order(kept, on, non, normal[p], order);
        /* A face that carries the smallest corner is not cut here: it is
         * covered by the sides of the fan instead, and comes out cut as a fan
         * from the same corner, which is the face's own smallest. */
        if (on[order[0]] == smallest) {
            continue;
        }
        for (k = 1; k < non - 1; ++k) {
            if (total >= limit) {
                return -1;
            }
            filled[4 * total] = smallest;
            filled[4 * total + 1] = on[order[0]];
            filled[4 * total + 2] = on[order[k]];
            filled[4 * total + 3] = on[order[k + 1]];
            ++total;
        }
    }
    return total;
}

PyDoc_STRVAR(tetrahedron_box_region_doc,
"tetrahedron_box_region(tet, bounds, tolerance=0.0, /)\n\
--\n\
\n\
Decomposes the region a tetrahedron and a box share into tetrahedra.\n\
\n\
The region is a convex solid whose faces lie in the box's six planes and the\n\
tetrahedron's four, so it is filled by taking each of those planes in turn,\n\
cutting the face it carries into triangles, and joining each triangle to the\n\
region's own smallest corner. No hull is found, no connectivity is guessed at,\n\
and no corner is added.\n\
\n\
Parameters\n\
----------\n\
tet : numpy.ndarray\n\
    A ``(3, 4)`` matrix of the tetrahedron's corners.\n\
bounds : numpy.ndarray\n\
    A ``(3, 2)`` box, such as one voxel of a grid.\n\
tolerance : float, optional\n\
    How far outside a plane a corner may lie and still be counted.\n\
\n\
Returns\n\
-------\n\
vertices : numpy.ndarray\n\
    A ``(3, V)`` matrix of the region's corners.\n\
tetrahedra : numpy.ndarray\n\
    A ``(4, T)`` integer matrix of the tetrahedra that fill the region.\n");

static PyObject *
core_tetrahedron_box_region(PyObject *self, PyObject *args,
                            PyObject *keywords)
{
    static char *names[] = {"tet", "bounds", "tolerance", NULL};
    PyObject *tet_obj = NULL;
    PyObject *bounds_obj = NULL;
    PyArrayObject *tet = NULL;
    PyArrayObject *bounds = NULL;
    PyArrayObject *verts_out = NULL;
    PyArrayObject *tets_out = NULL;
    PyObject *result = NULL;
    double tolerance = 0.0;
    double normal[NPLANES][3];
    double offset[NPLANES];
    double kept[NTRIPLES][3];
    npy_intp *filled = NULL;
    npy_intp limit = 0;
    npy_intp total = 0;
    int nkept = 0;
    int t, p;

    if (!PyArg_ParseTupleAndKeywords(args, keywords, "OO|d", names, &tet_obj,
                                     &bounds_obj, &tolerance)) {
        return NULL;
    }
    if (as_f64_matrix(tet_obj, &tet, 3, 4, "tet") != 0) {
        return NULL;
    }
    if (as_f64_matrix(bounds_obj, &bounds, 3, 2, "bounds") != 0) {
        Py_DECREF(tet);
        return NULL;
    }
    region_corners((const double *)PyArray_DATA(tet),
                   (const double *)PyArray_DATA(bounds), tolerance,
                   kept, &nkept, normal, offset);
    /* Each face gives at most one tetrahedron per corner it has beyond the
     * second, and there are ten faces, so this bounds the answer. */
    limit = (npy_intp)NPLANES * (MAXFACE - 2);
    filled = (npy_intp *)PyMem_Malloc((size_t)limit * 4 * sizeof(npy_intp));
    if (filled == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    if (nkept == 4) {
        /* A region of four corners is a tetrahedron already, and is returned as
         * one rather than as a fan of itself. */
        double edges[3][3];
        double volume;
        int q;
        for (p = 0; p < 3; ++p) {
            for (q = 0; q < 3; ++q) {
                edges[p][q] = kept[p + 1][q] - kept[0][q];
            }
        }
        volume = det3(edges[0], edges[1], edges[2]) / 6.0;
        if (fabs(volume) > EPSILON) {
            filled[0] = 0;
            filled[1] = 1;
            filled[2] = 2;
            filled[3] = 3;
            total = 1;
        } else {
            total = 0;
        }
    } else {
        total = region_fill(kept, nkept, normal, offset, tolerance, filled,
                            limit);
    }
    if (total < 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "the region needed more tetrahedra than it can have");
        goto done;
    }
    {
        npy_intp dims[2];
        dims[0] = 3;
        dims[1] = nkept;
        verts_out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        if (verts_out == NULL) {
            goto done;
        }
        {
            double *w = (double *)PyArray_DATA(verts_out);
            for (t = 0; t < nkept; ++t) {
                for (p = 0; p < 3; ++p) {
                    w[p * nkept + t] = kept[t][p];
                }
            }
        }
        dims[0] = 4;
        dims[1] = (npy_intp)total;
        tets_out = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_INTP);
        if (tets_out == NULL) {
            goto done;
        }
        {
            npy_intp *w = (npy_intp *)PyArray_DATA(tets_out);
            for (t = 0; t < total; ++t) {
                for (p = 0; p < 4; ++p) {
                    w[p * total + t] = filled[4 * t + p];
                }
            }
        }
    }
    result = Py_BuildValue("NN", (PyObject *)verts_out, (PyObject *)tets_out);
    verts_out = NULL;
    tets_out = NULL;

done:
    PyMem_Free(filled);
    Py_XDECREF(verts_out);
    Py_XDECREF(tets_out);
    Py_DECREF(tet);
    Py_DECREF(bounds);
    return result;
}


/* Module *********************************************************************/

static PyMethodDef core_methods[] = {
    {"split_cells", (PyCFunction)(void (*)(void))core_split_cells,
     METH_VARARGS | METH_KEYWORDS, split_cells_doc},
    {"tetrahedron_box_vertices",
     (PyCFunction)(void (*)(void))core_tetrahedron_box_vertices,
     METH_VARARGS | METH_KEYWORDS, tetrahedron_box_vertices_doc},
    {"tetrahedron_box_region",
     (PyCFunction)(void (*)(void))core_tetrahedron_box_region,
     METH_VARARGS | METH_KEYWORDS, tetrahedron_box_region_doc},
    {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(core_doc,
"The compiled kernels of the euclib library.\n\
\n\
Each of these has a pure-Python counterpart in euclib.utils._pycore, which is\n\
the definition of correct behavior: the tests run each kernel against the other\n\
and require them to agree. See euclib.utils._dispatch for how one is chosen.\n");

static struct PyModuleDef core_module = {
    PyModuleDef_HEAD_INIT,
    "euclib._c._core",
    core_doc,
    -1,
    core_methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__core(void)
{
    PyObject *module = NULL;
    import_array();
    module = PyModule_Create(&core_module);
    if (module == NULL) {
        return NULL;
    }
    return module;
}
