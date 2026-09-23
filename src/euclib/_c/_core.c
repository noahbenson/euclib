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
 *   split_cells                the bisection a quadtree or an octree repeats
 *   tetrahedron_box_vertices   the region a tetrahedron and a box share
 *
 * The second is the one that pays: it solves a 3 by 3 system for each triple of
 * the ten planes that bound the two objects, of which there are 120, and a
 * voxelized mesh means one such call per overlapping pair.
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
    double found[NTRIPLES][3];
    double kept[NTRIPLES][3];
    int nfound = 0;
    int nkept = 0;
    double step = 0.0;
    double scale = 0.0;
    int f, i, j, k, p, t;

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

    {
        const double *c = (const double *)PyArray_DATA(tet);
        const double *b = (const double *)PyArray_DATA(bounds);

        /* The four faces of the tetrahedron, each oriented so that the corner
         * it omits is on the inside. */
        for (f = 0; f < 4; ++f) {
            int others[3];
            double u[3], v[3], w[3];
            int n = 0;
            for (j = 0; j < 4; ++j) {
                if (j != f) {
                    others[n++] = j;
                }
            }
            for (i = 0; i < 3; ++i) {
                u[i] = TET_AT(c, i, others[1]) - TET_AT(c, i, others[0]);
                v[i] = TET_AT(c, i, others[2]) - TET_AT(c, i, others[0]);
                w[i] = TET_AT(c, i, f) - TET_AT(c, i, others[0]);
            }
            normal[f][0] = u[1] * v[2] - u[2] * v[1];
            normal[f][1] = u[2] * v[0] - u[0] * v[2];
            normal[f][2] = u[0] * v[1] - u[1] * v[0];
            if (dot3(normal[f], w) > 0.0) {
                normal[f][0] = -normal[f][0];
                normal[f][1] = -normal[f][1];
                normal[f][2] = -normal[f][2];
            }
            offset[f] = normal[f][0] * TET_AT(c, 0, others[0])
                      + normal[f][1] * TET_AT(c, 1, others[0])
                      + normal[f][2] * TET_AT(c, 2, others[0]);
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


/* Module *********************************************************************/

static PyMethodDef core_methods[] = {
    {"split_cells", (PyCFunction)(void (*)(void))core_split_cells,
     METH_VARARGS | METH_KEYWORDS, split_cells_doc},
    {"tetrahedron_box_vertices",
     (PyCFunction)(void (*)(void))core_tetrahedron_box_vertices,
     METH_VARARGS | METH_KEYWORDS, tetrahedron_box_vertices_doc},
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
