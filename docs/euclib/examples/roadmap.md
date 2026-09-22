# Roadmap: Operations Not Yet Implemented

`euclib` is under active development, and several operations that other geometry
libraries demonstrate are not implemented yet. This page records them, with the
upstream test that would make a good example page once the operation lands, so
that the gap between the two libraries' feature sets is explicit rather than
implied.

None of the operations below are documented with example pages yet, because an
example page must show working code. When one is implemented, it gets a page in
the [gallery](index.md) following the standard template.

## Not yet implemented

| Operation | Upstream test that would demonstrate it |
|---|---|
| Boolean union / intersection / difference | pyvista `tests/core/test_polydata.py::test_boolean_union_intersection`, `::test_boolean_difference`; trimesh `tests/test_boolean.py::test_boolean`, `::test_multiple` |
| Geodesic distance over a surface | pyvista `tests/core/test_polydata.py::test_geodesic_distance` |
| Ray casting (many rays from arbitrary origins) | trimesh `tests/test_ray.py::test_rps` |
| Delaunay triangulation and tetrahedralization | pyvista `test_polydata.py::test_delaunay_2d`, `test_dataset_filters.py::test_delaunay_3d_grid` |
| Marching cubes / surface reconstruction | trimesh `tests/test_voxel.py::test_marching_points` |
| Clipping and sectioning a volume | pyvista `test_dataset_filters.py::test_clip_closed_surface`; trimesh `tests/test_section.py::test_section`, `::test_slice` |
| Rasterization of 2D paths | trimesh `tests/test_raster.py::test_rasterize` |
| Convex hull, oriented bounding boxes, principal axes | trimesh `tests/test_convex.py::test_convex`, `tests/test_bounds.py::test_obb_mesh`; pyvista `test_dataset_filters.py::test_oriented_bounding_box`, `test_utilities.py::test_principal_axes` |
| Plane and line fitting | pyvista `test_utilities.py::test_fit_plane_to_points`, `::test_fit_line_to_points` |
| Mesh remeshing, subdivision, and simplification | trimesh `tests/test_remesh.py::test_subdivide`, `tests/test_simplify.py::test_merge_colinear` |
| Mass properties (inertia tensors) | trimesh `tests/test_inertia.py::test_primitives` |
| Vector-field integration (streamlines) | pyvista `test_dataset_filters.py::test_streamlines_evenly_spaced_2d` |

## Recently implemented, example page pending

`euclib` has recently gained operations that do not have gallery pages yet.
They are listed here rather than in the table above because they *are*
implemented and a page is simply still to be written:

- **Crossings and intersections** — `euclib.ops.path_crossings` (where a path
  crosses a triangle mesh) and `euclib.ops.path_intersections` (where two paths
  cross). These would make a good page adapted from trimesh
  `tests/test_ray.py` and pyvista `tests/core/test_polydata.py::test_intersection`.
- **Containment** — `euclib.ops.contains` reports whether positions lie on or
  within a geometry, which corresponds to trimesh's
  `test_primitives.py::test_sample` containment check.

## Partly implemented

A few operations exist but are incomplete, so their example pages will be
written to state the limitation rather than hide it:

- **Transform decomposition** — `euclib.types.Transform` supports composition,
  inversion, and application; decomposing a matrix back into translation,
  rotation, scale, and shear is not implemented (pyvista
  `test_utilities.py::test_transform_decompose`).
- **Separation between geometries** — `euclib.ops.separation` is exact only when
  the closest approach is at a vertex of one geometry; the general
  simplex-to-simplex case is pending.
- **Interpolation order** — only nearest-neighbour (order 0) and linear
  (order 1) interpolation are supported. Quadratic and cubic interpolation are
  declared but raise `NotImplementedError`.

:::{seealso}
- [The example gallery](index.md) for the pages that *are* written.
:::
