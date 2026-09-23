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
| Primitive generators (boxes, spheres, and so on from a size) | pyvista `tests/core/test_geometric_objects.py::test_cube`, `::test_icosphere`; trimesh `tests/test_creation.py::test_box`, `::test_spheres` |

## Recently implemented and documented

Operations that have landed and now have a page. They are listed here so that a
reader who remembers them being absent knows where they went:

- **Geodesic distance** — `euclib.geodesic`, see
  [Geodesic distance along a surface](surfaces/geodesic.md).
- **Surface-surface intersection** — `euclib.mesh_intersections`, see
  [Where two surfaces meet](intersections/mesh-mesh.md).
- **Segment-mesh crossings** — `euclib.path_crossings`, see
  [Where a path crosses a surface](intersections/path-mesh.md).
- **Segment-segment crossings** — `euclib.path_intersections`, see
  [Where two paths cross](intersections/path-path.md).
- **Volume-grid overlap** — `euclib.voxel_intersections`, see
  [Resampling a volume onto a grid](intersections/mesh-grid.md).
- **Containment** — `euclib.contains`, see
  [Containment: on a surface, or within a volume](queries/containment.md).
- **One-line constructors** — `euclib.trimesh` and friends, see
  [One-line constructors](construction/constructors.md).
- **Metadata and transient editing** — see
  [Metadata, immutability, and transient edits](properties/immutability.md).

## Partly implemented

A few operations exist but are incomplete, so their example pages will be
written to state the limitation rather than hide it:

- **Transform decomposition** — `euclib.types.Transform` supports composition,
  inversion, and application; decomposing a matrix back into translation,
  rotation, scale, and shear is not implemented (pyvista
  `test_utilities.py::test_transform_decompose`).
- **Separation between geometries** — `euclib.separation` is exact only when
  the closest approach is at a vertex of one geometry; the general
  simplex-to-simplex case is pending.
- **Interpolation order** — only nearest-neighbour (order 0) and linear
  (order 1) interpolation are supported. Quadratic and cubic interpolation are
  declared but raise `NotImplementedError`.
- **Ray-style queries** — `euclib.path_crossings` answers "what does this
  segment hit?" for a whole path at once, but there is no ray-casting object
  that fires independent rays from arbitrary origins and directions, which is
  what the trimesh benchmark above exercises.

:::{seealso}
- [The example gallery](index.md) for the pages that *are* written.
:::
