# Example Gallery

Each page in this gallery documents a single example of `euclib` in use. The
examples are **adapted from the test suites of two established geometry
libraries**, [`pyvista`](https://github.com/pyvista/pyvista) and
[`trimesh`](https://github.com/trimesh/trimesh). No upstream test code is
copied: each page cites the exact upstream repository, file, function, and
revision it was adapted from, and explains in prose how the example was
rewritten for `euclib`.

:::{admonition} How to read the Provenance blocks
:class: note
Every example page ends with a **Provenance** admonition. It names the upstream
function and links to it at a *pinned commit*, so the citation does not drift as
the upstream project changes. The paragraph beneath the link states what was
changed and why — usually because `euclib` expresses the same idea with a
different construction.
:::

## Construction

| Page | Adapted from | Exercises |
|---|---|---|
| [One-line constructors](construction/constructors.md) | pyvista `test_geometric_objects.py::test_line`, `::test_cube`; trimesh `test_creation.py::test_box` | `points`, `segpath`, `trimesh`, `tetmesh`, `prismmesh`, `grid` |

## Primitives

| Page | Adapted from | Exercises |
|---|---|---|
| [A cube as a triangle mesh](primitives/cube.md) | pyvista `test_geometric_objects.py::test_cube` | `trimesh`, `withprop` |
| [Spheres and tessellation](primitives/spheres.md) | trimesh `test_creation.py::test_spheres` | parametric `TriMesh`, `measures`, invariants |
| [Circles and ellipses](primitives/circles-and-ellipses.md) | pyvista `test_geometric_objects.py::test_circle`, `::test_ellipse` | `segpath`, closed loops |
| [Circular arcs](primitives/circular-arcs.md) | pyvista `test_geometric_objects.py::test_circular_arc` | open `SegPath`, arc length vs chord |
| [Surfaces of revolution](primitives/revolution.md) | trimesh `test_creation.py::test_revolve`, `::test_torus`, `::test_annulus` | swept `TriMesh`, `measures` |
| [Prism sweeps](primitives/prism-sweeps.md) | pyvista `test_geometric_objects.py::test_tube`; trimesh `test_sweep.py::test_simple_extrude` | `PrismMesh`, `.elevation()`, `.to_tetmesh()` |
| [Tetrahedral meshes](primitives/tetrahedral-meshes.md) | pyvista `test_geometric_objects.py::test_solid_sphere` | `tetmesh`, volumes |

## Surfaces and paths

| Page | Adapted from | Exercises |
|---|---|---|
| [Geodesic distance along a surface](surfaces/geodesic.md) | pyvista `test_polydata.py::test_geodesic_distance` | `geodesic`, sources as a mask, distance fields |

## Intersections

| Page | Adapted from | Exercises |
|---|---|---|
| [Where two surfaces meet](intersections/mesh-mesh.md) | pyvista `test_polydata.py::test_intersection` | `mesh_intersections`, crossing curves |
| [Where a path crosses a surface](intersections/path-mesh.md) | trimesh `test_ray.py::test_rps` | `path_crossings`, crossing triangle indices |
| [Where two paths cross](intersections/path-path.md) | trimesh `test_intersect.py::test_line_line` | `path_intersections`, segment indices |
| [Resampling a volume onto a grid](intersections/mesh-grid.md) | pyvista `test_dataset_filters.py::test_delaunay_3d_grid`; trimesh `test_voxel.py::test_voxel` | `voxel_intersections`, cell indices |

## Transforms

| Page | Adapted from | Exercises |
|---|---|---|
| [Affine transforms](transforms/affine-transforms.md) | pyvista `test_utilities.py::test_transform_translate`, `::test_transform_scale`, `::test_transform_rotate` | `Affine`, `compose`, `@`, `inverse`, `.transformed` |
| [Rotation about an axis, and reflection](transforms/axis-angle-and-reflection.md) | pyvista `test_utilities.py::test_axis_angle_rotation`, `::test_reflection` | affine matrices, invariants |

## Properties and interpolation

| Page | Adapted from | Exercises |
|---|---|---|
| [Point clouds and their properties](properties/point-clouds.md) | pyvista `test_pointset.py::test_pointset_basic` | `VertexSet`, `withprop`, `dropprop`, indexing |
| [Interpolating a property within a geometry](properties/interpolation.md) | pyvista `test_dataset_filters.py::test_sample_over_line` | `prop(at=...)`, interp order, mask, extrapolation |
| [Transferring a property between geometries](properties/transfer.md) | pyvista `test_dataset_filters.py::test_implicit_distance` | `el.ops.transfer`, `el.ops.sample`, cross-representation |
| [Metadata, immutability, and transient edits](properties/immutability.md) | trimesh `test_meta.py::test_glb`; pyvista `test_utilities.py::test_transform_translate` | `withmeta`, `dropmeta`, `transient`, `persistent` |

## Grids and images

| Page | Adapted from | Exercises |
|---|---|---|
| [Image grids from an array and an affine](grids/grids-from-arrays.md) | pyvista `test_grid.py::test_init_from_numpy_arrays`, `::test_create_image_data_from_specs` | `grid`, image properties, `el.ops.positions_of` |

## Queries and measures

| Page | Adapted from | Exercises |
|---|---|---|
| [Measures: lengths, areas, and volumes](queries/measures.md) | pyvista `test_cell_lengths.py::test_cell_edge_lengths` | `measures`, `el.utils.simplex_measures` |
| [Nearest points on a triangle mesh](queries/nearest-points.md) | trimesh `test_proximity.py::test_nearest_naive` | `el.nearest`, `el.distance`, `to_local` |
| [Distance, separation, and containment](queries/distance-and-separation.md) | trimesh `test_proximity.py::test_coplanar_signed_distance` | `el.distance`, `el.separation`, `el.contains` |
| [Containment: on a surface, or within a volume](queries/containment.md) | trimesh `test_ray.py::test_contains` | `el.contains`, tolerance, surface vs volume |

## Planned pages

The gallery is still being built out. The rows below name the upstream test a
page will adapt and the `euclib` surface it will exercise. Rows marked
*pending* document an operation that `euclib` does not implement yet; see
[the roadmap](roadmap.md) for that list.

| Page | Adapted from | Exercises |
|---|---|---|
| Platonic solids | pyvista `test_geometric_objects.py::test_platonic_solids` | `TriMesh` symmetry |
| Capsules, cylinders, and cones | trimesh `test_creation.py::test_capsule`, `::test_cylinder`, `::test_cone` | quadric `TriMesh` |
| Grids to quads and hexes *(pending)* | pyvista `test_grid.py::test_to_quads`, `::test_to_hexahedra` | grid-to-mesh transfer |

:::{seealso}
- [The roadmap](roadmap.md) lists the operations `euclib` does not implement yet,
  together with the upstream tests that would demonstrate them.
:::
