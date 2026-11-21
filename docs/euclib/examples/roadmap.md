# Roadmap: Operations Not Yet Implemented

:::{admonition} Warning: This page was written with AI assistance.
:class: warning

This page was written using AI assistance and has not yet been carefully
reviewed by a human. Its code runs and its claims are checked against the test
suite, but a person has not read it end to end. Treat it as an unreviewed draft:
verify the behaviour you depend on against the API reference and the tests, and
report anything that looks wrong.
:::

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
| Delaunay triangulation and tetrahedralization | pyvista `test_polydata.py::test_delaunay_2d`, `test_dataset_filters.py::test_delaunay_3d_grid` |
| Marching cubes / surface reconstruction | trimesh `tests/test_voxel.py::test_marching_points` |
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
- **Line and plane fitting** — `euclib.fit_line` and `euclib.fit_plane`, see
  [Fitting a line or a plane](queries/fitting.md).
- **Principal axes and oriented bounding boxes** — `euclib.principal_axes` and
  `euclib.oriented_bounding_box`, see
  [Principal axes and oriented bounding boxes](queries/principal-axes.md). The
  box is the one aligned with the axes; for the smallest box of any
  orientation, see the minimum-volume boxes below.
- **Convex hulls** — `euclib.convex_hull`, see
  [Convex hulls](queries/convex-hull.md).
- **Boolean union, intersection, and difference** — `euclib.boolean_union`,
  `euclib.boolean_intersection`, and `euclib.boolean_difference`, see
  [Union, intersection, and difference of solids](queries/boolean-operations.md).
- **Clipping and sectioning a volume** — `euclib.clip` and `euclib.section`,
  see
  [Clipping and sectioning a volume](queries/clipping-and-sectioning.md).
- **Minimum-volume oriented bounding boxes** —
  `euclib.minimum_volume_bounding_box`, see
  [Minimum-volume oriented bounding boxes](queries/minimum-volume-boxes.md).
- **Vector-field integration** — `euclib.streamlines`, see
  [Integrating a vector field into streamlines](grids/streamlines.md).
- **Mesh subdivision and path simplification** — `euclib.subdivide` and
  `euclib.merge_colinear`, see
  [Subdividing a mesh and simplifying a path](surfaces/subdivision.md).
- **Transform decomposition** — `euclib.types.Affine.decompose`, see
  [Affine transforms](transforms/affine-transforms.md).
- **Grid cells as simplex meshes** — `euclib.Grid.to_trimesh` and
  `euclib.Grid.to_tetmesh`, see
  [Grid cells as triangles and tetrahedra](grids/grid-cells-as-meshes.md).
- **Rasterization of two-dimensional paths** — `euclib.rasterize`, see
  [Rasterizing a two-dimensional path](grids/rasterizing-paths.md).
- **One-line constructors** — `euclib.trimesh` and friends, see
  [One-line constructors](construction/constructors.md).
- **Metadata and transient editing** — see
  [Metadata, immutability, and transient edits](properties/immutability.md).
- **Interpolation, at every order and on every element.** A segment takes
  nearest, linear, quadratic and cubic, and the cardinal cubic of Catmull and
  Rom; a triangle takes those and adds Clough–Tocher's C¹ piecewise cubic and
  Powell–Sabin's C¹ piecewise quadratic; a tetrahedron takes the Bézier orders;
  and a grid takes the seven separable kernels. See
  [Bézier interpolation on a triangle](properties/bezier-triangle.md) and
  [on a tetrahedron](properties/bezier-tetrahedron.md),
  [Clough–Tocher](properties/clough-tocher-triangle.md),
  [Powell–Sabin](properties/powell-sabin-triangle.md),
  [Catmull–Rom along a path](properties/catmull-rom-path.md), and the grid pages
  [nearest and linear](properties/grid-linear.md),
  [the boundary](properties/grid-boundary.md),
  [cubic convolution](properties/grid-cubic.md),
  [splines](properties/grid-spline.md),
  [Lanczos](properties/grid-lanczos.md) and
  [the polynomial fit](properties/grid-polynomial.md).

## Known defects

Operations that work in the cases they are used for and not in others. These
are defects rather than gaps: a caller can get a wrong answer rather than a
refusal, so they are listed separately.

None at present.

:::{admonition} What was here, and why it went
:class: note
A defect was recorded against mesh smoothing for two reasons, and on
investigation neither was one.

The first was that the tetrahedral smoothing used the *cotangent* formula,
which is a triangle's. That was real, and it is fixed: the stiffness is the
linear finite-element one, ``V (grad phi_i . grad phi_j)``, which *is* the
cotangent formula on a triangle --- so the triangle path was right all along and
only the tetrahedral one was wrong.

The second was that a plane is not preserved when the boundary is free. That is
not a defect. The weak Laplacian's natural boundary condition is ``df/dn = 0``,
which a plane does not satisfy; a plane is preserved under a *Dirichlet*
boundary, which is what ``fixboundary=True`` supplies. Both behaviours are now
tested, so the next person meets them as specification rather than as surprise.

The test that would have caught the first half was vacuous:
``test_a_linear_field_is_preserved`` used the default ``fixboundary=True``, so
the boundary was held still and nothing moved whatever the operator was. It has
been replaced by one that asserts what each boundary treatment should do.
:::

## Partly implemented

None at present.

:::{admonition} What was here, and why it went
:class: note
Asking a geometry whose coordinates are a tensor for the derivative of an
interpolation whose *gradient* the library must estimate --- order 2 or higher
with no gradient supplied or carried --- used to come back `nan` on a mesh
whose neighbourhoods were all exactly symmetric. Three things had to be true at
once, and any one of them missing avoided it: tensor coordinates, an estimated
gradient, and a mesh with no asymmetry anywhere.

The reason was PyTorch's, not the fit's. Estimating a gradient wrote the fit in
a basis taken from a singular value decomposition, and `torch.linalg.svd` warns
that the gradients it computes "will only be finite when `A` does not have
repeated singular values", because they are computed through
$1 / \min_{i \neq j} (\sigma_i^2 - \sigma_j^2)$. A vertex whose neighbours are
arranged symmetrically has exactly equal singular values: one vertex of a flat
triangular lattice and its six neighbours give the displacements a
$7 \times 3$ matrix whose singular values are $\sqrt{3}$, $\sqrt{3}$, and zero.
The numerator of that derivative is zero too, but the arithmetic produces
`nan` rather than a limit.

The estimate now builds its stencil frame by Gram--Schmidt when the coordinates
require a gradient. Gram--Schmidt has no repeated-singular-value tie, it depends
smoothly on the displacements as long as the chosen ones are independent, and
the values it produces are the same to within rounding. The tests check the
derivative against a numerical one on a symmetric sheet and a collinear path.
:::

:::{seealso}
- [The example gallery](index.md) for the pages that *are* written.
:::
