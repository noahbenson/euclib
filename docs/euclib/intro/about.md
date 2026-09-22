# About `euclib`

`euclib` is a Python library for tracking data associated with 2D or 3D
Euclidean geometric space. Tools for representing images, meshes, and models are
included.

## Motivation

The library solves the problem of representing spatial 2D and 3D Euclidean data
— points, line segments and curves, triangle meshes and surfaces, tetrahedral
meshes and volumes, prism meshes, grid-based images, and voxel/pixel images — of
transforming and interpolating *properties* between these geometric objects, and
of calculating basic geometric operations such as distances, geodesics, and
intersections.

## The kinds of geometry

`euclib` handles a small set of geometric objects, in both 2D and 3D, all of
which inherit from the abstract base `euclib.abc.Geometry`. Every geometry has a
matrix of global coordinates giving the position of its components in space, and
a `topo` field holding a *topology* object that defines how those components are
connected.

**Simplex-based geometries.** Four types are built from simplices:

1. **Vertices** — a single point or a cloud of points
   (`euclib.types.VertexSet`). A point cloud can carry, for example, a
   `'temperature'` value per point.
2. **Segments** — individual line segments or paths of them
   (`euclib.types.SegPath`), with properties such as `'length'`.
3. **Triangles** — individual triangles or triangle meshes
   (`euclib.types.TriMesh`), with properties such as `'surface_area'`.
4. **Tetrahedra** — individual tetrahedra or tetrahedral meshes
   (`euclib.types.TetMesh`; 3D only), with properties such as `'volume'`.

**Grid-based geometries.** `euclib.types.Grid` covers image grids: voxels in 3D
and pixels in 2D. A grid is specified by minimal information — its shape and an
affine transformation that maps image indices to global coordinates — and its
properties are stored as images.

**Derived geometries.** A **prism mesh** (`euclib.types.PrismMesh`; 3D only) is
a pair of triangle meshes that share one topology but have different
coordinates, used to represent thin sheets such as the human cerebral cortex.
A prism knows about *elevations* between 0 and 1 that interpolate between its
two sides, and it converts to a collection of tetrahedra.

## Properties

The feature that most distinguishes `euclib` is its treatment of properties:
numerical values attached to a geometry under a unique name. A property may be
attached to a geometry's coordinates (a value per point) or to its simplices (a
value per segment, triangle, or tetrahedron). Properties carry metadata — how
to interpolate them within the geometry, how to extrapolate them beyond it,
which values are masked out — and, critically, they can be **interpolated
between representations**: a property sampled on one geometry can be transferred
onto another geometry that occupies the same space, even one of a different kind
(a mesh to a grid, say).

## Design choices

`euclib`'s central design choices are described in [Design
Principles](design.md). In brief:

- Data are treated as **immutable**, using explicitly immutable data structures
  wherever possible.
- Interfaces are **lazy**: a mesh does not compute its surface area until that
  area is requested, and the result is cached.
- The library is built on [`immlib`](https://noahbenson.github.io/immlib/),
  which provides the immutable, lazy computation graphs (`plan`s, `plandict`s,
  and `planobject`s) that `euclib` uses throughout.
- Operations are written to be compatible with both **NumPy** and **PyTorch**,
  as well as with `pint` quantities and units, so that they can take part in
  gradient-based optimization.
- Array shapes follow the convention `(C..., X...)` — channel dimensions first,
  spatial dimensions last — which matches what PyTorch's model-training code
  expects once a batch dimension is added.
