# `euclib`: a library for tracking data associated 2D/3D Euclidean space

A Python library for tracking data associated with 2D or 3D Euclidean
geoemetric space. Tools for representing images, meshes, and models are
included.


## Motivation

This library solves the problem of representing spatial 2D and 3D Euclidean
data such as points, line segments/curves, triangle meshes/surfaces,
tetrahedral meshes/volumes, prism-meshes, grid-based images, and
voxel/pixel-based images; of transforming and interpolating properties between
these geometric objects, and of calculating basic geometric operations like
distances, geodesics, and intersections.


## High Level Design Choices

* The `euclib` library prefers to treat data, including matrices of points and
  properties assigned to those points, as immutable, and it prefers whenever
  possible to represent data using explicitly immutable data structures and
  collections.
* When possible, `euclib` prefers to provide lazily-loaded interfaces to data;
  for example, a mesh should not calculate its surface area until its surface
  area has been requested. However, `euclib` makes use of in-memory caching
  for such lazy computations.
* The `euclib` library heavily uses the `immlib` library to create immutable
  lazy workflows (directed acyclic computation graphs) to represent data in an
  immutable and lazy way. By using `immlib`, which relies on `pcollections`, it
  can maintain thread safety while doing this.
* The `euclib` library is designed to work well with both `numpy` and `pytorch`
  as well as with `pint` quantities and units. It uses `immlib`'s utilities to
  do this also. In particular, `euclib` operations should be compatible with
  PyTorch gradient descent optimization if they are used in models or loss
  functions.
* Geometric objects heavily encourage and enable the use of 'properties':
  numerical values attached to the geometric objects themselves. For example, a
  point cloud might have, as a property, `'temperature'`, which gives the
  temperature in Celsius at each point in the cloud. The points in a mesh might
  have properties like `'curvature'` while the faces might have properties like
  `'surface_area'`. Properties can be easily interpolated between
  representations of different geometric objects.
* Because `euclib` values PyTorch interoperability, it prefers to order its
  arrays and tensors such that their shapes are `(C..., X...)` where `C...`
  represents the channel dimensions and `X...` represents the spatial
  dimensions. PyTorch typically expects data used in model training to have a
  shape of `(N, C, X...)` where `N` is the batch size; `euclib` assumes that
  batches will contain several geometric objects whose properties or
  coordinates will make up the channels and additional dimensions. Effectively,
  this means that if a geometric object contains `N` coordinates, then its
  coordinate matrix will have a shape of `(D, N)` (where `D` is the
  dimensionality, 2 or 3), with `D` being the one channel dimension and `N`
  being the one spatial dimensions. The shape of a coordinate property array
  for this same object where the property encodes a shape of `A...` for each
  vertex would have a shape of `(A..., N)`. For a voxel image whose shape is
  `(R, C, S)`, a property consisting of a `2x3` Jacobian matrix per voxel would
  have a shape of `(2, 3, R, C, S)`; here the `(2,3)` are the channel
  dimensions and the `(R,C,S)` are the spatial dimensions.


## Architecture Overview

The `euclib` library is primarily written in pure Python. A small core of
functions that highly benefit from low-level implementations are included in an
external C module `euclib._c`. Examples of low-level C functions include
functions for calculating the intersection points of line segments and meshes,
line segments with each other, or calculating the set of tetrahedrons that form
the intersection of another tetrahedron with a voxel. These functions are
implemented in C only when a performant alternative using NumPy/PyTorch could
not be found.

`euclid` heavily leans into immutable paradigms using the `immlib` library
([docs](https://noahbenson.github.io/immlib/); [GitHub:
`noahbenson/immlib`](https://github.com/noahbenson/immlib)), specifically
`plan`s, `plandict`s, and `planobject`s. In alignment with this paradigm, it
does not allow the modification of its own types, but notably `immlib` allows
its plan-based types to create new transient copies that can be modified
in-place. These transients are thread-safe only so long as they are
thread-local, but the persistent objects are of course thread-safe universally
because they cannot be updated. Operations on `euclid` objects must typically
done by efficiently making a copy of the object. For example, if you want to
warm a triangle mesh into a new arrangement, you would need to make a copy of
the mesh that uses a new coordinate matrix.


## Global and Local Coordinates

The `euclib` library uses a single set of standard Euclidean/Cartesian
coordinates for representing global position in space. Global coordinates can
be represented in `euclib` as follows:
* For 2D spaces, a single coordinate is represented as a 2D vector while a
  collection of `N` coordinates is represented by a matrix of shape `(2, N)`.
  Individual points and collections of points can also be represented as a
  mapping with keys `'x'` and `'y'`.
* For 3D spaces, a single coordinate is represented as a 3D vector while a
  collection of `N` coordinates is represented by a matrix of shape `(3, N)`.
  Individual points and collections of points can also be represented as a
  mapping with keys `'x'`, `'y'`, and `'z'`.

In addition, each topology implicitly defines its own local coordinate system
that addresses its entire internal space. Like geometric objects, local
coordinate systems are linked to their topologies but can only be instantiated
with coordinates: one can lookup the local coordinates of a global coordinate
within a given geometric object, and the returned local coordinate will be
valid for any geometric object that shares its topology (but may not map to the
same global coordinate in another object). Local coordinates are not guaranteed
to be continuous everywhere, and they are not guaranteed to have linear
relationships to each other or even straightforward relationships. For example,
in a 3D triangle mesh (a simplex-based geometric object), the local coordinates
identify the position of the global coordinate on the mesh surface by using
three numbers: the index of the triangle and the first two barycentric
coordinates within it.

Because local coordinates do not all have the same structure, there is no
single representation of local coordinates. Throughout the `euclid` library,
functions that expect local coordinates use the parameter name `locs` while
functions that expect global coordinates use the parameter name `coords`.


## Geometric Objects

The `euclib` library primarily handles a few kinds of geometric objects in both
2D and 3D. All geometric objects include a matrix of global coordinates that
provides the position of its components in space; the objects themselves are
simplices or collections of simplices. There are four simplex-based geometric
types plus a grid-based type, and one derived type. All geometric objects
inherit from the abstract class `euclib.abc.Geometry`.

All geometric objects have a field, `topo`, which stores an topology object
that defines how the object's connectivity is structured. Topology objects
inherit from `euclib.abc.Topology`. Specific geometry objects are essentially
reifications of their topologies and can be seen as separate warpings or
transformations of the same topology.

### Simplex-based Geometric Objects

All simplex-based geometric objects inherit from the abstract class
`euclid.abc.SimplexGeometry`, which enforces a relationship with the
`euclid.abc.SimplexTopology` class. The `SimplexTopology` class is described in
greater detail below, but it contains information about how simplices are
connected to each other (i.e., the connection or corner matrices).

There are four fundamental simplex-based geometries in `euclid`:
1. **Vertices** (0-dimensional simplices) can be a singe point or a cloud of
   points. Each point can be given a property value (for example, the
   `'temperature'` measured at each point). The simplex matrix for vertices is
   just a `1 x N` matrix of coordinate indices. Vertices use the
   `euclib.VertexGeometry` and `eublib.VertexTopology` types.
2. **Segments** (1-dimensional simplex) can be individual line segments or
   collections or paths of such segments. Segments can have properties like
   `'length'` or `'spring_constant'`. The simplex matrix for segments is a `2 x
   N` matrix of start and end coordinates for each segment. Segments use the
   `euclib.SegGeometry` and `euclib.SegTopology` types.
3. **Triangles** (2-dimensional simplex) can be individual triangles or
   collections (meshes) of such triangles. Triangles can have properties like
   `'surface_area'`. The simplex matrix for each triangles object is a `3 x N`
   matrix of the triangle corners. Triangles use the `euclib.TriGeometry`
   and `euclib.TriTopology` types.
4. **Tetrahedra** (3-dimensional simplex; 3D spaces only) can be individual
   tetrahedrons or collections of such tetrahedra. Tetrahedra can have
   properties like `'volume'`. The simplex matrix for each tetrahedron object
   is a `4 x N` matrix of the tatrahedral corners. Tetrahedra use the
   `euclib.TetGeometry` and `euclib.TetTopology` types.

### Grid-based Geometric Objects

Grid-based Geometric Objects are objects that are laid out on an image
grid. Such objects can be specified using minimal information: the global
coordinates of the corner of the grid, the spacing of the grid, the orientation
of the grid, and the dimensions of the grid.

1. **Voxels** (3-dimensional grid; 3D spaces only) can be individual boxes or
   grids of boxes. Voxels can have properties, for example `'T1w_intensity'`,
   which are stored as 3D images.
2. **Pixels** (2-dimensional grid; 2D spaces only) can be individual boxes or
   grids of boxes. Pixels can have properties, which are stored as 2D images.

The `euclib.GridTopology` type stores grid topological information,
specifically the shape of the represented image matrix. The
`euclib.GridGeometry` type stores a `GridTopology` object for its `topo` field
and an `affine` transformation that describes how image indices are translated
into global Euclidean Cartesian coordinates.

### Derived Geometric Objects

**Prisms** (3-dimensional volume; 3D spaces only) can be individual prisms or
collections of prisms. A prism is essentially a pair of triangles whose corners
are connected to form a volume. Prism meshes are to represent layers of objects
that resemble thin sheets (such as the human cerebral cortex) where it is
convenient to represent the two sides of the sheet using the same tesselation,
just different vertex positions. Prisms are represented internally using sets
of tetrahedra and are sometimes represented as triangle meshes whose vertices
are line segments.

Prism objects use a `euclib.PrismTopology` type for their `topo` which inherts
from the `euclib.TriTopology` type. The prism topology is largely similar to a
triangle topology&mdash;in fact, a prism object is basically two triangle mesh
objects that share the same triangle topology&mdash;but prism topologies have
3D local coordinates while triangle topologies have 2D local coordinates. A
prism geometry object stores its coordinates as a tuple.

```
(coords0, coords1) = prism_obj.coords
```

In the above code-block, `coords0` is the matrix of the first side of the prism
mesh and `coords1` is the matrix of the other side. The prism object recognized
elevation values that indicate the height within the object between `coords0`
and `coords1`. Essentially each prism has, for any given elevation between 0
and 1, and triangle mesh geometric object whose topology uses the same triangle
matrix as the prism topology and whose coordinates are 
`coords0 * (1-elev) + coords1 * elev`.


### Properties of Geometric Objects

All geometric objects allow user-defined properties to be attached to their
data under a unique name. All such objects have a field `properties` that holds
a lazy dictionary (`pcollections.ldict`) of the properties associated with the
object. Properties must be given unique hashable names. 

All geometric objects overload the `__getitem__` method so that a property
array/tensor can be extracted from the object by name. For example,
`point_cloud['temperature']` should extract the `'temperature'` property from
each coordinate in the `point_cloud` geometric object. These overloads also
allow for the selection of multiple properties at once
(`point_cloud[['temperature','pressure']]` would return
`(temperature_prop,pressure_prop)`) and for indexing the spatial dimensions of
all selected properties. For example, in a triangle mesh with `N` coordinates,
`mesh['flux', ii]` for a boolean mask `ii` (or any other valid numpy/pytorch
index into the final dimension of the coordinate matrix such as a list of
indices) returns the `'flux'` property for the indices given by `ii`. The
property is still returned with it's typical shape, however. Suppose we have a
3D grid-image object `im` whose grid shape is `(10, 12, 15)`, and for the
property `'jacobian'` there is a `3x2` matrix represented for each voxel. The
shape of the `'jacobian'` property array would be `(3,2,10,12,15)`, and the
following would be a correct way to extract the jacobian matrix values for the
very first voxel.

```
((dfx_dx,dfx_dy), (dfy_dx,dfy_dy), (dfz_dx,dfz_dy)) = im['jacobian', 0, 0, 0]
```

Prism meshes have more flexible representations of properties than other
geometric objects. They allow properties to be specified at multiple
elevations. Elevations are numbers between 0 and 1 (inclusive), that indicate
how relatively close the property is in space to either the first or second
side of the prisms. This comports with the interpretation of a prism as a 2D
triangle mesh sheet embedded in 3D whose triangle coordinates are actually line
segments instead of points in space. For example, if a prism mesh represented a
sheet of metal 10 cm thick in a finite element simulation, the `'temperature'`
might be very different on either side of the prisms, and the user might want
to simulate how the temperature propogates across the sheet at a resolution of
mm. The temperature can be specified as a matrix of temperature values with a
shape of `(100, N)`; the `100` indicating the elevation dimension. If the
specific elevations need to be specified (i.e., they are not just evenly spaced
from 0 to 1), a property can be provided as a tuple `(elevs, values)` where
`elevs` is a matrix of vector of the elevations. If a vector is given, then it
is used for all spatial locations across the prism sheet; if a matrix is given,
then it must be the same shape as the values and provide the elevation of every
individual value given, in monotonically increasing order per spatial position.

The convenience method `geom_obj.prop` is the general purpose property
extraction method that facilitates the most common kinds of property
lookups. This method uses the following signature:

```
def prop(self, property, /, at=Ellipsis, **kw):
```

The `**kw` allows several property meta-data options to be specified (those not
specified explicitly are extracted from the metadata on the property that is
being queried). The `prop` function works only with coordinate-based (not
simplex-based) properties. The function itself works as follows.
 1. Some dispatch is performed on the kind of property.
    * **If the object is a simplex-based object** then the property argument
      may be a tuple `(k, property)` where `k` specifies the simplex order of
      the property. This value may be 0, 1, 2, 3, or `None`, indicating that
      the `property` belongs to the object's vertices, segments, triangles,
      tetrahedrons, or coordinates, respectively. If the first argument is
      `Ellipsis`, then the `Ellipsis` is ignored, as if only the `property`
      were provided.

      When either `Ellipsis` is given or no tuple is given, then the property
      is either interpreted as a coordinate or a vertex properties. (Whether
      coordinate or vertex can be determined by its final shape dimension,
      which must either match the number of coordinates, the number of vertices
      or both, in which case it doesn't matter which.)
    * **If the object is a prism-based object** then the property is processed
      as for simplex-based objects. Prism-based objects are basically
      simplex-based objects whose properties and simplices have an elevation
      dimension.
    * **If the object is a grid-based object** then the property argument
      needs no special processing.
 * The `at` argument is processed next. This argument tells `prop` where in
   space the property is being obtained at. It can specify the location in
   global or local coordinates, or it can extract the property at specific
   coordinate or vertex indices. The following formats are allowed:
    * If the `at` argument is a boolean mask with the same dimensions as the
      property requested or a sequence of integer indices, then it is treated
      as indices into the coordinate or vertex property.
    * If the `at` argument is a set of global coordinate (for example, a `3xN`
      matrix), then the property values are interpolated at the given
      coordinates according to the arguments or property metadata. For points
      that are not in the object, extrapolation may be used if requested.
    * Otherwise, the `at` argument is assumed to be a set of local coordinates,
      and the property values are interpolated at these coordiantes according
      to the arguments or property metadata.

Properties can be added to an object using the `withprop(new_name, new_values)`
method and removed from an object using the `dropprop(name)` method (both
return immutable copies of the geometric object with/without the
property). Both allow the property name argument to be a tuple like `(2,
'surface_area')` to refer to a simplex property instead of a coordinate
property in the case of simplex-based geometry objects (see the description of
simplex-based properties below). The `withprop` method also allows metadata
about how the property should be handled to be passed in as optional
arguments. The metadata for a property are also valid options for the `at`
function, when called on a simplex object, and can specify a few things:
 * `interp`: How to interpolate the property within the object
   itself. Interpolation is used when, for example, a user requests the value
   of a vertex property at a position within one of the triangles of a triangle
   mesh. The value can be interpolated using nearest neighbor, linear,
   quadratic, or cubic inteprolation, or the property could be considered
   uninterpolable, meaning an interpolation would give an `NA` result. The
   values `None`, `0`, `1`, `2`, and `3` stand for no interpolation, then
   nearest-neighbor, linear, quadratic, and cubic interpolation respectively.
   This can also be set to `Ellipsis`, which is the default value. `Ellipsis`
   means that nearest-neighbor interpolation should be used if the property is
   a non-continuous value such as an integer or boolean and otherwise cubic (3)
   interpolation should be used. All `NA` values (typically `NaN` for real
   numbers) always propogate to their nearest neighbor points, regardless of
   interpolation order.
 * `extrap`: How to extrapolate the property beyond the object. If a user
   requests a property at a position that is not part of the object, such as
   requesting a value at a point that is outside of a triangle mesh, how should
   it be estimated? This option may only be `None` or `0`: there is no method
   for performing higher-order interpolations. When the value is `None`, then
   no extrapolation will be performed and any result is just `NA`. If `0` is
   provided, then the interpolated value of the nearest point on the object
   will be returned. By default this is `None`.
 * `dtype`: What dtype to use. The default, `None`, uses the natural dtype of
   the property array that is given.
 * `mask`: An optional mask for the property. Masks are described below and can
   be used to indicate that certain property values should be considered
   missing, and instead of attempting to interpolate these values, a null value
   should be returned. For example, one might call `mesh.withprop('param',
   paramvec, mask=valid_vertices)` where `'param'` is the name of the property
   being added to the mesh's coordinates, `paramvec` is the vector of
   parameters, one per mesh coordinate, and `valid_vertices` is a boolean mask
   where all vertices whose parameters are valid (by some metric based on the
   noise in the measurements, perhaps) are `True`. If one interpolates the
   `'param'` property without providing explicit parameter overrides, then any
   value that would be interpolated from an invalid vertex will instead return
   the null value.
 * `null`: What value to use to indicate missing or not-available (`NA`) data
   in a masked interpolation.  When this option is not explicitly provided,
   `NaN` is used for floating point data, and `dtype.type()` is used for any
   other data (which implies 0 for integers and `None` for generic object
   arrays). If the value is explicitly provided, then it must be a valid
   instance of the property's dtype.

A user duplicate a mesh while updating a property's metadata via:

```
# Change the interpolation order from its previous value to 1 without
# providing new property values.
dup_mesh = mesh.withprop('param', interp=1)
```

Grid-based objects, simplex-based objects, and prisms all represent their
properties somewhat differently.

#### Properties of Simplex-based Objects

All simplex-based geometric objects are of type `SimplexGeometry` store their
global cartesian coordinates in a `D x N` matrix `geom_obj.coords` (where `D`
is the dimensionality of the space, either 2 or 3, and `N` is the number of
coordinates). The simplices of the object are defined in `geom_obj.topo`, an
object of type `euclib.SimplexTopology`. The `SimplexTopology` type primarily
stores a matrix of simplex corners, `geom_obj.topo.indices`, each of which is
an index into the `coordinates` matrix of the `geom_obj`. The simplex data are
separated from the coordinate data because many objects frequently share the
same simplices but have different coordinates&mdash;prisms in particular are
built to exploit this principle. For a point-cloud simplex, the `indices`
matrix is a `1 x N` matrix and would typically just contain something
equivalent to `np.arange(N)[None,:]`.

The `SimplexTopology` type represents its primary set of simplex indices in the
`indices` field, but for simplex types other than point clouds, there are
implied simplices as well. For example, in a 3D triangle mesh, the primary
simplices are the triangles, so `mesh.topo.indices` would be a `3 x M` matrix
(where `M` is the number of triangles), but there are also edges (or line
segments) implied by the sides of each triangle, which are represented in a `2
x E` matrix (`E` is the number of unique edges in the mesh). There is further a
point cloud simplex collection that represents all the vertices included in the
mesh's triangles (typically equivalent to `np.unique(indices)`). These implied
simplices are stored in `geom_obj.topo.simplices`, which is a lazy list
(`pcollections.llist`) object whose length is equal to the dimensionality of
the primary simplex type plus one. For example, for a triangle mesh, the
primary simplex type has a dimensionality of 2, so the simplices list has
indices 0 (the indices of points in the mesh), 1 (the edges/segments), and 2
(the triangle indices, identical to the `indices` field). These simplex
matrices are calculated lazily because they are often unused.

Not all coordinates have to be referenced in the `topo` object&mdash;it is
considered valid for a simplex to have more coordinates than it uses (and this
is sometimes exploited for objects like submeshes). Extra coordinates are not
compared when examining object equality. Because a topology object might be
designed to go with a matrix of coordinates that is far larger than its set of
simplices, topology objects may specify that their coordinate matrices have a
specific size (`topo.coord_count`). The coordinate count is distinct from the
`topo.vertex_count` (the number of coordinates actually included in the object
simplices as vertices), and `topo.simplex_count` which is itelf an `llist` that
is essentially equivalent to `[s.shape[-1] for s in topo.simplices]`.

User-defined properties of simplex objects come in two basic forms, both of
which are treated similarly: coordinate properties and simplex
properties. Coordinate properties are the primary properties for any geometric
object, and they are associated with the coordinates of the object. If there
are `N` coordinates in an object then a coordinate property array/tensor must
have a final dimension of `N`. Simplex properties are associated with entire
simplices and thus their final dimension (simplex count) must match the number
of columns in their simplex matrix. They are stored in
`geom_obj.simplex_properties` which is a tuple of `pcollections.ldict` property
dictionary objects for each simplex type in order.

For simplex-based geometric objects, the `__getitem__` method has special
behavior. The user may specify the simplex using its dimensionality as the
first argument in a tuple passed to `__getitem__`. So, for example, for a
triangle mesh, `mesh['x']` would retrieve the coordinate property `'x'`, whose
dimensionality will match the dimensionality of `mesh.coords`. But
`mesh[2,'x']` is equivalent to `mesh.simplex_properties[2]['x']`: it extracts
the triangle-property named `'x'`, whose last dimension will match
`mesh.topo.simplex_count[2]`. A special case here is that `mesh[0, prop]` and
`mesh[prop]` are slightly different. The former extracts the property values
for points included in the mesh while ignoring the properties for coordinates
not included in the mesh's triangles while the latter extracts the property for
all coordinates. If a property exists for the coordinates but not for the
vertex simplices, then `mesh[0, prop]` will correctly extract the coordinate
property at the active vertices.


## API and Usage Examples

### Getting Started

```python
# Import the library:
import euclib as el
```


## About

**Author**: Noah C. Benson &lt;[nben@uw.edu](nben@uw.edu)&gt;


## License

MIT License

Copyright (c) 2024-2025 Noah C. Benson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


