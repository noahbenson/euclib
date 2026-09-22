# `euclib`: Data in 2D and 3D Euclidean Space

`euclib` is a Python library for tracking data associated with 2D and 3D
Euclidean geometric space. It represents points, line segments and curves,
triangle meshes and surfaces, tetrahedral meshes and volumes, prism meshes,
grid-based images, and voxel/pixel images; it transforms and interpolates
properties between those representations; and it calculates basic geometric
quantities such as distances and nearest positions.

`euclib`'s defining commitments are that its data are **immutable**, that its
computations are **lazy** and cached, and that the values attached to a
geometry are first-class **properties** that travel with the geometry and can be
sampled from any other geometry that shares its space.

Start with the [introduction](intro/about.md), try the [getting-started
guide](intro/getting-started.md), or go straight to the [example
gallery](examples/index.md).
