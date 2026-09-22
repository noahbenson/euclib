---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python (euclib)
  language: python
  name: euclib
---
# Getting Started

## Installation

`euclib` is not yet on PyPI; it is installed from a checkout of its
[source](https://github.com/noahbenson/euclib):

```bash
$ git clone https://github.com/noahbenson/euclib
$ pip install -e ./euclib
```

## Importing

```{code-cell}
import numpy as np
import euclib as el
```

## A first geometry

The most direct way to build a geometry is to supply its coordinates and a
topology. Here is a triangle mesh with a single triangle — three corners and one
face:

```{code-cell}
from euclib import types as et

# A (3, N) matrix of corner coordinates...
coords = np.array([[0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0],
                   [0.0, 0.0, 0.0]])

# ...and a (3, M) matrix of triangle corners.
topo = et.TriTopology(np.array([[0], [1], [2]], dtype='int64'))

mesh = et.TriMesh(coords, topo)
mesh
```

Every geometry is immutable, so "modifying" one produces a new object. Adding a
property is the most common such change:

```{code-cell}
mesh = mesh.withprop('height', coords[2])
mesh['height']
```

## Where to go next

- The [example gallery](../examples/index.md) walks through complete examples,
  each adapted from a test in an established geometry library.
- The [API reference](../api.md) is generated from the library's docstrings.
