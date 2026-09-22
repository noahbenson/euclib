# Building the `euclib` documentation

The documentation is a **Jupyter Book 2** site (mystmd). Jupyter Book 2 uses the
`myst.yml` configuration format rather than Jupyter Book 1's `_config.yml` and
`_toc.yml`, so the site is configured by `docs/euclib/myst.yml`. The workflow
mirrors the one already used by
[`immlib`](https://github.com/noahbenson/immlib).

## Layout

- `euclib/` — the **site root**. `jupyter-book` must be run from this directory,
  because that is where `myst.yml` lives. `_build/` and the generated `api/`
  pages are gitignored.
- `euclib/examples/` — the example gallery. Each page adapts one test from an
  upstream library; see "Provenance" below.
- `euclib/examples/_figures/` — the static figures, committed on purpose. A page
  regenerates its figure when it is executed, but the committed copy lets a
  structural build (one without `--execute`) render the images too.
- `euclib/euclib_viz.py` — the single place where figures are drawn. It uses
  `k3d` when an interactive front-end is available and matplotlib otherwise.
- `generate_api.py` — generates the API-reference pages from the library's own
  docstrings. Run it before building.

## Environment

The build uses its own conda environment, `euclib` (Python 3.11). The local
checkouts of `immlib`, `pcollections`, and `docshare` are installed **editable
with `--no-deps`**, because `euclib` declares `immlib >= 1.0` / `docshare >= 1.0`
while the checkouts are 0.2.x, and the `--no-deps` avoids pip trying to replace
them from PyPI.

```bash
conda create -n euclib python=3.11 -y
conda activate euclib
pip install numpy ipykernel matplotlib "jupyter-book>=2" k3d \
            joblib pyyaml "cloudpathlib[s3,gs,azure]>=0.18,<0.26" \
            "pint>=0.24,<0.27" "scipy>=1.8"
pip install -e ../immlib -e ../pcollections -e ../docshare -e .. --no-deps
python -m ipykernel install --user --name euclib --display-name "Python (euclib)"
```

The example pages declare `name: euclib` as their kernel, so the kernelspec must
be installed before the book is built with `--execute`.

## Building

```bash
conda activate euclib
python docs/generate_api.py          # regenerate the API reference
cd docs/euclib
jupyter-book build --html            # structural build (no execution)
jupyter-book build --html --execute  # full build: runs the example pages
```

The HTML lands in `docs/euclib/_build/html`. To view it:

```bash
python3 -m http.server -d docs/euclib/_build/html 8000
```

## Provenance policy

Every example page must cite the upstream test it adapts in a **Provenance**
admonition, and must link to it at a **pinned commit** — never a branch name —
so the citation cannot drift. The admonition gives the repository, the file, the
function name and line, the permalink, and a prose paragraph explaining how the
example was adapted. No upstream test code is copied.

Pinned revisions used so far:

| Repository | Revision |
|---|---|
| `pyvista/pyvista` | `85fbb5c71c02943ebe9dd10051235fe4d05b39aa` |
| `trimesh/trimesh` | `fcf660feb0a14c68fd3945789e8ed77e260f9167` |

## Figures

`euclib_viz.show3d` and `euclib_viz.show2d` are the only drawing entry points a
page uses. `show3d` renders with `k3d` when `EUCLIB_DOCS_INTERACTIVE` is set to a
true value and with matplotlib otherwise; `show2d` always uses matplotlib. The
reason is that a `k3d.Plot` is a live Jupyter widget and does not render in a
statically built book — see the module docstring in `euclib_viz.py`.
