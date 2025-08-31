# modDR (modified Dimensionality Reduction)

[![PyPI version](https://img.shields.io/pypi/v/moddr.svg)](https://pypi.python.org/pypi/moddr)
[![Documentation Status](https://readthedocs.org/projects/moddr/badge/?version=latest)](https://moddr.readthedocs.io/en/latest/)

Modified Dimensionality Reduction (moddr) is a Python package for combining dimensionality reduction techniques with community detection and visualization capabilities.
This package presents a method for automatically modifying the positions of data points in low-dimensional spaces based on a feature selection to preserve both global structure and feature-driven similarity. The provided workflow uses graph theory concepts and layout methods to change the arrangement of a given DR-positioning in such a way that an additional similarity measure – based on selected features, for example – is integrated into the distance structure. 

## Documentation
Full documentation is available at: https://moddr.readthedocs.io

## Installation
The package is published on PyPI: https://pypi.python.org/pypi/moddr 

Install it via:
```bash
pip install moddr
```

or, if you are using the [uv package manager](https://docs.astral.sh/uv/):

```bash
uv add moddr
```

## Development
The package was developed with the [uv package manager](https://docs.astral.sh/uv/), which is required for local development. After cloning the repository, run the following commands to create a working development environment (if not inside an existing workspace):

```bash
uv init project-name
uv sync
uv pip install -e . # needed to make the package functions available locally
```

You can test the correct local installation by running:

```bash
uv run pytest
```

## Quick Start
The package consists of three modules:
- `processing` – computing modified embeddings
- `evaluation` – computing metrics for evaluation
- `visualization` – visualizing embeddings

An instance of the `EmbeddingState`-class allows you to access all computed information. Examples are available under `./examples` as jupyter-notebooks. A minimal example can be implemented as follows. The parameters may have to be adjusted for the used data set, as the default parameters may not be suited.


```python
import moddr

# Run the full moddr pipeline
embeddings = moddr.processing.run_pipeline(
    data=your_data,
    sim_features=your_feature_selection
    verbose=True
)

# Visualize the embeddings
moddr.visualization.display_embeddings(embeddings)
```