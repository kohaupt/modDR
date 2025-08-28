# moddr

[![PyPI version](https://img.shields.io/pypi/v/moddr.svg)](https://pypi.python.org/pypi/moddr)
[![Build Status](https://img.shields.io/travis/kohaupt/moddr.svg)](https://travis-ci.com/kohaupt/moddr)
[![Documentation Status](https://readthedocs.org/projects/moddr/badge/?version=latest)](https://moddr.readthedocs.io/en/latest/?version=latest)

Modified Dimensionality Reduction (moddr) is a Python package for advanced dimensionality reduction techniques with community detection and visualization capabilities.

- **Free software:** MIT license
- **Documentation:** https://moddr.readthedocs.io

## Features

- Dimensionality reduction using UMAP
- Community detection with Leiden algorithm
- Multiple graph layout algorithms (Kamada-Kawai, MDS, Fruchterman-Reingold)
- Comprehensive evaluation metrics for dimensionality reduction quality
- Advanced visualization tools for embeddings and communities
- Pipeline framework for reproducible experiments

## Installation

```bash
pip install moddr
```

## Quick Start

```python
from moddr.processing import run_pipeline
from moddr.evaluation import compute_metrics
from moddr.visualization import display_embeddings

# Run the full moddr pipeline
embeddings = run_pipeline(
    data=your_data,
    pipeline_config=config
)

# Evaluate the results
metrics = compute_metrics(embeddings)

# Visualize the embeddings
display_embeddings(embeddings)
```

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
