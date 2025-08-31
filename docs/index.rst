Documentation for the package modDR (modified Dimensionality Reduction)
=======================================================================

Modified Dimensionality Reduction (moddr) is a Python package for combining dimensionality reduction techniques with community detection and visualization capabilities. This package presents a method for automatically modifying the positions of data points in low-dimensional spaces based on a feature selection to preserve both global structure and feature-driven similarity. The provided workflow uses graph theory concepts and layout methods to change the arrangement of a given DR-positioning in such a way that an additional similarity measure – based on selected features, for example – is integrated into the distance structure.

Source Code
-----------

The source code is available at: `https://github.com/kohaupt/modDR <https://github.com/kohaupt/modDR>`_

Installation
------------

The package is published on PyPI: `https://pypi.python.org/pypi/moddr <https://pypi.python.org/pypi/moddr>`_

Install it via:

.. code-block:: bash

    pip install moddr

or, if you are using the `uv` package manager: `https://docs.astral.sh/uv/ <https://docs.astral.sh/uv/>`_

.. code-block:: bash

    uv add moddr

Development
-----------

The package was developed with the `uv` package manager, which is required for local development. After cloning the repository, run the following steps to create a working development environment (if not inside an existing workspace):

.. code-block:: bash

    uv init project-name
    uv sync
    uv pip install -e .

The command ``uv pip install -e .`` is mandatory to make the package functions available locally. You can test the correct local installation by running:

.. code-block:: bash

    uv run pytest

Quick Start
-----------

The package consists of three modules:

- ``processing`` – computing modified embeddings
- ``evaluation`` – computing metrics for evaluation
- ``visualization`` – visualizing embeddings

An instance of the ``EmbeddingState``-class allows you to access all computed information. A minimal example can be implemented as follows. The parameters may have to be adjusted for the used data set, as the default parameters may not be suited.

.. code-block:: python

    import moddr

    # Run the full moddr pipeline
    embeddings = moddr.processing.run_pipeline(
        data=your_data,
        sim_features=your_feature_selection,
        verbose=True
    )

    # Visualize the embeddings
    moddr.visualization.display_embeddings(embeddings)


.. toctree::
   :maxdepth: 2
   :titlesonly:

   API Reference <api_index>
