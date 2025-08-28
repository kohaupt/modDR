import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

from moddr.embedding_state import EmbeddingState
from moddr.processing import processing


@pytest.fixture
def sample_data():
    # Simple 100-point dataset with three features
    return pd.DataFrame(
        {
            "f1": [i for i in range(100)],
            "f2": [i**2 for i in range(100)],
            "f3": [i**3 for i in range(100)],
        }
    )


@pytest.fixture
def sample_embedding_state():
    # Create a dummy EmbeddingState with 100 nodes
    embedding = {i: np.array([float(i), float(i)]) for i in range(100)}
    graph = nx.complete_graph(100)
    nx.set_node_attributes(graph, 0, "community")
    return EmbeddingState(embedding=embedding, graph=graph, title="test")


def test_dimensionality_reduction_umap_runs(sample_data):
    emb = processing.dimensionality_reduction_umap(sample_data, n_neighbors=2)
    assert isinstance(emb, EmbeddingState)
    assert emb.embedding is not None
    assert emb.graph is not None
    assert len(emb.embedding) == len(sample_data)


def test_compute_pairwise_dists_basic(sample_data):
    dists = processing.compute_pairwise_dists(sample_data)
    assert dists.shape == (100, 100)
    assert np.allclose(np.diag(dists), 0)


def test_compute_pairwise_dists_invert_normalize(sample_data):
    dists = processing.compute_pairwise_dists(sample_data, invert=True, normalize=True)
    assert dists.shape == (100, 100)
    assert np.all((dists >= 0) & (dists <= 1))


def test_assign_graph_edge_weights(sample_embedding_state):
    sim_matrix = np.ones((100, 100), dtype=np.float32)
    emb = processing.assign_graph_edge_weights(sample_embedding_state, sim_matrix)
    for u, v in emb.graph.edges():
        assert emb.graph[u][v]["weight"] == 1.0


def test_assign_graph_edge_weights_shape_mismatch(sample_embedding_state):
    sim_matrix = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        processing.assign_graph_edge_weights(sample_embedding_state, sim_matrix)


def test_apply_balance_factor(sample_embedding_state):
    mod_pos = {i: np.array([i + 1, i + 2]) for i in range(100)}
    emb = processing.apply_balance_factor(sample_embedding_state, mod_pos, 0.5)
    for i in range(100):
        expected = 0.5 * sample_embedding_state.embedding[i] + 0.5 * mod_pos[i]
        assert np.allclose(emb.embedding[i], expected)


def test_apply_balance_factor_invalid_balance(sample_embedding_state):
    mod_pos = {i: np.array([i + 1, i + 2]) for i in range(100)}
    with pytest.raises(ValueError):
        processing.apply_balance_factor(sample_embedding_state, mod_pos, -0.1)
    with pytest.raises(ValueError):
        processing.apply_balance_factor(sample_embedding_state, mod_pos, 1.1)


def test_compute_distance_scaling():
    high = np.array([1, 2, 3], dtype=np.float32)
    low = np.array([2, 4, 6], dtype=np.float32)
    scale = processing.compute_distance_scaling(high, low)
    assert np.isclose(scale, np.dot(high, low) / np.dot(low, low))


def test_compute_distance_scaling_shape_mismatch():
    high = np.array([1, 2, 3], dtype=np.float32)
    low = np.array([2, 4], dtype=np.float32)
    with pytest.raises(ValueError):
        processing.compute_distance_scaling(high, low)


def test_compute_knn_graph(sample_data):
    graph, weights = processing.compute_knn_graph(sample_data, n_neighbors=2)
    assert isinstance(graph, nx.Graph)
    assert weights.shape[0] == graph.number_of_edges()


def test_compute_boundary_subgraph():
    graph = nx.path_graph(4)
    subgraph = graph.subgraph([0, 1]).copy()
    for u, v in graph.edges():
        graph[u][v]["weight"] = 1.0
    subgraph, boundary = processing.compute_boundary_subgraph(graph, subgraph)
    assert set(boundary) == {2}
    assert subgraph.has_node(2)
    assert subgraph.has_edge(1, 2)


def test_community_detection_leiden_basic(sample_embedding_state):
    # Assign weights to edges for Leiden algorithm
    for u, v in sample_embedding_state.graph.edges():
        sample_embedding_state.graph[u][v]["weight"] = 1.0
    emb = processing.community_detection_leiden(
        sample_embedding_state, resolution_parameter=0.5
    )
    assert isinstance(emb, EmbeddingState)
    assert hasattr(emb, "partition")
    assert isinstance(emb.partition, dict)
    # All nodes should have a 'community' attribute
    for node in emb.graph.nodes():
        assert "community" in emb.graph.nodes[node]


def test_community_detection_leiden_inplace(sample_embedding_state):
    for u, v in sample_embedding_state.graph.edges():
        sample_embedding_state.graph[u][v]["weight"] = 1.0
    emb = processing.community_detection_leiden(
        sample_embedding_state, resolution_parameter=0.5, inplace=True
    )
    assert emb is sample_embedding_state


def test_community_detection_leiden_no_graph():
    emb = EmbeddingState({})
    with pytest.raises(ValueError):
        processing.community_detection_leiden(emb, resolution_parameter=0.5)


def test_community_detection_leiden_metadata_update(sample_embedding_state):
    for u, v in sample_embedding_state.graph.edges():
        sample_embedding_state.graph[u][v]["weight"] = 1.0
    emb = processing.community_detection_leiden(
        sample_embedding_state, resolution_parameter=0.5
    )
    assert emb.metadata["com_detection"] == "Leiden"
    assert emb.metadata["com_detection_params"]["resolution"] == 0.5
    assert "Leiden" in emb.title


def test_compute_modified_positions_unbalanced(sample_embedding_state, sample_data):
    sample_embedding_state.partition = {0: list(range(50)), 1: list(range(50, 100))}
    sample_embedding_state.community_centers = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 1.0]),
    }

    targets = pdist(sample_data, metric="euclidean")
    emb, modified_positions = processing.compute_modified_positions(
        sample_embedding_state,
        layout_param=1.0,
        layout_method="MDS",
        targets=targets,
        inplace=False,
        verbose=False,
    )
    assert isinstance(emb, type(sample_embedding_state))
    assert emb.embedding is not None
    assert emb.metadata["layout_method"] == "MDS"
    assert emb.metadata["layout_params"]["balance factor"] == 1.0
    assert modified_positions is not None

    targets_mod = pdist(np.array(list(modified_positions.values())), metric="euclidean")
    assert not np.allclose(targets_mod, targets)


def test_compute_modified_positions_mds(sample_embedding_state, sample_data):
    sample_embedding_state.partition = {0: list(range(50)), 1: list(range(50, 100))}
    sample_embedding_state.community_centers = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 1.0]),
    }

    targets = pdist(sample_data, metric="euclidean")
    emb, modified_positions = processing.compute_modified_positions(
        sample_embedding_state,
        layout_param=0.5,
        layout_method="MDS",
        targets=targets,
        inplace=False,
        verbose=False,
    )
    assert isinstance(emb, type(sample_embedding_state))
    assert emb.embedding is not None
    assert emb.metadata["layout_method"] == "MDS"
    assert emb.metadata["layout_params"]["balance factor"] == 0.5
    assert modified_positions is not None

    targets_mod = pdist(np.array(list(modified_positions.values())), metric="euclidean")
    assert not np.allclose(targets_mod, targets)


def test_compute_modified_positions_kk(sample_embedding_state, sample_data):
    sample_embedding_state.partition = {0: list(range(50)), 1: list(range(50, 100))}
    sample_embedding_state.community_centers = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 1.0]),
    }

    targets = pdist(sample_data, metric="euclidean")
    emb, modified_positions = processing.compute_modified_positions(
        sample_embedding_state,
        layout_param=0.5,
        layout_method="KK",
        targets=targets,
        inplace=False,
        verbose=False,
    )
    assert isinstance(emb, type(sample_embedding_state))
    assert emb.embedding is not None
    assert emb.metadata["layout_method"] == "KK"
    assert emb.metadata["layout_params"]["balance factor"] == 0.5
    assert modified_positions is not None

    targets_mod = pdist(np.array(list(modified_positions.values())), metric="euclidean")
    assert not np.allclose(targets_mod, targets)


def test_compute_modified_positions_fr(sample_embedding_state, sample_data):
    sample_embedding_state.partition = {0: list(range(50)), 1: list(range(50, 100))}
    sample_embedding_state.community_centers = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 1.0]),
    }

    targets = pdist(sample_data, metric="euclidean")
    emb, modified_positions = processing.compute_modified_positions(
        sample_embedding_state,
        layout_param=10,
        layout_method="FR",
        targets=targets,
        inplace=False,
        verbose=False,
    )
    assert isinstance(emb, type(sample_embedding_state))
    assert emb.embedding is not None
    assert emb.metadata["layout_method"] == "FR"
    assert emb.metadata["layout_params"]["iterations"] == 10
    assert modified_positions is not None

    targets_mod = pdist(np.array(list(modified_positions.values())), metric="euclidean")
    assert not np.allclose(targets_mod, targets)


def test_compute_modified_positions_invalid_method(sample_embedding_state):
    sample_embedding_state.partition = {0: list(range(50)), 1: list(range(50, 100))}
    sample_embedding_state.community_centers = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 1.0]),
    }

    targets = np.ones((100, 100), dtype=np.float32)
    with pytest.raises(ValueError):
        processing.compute_modified_positions(
            sample_embedding_state,
            layout_param=0.5,
            layout_method="INVALID",
            targets=targets,
            inplace=False,
            verbose=False,
        )


def test_run_pipeline_smoke(sample_data):
    embeddings = processing.run_pipeline(
        sample_data,
        sim_features=["f1", "f2"],
        dr_method="UMAP",
        graph_method="DR",
        layout_method="FR",
        compute_metrics=True,
        verbose=False,
    )
    assert isinstance(embeddings, list)
    assert all(isinstance(e, EmbeddingState) for e in embeddings)

    for emb in embeddings[1:]:
        assert emb.embedding is not None
        assert emb.graph is not None
        assert emb.partition is not None
        assert emb.metadata is not None
        assert emb.metadata["dr_method"] == "UMAP"
        assert emb.metadata["layout_method"] == "FR"
        assert emb.metrics is not None
