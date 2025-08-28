import numpy as np
import pandas as pd
import pytest

from moddr import EmbeddingState, evaluation


def test_compute_kruskal_stress_basic(monkeypatch):
    # Monkeypatch processing.compute_distance_scaling to return 1.0
    monkeypatch.setattr(
        evaluation.processing, "compute_distance_scaling", lambda a, b: 1.0
    )
    dists_highdim = np.array([[0, 1], [1, 0]], dtype=np.float32)
    dists_lowdim = np.array([[0, 2], [2, 0]], dtype=np.float32)
    result = evaluation.compute_kruskal_stress(dists_highdim, dists_lowdim)
    # Expected: sqrt(((1-2)^2 + (1-2)^2) / (1^2 + 1^2)) = sqrt(2/2) = 1.0
    assert np.isclose(result, 1.0)


def test_compute_kruskal_stress_shape_mismatch():
    arr1 = np.ones((3, 3), dtype=np.float32)
    arr2 = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        evaluation.compute_kruskal_stress(arr1, arr2)


def test_compute_kruskal_stress_partition_shape_mismatch(monkeypatch):
    monkeypatch.setattr(evaluation, "compute_kruskal_stress", lambda a, b: 1.0)
    arr = np.ones((4, 4), dtype=np.float32)
    partition = {0: np.array([0, 1]), 1: np.array([2, 3])}
    with pytest.raises(ValueError):
        evaluation.compute_kruskal_stress_partition(arr, arr, partition)


def test_compute_coranking_matrix_shape_mismatch():
    r1 = np.zeros((3, 3), dtype=int)
    r2 = np.zeros((4, 4), dtype=int)
    with pytest.raises(ValueError):
        evaluation.compute_coranking_matrix(r1, r2)


def test_compute_coranking_matrix_basic():
    r1 = np.array([[0, 1], [1, 0]], dtype=int)
    r2 = np.array([[1, 0], [0, 1]], dtype=int)
    crm = evaluation.compute_coranking_matrix(r1, r2)
    assert crm.shape == (2, 2)
    assert np.sum(crm) == 4


def test_compute_trustworthiness_invalid_k():
    cr_matrix = np.ones((5, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        evaluation.compute_trustworthiness(cr_matrix, 0)
    with pytest.raises(ValueError):
        evaluation.compute_trustworthiness(cr_matrix, 5)


def test_compute_trustworthiness_basic():
    cr_matrix = np.eye(4, dtype=np.float32)
    result = evaluation.compute_trustworthiness(cr_matrix, 1)
    assert 0 <= result <= 1


def test_compute_continuity_invalid_k():
    cr_matrix = np.ones((5, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        evaluation.compute_continuity(cr_matrix, 0)
    with pytest.raises(ValueError):
        evaluation.compute_continuity(cr_matrix, 5)


def test_compute_continuity_basic():
    cr_matrix = np.eye(4, dtype=np.float32)
    result = evaluation.compute_continuity(cr_matrix, 1)
    assert 0 <= result <= 1


def test_compute_rnx_invalid_k():
    qnn = np.ones(5, dtype=np.float32)
    with pytest.raises(ValueError):
        evaluation.compute_rnx(qnn, 0)
    with pytest.raises(ValueError):
        evaluation.compute_rnx(qnn, 5)


def test_compute_rnx_basic():
    qnn = np.ones(4, dtype=np.float32)
    result = evaluation.compute_rnx(qnn, 1)
    assert isinstance(result, float)


def test_compute_rank_score_missing_metrics():
    emb = EmbeddingState({})
    with pytest.raises(ValueError):
        evaluation.compute_rank_score(emb)


def test_compute_rank_score_basic():
    emb = EmbeddingState({})
    emb.metrics = {"trustworthiness": 0.8, "continuity": 0.7, "rnx": 0.9}
    score = evaluation.compute_rank_score(emb)
    assert np.isclose(score, (0.8 + 0.7 + 0.9) / 3)


def test_compute_dist_score_missing_metrics():
    emb = EmbeddingState({})
    with pytest.raises(ValueError):
        evaluation.compute_dist_score(emb)


def test_compute_dist_score_basic():
    emb = EmbeddingState({})
    emb.metrics = {"sim_stress": 0.2, "sim_stress_com_diff": -0.5}
    score = evaluation.compute_dist_score(emb)
    sim_stress_com_diff_norm = (-0.5 + 1) / 2
    expected = 1 - ((0.2 + sim_stress_com_diff_norm) / 2)
    assert np.isclose(score, expected)


def test_compute_total_score_missing_metrics():
    emb = EmbeddingState({})
    with pytest.raises(ValueError):
        evaluation.compute_total_score(emb)


def test_compute_total_score_basic():
    emb = EmbeddingState({})
    emb.metrics = {"rank_score": 0.8, "distance_score": 0.6}
    score = evaluation.compute_total_score(emb)
    assert np.isclose(score, 0.7)


def test_compute_total_score_balance():
    emb = EmbeddingState({})
    emb.metrics = {"rank_score": 0.8, "distance_score": 0.6}
    balance = 0.3
    score = evaluation.compute_total_score(emb, balance)
    expected = balance * 0.8 + (1 - balance) * 0.6
    assert np.isclose(score, expected)


def test_create_report_empty():
    df = evaluation.create_report([], metadata=True, metrics=True)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_create_report_metadata_only():
    emb = EmbeddingState({}, obj_id="id1")
    emb.metadata = {"foo": "bar"}
    df = evaluation.create_report([emb], metadata=True, metrics=False)
    assert "obj_id" in df.columns and "foo" in df.columns


def test_create_report_metrics_only():
    emb = EmbeddingState({}, obj_id="id1")
    emb.metrics = {"trustworthiness": 0.8, "coranking_matrix": np.eye(2)}
    df = evaluation.create_report([emb], metadata=False, metrics=True)
    assert "obj_id" in df.columns and "trustworthiness" in df.columns
    assert "coranking_matrix" not in df.columns


def test_create_report_both():
    emb = EmbeddingState({}, obj_id="id1")
    emb.metadata = {"foo": "bar"}
    emb.metrics = {"trustworthiness": 0.8, "coranking_matrix": np.eye(2)}
    df = evaluation.create_report([emb], metadata=True, metrics=True)
    assert (
        "obj_id" in df.columns
        and "foo" in df.columns
        and "trustworthiness" in df.columns
    )
    assert "coranking_matrix" not in df.columns


def test_create_report_invalid_args():
    emb = EmbeddingState({}, obj_id="id1")
    with pytest.raises(ValueError):
        evaluation.create_report([emb], metadata=False, metrics=False)
