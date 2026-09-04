import sys
import types
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:
    import boto3  # noqa: F401
except ImportError:
    _boto3 = types.ModuleType("boto3")
    _boto3.client = MagicMock()
    sys.modules["boto3"] = _boto3

from pydantic import SecretStr, ValidationError

from vectordb_bench.backend.clients.s3_vectors.config import S3VectorsConfig
from vectordb_bench.backend.clients.s3_vectors.s3_vectors import S3Vectors


class FakeClient:
    def __init__(self):
        self.put_calls = []
        self.query_calls = []
        self.query_results = {}

    def put_vectors(self, **kwargs):
        self.put_calls.append(kwargs)

    def query_vectors(self, **kwargs):
        self.query_calls.append(kwargs)
        return {"vectors": self.query_results[kwargs["indexName"]]}


def _client(num_shards: int = 2) -> S3Vectors:
    db = object.__new__(S3Vectors)
    db.num_shards = num_shards
    db.index_name = "bench"
    db.index_names = S3Vectors._build_index_names(db.index_name, num_shards)
    db.bucket_name = "bucket"
    db.case_config = SimpleNamespace(data_type="float32")
    db.batch_size = 500
    db.with_scalar_labels = False
    db._scalar_id_field = "id"
    db._scalar_label_field = "label"
    db.client = FakeClient()
    db.filter = None
    db._executor = ThreadPoolExecutor(max_workers=num_shards) if num_shards > 1 else None
    return db


def _close(db: S3Vectors) -> None:
    if db._executor is not None:
        db._executor.shutdown(wait=True)


def test_config_defaults_to_single_shard_and_serializes_num_shards():
    cfg = S3VectorsConfig(
        access_key_id=SecretStr("key"),
        secret_access_key=SecretStr("secret"),
        bucket_name="bucket",
    )
    assert cfg.num_shards == 1
    assert cfg.to_dict()["num_shards"] == 1


@pytest.mark.parametrize("num_shards", [0, 10001])
def test_config_rejects_invalid_num_shards(num_shards: int):
    with pytest.raises(ValidationError):
        S3VectorsConfig(
            access_key_id=SecretStr("key"),
            secret_access_key=SecretStr("secret"),
            bucket_name="bucket",
            num_shards=num_shards,
        )


def test_index_names_preserve_single_index_and_suffix_shards():
    assert S3Vectors._build_index_names("bench", 1) == ["bench"]
    assert S3Vectors._build_index_names("bench", 3) == [
        "bench-shard-00",
        "bench-shard-01",
        "bench-shard-02",
    ]


def test_insert_routes_full_s3_batches_across_shards():
    db = _client(num_shards=2)
    try:
        inserted, error = db.insert_embeddings(
            embeddings=[[0.0]] * 1000,
            metadata=list(range(1000)),
        )
    finally:
        _close(db)

    assert inserted == 1000
    assert error is None

    calls = {call["indexName"]: call["vectors"] for call in db.client.put_calls}
    assert len(calls["bench-shard-00"]) == 500
    assert len(calls["bench-shard-01"]) == 500
    assert calls["bench-shard-00"][0]["key"] == "0"
    assert calls["bench-shard-00"][-1]["key"] == "499"
    assert calls["bench-shard-01"][0]["key"] == "500"
    assert calls["bench-shard-01"][-1]["key"] == "999"


def test_search_fans_out_and_merges_global_top_k_by_distance():
    db = _client(num_shards=2)
    db.client.query_results = {
        "bench-shard-00": [
            {"key": "10", "distance": 0.10},
            {"key": "11", "distance": 0.30},
        ],
        "bench-shard-01": [
            {"key": "20", "distance": 0.20},
            {"key": "21", "distance": 0.40},
        ],
    }

    try:
        result = db.search_embedding([1.0], k=2)
    finally:
        _close(db)

    assert result == [10, 20]
    assert {call["indexName"] for call in db.client.query_calls} == {
        "bench-shard-00",
        "bench-shard-01",
    }
    assert all(call["topK"] == 2 for call in db.client.query_calls)
    assert all(call["returnDistance"] is True for call in db.client.query_calls)
    assert all("filter" not in call for call in db.client.query_calls)


def test_single_shard_search_preserves_non_distance_query():
    db = _client(num_shards=1)
    db.client.query_results = {"bench": [{"key": "7"}]}

    result = db.search_embedding([1.0], k=1)

    assert result == [7]
    assert db.client.query_calls[0]["indexName"] == "bench"
    assert db.client.query_calls[0]["returnDistance"] is False
