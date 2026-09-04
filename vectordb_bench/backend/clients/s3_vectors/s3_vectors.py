"""Wrapper around Amazon S3 Vectors."""

import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any

import boto3

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import S3VectorsIndexConfig

log = logging.getLogger(__name__)

_MAX_INDEX_NAME_LENGTH = 63
_MAX_SHARD_WORKERS = 32


class S3Vectors(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: S3VectorsIndexConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the s3-vectors client."""
        self.db_config = db_config
        self.case_config = db_case_config
        self.with_scalar_labels = with_scalar_labels

        self.batch_size = 500

        self._scalar_id_field = "id"
        self._scalar_label_field = "label"
        self._vector_field = "vector"

        self.region_name = self.db_config.get("region_name")
        self.access_key_id = self.db_config.get("access_key_id")
        self.secret_access_key = self.db_config.get("secret_access_key")
        self.bucket_name = self.db_config.get("bucket_name")
        self.index_name = self.db_config.get("index_name")
        self.num_shards = int(self.db_config.get("num_shards", 1))
        self.index_names = self._build_index_names(self.index_name, self.num_shards)

        client = boto3.client(
            service_name="s3vectors",
            region_name=self.region_name,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
        )

        if drop_old:
            existing_index_names = self._list_index_names(client)

            for index_name in self.index_names:
                if index_name in existing_index_names:
                    log.info(f"drop old index: {index_name}")
                    client.delete_index(vectorBucketName=self.bucket_name, indexName=index_name)

            for index_name in self.index_names:
                client.create_index(
                    vectorBucketName=self.bucket_name,
                    indexName=index_name,
                    dataType=self.case_config.data_type,
                    dimension=dim,
                    distanceMetric=self.case_config.parse_metric(),
                )

        client.close()

    def _list_index_names(self, client: Any) -> set[str]:
        index_names: set[str] = set()
        request = {"vectorBucketName": self.bucket_name}
        while True:
            response = client.list_indexes(**request)
            index_names.update(index["indexName"] for index in response["indexes"])
            next_token = response.get("nextToken")
            if not next_token:
                return index_names
            request["nextToken"] = next_token

    @staticmethod
    def _build_index_names(index_name: str, num_shards: int) -> list[str]:
        if num_shards < 1:
            msg = "num_shards must be greater than 0"
            raise ValueError(msg)

        if num_shards == 1:
            return [index_name]

        width = max(2, len(str(num_shards - 1)))
        index_names = [f"{index_name}-shard-{shard_id:0{width}d}" for shard_id in range(num_shards)]
        too_long = next((name for name in index_names if len(name) > _MAX_INDEX_NAME_LENGTH), None)
        if too_long is not None:
            msg = (
                f"S3 Vectors index name '{too_long}' exceeds the {_MAX_INDEX_NAME_LENGTH}-character limit; "
                "use a shorter --index value"
            )
            raise ValueError(msg)
        return index_names

    @contextmanager
    def init(self):
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        self.client = boto3.client(
            service_name="s3vectors",
            region_name=self.region_name,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
        )
        self._executor = (
            ThreadPoolExecutor(max_workers=min(self.num_shards, _MAX_SHARD_WORKERS)) if self.num_shards > 1 else None
        )

        try:
            yield
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
            self.client.close()

    def optimize(self, **kwargs):
        return

    def need_normalize_cosine(self) -> bool:
        """Whether this database needs to normalize datasets to support COSINE."""
        return False

    def _make_vector(
        self,
        embedding: list[float],
        vector_id: int,
        label: str | None = None,
    ) -> dict:
        metadata = {self._scalar_id_field: vector_id}
        if self.with_scalar_labels:
            metadata[self._scalar_label_field] = label
        return {
            "key": str(vector_id),
            "data": {self.case_config.data_type: embedding},
            "metadata": metadata,
        }

    def _put_vectors(self, index_name: str, vectors: list[dict]) -> None:
        self.client.put_vectors(
            vectorBucketName=self.bucket_name,
            indexName=index_name,
            vectors=vectors,
        )

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into S3 Vectors. Call self.init() first."""
        assert self.client is not None
        assert len(embeddings) == len(metadata)

        if self.num_shards == 1:
            return self._insert_single_index(embeddings, metadata, labels_data)

        assert self._executor is not None
        shard_vectors: list[list[dict]] = [[] for _ in range(self.num_shards)]
        for i in range(len(embeddings)):
            # Stripe 500-vector key blocks so sequential benchmark batches stay full-sized PutVectors requests.
            shard_id = (metadata[i] // self.batch_size) % self.num_shards
            label = labels_data[i] if labels_data is not None else None
            shard_vectors[shard_id].append(self._make_vector(embeddings[i], metadata[i], label))

        requests: list[tuple[str, list[dict]]] = []
        for shard_id, vectors in enumerate(shard_vectors):
            for batch_start_offset in range(0, len(vectors), self.batch_size):
                requests.append(
                    (
                        self.index_names[shard_id],
                        vectors[batch_start_offset : batch_start_offset + self.batch_size],
                    )
                )

        futures = {
            self._executor.submit(self._put_vectors, index_name, vectors): len(vectors)
            for index_name, vectors in requests
        }
        insert_count = 0
        first_error = None
        for future in as_completed(futures):
            try:
                future.result()
                insert_count += futures[future]
            except Exception as e:
                if first_error is None:
                    first_error = e

        if first_error is not None:
            log.info(f"Failed to insert data: {first_error}")
            return insert_count, first_error
        return insert_count, None

    def _insert_single_index(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None,
    ) -> tuple[int, Exception]:
        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
                insert_data = [
                    self._make_vector(
                        embeddings[i],
                        metadata[i],
                        labels_data[i] if labels_data is not None else None,
                    )
                    for i in range(batch_start_offset, batch_end_offset)
                ]
                self._put_vectors(self.index_names[0], insert_data)
                insert_count += len(insert_data)
        except Exception as e:
            log.info(f"Failed to insert data: {e}")
            return insert_count, e
        return insert_count, None

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.filter = None
        elif filters.type == FilterOp.NumGE:
            self.filter = {self._scalar_id_field: {"$gte": filters.int_value}}
        elif filters.type == FilterOp.StrEqual:
            self.filter = {self._scalar_label_field: filters.label_value}
        else:
            msg = f"Not support Filter for S3Vectors - {filters}"
            raise ValueError(msg)

    def _query_index(
        self,
        index_name: str,
        query: list[float],
        k: int,
        return_distance: bool,
    ) -> list[dict]:
        query_args = {
            "vectorBucketName": self.bucket_name,
            "indexName": index_name,
            "queryVector": {"float32": query},
            "topK": k,
            "returnDistance": return_distance,
            "returnMetadata": False,
        }
        if self.filter is not None:
            query_args["filter"] = self.filter

        results = []
        while True:
            res = self.client.query_vectors(**query_args)
            results.extend(res["vectors"])
            if len(results) >= k:
                return results[:k]
            next_token = res.get("nextToken")
            if not next_token:
                return results
            query_args["nextToken"] = next_token

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
    ) -> list[int]:
        """Perform a search on a query embedding and return results."""
        assert self.client is not None

        if self.num_shards == 1:
            results = self._query_index(self.index_names[0], query, k, return_distance=False)
            return [int(result["key"]) for result in results]

        assert self._executor is not None
        futures = [
            self._executor.submit(self._query_index, index_name, query, k, True) for index_name in self.index_names
        ]
        candidates = [result for future in futures for result in future.result()]
        candidates.sort(key=lambda result: result["distance"])
        return [int(result["key"]) for result in candidates[:k]]
