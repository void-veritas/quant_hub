"""File store shared by the layers: a local directory, optionally mirrored to S3.

Keys are POSIX paths (``weights/<book>/<date>.json``). ``put`` writes locally first
(atomic rename) and then uploads; ``get`` prefers the bucket when configured so that
a freshly restarted executor sees what analytics wrote from another machine.
Each container is expected to hold credentials scoped to the prefixes it needs.
"""

from __future__ import annotations

import os
from pathlib import Path


class Store:
    """Local directory with optional S3 mirror (boto3 is imported lazily)."""

    def __init__(self, root: str | Path, bucket: str | None = None, prefix: str = ""):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._s3 = None

    def _client(self):
        if self._s3 is None:
            import boto3

            self._s3 = boto3.client("s3")
        return self._s3

    def _s3_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def path(self, key: str) -> Path:
        return self.root / key

    def put(self, key: str, data: bytes | str) -> Path:
        """Write atomically to the local root, then mirror to the bucket."""
        if isinstance(data, str):
            data = data.encode()
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        if self.bucket:
            self._client().put_object(Bucket=self.bucket, Key=self._s3_key(key), Body=data)
        return p

    def get(self, key: str) -> bytes | None:
        """Read from the bucket if configured (refreshing the local copy), else locally."""
        if self.bucket:
            try:
                body = (
                    self._client()
                    .get_object(Bucket=self.bucket, Key=self._s3_key(key))["Body"]
                    .read()
                )
            except self._client().exceptions.NoSuchKey:
                body = None
            if body is not None:
                p = self.path(key)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(body)
                return body
        p = self.path(key)
        return p.read_bytes() if p.exists() else None

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def delete(self, key: str) -> None:
        p = self.path(key)
        if p.exists():
            p.unlink()
        if self.bucket:
            self._client().delete_object(Bucket=self.bucket, Key=self._s3_key(key))

    def list(self, prefix: str) -> list[str]:
        """Keys under a prefix (bucket when configured, else local)."""
        if self.bucket:
            keys: list[str] = []
            token = None
            while True:
                kw = dict(Bucket=self.bucket, Prefix=self._s3_key(prefix))
                if token:
                    kw["ContinuationToken"] = token
                r = self._client().list_objects_v2(**kw)
                for o in r.get("Contents", []):
                    k = o["Key"]
                    keys.append(k[len(self.prefix) + 1 :] if self.prefix else k)
                if not r.get("IsTruncated"):
                    return sorted(keys)
                token = r["NextContinuationToken"]
        base = self.path(prefix)
        if not base.exists():
            return []
        return sorted(
            p.relative_to(self.root).as_posix()
            for p in base.rglob("*")
            if p.is_file() and not p.name.endswith(".tmp")
        )
