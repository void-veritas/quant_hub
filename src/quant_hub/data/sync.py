"""Mirror the raw data layer to/from an S3 bucket (boto3, no aws CLI needed).

    uv run python -m quant_hub.data.sync push --bucket yolo-research-ep
    uv run python -m quant_hub.data.sync pull --bucket yolo-research-ep [--dataset funding]

Keys mirror the local layout: raw/<dataset>/exchange=.../asset=.../year=.../data.parquet
plus external/ for third-party files. A file is copied only when its size differs
or it is missing on the destination, so repeated runs are cheap.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3

from quant_hub.data.storage import DATA_ROOT

PREFIXES = ("raw", "external")


def _local_files(root: Path, dataset: str | None):
    for prefix in PREFIXES:
        base = root / prefix
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and not p.name.startswith(".") and not p.suffix == ".tmp":
                rel = p.relative_to(root).as_posix()
                if dataset and not rel.startswith(f"raw/{dataset}/"):
                    continue
                yield rel, p


def _remote_index(s3, bucket: str, dataset: str | None) -> dict[str, int]:
    idx: dict[str, int] = {}
    prefix = f"raw/{dataset}/" if dataset else ""
    token = None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            idx[o["Key"]] = o["Size"]
        if not r.get("IsTruncated"):
            return idx
        token = r["NextContinuationToken"]


def push(bucket: str, dataset: str | None = None, workers: int = 8, root: Path = DATA_ROOT) -> int:
    s3 = boto3.client("s3")
    remote = _remote_index(s3, bucket, dataset)
    todo = [
        (rel, p) for rel, p in _local_files(root, dataset) if remote.get(rel) != p.stat().st_size
    ]

    def _up(item):
        rel, p = item
        s3.upload_file(str(p), bucket, rel)
        return p.stat().st_size

    with ThreadPoolExecutor(workers) as pool:
        total = sum(pool.map(_up, todo))
    print(f"push: {len(todo)} files, {total / 1e6:.1f} MB")
    return len(todo)


def pull(bucket: str, dataset: str | None = None, workers: int = 8, root: Path = DATA_ROOT) -> int:
    s3 = boto3.client("s3")
    remote = _remote_index(s3, bucket, dataset)
    local = {rel: p.stat().st_size for rel, p in _local_files(root, None)}
    todo = [k for k, size in remote.items() if local.get(k) != size]

    def _down(key):
        dest = root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dest))
        return remote[key]

    with ThreadPoolExecutor(workers) as pool:
        total = sum(pool.map(_down, todo))
    print(f"pull: {len(todo)} files, {total / 1e6:.1f} MB")
    return len(todo)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--dataset", help="only raw/<dataset>/")
    args = ap.parse_args()
    (push if args.action == "push" else pull)(args.bucket, args.dataset)


if __name__ == "__main__":
    main()
