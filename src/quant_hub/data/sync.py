r"""Mirror the raw data layer to/from an S3 bucket (boto3, no aws CLI needed).

    uv run python -m quant_hub.data.sync push --bucket yolo-research-ep [--dataset funding] \\
        [--force] [--dry-run]
    uv run python -m quant_hub.data.sync pull --bucket yolo-research-ep [--dataset funding] \\
        [--force] [--dry-run]

Keys mirror the local layout: raw/<dataset>/exchange=.../asset=.../year=.../data.parquet
plus external/ for third-party files. Files are compared by content: every upload
stores the file's SHA-256 in the object's metadata, and a file is copied only when it
is missing on the destination or its hash differs. A local file that differs from the
bucket and is newer than the bucket's copy is never overwritten by ``pull`` unless
``--force`` is given (it is reported instead); ``pull --merge`` instead merges such a
partition row by row (union on ``ts``, local rows win), which is the right call when
two machines backfilled the same partition. ``push --force`` re-uploads everything.
Credentials come from the environment or ``.env`` (AWS_ACCESS_KEY_ID / SECRET).
"""

from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from quant_hub.data.storage import DATA_ROOT
from quant_hub.utils.config import load_env

PREFIXES = ("raw", "external")
SKIP_SUFFIXES = (".tmp", ".lock")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _local_files(root: Path, dataset: str | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for prefix in PREFIXES:
        base = root / prefix
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and not p.name.startswith(".") and p.suffix not in SKIP_SUFFIXES:
                rel = p.relative_to(root).as_posix()
                if dataset and not rel.startswith(f"raw/{dataset}/"):
                    continue
                out[rel] = p
    return out


@dataclass
class _Remote:
    size: int
    modified: datetime
    sha: str | None = None  # filled lazily from object metadata


def _remote_index(s3, bucket: str, dataset: str | None) -> dict[str, _Remote]:
    idx: dict[str, _Remote] = {}
    prefix = f"raw/{dataset}/" if dataset else ""
    token = None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            idx[o["Key"]] = _Remote(o["Size"], o["LastModified"])
        if not r.get("IsTruncated"):
            return idx
        token = r["NextContinuationToken"]


def _remote_sha(s3, bucket: str, key: str) -> str | None:
    meta = s3.head_object(Bucket=bucket, Key=key).get("Metadata", {})
    return meta.get("sha256")


def _client():
    import boto3

    load_env()
    return boto3.client("s3")


def push(
    bucket: str,
    dataset: str | None = None,
    workers: int = 8,
    root: Path = DATA_ROOT,
    force: bool = False,
    dry_run: bool = False,
    s3=None,
) -> int:
    """Upload local files that are missing in the bucket or whose content differs."""
    s3 = s3 or _client()
    remote = _remote_index(s3, bucket, dataset)
    local = _local_files(root, dataset)

    def _decide(item):
        rel, p = item
        r = remote.get(rel)
        sha = _sha256(p)
        if force or r is None or r.size != p.stat().st_size:
            return rel, p, sha
        if _remote_sha(s3, bucket, rel) != sha:  # same size: compare content
            return rel, p, sha
        return None

    with ThreadPoolExecutor(workers) as pool:
        todo = [t for t in pool.map(_decide, local.items()) if t]
    total = sum(p.stat().st_size for _, p, _ in todo)
    if dry_run:
        for rel, _, _ in todo:
            print("would push", rel)
        print(f"push (dry run): {len(todo)} files, {total / 1e6:.1f} MB")
        return len(todo)

    def _up(item):
        rel, p, sha = item
        with open(p, "rb") as fh:
            s3.put_object(Bucket=bucket, Key=rel, Body=fh, Metadata={"sha256": sha})
        return p.stat().st_size

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(_up, todo))
    print(f"push: {len(todo)} files, {total / 1e6:.1f} MB")
    return len(todo)


def pull(
    bucket: str,
    dataset: str | None = None,
    workers: int = 8,
    root: Path = DATA_ROOT,
    force: bool = False,
    dry_run: bool = False,
    s3=None,
    merge: bool = False,
) -> int:
    """Download bucket files that are missing locally or whose content differs.

    A local file that differs AND is newer than the bucket copy is left alone and
    listed, unless `force` is set: this protects a fresh local backfill from being
    overwritten by an older mirror. With `merge`, such a raw parquet partition is
    instead merged row by row with the bucket copy (union on `ts`, local rows win on
    a duplicate timestamp); a following `push` then uploads the union.
    """
    s3 = s3 or _client()
    remote = _remote_index(s3, bucket, dataset)
    local = _local_files(root, None)

    def _decide(key):
        p = local.get(key)
        r = remote[key]
        if p is None:
            return key, "download"
        if p.stat().st_size == r.size:
            rsha = _remote_sha(s3, bucket, key)
            if rsha is None or rsha == _sha256(p):
                return None  # same content (or legacy object without a hash: assume same)
        local_newer = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC) > r.modified
        if local_newer and not force:
            if merge and key.startswith("raw/") and key.endswith(".parquet"):
                return key, "merge"
            return key, "skip-local-newer"
        return key, "download"

    with ThreadPoolExecutor(workers) as pool:
        decisions = [d for d in pool.map(_decide, remote) if d]
    skipped = [k for k, a in decisions if a == "skip-local-newer"]
    todo = [k for k, a in decisions if a == "download"]
    merges = [k for k, a in decisions if a == "merge"]
    for k in skipped:
        print(f"kept local (newer than bucket, differs): {k}  [pull --merge or --force]")
    total = sum(remote[k].size for k in todo + merges)
    if dry_run:
        for k in todo:
            print("would pull", k)
        for k in merges:
            print("would merge", k)
        print(
            f"pull (dry run): {len(todo)} files, {total / 1e6:.1f} MB, "
            f"{len(merges)} merged, {len(skipped)} kept"
        )
        return len(todo) + len(merges)

    def _down(key):
        dest = root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        s3.download_file(bucket, key, str(tmp))
        tmp.replace(dest)

    def _merge(key):
        dest = root / key
        tmp = dest.with_suffix(dest.suffix + ".remote.tmp")
        s3.download_file(bucket, key, str(tmp))
        _merge_partition(dest, tmp)
        tmp.unlink(missing_ok=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(_down, todo))
        list(pool.map(_merge, merges))
    print(
        f"pull: {len(todo)} files, {total / 1e6:.1f} MB, {len(merges)} merged, "
        f"{len(skipped)} kept local"
    )
    return len(todo) + len(merges)


def _merge_partition(local: Path, remote: Path) -> None:
    """Union of two copies of one partition on `ts`; local rows win on duplicates."""
    import pandas as pd

    a = pd.read_parquet(local)
    b = pd.read_parquet(remote)
    for df in (a, b):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    merged = (
        pd.concat([a, b], ignore_index=True)
        .drop_duplicates(subset="ts", keep="first")
        .sort_values("ts")
    )
    tmp = local.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    tmp.replace(local)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--dataset", help="only raw/<dataset>/")
    ap.add_argument(
        "--force",
        action="store_true",
        help="push: re-upload all; pull: overwrite newer local files",
    )
    ap.add_argument("--dry-run", action="store_true", help="list what would be copied")
    ap.add_argument(
        "--merge",
        action="store_true",
        help="pull: merge a conflicting raw partition row by row instead of keeping local",
    )
    args = ap.parse_args()
    if args.action == "push":
        push(args.bucket, args.dataset, force=args.force, dry_run=args.dry_run)
    else:
        pull(args.bucket, args.dataset, force=args.force, dry_run=args.dry_run, merge=args.merge)


if __name__ == "__main__":
    main()
