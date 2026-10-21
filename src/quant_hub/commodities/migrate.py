"""One-off move of commodity objects out of the crypto bucket (reproducible, idempotent).

    uv run python -m quant_hub.commodities.migrate [--dry-run] [--keep-source]

Moves, key for key, `raw/{ohlcv_1d,funding}/exchange=hyperliquid_hip3/...` from
`yolo-research-ep` to `yolo-commodities-ep`, and `external/tcrc/...` to
`reference/tcrc/...`. Each object is copied server-side with its metadata (the
SHA-256 the sync layer compares), verified by size and hash on the destination,
and only then deleted from the source. Local copies under `data/raw` and
`data/external` are moved into `data/commodities/` the same way.

Why: the crypto `quant_hub.data.health` saw the dead `flx` markets as stale and went
red for everyone; the commodities domain owns them now (note 028 in yolo-microstructure).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from quant_hub.commodities.storage import BUCKET, COMMODITIES_ROOT
from quant_hub.data.storage import DATA_ROOT
from quant_hub.data.sync import _client, _remote_index, _sha256

SOURCE_BUCKET = "yolo-research-ep"
# source prefix -> destination prefix (keys keep their tail)
MOVES = {
    "raw/ohlcv_1d/exchange=hyperliquid_hip3/": "raw/ohlcv_1d/exchange=hyperliquid_hip3/",
    "raw/funding/exchange=hyperliquid_hip3/": "raw/funding/exchange=hyperliquid_hip3/",
    "external/tcrc/": "reference/tcrc/",
}


def destination_key(key: str) -> str | None:
    """Destination key of a source key, or None when it is not part of the move."""
    for src, dst in MOVES.items():
        if key.startswith(src):
            return dst + key[len(src) :]
    return None


def move_objects(
    s3, src_bucket: str = SOURCE_BUCKET, dst_bucket: str = BUCKET, dry_run=False, keep_source=False
) -> list[str]:
    """Copy every matching object, verify it, delete the source copy. Returns moved keys."""
    moved = []
    for key, meta in sorted(_remote_index(s3, src_bucket, None).items()):
        dst = destination_key(key)
        if dst is None:
            continue
        if dry_run:
            print(f"would move s3://{src_bucket}/{key} -> s3://{dst_bucket}/{dst}")
            moved.append(key)
            continue
        s3.copy_object(
            Bucket=dst_bucket,
            Key=dst,
            CopySource={"Bucket": src_bucket, "Key": key},
            MetadataDirective="COPY",
        )
        src_head = s3.head_object(Bucket=src_bucket, Key=key)
        dst_head = s3.head_object(Bucket=dst_bucket, Key=dst)
        same_size = src_head.get("ContentLength", meta.size) == dst_head.get("ContentLength")
        same_sha = src_head.get("Metadata", {}).get("sha256") == dst_head.get("Metadata", {}).get(
            "sha256"
        )
        if not (same_size and same_sha):
            raise RuntimeError(f"verification failed for {key} -> {dst}; source kept")
        if not keep_source:
            s3.delete_object(Bucket=src_bucket, Key=key)
        print(f"moved {key} -> {dst}")
        moved.append(key)
    return moved


def move_local(
    root: Path = DATA_ROOT, dest_root: Path = COMMODITIES_ROOT, dry_run=False
) -> list[str]:
    """Move local files of the same prefixes from the crypto layer to the commodities one."""
    moved = []
    for src_prefix, dst_prefix in MOVES.items():
        base = root / src_prefix
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(base).as_posix()
            dest = dest_root / dst_prefix / rel
            if dry_run:
                print(f"would move {p} -> {dest}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() and _sha256(dest) != _sha256(p):
                    raise RuntimeError(f"{dest} exists with different content; not overwritten")
                shutil.move(str(p), str(dest))
            moved.append(rel)
        if not dry_run:
            for d in sorted((x for x in base.rglob("*") if x.is_dir()), reverse=True):
                if not any(d.iterdir()):
                    d.rmdir()
            if base.exists() and not any(base.iterdir()):
                base.rmdir()
    return moved


def main(argv=None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-bucket", default=SOURCE_BUCKET)
    ap.add_argument("--bucket", default=BUCKET, help="destination (commodities) bucket")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-source", action="store_true", help="copy and verify, do not delete")
    args = ap.parse_args(argv)
    keys = move_objects(
        _client(),
        args.source_bucket,
        args.bucket,
        dry_run=args.dry_run,
        keep_source=args.keep_source,
    )
    local = move_local(dry_run=args.dry_run)
    verb = "would move" if args.dry_run else "moved"
    print(f"{verb}: {len(keys)} objects, {len(local)} local files")


if __name__ == "__main__":
    main()
