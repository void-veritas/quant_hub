r"""Mirror the commodities domain to/from its own bucket.

    uv run python -m quant_hub.commodities.sync push [--bucket yolo-commodities-ep] \\
        [--dataset futures_1d] [--force] [--dry-run]
    uv run python -m quant_hub.commodities.sync pull [--bucket yolo-commodities-ep] \\
        [--dataset futures_1d] [--force] [--dry-run] [--merge]

Same engine and rules as `quant_hub.data.sync` (content hashes, newer local files
protected, `--merge` unions a partition); only the root (`data/commodities/`), the
layers (raw / reference / curated) and the default bucket differ. The crypto bucket
never sees commodity files and vice versa.
"""

from __future__ import annotations

import argparse

from quant_hub.commodities.storage import BUCKET, COMMODITIES_ROOT, LAYERS
from quant_hub.data import sync as _sync


def push(bucket: str = BUCKET, dataset: str | None = None, **kw) -> int:
    """Upload commodities files missing in the bucket or whose content differs."""
    return _sync.push(bucket, dataset, root=COMMODITIES_ROOT, prefixes=LAYERS, **kw)


def pull(bucket: str = BUCKET, dataset: str | None = None, **kw) -> int:
    """Download commodities files missing locally or whose content differs."""
    return _sync.pull(bucket, dataset, root=COMMODITIES_ROOT, prefixes=LAYERS, **kw)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--dataset", help="only raw/<dataset>/")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--merge", action="store_true", help="pull: merge conflicting partitions")
    args = ap.parse_args()
    if args.action == "push":
        push(args.bucket, args.dataset, force=args.force, dry_run=args.dry_run)
    else:
        pull(args.bucket, args.dataset, force=args.force, dry_run=args.dry_run, merge=args.merge)


if __name__ == "__main__":
    main()
