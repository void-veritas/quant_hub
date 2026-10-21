"""Roots of the commodities domain; partition functions are the crypto layer's."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_hub.data import storage as _storage
from quant_hub.data.storage import DATA_ROOT

COMMODITIES_ROOT = DATA_ROOT / "commodities"
RAW_ROOT = COMMODITIES_ROOT / "raw"
REFERENCE_ROOT = COMMODITIES_ROOT / "reference"
CURATED_ROOT = COMMODITIES_ROOT / "curated"
LAYERS = ("raw", "reference", "curated")
BUCKET = "yolo-commodities-ep"


def write_partition(df: pd.DataFrame, dataset: str, exchange: str, asset: str) -> int:
    """`quant_hub.data.storage.write_partition` on the commodities raw root."""
    return _storage.write_partition(df, dataset, exchange, asset, root=RAW_ROOT)


def list_partitions(dataset: str) -> pd.DataFrame:
    """Inventory of one commodities raw dataset."""
    if not (RAW_ROOT / dataset).exists():
        return pd.DataFrame(columns=["exchange", "asset", "year", "path"])
    return _storage.list_partitions(dataset, root=RAW_ROOT)


def coverage(dataset: str) -> pd.DataFrame:
    """Per exchange/asset first/last timestamp and rows of one commodities raw dataset."""
    return _storage.coverage(dataset, root=RAW_ROOT)


def dataset_glob(dataset: str) -> str:
    """Glob of every partition file of a commodities raw dataset (DuckDB/pyarrow)."""
    return _storage.dataset_glob(dataset, root=RAW_ROOT)


def layer_path(layer: str, *parts: str) -> Path:
    """Path inside one layer (`raw`, `reference`, `curated`), parents created."""
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}; one of {LAYERS}")
    path = COMMODITIES_ROOT.joinpath(layer, *parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
