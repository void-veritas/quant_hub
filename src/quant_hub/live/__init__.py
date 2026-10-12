"""Live trading layer: contracts between analytics and execution, OMS, risk, policy, adapters.

Design (yolo-microstructure docs/STATUS.md, M7): three layers linked only by files in a
store (local directory mirrored to S3). Analytics writes ``weights/<book>/<as_of>.json``;
the executor reads it, trades inside the book's window and writes
``fills/<book>/<as_of>.parquet`` and ``state/<book>/<as_of>.json``. Remote commands
only move the system toward safety (``flags/kill``, ``flags/<book>/pause``).
"""
