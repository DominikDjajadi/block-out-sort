"""CLI entry point for persisted held-out evaluation split manifests."""

from __future__ import annotations

from .cotraining.eval_split import main


if __name__ == "__main__":
    raise SystemExit(main())
