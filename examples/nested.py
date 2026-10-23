r"""Show examples of the ``coola.nested`` functions.

Run with: ``python examples/nested.py``
"""

from __future__ import annotations

import logging

from coola.display.colorlog import configure_colorlog_logging
from coola.nested import from_flat_dict, merge_mappings, to_flat_dict

logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    r"""Run the examples."""
    data = {"model": {"layers": [{"units": 8}, {"units": 4}]}, "lr": 0.1}
    flat = to_flat_dict(data)
    logger.info("flat keys: %s", sorted(flat))
    logger.info("round trip: %s", from_flat_dict(flat, restore_lists=True) == data)
    logger.info("merged: %s", merge_mappings([{"a": 1}, {"b": 2}, {"a": 1}]))


if __name__ == "__main__":
    configure_colorlog_logging(level=logging.INFO)
    main()
