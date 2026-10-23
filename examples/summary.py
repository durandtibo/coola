r"""Show examples of the ``coola.summary`` and ``coola.reducer``
functions.

Run with: ``python examples/summary.py``
"""

from __future__ import annotations

import logging

from coola.display.colorlog import configure_colorlog_logging
from coola.reducer import NativeReducer
from coola.summary import summarize

logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    r"""Run the examples."""
    logger.info("%s", summarize({"a": 1, "b": [1, 2, 3]}, max_depth=2))
    reducer = NativeReducer()
    logger.info("median: %s", reducer.median([3, 1, 2]))
    logger.info("max: %s", reducer.max([3, 1, 2]))


if __name__ == "__main__":
    configure_colorlog_logging(level=logging.INFO)
    main()
