r"""Show examples of the ``coola.rounding functions``.

Run with: ``python examples/rounding.py``
"""

from __future__ import annotations

import logging

from coola.display.colorlog import configure_colorlog_logging
from coola.rounding import round_values

logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    r"""Run the examples."""
    logger.info(
        "rounded: %s", round_values({"a": 2.567, "b": [1.234, 5.678]}, decimals=1, recursive=True)
    )
    logger.info("floored: %s", round_values(2.567, mode="floor", decimals=1))


if __name__ == "__main__":
    configure_colorlog_logging(level=logging.INFO)
    main()
