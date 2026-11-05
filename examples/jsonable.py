r"""Show examples of the ``coola.jsonable functions``.

Run with: ``python examples/jsonable.py``
"""

from __future__ import annotations

import logging

from coola.display.colorlog import configure_colorlog_logging
from coola.jsonable import to_jsonable

logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    r"""Run the examples."""
    data = {"tags": {"b", "a"}, "point": (1, 2), "name": "coola"}
    logger.info("jsonable: %s", to_jsonable(data))


if __name__ == "__main__":
    configure_colorlog_logging(level=logging.INFO)
    main()
