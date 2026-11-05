r"""Show examples of the ``coola.identifier functions``.

Run with: ``python examples/identifier.py``
"""

from __future__ import annotations

import logging

from coola.display.colorlog import configure_colorlog_logging
from coola.identifier import generate_stable_content_id, generate_ulid

logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    r"""Run the examples."""
    first = generate_stable_content_id({"source": "cats.txt", "page": 1})
    second = generate_stable_content_id({"page": 1, "source": "cats.txt"})
    logger.info("same content id for the same content: %s", first == second)
    logger.info("ulid has 26 characters: %s", len(generate_ulid()) == 26)


if __name__ == "__main__":
    configure_colorlog_logging(level=logging.INFO)
    main()
