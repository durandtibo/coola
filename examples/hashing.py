r"""Show examples of the ``coola.hashing`` and ``coola.identifier`` functions.

Run with: ``python examples/hashing.py``
"""

from __future__ import annotations

import logging

from coola.display.colorlog import configure_colorlog_logging
from coola.hashing import hash_object
from coola.identifier import generate_stable_content_id

logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    r"""Run the examples."""
    data = {"a": [1, 2, 3], "b": {"c": "text"}}
    same = {"b": {"c": "text"}, "a": [1, 2, 3]}
    other = {"a": [1, 2, 4], "b": {"c": "text"}}
    logger.info("same hash for the same content: %s", hash_object(data) == hash_object(same))
    logger.info("different hash for other content: %s", hash_object(data) != hash_object(other))
    logger.info(
        "content id has 8 characters: %s", len(generate_stable_content_id(data, length=8)) == 8
    )


if __name__ == "__main__":
    configure_colorlog_logging(level=logging.INFO)
    main()
