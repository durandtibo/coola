r"""Show examples of the ``coola.io functions``.

Run with: ``python examples/io.py``
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from coola.display.colorlog import configure_colorlog_logging
from coola.io import load_json, save_json

logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    r"""Run the examples."""
    data = {"key1": [1, 2, 3], "key2": "abc"}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir).joinpath("data.json")
        save_json(data, path)
        logger.info("round trip: %s", load_json(path) == data)


if __name__ == "__main__":
    configure_colorlog_logging(level=logging.INFO)
    main()
