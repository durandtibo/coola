r"""Show examples of the ``coola.equality`` functions and their messages.

Run with: ``python debug/equality_examples.py``
"""

# ruff: noqa: D103
from __future__ import annotations

import logging

from coola.display.colorlog import configure_colorlog_logging
from coola.equality import (
    assert_objects_allclose,
    assert_objects_equal,
    compare,
    objects_are_allclose,
    objects_are_equal,
)
from coola.utils.imports import is_numpy_available, is_torch_available

if is_numpy_available():
    import numpy as np
if is_torch_available():
    import torch

logger: logging.Logger = logging.getLogger(__name__)


def section(title: str) -> None:
    logger.info(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def cases() -> list[tuple[str, object, object]]:
    data = [
        ("equal nested", {"a": [1, {"b": 2}]}, {"a": [1, {"b": 2}]}),
        ("different int at root", 1, 2),
        ("different types", [1, 2], (1, 2)),
        ("different list value", [1, 2, 3], [1, 2, 4]),
        ("different list length", [1, 2, 3], [1, 2]),
        ("different dict keys", {"a": 1, "b": 2}, {"a": 1, "c": 3}),
        ("different nested value", {"a": [1, {"b": 2}]}, {"a": [1, {"b": 3}]}),
        ("different string", {"name": "alice"}, {"name": "bob"}),
        ("float within tolerance only", [1.0, 2.0], [1.0, 2.0 + 1e-3]),
        ("nan", [float("nan")], [float("nan")]),
    ]
    if is_numpy_available():
        data += [
            ("numpy different values", {"x": np.array([1, 2, 3])}, {"x": np.array([1, 2, 4])}),
            ("numpy different shape", np.zeros((2, 3)), np.zeros((3, 2))),
            ("numpy different dtype", np.zeros(3, dtype=int), np.zeros(3, dtype=float)),
        ]
    if is_torch_available():
        data += [
            ("torch different values", [torch.ones(2)], [torch.zeros(2)]),
            ("torch different dtype", torch.ones(2), torch.ones(2, dtype=torch.float64)),
        ]
    return data


def main() -> None:
    section("objects_are_equal(..., show_difference=True)  -- messages are logged")
    for name, actual, expected in cases():
        logger.info(f"\n# {name}")
        logger.info(f"result: {objects_are_equal(actual, expected, show_difference=True)}")

    section("objects_are_allclose(..., show_difference=True)  -- atol=1e-5")
    for name, actual, expected in cases():
        logger.info(f"\n# {name}")
        logger.info(
            f"result: {objects_are_allclose(actual, expected, atol=1e-5, show_difference=True)}"
        )

    section("compare(...)  -- structured result (nothing is logged)")
    for name, actual, expected in cases():
        result = compare(actual, expected)
        logger.info(f"\n# {name}")
        logger.info(f"equal  : {bool(result)}")
        logger.info(f"path   : {result.path!r}  ->  {result.format_path('data')!r}")
        logger.info(f"reason : {result.reason}")

    section("assert_objects_equal(...)  -- AssertionError messages")
    for name, actual, expected in cases():
        logger.info(f"\n# {name}")
        try:
            assert_objects_equal(actual, expected, root="data")
            logger.info("OK")
        except AssertionError as exc:
            logger.info(f"AssertionError: {exc}")

    section("assert_objects_allclose(...)  -- AssertionError messages (atol=1e-2)")
    for name, actual, expected in cases():
        logger.info(f"\n# {name}")
        try:
            assert_objects_allclose(actual, expected, atol=1e-2, root="data")
            logger.info("OK")
        except AssertionError as exc:
            logger.info(f"AssertionError: {exc}")


if __name__ == "__main__":
    configure_colorlog_logging(level=logging.INFO)
    main()
