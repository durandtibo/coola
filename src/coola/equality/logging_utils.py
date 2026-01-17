r"""Implement utilities for logging equality differences."""

from __future__ import annotations

__all__ = ["setup_difference_logging"]

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_logging_setup_done = False


def setup_difference_logging() -> None:
    r"""Set up logging to display equality differences.

    This function configures the logging system to show INFO level
    messages from the coola.equality module. It only sets up logging
    if it hasn't been configured already.

    This is automatically called when ``show_difference=True`` is
    used in ``objects_are_equal`` or ``objects_are_allclose`` to
    ensure difference messages are visible to users.

    Example:
        ```pycon
        >>> from coola.equality.logging_utils import setup_difference_logging
        >>> setup_difference_logging()

        ```
    """
    global _logging_setup_done  # noqa: PLW0603
    
    # Check if logging has already been configured
    root_logger = logging.getLogger()
    
    # Only auto-configure if no handlers exist or only NullHandler exists
    has_real_handlers = any(
        not isinstance(h, logging.NullHandler) for h in root_logger.handlers
    )
    
    if not has_real_handlers and not _logging_setup_done:
        # Set up a simple handler for coola.equality loggers
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.INFO)
        
        # Simple format focusing on the message
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        
        # Apply to coola.equality module and submodules
        coola_logger = logging.getLogger("coola.equality")
        coola_logger.addHandler(handler)
        coola_logger.setLevel(logging.INFO)
        coola_logger.propagate = False  # Don't propagate to root
        
        _logging_setup_done = True
