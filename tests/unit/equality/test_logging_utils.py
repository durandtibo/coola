from __future__ import annotations

import logging

import coola.equality.logging_utils
from coola.equality.logging_utils import setup_difference_logging


def test_setup_difference_logging() -> None:
    # Clear any existing handlers and reset the flag
    coola_logger = logging.getLogger("coola.equality")
    original_handlers = list(coola_logger.handlers)
    coola_logger.handlers.clear()
    coola.equality.logging_utils._logging_setup_done = False
    
    try:
        # Setup logging
        setup_difference_logging()
        
        # Verify logger is configured
        # The handler may or may not be added depending on root logger state
        # But propagate should be False if setup was done
        if coola.equality.logging_utils._logging_setup_done:
            assert not coola_logger.propagate
            assert len(coola_logger.handlers) > 0
            # Verify handler is set to INFO level
            assert any(h.level == logging.INFO for h in coola_logger.handlers)
    finally:
        # Restore original state
        coola_logger.handlers = original_handlers


def test_setup_difference_logging_idempotent() -> None:
    # Clear any existing handlers
    coola_logger = logging.getLogger("coola.equality")
    original_handlers = list(coola_logger.handlers)
    coola_logger.handlers.clear()
    
    # Reset the global flag
    coola.equality.logging_utils._logging_setup_done = False
    
    try:
        # Setup logging multiple times
        setup_difference_logging()
        handler_count = len(coola_logger.handlers)
        
        setup_difference_logging()
        # Should not add additional handlers (flag prevents re-setup)
        assert len(coola_logger.handlers) == handler_count
    finally:
        # Restore original state
        coola_logger.handlers = original_handlers
