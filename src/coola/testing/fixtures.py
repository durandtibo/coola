r"""Define a pytest fixture to isolate the default registries.

`pytest` is required to use this fixture.
"""

from __future__ import annotations

__all__ = ["isolated_default_registries"]

from typing import TYPE_CHECKING

import pytest

from coola.registry.default import temporary_default_registries

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def isolated_default_registries() -> Generator[None]:
    r"""Restore the default registries after the test.

    A test that registers a type in a default registry (for example
    with ``register_equality_testers``) changes it for the rest of the
    process. With this fixture, the test works on copies of the default
    registries and the original ones are restored when it ends. Load
    the module as a plugin in the top-level ``conftest.py``
    (``pytest_plugins = ["coola.testing.fixtures"]``) and use the
    fixture with
    ``pytest.mark.usefixtures("isolated_default_registries")``.

    Yields:
        ``None``. The default registries are restored when the test
            ends.

    Example:
        ```pycon
        >>> from coola.testing.fixtures import isolated_default_registries
        >>> isolated_default_registries.__name__
        'isolated_default_registries'

        ```
    """
    with temporary_default_registries():
        yield
