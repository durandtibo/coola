r"""Define a pytest fixture to isolate the default registries.

`pytest` is required to use this fixture.
"""

from __future__ import annotations

__all__ = ["isolated_default_registries"]

from typing import TYPE_CHECKING

import pytest

from coola.utils.singleton import temporary_default_registries

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def isolated_default_registries() -> Generator[None]:
    r"""Restore the default registries after the test.

    A test that registers a type in a default registry (for example
    with ``register_equality_testers``) changes it for the rest of the
    process. With this fixture, the test works on copies of the default
    registries and the original ones are restored when it ends. Import
    it in a ``conftest.py`` (``from coola.testing.registries import
    isolated_default_registries``) and use it with
    ``pytest.mark.usefixtures("isolated_default_registries")``.

    Example:
        ```pycon
        >>> from coola.testing.registries import isolated_default_registries

        ```
    """
    with temporary_default_registries():
        yield
