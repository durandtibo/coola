from __future__ import annotations

import coola.nested


def test_all_names_are_importable() -> None:
    r"""Every name listed in ``__all__`` must be an attribute of the
    package (regression test: ``__all__`` used to list ``remove_keys``,
    which does not exist, breaking ``from coola.nested import
    remove_keys`` and ``from coola.nested import *``)."""
    for name in coola.nested.__all__:
        assert hasattr(coola.nested, name), f"{name!r} is listed in __all__ but does not exist"


def test_remove_keys_not_in_all() -> None:
    r"""``remove_keys`` does not exist; only the
    ``_if``/``_containing``/ ``_starting_with`` variants do."""
    assert "remove_keys" not in coola.nested.__all__
    assert "remove_keys_if" in coola.nested.__all__
    assert "remove_keys_containing" in coola.nested.__all__
    assert "remove_keys_starting_with" in coola.nested.__all__
