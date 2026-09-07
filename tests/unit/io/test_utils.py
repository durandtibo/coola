from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

from coola.io import add_uuid_suffix

if TYPE_CHECKING:
    from pathlib import Path


#####################################
#     Tests for add_uuid_suffix     #
#####################################


def test_add_uuid_suffix_no_suffix(tmp_path: Path) -> None:
    with patch("coola.io.utils.uuid.uuid4", lambda: Mock(hex="a1b2c3")):
        assert add_uuid_suffix(tmp_path.joinpath("data")) == tmp_path.joinpath("data-a1b2c3")


def test_add_uuid_suffix_one_suffix(tmp_path: Path) -> None:
    with patch("coola.io.utils.uuid.uuid4", lambda: Mock(hex="a1b2c3")):
        assert add_uuid_suffix(tmp_path.joinpath("data.json")) == tmp_path.joinpath(
            "data-a1b2c3.json"
        )


def test_add_uuid_suffix_two_suffixes(tmp_path: Path) -> None:
    with patch("coola.io.utils.uuid.uuid4", lambda: Mock(hex="a1b2c3")):
        assert add_uuid_suffix(tmp_path.joinpath("data.tar.gz")) == tmp_path.joinpath(
            "data-a1b2c3.tar.gz"
        )


def test_add_uuid_suffix_dir(tmp_path: Path) -> None:
    with patch("coola.io.utils.uuid.uuid4", lambda: Mock(hex="a1b2c3")):
        assert add_uuid_suffix(tmp_path.joinpath("data/")) == tmp_path.joinpath("data-a1b2c3")
