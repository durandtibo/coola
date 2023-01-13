from unittest.mock import patch

from pytest import raises

from coola.import_utils import check_torch


def test_check_torch_with_package():
    with patch("coola.import_utils.is_torch_available", lambda *args: True):
        check_torch()


def test_check_torch_without_package():
    with patch("coola.import_utils.is_torch_available", lambda *args: False):
        with raises(RuntimeError):
            check_torch()
