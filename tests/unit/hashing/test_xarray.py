from __future__ import annotations

from coola.hashing import XarrayHasher
from coola.testing.fixtures import xarray_available

###################################
#     Tests for XarrayHasher      #
###################################


@xarray_available
def test_xarray_hasher_repr() -> None:
    assert repr(XarrayHasher("DataArray")) == "XarrayHasher(tag='DataArray')"
