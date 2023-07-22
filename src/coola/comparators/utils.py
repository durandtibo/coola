from __future__ import annotations

__all__ = ["get_mapping_allclose"]


from coola.comparators.base import BaseAllCloseOperator


def get_mapping_allclose() -> dict[type[object], BaseAllCloseOperator]:
    r"""Gets a default mapping between the types and the allclose
    operators.

    Returns:
        dict: The mapping between the types and the allclose
            operators.
    """
    from coola import comparators as cmp  # Local import to avoid cyclic dependencies

    return (
        cmp.allclose.get_mapping_allclose()
        | cmp.numpy_.get_mapping_allclose()
        | cmp.pandas_.get_mapping_allclose()
        | cmp.polars_.get_mapping_allclose()
        | cmp.torch_.get_mapping_allclose()
        | cmp.xarray_.get_mapping_allclose()
    )
