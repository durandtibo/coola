from __future__ import annotations

__all__ = ["get_mapping_allclose", "get_mapping_equality"]


from coola.comparators.base import BaseAllCloseOperator, BaseEqualityOperator


def get_mapping_allclose() -> dict[type[object], BaseAllCloseOperator]:
    r"""Gets a default mapping between the types and the allclose
    operators.

    Returns:
    -------
        dict: The mapping between the types and the allclose
            operators.
    """
    from coola import comparators  # Local import to avoid cyclic dependencies

    return (
        comparators.allclose.get_mapping_allclose()
        | comparators.jax_.get_mapping_allclose()
        | comparators.numpy_.get_mapping_allclose()
        | comparators.pandas_.get_mapping_allclose()
        | comparators.polars_.get_mapping_allclose()
        | comparators.torch_.get_mapping_allclose()
        | comparators.xarray_.get_mapping_allclose()
    )


def get_mapping_equality() -> dict[type[object], BaseEqualityOperator]:
    r"""Gets a default mapping between the types and the equality
    operators.

    Returns:
    -------
        dict: The mapping between the types and the equality
            operators.
    """
    from coola import comparators  # Local import to avoid cyclic dependencies

    return (
        comparators.equality.get_mapping_equality()
        | comparators.jax_.get_mapping_equality()
        | comparators.numpy_.get_mapping_equality()
        | comparators.pandas_.get_mapping_equality()
        | comparators.polars_.get_mapping_equality()
        | comparators.torch_.get_mapping_equality()
        | comparators.xarray_.get_mapping_equality()
    )
