from __future__ import annotations

from coola.registry import Registry

##############################
#     Tests for Registry     #
##############################


def test_registry_full_workflow() -> None:
    """Test a complete workflow with multiple operations."""
    registry = Registry[str, int]()

    # Register multiple items
    registry.register("a", 1)
    registry.register("b", 2)
    registry["c"] = 3

    # Check state
    assert len(registry) == 3
    assert "a" in registry

    # Update existing
    registry["b"] = 20
    assert registry.get("b") == 20

    # Unregister
    value = registry.unregister("a")
    assert value == 1
    assert len(registry) == 2

    # Clear all
    registry.clear()
    assert len(registry) == 0
