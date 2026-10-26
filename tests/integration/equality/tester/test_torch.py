from __future__ import annotations

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    TorchPackedSequenceEqualityTester,
    TorchTensorEqualityTester,
)
from coola.testing.fixtures import torch_available, torch_not_available
from coola.utils.imports import MissingPackageError, is_torch_available
from tests.unit.equality.utils import run_tester

if is_torch_available():
    import torch


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


#######################################################
#     Tests for TorchPackedSequenceEqualityTester     #
#######################################################


@torch_available
def test_torch_packed_sequence_equality_tester_with_torch(config: EqualityConfig) -> None:
    assert run_tester(
        TorchPackedSequenceEqualityTester(),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        config,
    )


@torch_not_available
def test_torch_packed_sequence_equality_tester_without_torch() -> None:
    with pytest.raises(
        MissingPackageError, match=r"'torch' package is required but not installed."
    ):
        TorchPackedSequenceEqualityTester()


###############################################
#     Tests for TorchTensorEqualityTester     #
###############################################


@torch_available
def test_torch_tensor_equality_tester_with_torch(config: EqualityConfig) -> None:
    assert run_tester(TorchTensorEqualityTester(), torch.ones(2, 3), torch.ones(2, 3), config)


@torch_not_available
def test_torch_tensor_equality_tester_without_torch() -> None:
    with pytest.raises(
        MissingPackageError, match=r"'torch' package is required but not installed."
    ):
        TorchTensorEqualityTester()
