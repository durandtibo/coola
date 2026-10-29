from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from coola.equality import compare
from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    TorchPackedSequenceEqualityTester,
    TorchTensorEqualityTester,
)
from coola.equality.tolerance import Tolerance
from coola.testing.marks import (
    torch_available,
    torch_cuda_available,
    torch_mps_available,
)
from coola.utils.imports import MissingPackageError, is_torch_available
from tests.unit.equality.utils import ExamplePair, explaining, messages, run_tester

if is_torch_available():
    import torch
else:
    torch = Mock()


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


TORCH_PACKED_SEQUENCE_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10, dtype=torch.float).view(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            expected=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10, dtype=torch.float).view(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
        ),
        id="float dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10, dtype=torch.long).view(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            expected=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10, dtype=torch.long).view(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
        ),
        id="long dtype",
    ),
]
TORCH_PACKED_SEQUENCE_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10, dtype=torch.float).view(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            expected=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10).view(2, 5).add(1).float(),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            expected_message="values are different:",
        ),
        id="different data",
    ),
    pytest.param(
        ExamplePair(
            actual=torch.nn.utils.rnn.PackedSequence(
                data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
                batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
                sorted_indices=None,
                unsorted_indices=None,
            ),
            expected=torch.nn.utils.rnn.PackedSequence(
                data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
                batch_sizes=torch.tensor([2, 2, 2, 2, 0]),
                sorted_indices=None,
                unsorted_indices=None,
            ),
            expected_message="values are different:",
        ),
        id="different batch_sizes",
    ),
    pytest.param(
        ExamplePair(
            actual=torch.nn.utils.rnn.PackedSequence(
                data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
                batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
                sorted_indices=torch.tensor([0, 1]),
                unsorted_indices=None,
            ),
            expected=torch.nn.utils.rnn.PackedSequence(
                data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
                batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
                sorted_indices=None,
                unsorted_indices=None,
            ),
            expected_message="objects have different types:",
        ),
        id="different sorted_indices",
    ),
    pytest.param(
        ExamplePair(
            actual=torch.nn.utils.rnn.PackedSequence(
                data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
                batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
                sorted_indices=None,
                unsorted_indices=torch.tensor([0, 1]),
            ),
            expected=torch.nn.utils.rnn.PackedSequence(
                data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
                batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
                sorted_indices=None,
                unsorted_indices=None,
            ),
            expected_message="objects have different types:",
        ),
        id="different unsorted_indices",
    ),
    pytest.param(
        ExamplePair(
            actual=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10, dtype=torch.float).view(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            expected=torch.arange(10, dtype=torch.float).view(2, 5),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]
TORCH_PACKED_SEQUENCE_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(
            actual=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.ones(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            expected=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.full(size=(2, 5), fill_value=1.5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            atol=1.0,
        ),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.ones(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            expected=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.full(size=(2, 5), fill_value=1.05),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            atol=0.1,
        ),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.ones(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            expected=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.full(size=(2, 5), fill_value=1.005),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            atol=0.01,
        ),
        id="atol=0.01",
    ),
    # rtol
    pytest.param(
        ExamplePair(
            actual=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.ones(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            expected=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.full(size=(2, 5), fill_value=1.5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            rtol=1.0,
        ),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.ones(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            expected=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.full(size=(2, 5), fill_value=1.05),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            rtol=0.1,
        ),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.ones(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            expected=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.full(size=(2, 5), fill_value=1.005),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            rtol=0.01,
        ),
        id="rtol=0.01",
    ),
]

TORCH_TENSOR_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=torch.ones(2, 3, dtype=torch.float), expected=torch.ones(2, 3, dtype=torch.float)
        ),
        id="float dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=torch.ones(2, 3, dtype=torch.long), expected=torch.ones(2, 3, dtype=torch.long)
        ),
        id="long dtype",
    ),
    pytest.param(ExamplePair(actual=torch.ones(6), expected=torch.ones(6)), id="1d tensor"),
    pytest.param(ExamplePair(actual=torch.ones(2, 3), expected=torch.ones(2, 3)), id="2d tensor"),
]
TORCH_TENSOR_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=torch.ones(2, 3, dtype=torch.float),
            expected=torch.ones(2, 3, dtype=torch.long),
            expected_message="objects have different data types:",
        ),
        id="different data types",
    ),
    pytest.param(
        ExamplePair(
            actual=torch.ones(2, 3),
            expected=torch.ones(6),
            expected_message="objects have different shapes:",
        ),
        id="different shapes",
    ),
    pytest.param(
        ExamplePair(
            actual=torch.ones(2, 3),
            expected=torch.zeros(2, 3),
            expected_message="values are different:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=torch.ones(2, 3),
            expected="meow",
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]
TORCH_TENSOR_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(actual=torch.ones(2, 3), expected=torch.full((2, 3), 1.5), atol=1.0),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(actual=torch.ones(2, 3), expected=torch.full((2, 3), 1.05), atol=0.1),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(actual=torch.ones(2, 3), expected=torch.full((2, 3), 1.005), atol=0.01),
        id="atol=0.01",
    ),
    # rtol
    pytest.param(
        ExamplePair(actual=torch.ones(2, 3), expected=torch.full((2, 3), 1.5), rtol=1.0),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(actual=torch.ones(2, 3), expected=torch.full((2, 3), 1.05), rtol=0.1),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(actual=torch.ones(2, 3), expected=torch.full((2, 3), 1.005), rtol=0.01),
        id="rtol=0.01",
    ),
]

TORCH_EQUAL = TORCH_TENSOR_EQUAL + TORCH_PACKED_SEQUENCE_EQUAL
TORCH_NOT_EQUAL = TORCH_TENSOR_NOT_EQUAL + TORCH_PACKED_SEQUENCE_NOT_EQUAL
TORCH_EQUAL_TOLERANCE = TORCH_TENSOR_EQUAL_TOLERANCE + TORCH_PACKED_SEQUENCE_EQUAL_TOLERANCE


#######################################################
#     Tests for TorchPackedSequenceEqualityTester     #
#######################################################


@torch_available
def test_torch_packed_sequence_equality_tester_repr() -> None:
    assert repr(TorchPackedSequenceEqualityTester()) == "TorchPackedSequenceEqualityTester()"


@torch_available
def test_torch_packed_sequence_equality_tester_str() -> None:
    assert str(TorchPackedSequenceEqualityTester()) == "TorchPackedSequenceEqualityTester()"


@torch_available
def test_torch_packed_sequence_equality_tester_equal_true() -> None:
    assert TorchPackedSequenceEqualityTester() == TorchPackedSequenceEqualityTester()


@torch_available
def test_torch_packed_sequence_equality_tester_equal_false_different_type() -> None:
    assert TorchPackedSequenceEqualityTester() != 123


@torch_available
def test_torch_packed_sequence_equality_tester_equal_false_different_type_child() -> None:
    class Child(TorchPackedSequenceEqualityTester): ...

    assert TorchPackedSequenceEqualityTester() != Child()


@torch_available
def test_torch_packed_sequence_equality_tester_objects_are_equal_true_same_obj(
    config: EqualityConfig,
) -> None:
    obj = torch.nn.utils.rnn.pack_padded_sequence(
        input=torch.arange(10, dtype=torch.float).view(2, 5),
        lengths=torch.tensor([5, 3], dtype=torch.long),
        batch_first=True,
    )
    assert run_tester(TorchPackedSequenceEqualityTester(), obj, obj, config)


@torch_available
@pytest.mark.parametrize("example", TORCH_PACKED_SEQUENCE_EQUAL)
def test_torch_packed_sequence_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = TorchPackedSequenceEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@torch_available
@pytest.mark.parametrize("example", TORCH_PACKED_SEQUENCE_EQUAL)
def test_torch_packed_sequence_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = TorchPackedSequenceEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@torch_available
@pytest.mark.parametrize("example", TORCH_PACKED_SEQUENCE_EQUAL)
def test_torch_packed_sequence_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = TorchPackedSequenceEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@torch_available
@pytest.mark.parametrize("example", TORCH_PACKED_SEQUENCE_EQUAL)
def test_torch_packed_sequence_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = TorchPackedSequenceEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@torch_available
@pytest.mark.parametrize("example", TORCH_PACKED_SEQUENCE_NOT_EQUAL)
def test_torch_packed_sequence_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = TorchPackedSequenceEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@torch_available
@pytest.mark.parametrize("example", TORCH_PACKED_SEQUENCE_NOT_EQUAL)
def test_torch_packed_sequence_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = TorchPackedSequenceEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@torch_available
@pytest.mark.parametrize("example", TORCH_PACKED_SEQUENCE_NOT_EQUAL)
def test_torch_packed_sequence_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = TorchPackedSequenceEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@torch_available
@pytest.mark.parametrize("example", TORCH_PACKED_SEQUENCE_NOT_EQUAL)
def test_torch_packed_sequence_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = TorchPackedSequenceEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert messages(config)[0].startswith(example.expected_message)


@torch_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_torch_packed_sequence_equality_tester_objects_are_equal_nan_false(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config = replace(config, equal_nan=equal_nan)
    assert (
        run_tester(
            TorchPackedSequenceEqualityTester(),
            torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.tensor(
                    [[0.0, 1.0, 2.0, float("nan"), 4.0], [float("nan"), 6.0, 7.0, 8.0, 9.0]],
                    dtype=torch.float,
                ),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.tensor(
                    [[0.0, 1.0, 2.0, float("nan"), 4.0], [float("nan"), 6.0, 7.0, 8.0, 9.0]],
                    dtype=torch.float,
                ),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            config,
        )
        == equal_nan
    )


def test_torch_packed_sequence_equality_tester_no_torch() -> None:
    with (
        patch("coola.utils.imports.optional.is_torch_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'torch' package is required but not installed."),
    ):
        TorchPackedSequenceEqualityTester()


###############################################
#     Tests for TorchTensorEqualityTester     #
###############################################


@torch_available
def test_torch_tensor_equality_tester_repr() -> None:
    assert repr(TorchTensorEqualityTester()).startswith("TorchTensorEqualityTester(")


@torch_available
def test_torch_tensor_equality_tester_str() -> None:
    assert str(TorchTensorEqualityTester()).startswith("TorchTensorEqualityTester(")


@torch_available
def test_torch_tensor_equality_tester_equal_true() -> None:
    assert TorchTensorEqualityTester() == TorchTensorEqualityTester()


@torch_available
def test_torch_tensor_equality_tester_equal_false_different_type() -> None:
    assert TorchTensorEqualityTester() != 123


@torch_available
def test_torch_tensor_equality_tester_equal_false_different_type_child() -> None:
    class Child(TorchTensorEqualityTester): ...

    assert TorchTensorEqualityTester() != Child()


@torch_available
def test_torch_tensor_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    tensor = torch.ones(2, 3)
    assert run_tester(TorchTensorEqualityTester(), tensor, tensor, config)


@torch_available
@pytest.mark.parametrize("example", TORCH_TENSOR_EQUAL)
def test_torch_tensor_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = TorchTensorEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@torch_available
@pytest.mark.parametrize("example", TORCH_TENSOR_EQUAL)
def test_torch_tensor_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = TorchTensorEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@torch_available
@pytest.mark.parametrize("example", TORCH_TENSOR_EQUAL)
def test_torch_tensor_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = TorchTensorEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@torch_available
@pytest.mark.parametrize("example", TORCH_TENSOR_EQUAL)
def test_torch_tensor_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = TorchTensorEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@torch_available
@pytest.mark.parametrize("example", TORCH_TENSOR_NOT_EQUAL)
def test_torch_tensor_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = TorchTensorEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@torch_available
@pytest.mark.parametrize("example", TORCH_TENSOR_NOT_EQUAL)
def test_torch_tensor_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = TorchTensorEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@torch_available
@pytest.mark.parametrize("example", TORCH_TENSOR_NOT_EQUAL)
def test_torch_tensor_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = TorchTensorEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@torch_available
@pytest.mark.parametrize("example", TORCH_TENSOR_NOT_EQUAL)
def test_torch_tensor_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = TorchTensorEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert messages(config)[0].startswith(example.expected_message)


@torch_available
def test_torch_tensor_equality_tester_objects_are_equal_false_different_device_meta(
    config: EqualityConfig,
) -> None:
    # The meta device exists without a GPU, so the check of the devices always runs.
    config = explaining(config)
    tester = TorchTensorEqualityTester()
    assert not run_tester(
        tester,
        torch.ones(2, 3, device=torch.device("cpu")),
        torch.zeros(2, 3, device=torch.device("meta")),
        config,
    )


@torch_available
def test_torch_tensor_equality_tester_objects_are_equal_false_different_device_meta_logs(
    config: EqualityConfig,
) -> None:
    # The meta device exists without a GPU, so the check of the devices always runs.
    config = explaining(config)
    tester = TorchTensorEqualityTester()
    run_tester(
        tester,
        torch.ones(2, 3, device=torch.device("cpu")),
        torch.zeros(2, 3, device=torch.device("meta")),
        config,
    )
    assert messages(config)[0].startswith("torch.Tensors have different devices:")


@torch_available
@torch_cuda_available
def test_torch_tensor_equality_tester_objects_are_equal_false_different_device_cuda(
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = TorchTensorEqualityTester()
    assert not run_tester(
        tester,
        torch.ones(2, 3, device=torch.device("cpu")),
        torch.zeros(2, 3, device=torch.device("cuda:0")),
        config,
    )


@torch_available
@torch_cuda_available
def test_torch_tensor_equality_tester_objects_are_equal_false_different_device_cuda_logs(
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = TorchTensorEqualityTester()
    run_tester(
        tester,
        torch.ones(2, 3, device=torch.device("cpu")),
        torch.zeros(2, 3, device=torch.device("cuda:0")),
        config,
    )
    assert messages(config)[0].startswith("torch.Tensors have different devices:")


@torch_available
@torch_mps_available
def test_torch_tensor_equality_tester_objects_are_equal_false_different_device_mps(
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = TorchTensorEqualityTester()
    assert not run_tester(
        tester,
        torch.ones(2, 3, device=torch.device("cpu")),
        torch.zeros(2, 3, device=torch.device("mps:0")),
        config,
    )


@torch_available
@torch_mps_available
def test_torch_tensor_equality_tester_objects_are_equal_false_different_device_mps_logs(
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = TorchTensorEqualityTester()
    run_tester(
        tester,
        torch.ones(2, 3, device=torch.device("cpu")),
        torch.zeros(2, 3, device=torch.device("mps:0")),
        config,
    )
    assert messages(config)[0].startswith("torch.Tensors have different devices:")


@torch_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_torch_tensor_equality_tester_objects_are_equal_nan_true(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config = replace(config, equal_nan=equal_nan)
    assert (
        run_tester(
            TorchTensorEqualityTester(),
            torch.tensor([0.0, float("nan"), float("nan"), 1.2]),
            torch.tensor([0.0, float("nan"), float("nan"), 1.2]),
            config,
        )
        == equal_nan
    )


@torch_available
@pytest.mark.parametrize("example", TORCH_TENSOR_EQUAL_TOLERANCE)
def test_torch_tensor_equality_tester_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config = replace(config, tolerance=Tolerance(atol=example.atol, rtol=example.rtol))
    assert run_tester(TorchTensorEqualityTester(), example.actual, example.expected, config)


def test_torch_tensor_equality_tester_no_torch() -> None:
    with (
        patch("coola.utils.imports.optional.is_torch_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'torch' package is required but not installed."),
    ):
        TorchTensorEqualityTester()


#############################################
#     Tests for the location of a difference #
#############################################


@torch_available
def test_torch_tensor_compare_path_is_the_index() -> None:
    expected = torch.zeros(2, 3)
    expected[1, 2] = 5.0
    assert compare(torch.zeros(2, 3), expected).path == (1, 2)


@torch_available
def test_torch_tensor_compare_path_is_a_tuple_of_ints() -> None:
    expected = torch.zeros(2)
    expected[1] = 5.0
    assert type(compare(torch.zeros(2), expected).path[0]) is int


@torch_available
def test_torch_tensor_compare_path_of_a_0d_tensor() -> None:
    assert compare(torch.tensor(1.0), torch.tensor(2.0)).path == ()


@torch_available
def test_torch_tensor_compare_path_of_a_boolean_tensor() -> None:
    assert compare(torch.tensor([True, False]), torch.tensor([True, True])).path == (1,)


@torch_available
def test_torch_tensor_compare_path_with_a_tolerance() -> None:
    assert compare(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.5]), atol=0.1).path == (1,)


@torch_available
def test_torch_tensor_compare_path_of_an_integer_tensor_with_a_tolerance() -> None:
    assert compare(torch.tensor([1, 2]), torch.tensor([1, 5]), atol=1.0).path == (1,)


@torch_available
def test_torch_tensor_compare_path_of_a_uint64_tensor_with_a_tolerance() -> None:
    assert compare(
        torch.tensor([1, 2], dtype=torch.uint64), torch.tensor([1, 5], dtype=torch.uint64), atol=1.0
    ).path == (1,)


@torch_available
def test_torch_tensor_compare_path_of_a_sparse_tensor() -> None:
    assert compare(torch.zeros(2).to_sparse(), torch.tensor([0.0, 1.0]).to_sparse()).path == (1,)


@torch_available
def test_torch_tensor_compare_different_layouts() -> None:
    result = compare(torch.zeros(2), torch.zeros(2).to_sparse())
    assert (
        result.reason == "torch.Tensors have different layouts: torch.strided vs torch.sparse_coo"
    )


@torch_available
def test_torch_tensor_compare_nan_with_equal_nan() -> None:
    tensor = torch.tensor([1.0, float("nan")])
    assert compare(tensor, tensor.clone(), equal_nan=True).equal


@torch_available
def test_torch_tensor_compare_nan_without_equal_nan() -> None:
    tensor = torch.tensor([1.0, float("nan")])
    assert compare(tensor, tensor.clone()).path == (1,)


@torch_available
def test_torch_tensor_compare_meta_tensors_equal() -> None:
    assert compare(torch.zeros(2, device="meta"), torch.zeros(2, device="meta")).equal


@torch_available
def test_torch_tensor_compare_meta_tensors_different_shape() -> None:
    assert not compare(torch.zeros(2, device="meta"), torch.zeros(3, device="meta")).equal


@torch_available
def test_torch_tensor_compare_reason_is_bounded() -> None:
    expected = torch.zeros(1_000_000)
    expected[0] = 1.0
    reason = compare(torch.zeros(1_000_000), expected).reason
    assert reason is not None
    assert len(reason) < 100


@torch_available
def test_torch_tensor_compare_does_not_look_up_the_index_when_not_explaining(
    config: EqualityConfig,
) -> None:
    assert not run_tester(TorchTensorEqualityTester(), torch.zeros(3), torch.ones(3), config)


@torch_available
def test_torch_tensor_compare_different_devices() -> None:
    assert not compare(torch.zeros(2), torch.zeros(2, device="meta")).equal


@torch_available
def test_torch_packed_sequence_compare_path_of_the_data() -> None:
    def make(value: float) -> object:
        return torch.nn.utils.rnn.pack_padded_sequence(
            torch.tensor([[1.0, value], [3.0, 4.0]]),
            lengths=torch.tensor([2, 1]),
            batch_first=True,
            enforce_sorted=False,
        )

    assert compare(make(2.0), make(5.0)).path == ("data", 2)
