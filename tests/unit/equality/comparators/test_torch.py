from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola.equality import EqualityConfig
from coola.equality.comparators.torch_ import (
    TorchPackedSequenceEqualityComparator,
    TorchTensorEqualityComparator,
    get_type_comparator_mapping,
)
from coola.equality.testers import EqualityTester
from coola.testing import torch_available, torch_cuda_available, torch_mps_available
from coola.utils.imports import is_torch_available
from tests.unit.equality.comparators.utils import ExamplePair
from tests.unit.equality.handlers.test_torch import TORCH_TENSOR_EQUAL_TOLERANCE

if is_torch_available():
    import torch
else:
    torch = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


TORCH_PACKED_SEQUENCE_EQUAL = [
    pytest.param(
        ExamplePair(
            object1=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10, dtype=torch.float).view(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            object2=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10, dtype=torch.float).view(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
        ),
        id="float dtype",
    ),
    pytest.param(
        ExamplePair(
            object1=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10, dtype=torch.long).view(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            object2=torch.nn.utils.rnn.pack_padded_sequence(
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
            object1=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10, dtype=torch.float).view(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            object2=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10).view(2, 5).add(1).float(),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            expected_message="objects have different data:",
        ),
        id="different data",
    ),
    pytest.param(
        ExamplePair(
            object1=torch.nn.utils.rnn.PackedSequence(
                data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
                batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
                sorted_indices=None,
                unsorted_indices=None,
            ),
            object2=torch.nn.utils.rnn.PackedSequence(
                data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
                batch_sizes=torch.tensor([2, 2, 2, 2, 0]),
                sorted_indices=None,
                unsorted_indices=None,
            ),
            expected_message="objects have different batch_sizes:",
        ),
        id="different batch_sizes",
    ),
    pytest.param(
        ExamplePair(
            object1=torch.nn.utils.rnn.PackedSequence(
                data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
                batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
                sorted_indices=torch.tensor([0, 1]),
                unsorted_indices=None,
            ),
            object2=torch.nn.utils.rnn.PackedSequence(
                data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
                batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
                sorted_indices=None,
                unsorted_indices=None,
            ),
            expected_message="objects have different sorted_indices:",
        ),
        id="different sorted_indices",
    ),
    pytest.param(
        ExamplePair(
            object1=torch.nn.utils.rnn.PackedSequence(
                data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
                batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
                sorted_indices=None,
                unsorted_indices=torch.tensor([0, 1]),
            ),
            object2=torch.nn.utils.rnn.PackedSequence(
                data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
                batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
                sorted_indices=None,
                unsorted_indices=None,
            ),
            expected_message="objects have different unsorted_indices:",
        ),
        id="different unsorted_indices",
    ),
    pytest.param(
        ExamplePair(
            object1=torch.nn.utils.rnn.pack_padded_sequence(
                input=torch.arange(10, dtype=torch.float).view(2, 5),
                lengths=torch.tensor([5, 3], dtype=torch.long),
                batch_first=True,
            ),
            object2=torch.arange(10, dtype=torch.float).view(2, 5),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]


TORCH_TENSOR_EQUAL = [
    pytest.param(
        ExamplePair(
            object1=torch.ones(2, 3, dtype=torch.float), object2=torch.ones(2, 3, dtype=torch.float)
        ),
        id="float dtype",
    ),
    pytest.param(
        ExamplePair(
            object1=torch.ones(2, 3, dtype=torch.long), object2=torch.ones(2, 3, dtype=torch.long)
        ),
        id="long dtype",
    ),
    pytest.param(ExamplePair(object1=torch.ones(6), object2=torch.ones(6)), id="1d tensor"),
    pytest.param(ExamplePair(object1=torch.ones(2, 3), object2=torch.ones(2, 3)), id="2d tensor"),
]


TORCH_TENSOR_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            object1=torch.ones(2, 3, dtype=torch.float),
            object2=torch.ones(2, 3, dtype=torch.long),
            expected_message="objects have different data types:",
        ),
        id="different data types",
    ),
    pytest.param(
        ExamplePair(
            object1=torch.ones(2, 3),
            object2=torch.ones(6),
            expected_message="objects have different shapes:",
        ),
        id="different shapes",
    ),
    pytest.param(
        ExamplePair(
            object1=torch.ones(2, 3),
            object2=torch.zeros(2, 3),
            expected_message="torch.Tensors have different elements:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            object1=torch.ones(2, 3),
            object2="meow",
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]

TORCH_EQUAL = TORCH_TENSOR_EQUAL + TORCH_PACKED_SEQUENCE_EQUAL
TORCH_NOT_EQUAL = TORCH_TENSOR_NOT_EQUAL + TORCH_PACKED_SEQUENCE_NOT_EQUAL


###########################################################
#     Tests for TorchPackedSequenceEqualityComparator     #
###########################################################


@torch_available
def test_tensor_packed_sequence_equality_comparator_str() -> None:
    assert str(TorchPackedSequenceEqualityComparator()) == "TorchPackedSequenceEqualityComparator()"


@torch_available
def test_tensor_packed_sequence_equality_comparator__eq__true() -> None:
    assert TorchPackedSequenceEqualityComparator() == TorchPackedSequenceEqualityComparator()


@torch_available
def test_tensor_packed_sequence_equality_comparator__eq__false() -> None:
    assert TorchPackedSequenceEqualityComparator() != 123


@torch_available
def test_tensor_packed_sequence_equality_comparator_clone() -> None:
    op = TorchPackedSequenceEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@torch_available
def test_tensor_packed_sequence_equality_comparator_equal_true_same_obj(
    config: EqualityConfig,
) -> None:
    obj = torch.nn.utils.rnn.pack_padded_sequence(
        input=torch.arange(10, dtype=torch.float).view(2, 5),
        lengths=torch.tensor([5, 3], dtype=torch.long),
        batch_first=True,
    )
    assert TorchPackedSequenceEqualityComparator().equal(obj, obj, config)


@torch_available
@pytest.mark.parametrize("example", TORCH_PACKED_SEQUENCE_EQUAL)
def test_tensor_packed_sequence_equality_comparator_equal_yes(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = TorchPackedSequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@torch_available
@pytest.mark.parametrize("example", TORCH_PACKED_SEQUENCE_EQUAL)
def test_tensor_packed_sequence_equality_comparator_equal_yes_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = TorchPackedSequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@torch_available
@pytest.mark.parametrize("example", TORCH_PACKED_SEQUENCE_NOT_EQUAL)
def test_tensor_packed_sequence_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = TorchPackedSequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@torch_available
@pytest.mark.parametrize("example", TORCH_PACKED_SEQUENCE_NOT_EQUAL)
def test_tensor_packed_sequence_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = TorchPackedSequenceEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@torch_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_tensor_packed_sequence_equality_comparator_equal_nan_false(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        TorchPackedSequenceEqualityComparator().equal(
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


@torch_available
def test_tensor_packed_sequence_equality_comparator_no_torch() -> None:
    with patch(
        "coola.utils.imports.is_torch_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`torch` package is required but not installed."):
        TorchPackedSequenceEqualityComparator()


###################################################
#     Tests for TorchTensorEqualityComparator     #
###################################################


@torch_available
def test_torch_tensor_equality_comparator_str() -> None:
    assert str(TorchTensorEqualityComparator()).startswith("TorchTensorEqualityComparator(")


@torch_available
def test_torch_tensor_equality_comparator__eq__true() -> None:
    assert TorchTensorEqualityComparator() == TorchTensorEqualityComparator()


@torch_available
def test_torch_tensor_equality_comparator__eq__false_different_type() -> None:
    assert TorchTensorEqualityComparator() != 123


@torch_available
def test_torch_tensor_equality_comparator_clone() -> None:
    op = TorchTensorEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@torch_available
def test_torch_tensor_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    tensor = torch.ones(2, 3)
    assert TorchTensorEqualityComparator().equal(tensor, tensor, config)


@torch_available
@pytest.mark.parametrize("example", TORCH_TENSOR_EQUAL)
def test_torch_tensor_equality_comparator_equal_yes(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = TorchTensorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@torch_available
@pytest.mark.parametrize("example", TORCH_TENSOR_EQUAL)
def test_torch_tensor_equality_comparator_equal_yes_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = TorchTensorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@torch_available
@pytest.mark.parametrize("example", TORCH_TENSOR_NOT_EQUAL)
def test_torch_tensor_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = TorchTensorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@torch_available
@pytest.mark.parametrize("example", TORCH_TENSOR_NOT_EQUAL)
def test_torch_tensor_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = TorchTensorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@torch_available
@torch_cuda_available
def test_torch_tensor_equality_comparator_equal_false_different_device_cuda(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = TorchTensorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            torch.ones(2, 3, device=torch.device("cpu")),
            torch.zeros(2, 3, device=torch.device("cuda:0")),
            config,
        )
        assert caplog.messages[0].startswith("torch.Tensors have different devices:")


@torch_available
@torch_mps_available
def test_torch_tensor_equality_comparator_equal_false_different_device_mps(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = TorchTensorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            torch.ones(2, 3, device=torch.device("cpu")),
            torch.zeros(2, 3, device=torch.device("mps:0")),
            config,
        )
        assert caplog.messages[0].startswith("torch.Tensors have different devices:")


@torch_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_torch_tensor_equality_comparator_equal_nan_true(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        TorchTensorEqualityComparator().equal(
            object1=torch.tensor([0.0, float("nan"), float("nan"), 1.2]),
            object2=torch.tensor([0.0, float("nan"), float("nan"), 1.2]),
            config=config,
        )
        == equal_nan
    )


@torch_available
@pytest.mark.parametrize("example", TORCH_TENSOR_EQUAL_TOLERANCE)
def test_torch_tensor_equality_comparator_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert TorchTensorEqualityComparator().equal(
        object1=example.object1, object2=example.object2, config=config
    )


@torch_available
def test_torch_tensor_equality_comparator_no_torch() -> None:
    with patch(
        "coola.utils.imports.is_torch_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`torch` package is required but not installed."):
        TorchTensorEqualityComparator()


#################################################
#     Tests for get_type_comparator_mapping     #
#################################################


@torch_available
def test_get_type_comparator_mapping() -> None:
    assert get_type_comparator_mapping() == {
        torch.nn.utils.rnn.PackedSequence: TorchPackedSequenceEqualityComparator(),
        torch.Tensor: TorchTensorEqualityComparator(),
    }


def test_get_type_comparator_mapping_no_torch() -> None:
    with patch(
        "coola.equality.comparators.torch_.is_torch_available", lambda *args, **kwargs: False
    ):
        assert get_type_comparator_mapping() == {}
