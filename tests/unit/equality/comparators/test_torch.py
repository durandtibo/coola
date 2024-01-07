from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola import objects_are_equal
from coola.equality import EqualityConfig
from coola.equality.comparators.torch_ import (
    TorchPackedSequenceEqualityComparator,
    TorchTensorEqualityComparator,
    get_type_comparator_mapping,
)
from coola.testers import EqualityTester
from coola.testing import torch_available, torch_cuda_available
from coola.utils.imports import is_torch_available

if is_torch_available():
    import torch
else:
    torch = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


###########################################################
#     Tests for TorchPackedSequenceEqualityComparator     #
###########################################################


@torch_available
def test_objects_are_equal_packed_sequence() -> None:
    assert objects_are_equal(
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
    )


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
def test_tensor_packed_sequence_equality_comparator_equal_true(config: EqualityConfig) -> None:
    assert TorchPackedSequenceEqualityComparator().equal(
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
def test_tensor_packed_sequence_equality_comparator_equal_false_data(
    config: EqualityConfig,
) -> None:
    assert not TorchPackedSequenceEqualityComparator().equal(
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10).view(2, 5).add(1).float(),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        config,
    )


@torch_available
def test_tensor_packed_sequence_equality_comparator_equal_false_data_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = TorchPackedSequenceEqualityComparator()
    x1 = torch.nn.utils.rnn.pack_padded_sequence(
        input=torch.arange(10, dtype=torch.float).view(2, 5),
        lengths=torch.tensor([5, 3], dtype=torch.long),
        batch_first=True,
    )
    x2 = torch.nn.utils.rnn.pack_padded_sequence(
        input=torch.arange(10).view(2, 5).add(1).float(),
        lengths=torch.tensor([5, 3], dtype=torch.long),
        batch_first=True,
    )
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(x1, x2, config)
        assert caplog.messages[-1].startswith("objects have different data:")


@torch_available
def test_tensor_packed_sequence_equality_comparator_equal_false_batch_sizes(
    config: EqualityConfig,
) -> None:
    assert not TorchPackedSequenceEqualityComparator().equal(
        torch.nn.utils.rnn.PackedSequence(
            data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
            batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
            sorted_indices=None,
            unsorted_indices=None,
        ),
        torch.nn.utils.rnn.PackedSequence(
            data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
            batch_sizes=torch.tensor([2, 2, 2, 2, 0]),
            sorted_indices=None,
            unsorted_indices=None,
        ),
        config,
    )


@torch_available
def test_tensor_packed_sequence_equality_comparator_equal_false_batch_sizes_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = TorchPackedSequenceEqualityComparator()
    x1 = torch.nn.utils.rnn.PackedSequence(
        data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
        batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
        sorted_indices=None,
        unsorted_indices=None,
    )
    x2 = torch.nn.utils.rnn.PackedSequence(
        data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
        batch_sizes=torch.tensor([2, 2, 2, 2, 0]),
        sorted_indices=None,
        unsorted_indices=None,
    )
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(x1, x2, config)
        assert caplog.messages[-1].startswith("objects have different batch_sizes:")


@torch_available
def test_tensor_packed_sequence_equality_comparator_equal_false_sorted_indices(
    config: EqualityConfig,
) -> None:
    assert not TorchPackedSequenceEqualityComparator().equal(
        torch.nn.utils.rnn.PackedSequence(
            data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
            batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
            sorted_indices=torch.tensor([0, 1]),
            unsorted_indices=None,
        ),
        torch.nn.utils.rnn.PackedSequence(
            data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
            batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
            sorted_indices=None,
            unsorted_indices=None,
        ),
        config,
    )


@torch_available
def test_tensor_packed_sequence_equality_comparator_equal_false_sorted_indices_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = TorchPackedSequenceEqualityComparator()
    x1 = torch.nn.utils.rnn.PackedSequence(
        data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
        batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
        sorted_indices=torch.tensor([0, 1]),
        unsorted_indices=None,
    )
    x2 = torch.nn.utils.rnn.PackedSequence(
        data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
        batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
        sorted_indices=None,
        unsorted_indices=None,
    )
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(x1, x2, config)
        assert caplog.messages[-1].startswith("objects have different sorted_indices:")


@torch_available
def test_tensor_packed_sequence_equality_comparator_equal_false_unsorted_indices(
    config: EqualityConfig,
) -> None:
    assert not TorchPackedSequenceEqualityComparator().equal(
        torch.nn.utils.rnn.PackedSequence(
            data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
            batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
            sorted_indices=None,
            unsorted_indices=torch.tensor([0, 1]),
        ),
        torch.nn.utils.rnn.PackedSequence(
            data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
            batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
            sorted_indices=None,
            unsorted_indices=None,
        ),
        config,
    )


@torch_available
def test_tensor_packed_sequence_equality_comparator_equal_false_unsorted_indices_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = TorchPackedSequenceEqualityComparator()
    x1 = torch.nn.utils.rnn.PackedSequence(
        data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
        batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
        sorted_indices=None,
        unsorted_indices=torch.tensor([0, 1]),
    )
    x2 = torch.nn.utils.rnn.PackedSequence(
        data=torch.tensor([0.0, 5.0, 1.0, 6.0, 2.0, 7.0, 3.0, 4.0]),
        batch_sizes=torch.tensor([2, 2, 2, 1, 1]),
        sorted_indices=None,
        unsorted_indices=None,
    )
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(x1, x2, config)
        assert caplog.messages[-1].startswith("objects have different unsorted_indices:")


@torch_available
def test_tensor_packed_sequence_equality_comparator_equal_false_different_type(
    config: EqualityConfig,
) -> None:
    assert not TorchPackedSequenceEqualityComparator().equal(
        torch.nn.utils.rnn.pack_padded_sequence(
            input=torch.arange(10, dtype=torch.float).view(2, 5),
            lengths=torch.tensor([5, 3], dtype=torch.long),
            batch_first=True,
        ),
        torch.arange(10, dtype=torch.float).view(2, 5),
        config,
    )


@torch_available
def test_tensor_packed_sequence_equality_comparator_equal_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = TorchPackedSequenceEqualityComparator()
    x1 = torch.nn.utils.rnn.pack_padded_sequence(
        input=torch.arange(10, dtype=torch.float).view(2, 5),
        lengths=torch.tensor([5, 3], dtype=torch.long),
        batch_first=True,
    )
    x2 = torch.arange(10, dtype=torch.float).view(2, 5)
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(x1, x2, config)
        assert caplog.messages[0].startswith("objects have different types:")


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
def test_objects_are_equal_tensor() -> None:
    assert objects_are_equal(torch.ones(2, 3), torch.ones(2, 3))


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
def test_torch_tensor_equality_comparator_equal_true(config: EqualityConfig) -> None:
    assert TorchTensorEqualityComparator().equal(torch.ones(2, 3), torch.ones(2, 3), config)


@torch_available
def test_torch_tensor_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    array = torch.ones(2, 3)
    assert TorchTensorEqualityComparator().equal(array, array, config)


@torch_available
def test_torch_tensor_equality_comparator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = TorchTensorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            object1=torch.ones(2, 3),
            object2=torch.ones(2, 3),
            config=config,
        )
        assert not caplog.messages


@torch_available
def test_torch_tensor_equality_comparator_equal_false_different_dtype(
    config: EqualityConfig,
) -> None:
    assert not TorchTensorEqualityComparator().equal(
        torch.ones(2, 3, dtype=torch.float), torch.ones(2, 3, dtype=torch.int), config
    )


@torch_available
def test_torch_tensor_equality_comparator_equal_false_different_dtype_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = TorchTensorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=torch.ones(2, 3, dtype=torch.float),
            object2=torch.ones(2, 3, dtype=torch.int),
            config=config,
        )
        assert caplog.messages[0].startswith("objects have different data types:")


@torch_available
def test_torch_tensor_equality_comparator_equal_false_different_shape(
    config: EqualityConfig,
) -> None:
    assert not TorchTensorEqualityComparator().equal(torch.ones(2, 3), torch.zeros(6), config)


@torch_available
def test_torch_tensor_equality_comparator_equal_false_different_shape_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = TorchTensorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=torch.ones(2, 3),
            object2=torch.zeros(6),
            config=config,
        )
        assert caplog.messages[0].startswith("objects have different shapes:")


@torch_available
def test_torch_tensor_equality_comparator_equal_false_different_value(
    config: EqualityConfig,
) -> None:
    assert not TorchTensorEqualityComparator().equal(torch.ones(2, 3), torch.zeros(2, 3), config)


@torch_available
def test_torch_tensor_equality_comparator_equal_false_different_value_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = TorchTensorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=torch.ones(2, 3),
            object2=torch.zeros(2, 3),
            config=config,
        )
        assert caplog.messages[0].startswith("torch.Tensors have different elements:")


@torch_available
def test_torch_tensor_equality_comparator_equal_false_different_type(
    config: EqualityConfig,
) -> None:
    assert not TorchTensorEqualityComparator().equal(
        object1=torch.ones(2, 3), object2=42, config=config
    )


@torch_available
def test_torch_tensor_equality_comparator_equal_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = TorchTensorEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=torch.ones(2, 3),
            object2=42,
            config=config,
        )
        assert caplog.messages[0].startswith("objects have different types:")


@torch_available
@torch_cuda_available
def test_torch_tensor_equality_comparator_equal_false_different_device(
    config: EqualityConfig,
) -> None:
    assert not TorchTensorEqualityComparator().equal(
        torch.ones(2, 3, device=torch.device("cpu")),
        torch.zeros(2, 3, device=torch.device("cuda:0")),
        config,
    )


@torch_available
def test_torch_tensor_equality_comparator_equal_false_different_device_mock(
    config: EqualityConfig,
) -> None:
    assert not TorchTensorEqualityComparator().equal(
        Mock(spec=torch.Tensor, dtype=torch.float, shape=(2, 3), device=torch.device("cpu")),
        Mock(spec=torch.Tensor, dtype=torch.float, shape=(2, 3), device=torch.device("cuda:0")),
        config,
    )


@torch_available
def test_torch_tensor_equality_comparator_equal_nan_false(config: EqualityConfig) -> None:
    assert not TorchTensorEqualityComparator().equal(
        object1=torch.tensor([0.0, float("nan"), float("nan"), 1.2]),
        object2=torch.tensor([0.0, float("nan"), float("nan"), 1.2]),
        config=config,
    )


@torch_available
def test_torch_tensor_equality_comparator_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert TorchTensorEqualityComparator().equal(
        object1=torch.tensor([0.0, float("nan"), float("nan"), 1.2]),
        object2=torch.tensor([0.0, float("nan"), float("nan"), 1.2]),
        config=config,
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
