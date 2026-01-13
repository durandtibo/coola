# Code Examples

This page provides practical examples of using `coola` in real-world scenarios.

## Basic Examples

### Comparing Configuration Dictionaries

```python
import yaml
from coola import objects_are_equal

# Load configuration files
with open("config1.yaml") as f:
    config1 = yaml.safe_load(f)

with open("config2.yaml") as f:
    config2 = yaml.safe_load(f)

# Compare configurations
if objects_are_equal(config1, config2, show_difference=True):
    print("Configurations are identical")
else:
    print("Configurations differ")
```

### Validating Model Outputs

```python
import torch
from coola import objects_are_allclose


def test_model_inference():
    model = load_model()
    test_input = torch.randn(1, 3, 224, 224)

    # Get actual output
    actual_output = model(test_input)

    # Load expected output
    expected_output = torch.load("expected_output.pt")

    # Compare with tolerance
    assert objects_are_allclose(
        actual_output, expected_output, atol=1e-5, rtol=1e-4, show_difference=True
    ), "Model output differs from expected"
```

## Machine Learning Examples

### Comparing Training Checkpoints

```python
import torch
from coola import objects_are_equal


def compare_checkpoints(checkpoint1_path, checkpoint2_path):
    """Compare two PyTorch checkpoint files."""
    checkpoint1 = torch.load(checkpoint1_path)
    checkpoint2 = torch.load(checkpoint2_path)

    # Compare model state dicts
    if not objects_are_equal(
        checkpoint1["model_state_dict"], checkpoint2["model_state_dict"]
    ):
        print("Model state dicts differ")
        return False

    # Compare optimizer state dicts
    if not objects_are_equal(
        checkpoint1["optimizer_state_dict"], checkpoint2["optimizer_state_dict"]
    ):
        print("Optimizer state dicts differ")
        return False

    # Compare other metadata
    metadata_keys = ["epoch", "loss", "accuracy"]
    for key in metadata_keys:
        if key in checkpoint1 and key in checkpoint2:
            if checkpoint1[key] != checkpoint2[key]:
                print(f"Metadata '{key}' differs")
                return False

    return True
```

### Validating Data Preprocessing

```python
import numpy as np
from coola import objects_are_allclose


def test_preprocessing_pipeline():
    """Test that preprocessing is deterministic."""
    # Sample data
    raw_data = load_raw_data()

    # Process twice
    processed_1 = preprocessing_pipeline(raw_data, seed=42)
    processed_2 = preprocessing_pipeline(raw_data, seed=42)

    # Should be identical when using same seed
    assert objects_are_allclose(
        processed_1, processed_2, equal_nan=True
    ), "Preprocessing is not deterministic"
```

### Comparing Model Predictions

```python
import torch
from coola import objects_are_allclose


def compare_model_versions(model_v1, model_v2, test_data):
    """Compare predictions from two model versions."""
    model_v1.eval()
    model_v2.eval()

    differences = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_data):
            pred_v1 = model_v1(inputs)
            pred_v2 = model_v2(inputs)

            # Check if predictions are close
            if not objects_are_allclose(pred_v1, pred_v2, atol=1e-4):
                differences.append(
                    {
                        "batch_idx": batch_idx,
                        "max_diff": (pred_v1 - pred_v2).abs().max().item(),
                    }
                )

    return differences
```

## Data Science Examples

### Comparing DataFrames

```python
import pandas as pd
from coola import objects_are_equal


def compare_dataframes(df1, df2, ignore_index=False):
    """Compare two pandas DataFrames."""
    if ignore_index:
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)

    return objects_are_equal(df1, df2, show_difference=True)


# Example usage
df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
df2 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

if compare_dataframes(df1, df2):
    print("DataFrames are equal")
```

### Validating Data Transformations

```python
import numpy as np
import pandas as pd
from coola import objects_are_allclose


def test_data_transformation():
    """Test that data transformation preserves statistical properties."""
    original = pd.DataFrame({"values": np.random.randn(1000)})

    # Transform data
    transformed = transform_data(original)

    # Transform back
    restored = inverse_transform_data(transformed)

    # Check if original and restored are close
    assert objects_are_allclose(
        original.values, restored.values, atol=1e-6, rtol=1e-5
    ), "Transformation is not reversible"
```

### Comparing Time Series Data

```python
import pandas as pd
import numpy as np
from coola import objects_are_allclose


def compare_time_series(series1, series2, tolerance=1e-6):
    """Compare two time series with tolerance for floating-point errors."""
    # Ensure same index
    if not series1.index.equals(series2.index):
        print("Time series have different indices")
        return False

    # Compare values with tolerance
    return objects_are_allclose(
        series1.values,
        series2.values,
        atol=tolerance,
        equal_nan=True,  # Treat NaN as equal
    )
```

## Testing Examples

### pytest Integration

```python
import pytest
import torch
from coola import objects_are_equal, objects_are_allclose


@pytest.fixture
def sample_tensor():
    return torch.randn(10, 10)


def test_tensor_transformation(sample_tensor):
    """Test that transformation produces expected output."""
    result = my_transformation(sample_tensor)
    expected = load_expected_result("transformation_output.pt")

    assert objects_are_allclose(result, expected, atol=1e-6, show_difference=True)


def test_data_equality(sample_tensor):
    """Test data equality with coola."""
    processed = process_data(sample_tensor)

    # Load expected structure
    expected = {
        "data": torch.zeros(10, 10),
        "metadata": {"shape": (10, 10), "dtype": "float32"},
    }

    assert objects_are_equal(processed, expected, show_difference=True)
```

### Unittest Integration

```python
import unittest
import numpy as np
from coola import objects_are_equal, objects_are_allclose


class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.test_data = np.random.randn(100, 100)

    def test_data_normalization(self):
        """Test data normalization."""
        normalized = normalize_data(self.test_data)

        # Check that mean is close to 0 and std is close to 1
        self.assertTrue(objects_are_allclose(normalized.mean(), 0.0, atol=1e-6))
        self.assertTrue(objects_are_allclose(normalized.std(), 1.0, atol=1e-6))

    def test_data_structure(self):
        """Test that data structure matches expected."""
        result = create_data_structure(self.test_data)
        expected = {
            "data": self.test_data,
            "mean": self.test_data.mean(),
            "std": self.test_data.std(),
        }

        self.assertTrue(objects_are_equal(result, expected, show_difference=True))
```

## Advanced Examples

### Custom Comparator for Custom Classes

```python
from typing import Any
from coola import objects_are_equal
from coola.equality.config import EqualityConfig2
from coola.equality.comparators import BaseEqualityComparator
from coola.equality.testers import EqualityTester


class Vector3D:
    """A simple 3D vector class."""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Vector3DComparator(BaseEqualityComparator):
    """Custom comparator for Vector3D objects."""

    def clone(self):
        return self.__class__()

    def equal(self, actual: Vector3D, expected: Any, config: EqualityConfig2) -> bool:
        if not isinstance(expected, Vector3D):
            if config.show_difference:
                print(f"Types differ: {type(actual)} vs {type(expected)}")
            return False

        # Compare components
        equal = (
            actual.x == expected.x and actual.y == expected.y and actual.z == expected.z
        )

        if not equal and config.show_difference:
            print(
                f"Vectors differ: ({actual.x}, {actual.y}, {actual.z}) "
                f"vs ({expected.x}, {expected.y}, {expected.z})"
            )

        return equal


# Register the comparator
tester = EqualityTester.local_copy()
tester.add_comparator(Vector3D, Vector3DComparator())

# Use it
v1 = Vector3D(1, 2, 3)
v2 = Vector3D(1, 2, 3)
v3 = Vector3D(1, 2, 4)

print(objects_are_equal(v1, v2, tester=tester))  # True
print(objects_are_equal(v1, v3, tester=tester, show_difference=True))  # False
```

### Comparing Nested Complex Structures

```python
import torch
import numpy as np
import pandas as pd
from coola import objects_are_equal


def compare_ml_experiment_results(result1, result2):
    """Compare complete ML experiment results."""

    experiment1 = {
        "config": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam",
        },
        "metrics": {
            "train": pd.DataFrame(
                {"loss": [0.5, 0.4, 0.3], "accuracy": [0.8, 0.85, 0.9]}
            ),
            "val": pd.DataFrame(
                {"loss": [0.6, 0.5, 0.4], "accuracy": [0.75, 0.8, 0.85]}
            ),
        },
        "model_weights": {
            "layer1": torch.randn(100, 50),
            "layer2": torch.randn(50, 10),
        },
        "predictions": {
            "test": np.random.rand(100, 10),
            "metadata": {"num_samples": 100, "num_classes": 10},
        },
    }

    experiment2 = {
        # Similar structure
        "config": experiment1["config"].copy(),
        "metrics": {
            "train": experiment1["metrics"]["train"].copy(),
            "val": experiment1["metrics"]["val"].copy(),
        },
        "model_weights": {
            "layer1": experiment1["model_weights"]["layer1"].clone(),
            "layer2": experiment1["model_weights"]["layer2"].clone(),
        },
        "predictions": {
            "test": experiment1["predictions"]["test"].copy(),
            "metadata": experiment1["predictions"]["metadata"].copy(),
        },
    }

    return objects_are_equal(experiment1, experiment2, show_difference=True)
```

### Conditional Comparison Based on Type

```python
import torch
import numpy as np
from coola import objects_are_equal, objects_are_allclose


def smart_compare(obj1, obj2, numeric_tolerance=1e-6):
    """Smart comparison that uses appropriate method based on type."""

    # For numeric types, use allclose
    if isinstance(obj1, (torch.Tensor, np.ndarray)):
        return objects_are_allclose(
            obj1, obj2, atol=numeric_tolerance, rtol=numeric_tolerance
        )

    # For dictionaries, check each value
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        if obj1.keys() != obj2.keys():
            return False

        return all(
            smart_compare(obj1[k], obj2[k], numeric_tolerance) for k in obj1.keys()
        )

    # For lists/tuples, check each element
    if isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
        if len(obj1) != len(obj2):
            return False

        return all(smart_compare(a, b, numeric_tolerance) for a, b in zip(obj1, obj2))

    # For other types, use exact equality
    return objects_are_equal(obj1, obj2)
```

## Integration Examples

### With Logging

```python
import logging
from coola import objects_are_equal

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def compare_with_logging(obj1, obj2, context=""):
    """Compare objects with detailed logging."""
    logger = logging.getLogger(__name__)

    logger.info(f"Starting comparison{f' for {context}' if context else ''}")

    result = objects_are_equal(obj1, obj2, show_difference=True)

    if result:
        logger.info(f"Objects are equal{f' ({context})' if context else ''}")
    else:
        logger.warning(f"Objects differ{f' ({context})' if context else ''}")

    return result
```

### With Context Managers

```python
import time
from contextlib import contextmanager
from coola import objects_are_equal


@contextmanager
def timed_comparison(description=""):
    """Time the comparison operation."""
    start = time.time()
    print(f"Starting comparison{f': {description}' if description else ''}...")
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"Comparison completed in {elapsed:.4f} seconds")


# Usage
with timed_comparison("model checkpoints"):
    result = objects_are_equal(checkpoint1, checkpoint2, show_difference=True)
```

## Best Practices

### 2. Use Appropriate Tolerance for Numerical Data

```python
from coola import objects_are_allclose

# For single-precision floats
objects_are_allclose(float32_data1, float32_data2, atol=1e-6, rtol=1e-5)

# For double-precision floats
objects_are_allclose(float64_data1, float64_data2, atol=1e-12, rtol=1e-10)
```

### 3. Handle NaN Values Explicitly

```python
from coola import objects_are_allclose

# Explicitly decide how to handle NaN
result = objects_are_allclose(
    data1,
    data2,
    equal_nan=True,  # or False, depending on requirements
    show_difference=True,
)
```

### 4. Compare Metadata Before Large Data

```python
from coola import objects_are_equal


def efficient_compare(obj1, obj2):
    # Quick checks first
    if obj1.metadata != obj2.metadata:
        return False

    # Expensive comparison last
    return objects_are_equal(obj1.data, obj2.data)
```

## See Also

- [Quickstart Guide](quickstart.md) - Basic usage examples
- [FAQ](faq.md) - Frequently asked questions
