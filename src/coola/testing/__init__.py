r"""Provide ``pytest`` fixtures to skip or require tests based on the
availability of optional dependencies (e.g. NumPy, pandas, PyTorch).

Import a fixture and use it as a test decorator, e.g.
``@numpy_available`` skips the test unless NumPy is installed.
"""
