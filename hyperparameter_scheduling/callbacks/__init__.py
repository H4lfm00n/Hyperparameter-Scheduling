"""
Callback system for easy integration with any deep learning framework.

This module provides callback interfaces that can be used with PyTorch Lightning,
TensorFlow/Keras, or any custom training loop.
"""

from .base import BaseCallback
from .pytorch import PyTorchCallback
from .generic import GenericCallback

# Lazy import for TensorFlow to avoid segfaults on macOS
def _get_tensorflow_callback():
    """Lazy import for TensorFlowCallback."""
    from .tensorflow import TensorFlowCallback
    return TensorFlowCallback

# Make TensorFlowCallback available but lazy-loaded
def __getattr__(name):
    if name == 'TensorFlowCallback':
        return _get_tensorflow_callback()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'BaseCallback',
    'PyTorchCallback', 
    'TensorFlowCallback',
    'GenericCallback'
]


