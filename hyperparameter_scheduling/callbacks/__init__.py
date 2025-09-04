"""
Callback system for easy integration with any deep learning framework.

This module provides callback interfaces that can be used with PyTorch Lightning,
TensorFlow/Keras, or any custom training loop.
"""

from .base import BaseCallback
from .pytorch import PyTorchCallback
from .tensorflow import TensorFlowCallback
from .generic import GenericCallback

__all__ = [
    'BaseCallback',
    'PyTorchCallback', 
    'TensorFlowCallback',
    'GenericCallback'
]


