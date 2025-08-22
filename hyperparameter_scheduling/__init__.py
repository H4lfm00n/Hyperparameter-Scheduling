"""
Automatic Hyperparameter Scheduling Library

A sophisticated library that learns optimal hyperparameter schedules based on training dynamics
and generalizes across similar problems.
"""

from .core.scheduler import AutoScheduler
from .core.base import BaseScheduler
from .dynamics.analyzer import TrainingDynamicsAnalyzer
from .learners.meta_learner import MetaLearner
from .transfer.transfer_learner import TransferLearner
from .optimizers.multi_objective import MultiObjectiveOptimizer

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "AutoScheduler",
    "BaseScheduler", 
    "TrainingDynamicsAnalyzer",
    "MetaLearner",
    "TransferLearner",
    "MultiObjectiveOptimizer",
]
