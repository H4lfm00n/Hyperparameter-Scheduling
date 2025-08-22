"""
Base classes for hyperparameter scheduling.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum


class ObjectiveType(Enum):
    """Types of optimization objectives."""
    CONVERGENCE_SPEED = "convergence_speed"
    FINAL_ACCURACY = "final_accuracy"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"
    GENERALIZATION = "generalization"
    STABILITY = "stability"


@dataclass
class HyperparameterConfig:
    """Configuration for a hyperparameter."""
    name: str
    min_value: float
    max_value: float
    default_value: float
    scale: str = "log"  # 'linear' or 'log'
    constraints: Optional[List[str]] = None


@dataclass
class TrainingState:
    """Current state of training."""
    epoch: int
    step: int
    loss: float
    accuracy: float
    learning_rate: float
    batch_size: int
    gradient_norm: float
    validation_metrics: Dict[str, float]
    training_time: float
    memory_usage: float


@dataclass
class ScheduleDecision:
    """Decision made by the scheduler."""
    hyperparameters: Dict[str, float]
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]


class BaseScheduler(ABC):
    """
    Base class for all hyperparameter schedulers.
    
    This class defines the interface that all schedulers must implement.
    """
    
    def __init__(
        self,
        hyperparameters: List[str],
        objectives: List[ObjectiveType],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base scheduler.
        
        Args:
            hyperparameters: List of hyperparameter names to schedule
            objectives: List of optimization objectives
            config: Additional configuration parameters
        """
        self.hyperparameters = hyperparameters
        self.objectives = objectives
        self.config = config or {}
        self.history = []
        self.current_state = None
        
    @abstractmethod
    def get_schedule(
        self, 
        current_state: TrainingState,
        history: List[TrainingState]
    ) -> ScheduleDecision:
        """
        Get the next hyperparameter schedule based on current state and history.
        
        Args:
            current_state: Current training state
            history: Historical training states
            
        Returns:
            ScheduleDecision with the next hyperparameter values
        """
        pass
    
    @abstractmethod
    def update(
        self,
        state: TrainingState,
        performance: Dict[str, float]
    ) -> None:
        """
        Update the scheduler with new training state and performance.
        
        Args:
            state: Current training state
            performance: Performance metrics
        """
        pass
    
    @abstractmethod
    def fit(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        epochs: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fit the scheduler to a model and dataset.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            **kwargs: Additional arguments
            
        Returns:
            Training results and learned schedule
        """
        pass
    
    def save(self, path: str) -> None:
        """Save the scheduler state."""
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        """Load the scheduler state."""
        raise NotImplementedError
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the scheduler's performance."""
        return {
            "hyperparameters": self.hyperparameters,
            "objectives": [obj.value for obj in self.objectives],
            "history_length": len(self.history),
            "config": self.config
        }


class FixedScheduler(BaseScheduler):
    """
    Simple scheduler that maintains fixed hyperparameter values.
    Useful for baseline comparisons.
    """
    
    def __init__(
        self,
        hyperparameters: List[str],
        fixed_values: Dict[str, float],
        objectives: List[ObjectiveType]
    ):
        super().__init__(hyperparameters, objectives)
        self.fixed_values = fixed_values
    
    def get_schedule(
        self, 
        current_state: TrainingState,
        history: List[TrainingState]
    ) -> ScheduleDecision:
        return ScheduleDecision(
            hyperparameters=self.fixed_values,
            confidence=1.0,
            reasoning="Fixed schedule",
            metadata={}
        )
    
    def update(
        self,
        state: TrainingState,
        performance: Dict[str, float]
    ) -> None:
        self.history.append(state)
    
    def fit(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        epochs: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        # Implementation would go here
        return {"status": "completed", "schedule": self.fixed_values}
