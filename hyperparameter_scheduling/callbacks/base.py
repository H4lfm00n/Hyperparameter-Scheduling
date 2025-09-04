"""
Base callback class for hyperparameter scheduling.

This defines the interface that all framework-specific callbacks must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging

from ..core.scheduler import AutoScheduler
from ..core.base import TrainingState, ObjectiveType


class BaseCallback(ABC):
    """
    Base callback class for hyperparameter scheduling.
    
    This class defines the interface that all framework-specific callbacks
    must implement. It provides a common interface for integrating the
    hyperparameter scheduler with any deep learning framework.
    """
    
    def __init__(
        self,
        hyperparameters: List[str],
        objectives: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        update_frequency: int = 1,
        log_level: str = "INFO"
    ):
        """
        Initialize the base callback.
        
        Args:
            hyperparameters: List of hyperparameter names to schedule
            objectives: List of optimization objectives (optional)
            config: Configuration dictionary for the scheduler
            update_frequency: How often to update hyperparameters (every N steps)
            log_level: Logging level for the callback
        """
        self.hyperparameters = hyperparameters
        self.objectives = objectives or ['convergence_speed', 'final_accuracy']
        self.config = config or {}
        self.update_frequency = update_frequency
        
        # Initialize scheduler
        self.scheduler = AutoScheduler(
            hyperparameters=hyperparameters,
            objectives=[ObjectiveType(obj) for obj in self.objectives],
            config=config
        )
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # State tracking
        self.current_epoch = 0
        self.current_step = 0
        self.training_history = []
        self.schedule_history = []
        
        # Performance tracking
        self.best_performance = {}
        self.current_performance = {}
    
    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the beginning of each epoch.
        
        Args:
            epoch: Current epoch number
            logs: Additional logging information
        """
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of each epoch.
        
        Args:
            epoch: Current epoch number
            logs: Additional logging information
        """
        pass
    
    @abstractmethod
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the beginning of each batch.
        
        Args:
            batch: Current batch number
            logs: Additional logging information
        """
        pass
    
    @abstractmethod
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of each batch.
        
        Args:
            batch: Current batch number
            logs: Additional logging information
        """
        pass
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the beginning of training.
        
        Args:
            logs: Additional logging information
        """
        self.logger.info("Starting hyperparameter scheduling")
        self.current_epoch = 0
        self.current_step = 0
        self.training_history = []
        self.schedule_history = []
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of training.
        
        Args:
            logs: Additional logging information
        """
        self.logger.info("Training completed")
        self.logger.info(f"Best performance: {self.best_performance}")
        self.logger.info(f"Schedule changes: {len(self.schedule_history)}")
    
    def update_hyperparameters(self, current_state: TrainingState) -> Dict[str, float]:
        """
        Update hyperparameters using the scheduler.
        
        Args:
            current_state: Current training state
            
        Returns:
            Dictionary of new hyperparameter values
        """
        try:
            # Get schedule decision
            schedule_decision = self.scheduler.get_schedule(
                current_state=current_state,
                history=self.training_history
            )
            
            # Extract new hyperparameter values
            new_schedule = schedule_decision.schedule
            
            # Log the change
            self.logger.debug(f"New schedule: {new_schedule}")
            
            # Store schedule history
            self.schedule_history.append({
                'epoch': current_state.epoch,
                'step': current_state.step,
                'schedule': new_schedule,
                'confidence': schedule_decision.confidence,
                'reasoning': schedule_decision.reasoning
            })
            
            return new_schedule
            
        except Exception as e:
            self.logger.error(f"Failed to update hyperparameters: {e}")
            return {}
    
    def get_current_schedule(self) -> Dict[str, float]:
        """
        Get the current hyperparameter schedule.
        
        Returns:
            Dictionary of current hyperparameter values
        """
        if self.schedule_history:
            return self.schedule_history[-1]['schedule']
        return {}
    
    def get_schedule_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete schedule history.
        
        Returns:
            List of schedule decisions
        """
        return self.schedule_history
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of training performance.
        
        Returns:
            Dictionary with performance summary
        """
        return {
            'best_performance': self.best_performance,
            'current_performance': self.current_performance,
            'schedule_changes': len(self.schedule_history),
            'training_history_length': len(self.training_history)
        }
    
    def save_scheduler_state(self, filepath: str) -> None:
        """
        Save the scheduler state to a file.
        
        Args:
            filepath: Path to save the scheduler state
        """
        try:
            self.scheduler.save(filepath)
            self.logger.info(f"Scheduler state saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save scheduler state: {e}")
    
    def load_scheduler_state(self, filepath: str) -> None:
        """
        Load the scheduler state from a file.
        
        Args:
            filepath: Path to load the scheduler state from
        """
        try:
            self.scheduler.load(filepath)
            self.logger.info(f"Scheduler state loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load scheduler state: {e}")
    
    def set_log_level(self, level: str) -> None:
        """
        Set the logging level for the callback.
        
        Args:
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        self.logger.setLevel(getattr(logging, level.upper()))
    
    def get_scheduler(self) -> AutoScheduler:
        """
        Get the underlying scheduler instance.
        
        Returns:
            AutoScheduler instance
        """
        return self.scheduler


