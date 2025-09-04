"""
TensorFlow callback for hyperparameter scheduling.

This callback is designed to work with TensorFlow/Keras training.
"""

from typing import Dict, Any, Optional, List
import tensorflow as tf
import time
import psutil

from .base import BaseCallback
from ..core.base import TrainingState


class TensorFlowCallback(BaseCallback):
    """
    TensorFlow callback for hyperparameter scheduling.
    
    This callback is designed to work with TensorFlow/Keras training
    and can be used as a Keras callback.
    """
    
    def __init__(
        self,
        hyperparameters: List[str],
        objectives: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        update_frequency: int = 1,
        log_level: str = "INFO",
        model: Optional[tf.keras.Model] = None,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None
    ):
        """
        Initialize the TensorFlow callback.
        
        Args:
            hyperparameters: List of hyperparameter names to schedule
            objectives: List of optimization objectives
            config: Configuration dictionary for the scheduler
            update_frequency: How often to update hyperparameters
            log_level: Logging level for the callback
            model: TensorFlow model (optional, can be set later)
            optimizer: TensorFlow optimizer (optional, can be set later)
        """
        super().__init__(hyperparameters, objectives, config, update_frequency, log_level)
        
        self.model = model
        self.optimizer = optimizer
        
        # Current state tracking
        self.current_loss = 0.0
        self.current_accuracy = 0.0
        self.current_learning_rate = 0.001
        self.current_batch_size = 32
        self.current_weight_decay = 0.0
        
        # Performance tracking
        self.epoch_start_time = time.time()
        self.batch_start_time = time.time()
    
    def set_model(self, model: tf.keras.Model) -> None:
        """
        Set the TensorFlow model.
        
        Args:
            model: TensorFlow model
        """
        self.model = model
    
    def set_optimizer(self, optimizer: tf.keras.optimizers.Optimizer) -> None:
        """
        Set the TensorFlow optimizer.
        
        Args:
            optimizer: TensorFlow optimizer
        """
        self.optimizer = optimizer
        if hasattr(optimizer, 'learning_rate'):
            self.current_learning_rate = optimizer.learning_rate.numpy()
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.logger.info(f"Starting epoch {epoch}")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each epoch."""
        epoch_time = time.time() - self.epoch_start_time
        
        # Update performance metrics
        if logs:
            self.current_performance.update(logs)
            
            # Track best performance
            for key, value in logs.items():
                if key not in self.best_performance or value > self.best_performance[key]:
                    self.best_performance[key] = value
        
        self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each batch."""
        self.current_step += 1
        self.batch_start_time = time.time()
        
        # Update hyperparameters if needed
        if self.current_step % self.update_frequency == 0:
            self._update_schedule()
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each batch."""
        batch_time = time.time() - self.batch_start_time
        
        # Update current metrics
        if logs:
            self.current_loss = logs.get('loss', self.current_loss)
            self.current_accuracy = logs.get('accuracy', self.current_accuracy)
    
    def _update_schedule(self) -> None:
        """Update the hyperparameter schedule."""
        # Compute gradient norm if model is available
        gradient_norm = 0.0
        if self.model is not None:
            try:
                # Get gradients from the model
                gradients = self.model.optimizer.get_gradients(
                    self.model.total_loss, self.model.trainable_weights
                )
                gradient_norm = tf.sqrt(sum(tf.reduce_sum(g ** 2) for g in gradients)).numpy()
            except:
                gradient_norm = 0.0
        
        # Create current training state
        current_state = TrainingState(
            epoch=self.current_epoch,
            step=self.current_step,
            loss=self.current_loss,
            accuracy=self.current_accuracy,
            learning_rate=self.current_learning_rate,
            batch_size=self.current_batch_size,
            gradient_norm=gradient_norm,
            validation_metrics=self.current_performance,
            training_time=time.time() - self.epoch_start_time,
            memory_usage=psutil.Process().memory_info().rss / 1024 / 1024  # MB
        )
        
        # Add to training history
        self.training_history.append(current_state)
        
        # Get new schedule
        new_schedule = self.update_hyperparameters(current_state)
        
        # Apply new schedule
        if new_schedule:
            self._apply_schedule(new_schedule)
    
    def _apply_schedule(self, schedule: Dict[str, float]) -> None:
        """
        Apply the new hyperparameter schedule.
        
        Args:
            schedule: Dictionary of new hyperparameter values
        """
        if 'learning_rate' in schedule and self.optimizer is not None:
            new_lr = schedule['learning_rate']
            self.optimizer.learning_rate.assign(new_lr)
            self.current_learning_rate = new_lr
            self.logger.info(f"Learning rate updated to: {new_lr}")
        
        if 'batch_size' in schedule:
            self.current_batch_size = int(schedule['batch_size'])
            self.logger.info(f"Batch size updated to: {self.current_batch_size}")
        
        if 'weight_decay' in schedule and self.optimizer is not None:
            new_wd = schedule['weight_decay']
            # Note: Weight decay in TensorFlow is typically handled by the optimizer
            # This would need to be implemented based on the specific optimizer
            self.current_weight_decay = new_wd
            self.logger.info(f"Weight decay updated to: {new_wd}")
    
    def get_current_hyperparameters(self) -> Dict[str, float]:
        """
        Get the current hyperparameter values.
        
        Returns:
            Dictionary of current hyperparameter values
        """
        return {
            'learning_rate': self.current_learning_rate,
            'batch_size': self.current_batch_size,
            'weight_decay': self.current_weight_decay
        }
    
    def set_current_metrics(self, loss: float, accuracy: float) -> None:
        """
        Set the current training metrics.
        
        Args:
            loss: Current loss value
            accuracy: Current accuracy value
        """
        self.current_loss = loss
        self.current_accuracy = accuracy
    
    def step(self) -> None:
        """
        Manually trigger a step update.
        
        This can be called in custom training loops to update hyperparameters.
        """
        self.current_step += 1
        if self.current_step % self.update_frequency == 0:
            self._update_schedule()
    
    def update_metrics(self, **metrics) -> None:
        """
        Update training metrics.
        
        Args:
            **metrics: Metric name-value pairs
        """
        if 'loss' in metrics:
            self.current_loss = metrics['loss']
        if 'accuracy' in metrics:
            self.current_accuracy = metrics['accuracy']
        
        # Update performance tracking
        self.current_performance.update(metrics)
    
    def compute_gradient_norm(self) -> float:
        """
        Compute the gradient norm of the model.
        
        Returns:
            Gradient norm value
        """
        if self.model is None:
            return 0.0
        
        try:
            # Get gradients from the model
            gradients = self.model.optimizer.get_gradients(
                self.model.total_loss, self.model.trainable_weights
            )
            return tf.sqrt(sum(tf.reduce_sum(g ** 2) for g in gradients)).numpy()
        except:
            return 0.0
    
    def get_optimizer_state(self) -> Dict[str, Any]:
        """
        Get the current optimizer state.
        
        Returns:
            Dictionary with optimizer state information
        """
        if self.optimizer is None:
            return {}
        
        state = {
            'learning_rate': self.optimizer.learning_rate.numpy(),
            'optimizer_type': type(self.optimizer).__name__
        }
        
        # Add optimizer-specific state
        if hasattr(self.optimizer, 'momentum'):
            state['momentum'] = self.optimizer.momentum.numpy()
        if hasattr(self.optimizer, 'beta_1'):
            state['beta_1'] = self.optimizer.beta_1.numpy()
        if hasattr(self.optimizer, 'beta_2'):
            state['beta_2'] = self.optimizer.beta_2.numpy()
        
        return state
