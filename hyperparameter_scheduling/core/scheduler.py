"""
Main AutoScheduler class that orchestrates automatic hyperparameter scheduling.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base import (
    BaseScheduler, 
    TrainingState, 
    ScheduleDecision, 
    ObjectiveType,
    HyperparameterConfig
)
from ..dynamics.analyzer import TrainingDynamicsAnalyzer
from ..learners.meta_learner import MetaLearner
from ..transfer.transfer_learner import TransferLearner
from ..optimizers.multi_objective import MultiObjectiveOptimizer
from ..utils.metrics import compute_training_metrics


class AutoScheduler(BaseScheduler):
    """
    Main automatic hyperparameter scheduler that learns optimal schedules
    based on training dynamics and generalizes across similar problems.
    """
    
    def __init__(
        self,
        hyperparameters: List[str],
        objectives: List[Union[str, ObjectiveType]],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the AutoScheduler.
        
        Args:
            hyperparameters: List of hyperparameter names to schedule
            objectives: List of optimization objectives
            config: Configuration dictionary
        """
        # Convert string objectives to ObjectiveType
        obj_types = []
        for obj in objectives:
            if isinstance(obj, str):
                obj_types.append(ObjectiveType(obj))
            else:
                obj_types.append(obj)
        
        super().__init__(hyperparameters, obj_types, config)
        
        # Initialize components
        self.dynamics_analyzer = TrainingDynamicsAnalyzer()
        self.meta_learner = MetaLearner(
            hyperparameters=hyperparameters,
            objectives=obj_types,
            config=self.config.get("meta_learner", {})
        )
        self.transfer_learner = TransferLearner(
            config=self.config.get("transfer_learner", {})
        )
        self.multi_objective_optimizer = MultiObjectiveOptimizer(
            objectives=obj_types,
            config=self.config.get("multi_objective", {})
        )
        
        # Training state tracking
        self.current_epoch = 0
        self.current_step = 0
        self.best_performance = {}
        self.schedule_history = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
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
        # Analyze training dynamics
        dynamics_features = self.dynamics_analyzer.extract_features(
            current_state, history
        )
        
        # Get meta-learner prediction
        meta_prediction = self.meta_learner.predict(
            dynamics_features, current_state, history
        )
        
        # Apply transfer learning if applicable
        if self.transfer_learner.has_similar_problems():
            transfer_adjustment = self.transfer_learner.get_adjustment(
                dynamics_features, current_state
            )
            # Combine meta prediction with transfer adjustment
            final_prediction = self._combine_predictions(
                meta_prediction, transfer_adjustment
            )
        else:
            final_prediction = meta_prediction
        
        # Optimize for multiple objectives
        optimized_schedule = self.multi_objective_optimizer.optimize(
            final_prediction, current_state, history
        )
        
        # Compute confidence and reasoning
        confidence = self._compute_confidence(
            dynamics_features, meta_prediction, optimized_schedule
        )
        reasoning = self._generate_reasoning(
            dynamics_features, meta_prediction, optimized_schedule
        )
        
        return ScheduleDecision(
            hyperparameters=optimized_schedule,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "dynamics_features": dynamics_features,
                "meta_prediction": meta_prediction,
                "transfer_adjustment": transfer_adjustment if self.transfer_learner.has_similar_problems() else None
            }
        )
    
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
        # Update history
        self.history.append(state)
        self.current_state = state
        
        # Update best performance tracking
        for metric, value in performance.items():
            if metric not in self.best_performance or value > self.best_performance[metric]:
                self.best_performance[metric] = value
        
        # Update meta-learner
        self.meta_learner.update(state, performance)
        
        # Update transfer learner
        self.transfer_learner.update(state, performance)
        
        # Log progress
        self.logger.info(
            f"Epoch {state.epoch}, Step {state.step}: "
            f"Loss={state.loss:.4f}, Accuracy={state.accuracy:.4f}, "
            f"LR={state.learning_rate:.6f}"
        )
    
    def fit(
        self,
        model: nn.Module,
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
        self.logger.info("Starting automatic hyperparameter scheduling...")
        
        # Initialize training
        device = next(model.parameters()).device
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Get current training state
            current_state = self._get_current_state(model, train_loader, val_loader)
            
            # Get schedule for this epoch
            schedule_decision = self.get_schedule(current_state, self.history)
            
            # Apply schedule
            self._apply_schedule(model, optimizer, schedule_decision.hyperparameters)
            
            # Train for one epoch
            epoch_loss, epoch_accuracy = self._train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            
            # Evaluate on validation set
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._evaluate_model(model, val_loader, criterion, device)
            
            # Update scheduler
            performance = {
                "train_loss": epoch_loss,
                "train_accuracy": epoch_accuracy,
                **val_metrics
            }
            self.update(current_state, performance)
            
            # Store schedule decision
            self.schedule_history.append(schedule_decision)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss={epoch_loss:.4f}, Train Acc={epoch_accuracy:.4f}, "
                f"Schedule Confidence={schedule_decision.confidence:.3f}"
            )
        
        # Return results
        return {
            "final_model": model,
            "schedule_history": self.schedule_history,
            "training_history": self.history,
            "best_performance": self.best_performance,
            "meta_learner_state": self.meta_learner.get_state(),
            "transfer_learner_state": self.transfer_learner.get_state()
        }
    
    def _get_current_state(
        self, 
        model: nn.Module, 
        train_loader: Any, 
        val_loader: Optional[Any]
    ) -> TrainingState:
        """Get current training state."""
        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        gradient_norm = total_norm ** (1. / 2)
        
        # Get current hyperparameters (use defaults for now)
        current_lr = 0.001  # Default learning rate
        current_batch_size = train_loader.batch_size if hasattr(train_loader, 'batch_size') else 32
        
        # Get validation metrics
        val_metrics = {}
        if val_loader is not None:
            device = next(model.parameters()).device
            val_metrics = self._evaluate_model(model, val_loader, nn.CrossEntropyLoss(), device)
        
        return TrainingState(
            epoch=self.current_epoch,
            step=self.current_step,
            loss=0.0,  # Will be updated after training
            accuracy=0.0,  # Will be updated after training
            learning_rate=current_lr,
            batch_size=current_batch_size,
            gradient_norm=gradient_norm,
            validation_metrics=val_metrics,
            training_time=0.0,
            memory_usage=0.0
        )
    
    def _apply_schedule(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        hyperparameters: Dict[str, float]
    ) -> None:
        """Apply hyperparameter schedule to model and optimizer."""
        if 'learning_rate' in hyperparameters:
            for param_group in optimizer.param_groups:
                param_group['lr'] = hyperparameters['learning_rate']
        
        if 'weight_decay' in hyperparameters:
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = hyperparameters['weight_decay']
        
        # Note: Batch size changes would require recreating the data loader
        # This is a simplified implementation
    
    def _train_epoch(
        self, 
        model: nn.Module, 
        train_loader: Any, 
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module, 
        device: torch.device
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Handle both classification and regression
            if len(output.shape) > 1 and output.shape[1] > 1:
                # Classification task
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                # Regression task - use MSE as accuracy proxy
                mse = ((output.squeeze() - target) ** 2).mean().item()
                correct += (1.0 / (1.0 + mse)) * target.size(0)  # Convert MSE to accuracy-like metric
            
            total += target.size(0)
            self.current_step += 1
        
        return total_loss / len(train_loader), correct / total
    
    def _evaluate_model(
        self, 
        model: nn.Module, 
        val_loader: Any, 
        criterion: nn.Module, 
        device: torch.device
    ) -> Dict[str, float]:
        """Evaluate model on validation set."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                total_loss += criterion(output, target).item()
                
                # Handle both classification and regression
                if len(output.shape) > 1 and output.shape[1] > 1:
                    # Classification task
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                else:
                    # Regression task - use MSE as accuracy proxy
                    mse = ((output.squeeze() - target) ** 2).mean().item()
                    correct += (1.0 / (1.0 + mse)) * target.size(0)  # Convert MSE to accuracy-like metric
                
                total += target.size(0)
        
        return {
            "val_loss": total_loss / len(val_loader),
            "val_accuracy": correct / total
        }
    
    def _combine_predictions(
        self, 
        meta_prediction: Dict[str, float], 
        transfer_adjustment: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine meta-learner prediction with transfer adjustment."""
        combined = {}
        for param in self.hyperparameters:
            if param in meta_prediction and param in transfer_adjustment:
                # Weighted combination
                weight = self.config.get("transfer_weight", 0.3)
                combined[param] = (
                    (1 - weight) * meta_prediction[param] + 
                    weight * transfer_adjustment[param]
                )
            elif param in meta_prediction:
                combined[param] = meta_prediction[param]
            elif param in transfer_adjustment:
                combined[param] = transfer_adjustment[param]
        
        return combined
    
    def _compute_confidence(
        self, 
        dynamics_features: Dict[str, float], 
        meta_prediction: Dict[str, float], 
        optimized_schedule: Dict[str, float]
    ) -> float:
        """Compute confidence in the schedule decision."""
        # Simple confidence based on feature stability
        feature_stability = np.mean([
            abs(v) for v in dynamics_features.values() 
            if isinstance(v, (int, float))
        ])
        
        # Normalize to [0, 1] range
        confidence = 1.0 / (1.0 + feature_stability)
        
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_reasoning(
        self, 
        dynamics_features: Dict[str, float], 
        meta_prediction: Dict[str, float], 
        optimized_schedule: Dict[str, float]
    ) -> str:
        """Generate human-readable reasoning for the schedule decision."""
        reasoning_parts = []
        
        # Analyze gradient norm
        if "gradient_norm" in dynamics_features:
            grad_norm = dynamics_features["gradient_norm"]
            if grad_norm > 1.0:
                reasoning_parts.append("High gradient norm detected, reducing learning rate")
            elif grad_norm < 0.1:
                reasoning_parts.append("Low gradient norm detected, increasing learning rate")
        
        # Analyze loss trend
        if "loss_trend" in dynamics_features:
            loss_trend = dynamics_features["loss_trend"]
            if loss_trend > 0:
                reasoning_parts.append("Loss increasing, adjusting schedule for stability")
            elif loss_trend < -0.1:
                reasoning_parts.append("Loss decreasing rapidly, maintaining current schedule")
        
        if not reasoning_parts:
            reasoning_parts.append("Maintaining current schedule based on stable dynamics")
        
        return "; ".join(reasoning_parts)
    
    def save(self, path: str) -> None:
        """Save the scheduler state."""
        import pickle
        state = {
            "meta_learner": self.meta_learner.get_state(),
            "transfer_learner": self.transfer_learner.get_state(),
            "history": self.history,
            "schedule_history": self.schedule_history,
            "best_performance": self.best_performance,
            "config": self.config
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str) -> None:
        """Load the scheduler state."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.meta_learner.load_state(state["meta_learner"])
        self.transfer_learner.load_state(state["transfer_learner"])
        self.history = state["history"]
        self.schedule_history = state["schedule_history"]
        self.best_performance = state["best_performance"]
        self.config.update(state["config"])
