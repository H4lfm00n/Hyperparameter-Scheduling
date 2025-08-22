"""
Multi-objective optimizer for balancing multiple objectives in hyperparameter scheduling.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.optimize import minimize, differential_evolution
from sklearn.preprocessing import StandardScaler
import torch

from ..core.base import TrainingState, ObjectiveType


class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer that balances multiple objectives when
    optimizing hyperparameter schedules.
    
    This component uses various optimization techniques to find Pareto-optimal
    solutions that balance different objectives like convergence speed,
    final accuracy, and computational efficiency.
    """
    
    def __init__(
        self,
        objectives: List[ObjectiveType],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the multi-objective optimizer.
        
        Args:
            objectives: List of optimization objectives
            config: Configuration dictionary
        """
        self.objectives = objectives
        self.config = config or {}
        
        # Objective weights (can be learned or set manually)
        self.objective_weights = self._initialize_weights()
        
        # Optimization parameters
        self.optimization_method = self.config.get("method", "weighted_sum")
        self.max_iterations = self.config.get("max_iterations", 100)
        self.tolerance = self.config.get("tolerance", 1e-6)
        
        # Constraint handling
        self.constraints = self.config.get("constraints", {})
        self.penalty_factor = self.config.get("penalty_factor", 1000.0)
        
        # Pareto front tracking
        self.pareto_front = []
        self.pareto_history = []
        
    def optimize(
        self,
        initial_schedule: Dict[str, float],
        current_state: TrainingState,
        history: List[TrainingState]
    ) -> Dict[str, float]:
        """
        Optimize hyperparameter schedule for multiple objectives.
        
        Args:
            initial_schedule: Initial hyperparameter schedule
            current_state: Current training state
            history: Historical training states
            
        Returns:
            Optimized hyperparameter schedule
        """
        if self.optimization_method == "weighted_sum":
            return self._weighted_sum_optimization(initial_schedule, current_state, history)
        elif self.optimization_method == "pareto_front":
            return self._pareto_front_optimization(initial_schedule, current_state, history)
        elif self.optimization_method == "evolutionary":
            return self._evolutionary_optimization(initial_schedule, current_state, history)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
    
    def _weighted_sum_optimization(
        self,
        initial_schedule: Dict[str, float],
        current_state: TrainingState,
        history: List[TrainingState]
    ) -> Dict[str, float]:
        """
        Optimize using weighted sum of objectives.
        
        Args:
            initial_schedule: Initial hyperparameter schedule
            current_state: Current training state
            history: Historical training states
            
        Returns:
            Optimized hyperparameter schedule
        """
        # Define objective function
        def objective_function(x):
            schedule = self._vector_to_schedule(x, list(initial_schedule.keys()))
            return self._compute_weighted_objective(schedule, current_state, history)
        
        # Define bounds for each hyperparameter
        bounds = self._get_bounds(list(initial_schedule.keys()), current_state)
        
        # Initial guess
        x0 = self._schedule_to_vector(initial_schedule)
        
        # Optimize
        try:
            result = minimize(
                objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': self.max_iterations}
            )
            
            if result.success:
                optimized_schedule = self._vector_to_schedule(result.x, list(initial_schedule.keys()))
                return optimized_schedule
            else:
                # Fallback to initial schedule if optimization fails
                return initial_schedule
                
        except Exception as e:
            print(f"Optimization failed: {e}")
            return initial_schedule
    
    def _pareto_front_optimization(
        self,
        initial_schedule: Dict[str, float],
        current_state: TrainingState,
        history: List[TrainingState]
    ) -> Dict[str, float]:
        """
        Optimize using Pareto front approach.
        
        Args:
            initial_schedule: Initial hyperparameter schedule
            current_state: Current training state
            history: Historical training states
            
        Returns:
            Optimized hyperparameter schedule
        """
        # Generate multiple weight combinations
        weight_combinations = self._generate_weight_combinations()
        
        best_schedule = initial_schedule
        best_score = float('inf')
        
        for weights in weight_combinations:
            # Temporarily set weights
            original_weights = self.objective_weights.copy()
            self.objective_weights = weights
            
            # Optimize with these weights
            schedule = self._weighted_sum_optimization(initial_schedule, current_state, history)
            
            # Evaluate schedule
            score = self._compute_weighted_objective(schedule, current_state, history)
            
            if score < best_score:
                best_score = score
                best_schedule = schedule
            
            # Restore original weights
            self.objective_weights = original_weights
        
        return best_schedule
    
    def _evolutionary_optimization(
        self,
        initial_schedule: Dict[str, float],
        current_state: TrainingState,
        history: List[TrainingState]
    ) -> Dict[str, float]:
        """
        Optimize using evolutionary algorithms.
        
        Args:
            initial_schedule: Initial hyperparameter schedule
            current_state: Current training state
            history: Historical training states
            
        Returns:
            Optimized hyperparameter schedule
        """
        # Define objective function
        def objective_function(x):
            schedule = self._vector_to_schedule(x, list(initial_schedule.keys()))
            return self._compute_weighted_objective(schedule, current_state, history)
        
        # Define bounds
        bounds = self._get_bounds(list(initial_schedule.keys()), current_state)
        
        # Run differential evolution
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=self.max_iterations,
                tol=self.tolerance,
                seed=42
            )
            
            if result.success:
                optimized_schedule = self._vector_to_schedule(result.x, list(initial_schedule.keys()))
                return optimized_schedule
            else:
                return initial_schedule
                
        except Exception as e:
            print(f"Evolutionary optimization failed: {e}")
            return initial_schedule
    
    def _compute_weighted_objective(
        self,
        schedule: Dict[str, float],
        current_state: TrainingState,
        history: List[TrainingState]
    ) -> float:
        """
        Compute weighted sum of objectives for a given schedule.
        
        Args:
            schedule: Hyperparameter schedule
            current_state: Current training state
            history: Historical training states
            
        Returns:
            Weighted objective value
        """
        objective_values = {}
        
        # Compute each objective
        for objective in self.objectives:
            if objective == ObjectiveType.CONVERGENCE_SPEED:
                objective_values[objective] = self._compute_convergence_speed(schedule, current_state, history)
            elif objective == ObjectiveType.FINAL_ACCURACY:
                objective_values[objective] = self._compute_final_accuracy(schedule, current_state, history)
            elif objective == ObjectiveType.COMPUTATIONAL_EFFICIENCY:
                objective_values[objective] = self._compute_computational_efficiency(schedule, current_state, history)
            elif objective == ObjectiveType.GENERALIZATION:
                objective_values[objective] = self._compute_generalization(schedule, current_state, history)
            elif objective == ObjectiveType.STABILITY:
                objective_values[objective] = self._compute_stability(schedule, current_state, history)
        
        # Compute weighted sum (minimization problem)
        weighted_sum = 0.0
        for objective, value in objective_values.items():
            weight = self.objective_weights.get(objective, 1.0)
            weighted_sum += weight * value
        
        # Add penalty for constraint violations
        penalty = self._compute_constraint_penalty(schedule, current_state)
        weighted_sum += penalty
        
        return weighted_sum
    
    def _compute_convergence_speed(
        self,
        schedule: Dict[str, float],
        current_state: TrainingState,
        history: List[TrainingState]
    ) -> float:
        """Compute convergence speed objective."""
        if len(history) < 2:
            return 0.0
        
        # Estimate convergence rate based on recent history
        recent_losses = [state.loss for state in history[-10:]]
        
        if len(recent_losses) > 1:
            # Fit exponential decay model
            epochs = np.arange(len(recent_losses))
            try:
                # Simple linear fit on log of losses
                log_losses = np.log(np.array(recent_losses) + 1e-8)
                convergence_rate = -np.polyfit(epochs, log_losses, 1)[0]
                return -convergence_rate  # Negative because we want to maximize
            except:
                return 0.0
        
        return 0.0
    
    def _compute_final_accuracy(
        self,
        schedule: Dict[str, float],
        current_state: TrainingState,
        history: List[TrainingState]
    ) -> float:
        """Compute final accuracy objective."""
        if len(history) > 0:
            # Use recent accuracy as proxy for final accuracy
            recent_accuracies = [state.accuracy for state in history[-5:]]
            return -np.mean(recent_accuracies)  # Negative because we want to maximize
        return 0.0
    
    def _compute_computational_efficiency(
        self,
        schedule: Dict[str, float],
        current_state: TrainingState,
        history: List[TrainingState]
    ) -> float:
        """Compute computational efficiency objective."""
        # Consider factors like batch size, learning rate, etc.
        efficiency_score = 0.0
        
        # Batch size efficiency (larger batches are more efficient)
        if 'batch_size' in schedule:
            batch_size = schedule['batch_size']
            efficiency_score += batch_size / 1024.0  # Normalize by max batch size
        
        # Learning rate efficiency (appropriate learning rate)
        if 'learning_rate' in schedule:
            lr = schedule['learning_rate']
            # Penalize very small or very large learning rates
            if lr < 1e-6 or lr > 1.0:
                efficiency_score -= 1.0
        
        return -efficiency_score  # Negative because we want to maximize
    
    def _compute_generalization(
        self,
        schedule: Dict[str, float],
        current_state: TrainingState,
        history: List[TrainingState]
    ) -> float:
        """Compute generalization objective."""
        if len(history) < 2:
            return 0.0
        
        # Use validation accuracy as proxy for generalization
        val_accuracies = []
        for state in history[-10:]:
            if 'val_accuracy' in state.validation_metrics:
                val_accuracies.append(state.validation_metrics['val_accuracy'])
        
        if val_accuracies:
            return -np.mean(val_accuracies)  # Negative because we want to maximize
        
        return 0.0
    
    def _compute_stability(
        self,
        schedule: Dict[str, float],
        current_state: TrainingState,
        history: List[TrainingState]
    ) -> float:
        """Compute training stability objective."""
        if len(history) < 3:
            return 0.0
        
        # Compute stability based on loss variance
        recent_losses = [state.loss for state in history[-10:]]
        loss_variance = np.var(recent_losses)
        
        # Also consider gradient norm stability
        recent_gradients = [state.gradient_norm for state in history[-10:]]
        gradient_variance = np.var(recent_gradients)
        
        stability_score = 1.0 / (1.0 + loss_variance + gradient_variance)
        return -stability_score  # Negative because we want to maximize
    
    def _compute_constraint_penalty(
        self,
        schedule: Dict[str, float],
        current_state: TrainingState
    ) -> float:
        """Compute penalty for constraint violations."""
        penalty = 0.0
        
        for param, value in schedule.items():
            param_constraints = self.constraints.get(param, {})
            
            # Check bounds
            min_val = param_constraints.get("min", 0.0)
            max_val = param_constraints.get("max", float('inf'))
            
            if value < min_val:
                penalty += self.penalty_factor * (min_val - value) ** 2
            if value > max_val:
                penalty += self.penalty_factor * (value - max_val) ** 2
            
            # Check smoothness constraints
            smoothness = param_constraints.get("smoothness", 0.5)
            if param == "learning_rate":
                current_val = current_state.learning_rate
                max_change = current_val * smoothness
                if abs(value - current_val) > max_change:
                    penalty += self.penalty_factor * (abs(value - current_val) - max_change) ** 2
        
        return penalty
    
    def _initialize_weights(self) -> Dict[ObjectiveType, float]:
        """Initialize objective weights."""
        default_weights = {
            ObjectiveType.CONVERGENCE_SPEED: 0.3,
            ObjectiveType.FINAL_ACCURACY: 0.4,
            ObjectiveType.COMPUTATIONAL_EFFICIENCY: 0.2,
            ObjectiveType.GENERALIZATION: 0.3,
            ObjectiveType.STABILITY: 0.2,
        }
        
        # Override with config weights if provided
        config_weights = self.config.get("objective_weights", {})
        for obj, weight in config_weights.items():
            if isinstance(obj, str):
                obj = ObjectiveType(obj)
            default_weights[obj] = weight
        
        return default_weights
    
    def _generate_weight_combinations(self) -> List[Dict[ObjectiveType, float]]:
        """Generate different weight combinations for Pareto front optimization."""
        combinations = []
        
        # Generate grid of weight combinations
        weight_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for conv_weight in weight_values:
            for acc_weight in weight_values:
                for eff_weight in weight_values:
                    # Normalize weights
                    total_weight = conv_weight + acc_weight + eff_weight
                    if total_weight > 0:
                        weights = {
                            ObjectiveType.CONVERGENCE_SPEED: conv_weight / total_weight,
                            ObjectiveType.FINAL_ACCURACY: acc_weight / total_weight,
                            ObjectiveType.COMPUTATIONAL_EFFICIENCY: eff_weight / total_weight,
                        }
                        combinations.append(weights)
        
        return combinations
    
    def _get_bounds(self, param_names: List[str], current_state: TrainingState) -> List[Tuple[float, float]]:
        """Get bounds for optimization variables."""
        bounds = []
        
        for param in param_names:
            param_constraints = self.constraints.get(param, {})
            
            if param == "learning_rate":
                min_val = param_constraints.get("min", 1e-6)
                max_val = param_constraints.get("max", 1.0)
            elif param == "batch_size":
                min_val = param_constraints.get("min", 1)
                max_val = param_constraints.get("max", 1024)
            elif param == "weight_decay":
                min_val = param_constraints.get("min", 0.0)
                max_val = param_constraints.get("max", 1.0)
            else:
                min_val = param_constraints.get("min", 0.0)
                max_val = param_constraints.get("max", 100.0)
            
            bounds.append((min_val, max_val))
        
        return bounds
    
    def _schedule_to_vector(self, schedule: Dict[str, float]) -> np.ndarray:
        """Convert schedule dictionary to vector."""
        return np.array(list(schedule.values()))
    
    def _vector_to_schedule(self, vector: np.ndarray, param_names: List[str]) -> Dict[str, float]:
        """Convert vector to schedule dictionary."""
        return dict(zip(param_names, vector))
    
    def update_weights(self, performance_history: List[Dict[str, float]]) -> None:
        """Update objective weights based on performance history."""
        if len(performance_history) < 5:
            return
        
        # Simple adaptive weight adjustment
        # In practice, this could be more sophisticated
        
        # Analyze recent performance trends
        recent_performance = performance_history[-5:]
        
        # Adjust weights based on which objectives are performing poorly
        for objective in self.objectives:
            if objective == ObjectiveType.FINAL_ACCURACY:
                accuracies = [p.get("train_accuracy", 0.0) for p in recent_performance]
                if np.mean(accuracies) < 0.5:
                    self.objective_weights[objective] *= 1.2  # Increase weight
            elif objective == ObjectiveType.CONVERGENCE_SPEED:
                losses = [p.get("train_loss", 1.0) for p in recent_performance]
                if np.mean(losses) > 0.5:
                    self.objective_weights[objective] *= 1.1  # Increase weight
        
        # Normalize weights
        total_weight = sum(self.objective_weights.values())
        for objective in self.objective_weights:
            self.objective_weights[objective] /= total_weight
    
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Get the current Pareto front."""
        return self.pareto_front
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization performance."""
        return {
            "method": self.optimization_method,
            "objectives": [obj.value for obj in self.objectives],
            "weights": {obj.value: weight for obj, weight in self.objective_weights.items()},
            "pareto_front_size": len(self.pareto_front),
            "constraints": self.constraints,
        }
