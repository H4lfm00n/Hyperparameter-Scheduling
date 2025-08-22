"""
Meta-learner for predicting optimal hyperparameter schedules based on training dynamics.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from ..core.base import TrainingState, ObjectiveType


class MetaLearner:
    """
    Meta-learner that learns to predict optimal hyperparameter schedules
    based on training dynamics and historical performance.
    
    This component uses machine learning models to learn the mapping from
    training dynamics features to optimal hyperparameter values.
    """
    
    def __init__(
        self,
        hyperparameters: List[str],
        objectives: List[ObjectiveType],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the meta-learner.
        
        Args:
            hyperparameters: List of hyperparameter names to predict
            objectives: List of optimization objectives
            config: Configuration dictionary
        """
        self.hyperparameters = hyperparameters
        self.objectives = objectives
        self.config = config or {}
        
        # Initialize models for each hyperparameter
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        
        # Training data storage
        self.training_data = {
            'features': [],
            'targets': {},
            'performance': []
        }
        
        # Model configuration
        self.model_type = self.config.get("model_type", "random_forest")
        self.min_samples = self.config.get("min_samples", 10)
        self.update_frequency = self.config.get("update_frequency", 5)
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize prediction models for each hyperparameter."""
        for param in self.hyperparameters:
            if self.model_type == "random_forest":
                self.models[param] = RandomForestRegressor(
                    n_estimators=self.config.get("n_estimators", 100),
                    max_depth=self.config.get("max_depth", 10),
                    random_state=42
                )
            elif self.model_type == "gradient_boosting":
                self.models[param] = GradientBoostingRegressor(
                    n_estimators=self.config.get("n_estimators", 100),
                    max_depth=self.config.get("max_depth", 6),
                    learning_rate=self.config.get("learning_rate", 0.1),
                    random_state=42
                )
            elif self.model_type == "neural_network":
                self.models[param] = MLPRegressor(
                    hidden_layer_sizes=self.config.get("hidden_layers", (100, 50)),
                    max_iter=self.config.get("max_iter", 1000),
                    random_state=42
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.scalers[param] = StandardScaler()
            self.training_data['targets'][param] = []
    
    def predict(
        self,
        dynamics_features: Dict[str, float],
        current_state: TrainingState,
        history: List[TrainingState]
    ) -> Dict[str, float]:
        """
        Predict optimal hyperparameter values based on training dynamics.
        
        Args:
            dynamics_features: Extracted training dynamics features
            current_state: Current training state
            history: Historical training states
            
        Returns:
            Dictionary of predicted hyperparameter values
        """
        # Convert features to array
        feature_vector = self._extract_feature_vector(dynamics_features, current_state, history)
        
        predictions = {}
        
        for param in self.hyperparameters:
            if len(self.training_data['targets'][param]) >= self.min_samples:
                # Make prediction using trained model
                try:
                    # Scale features
                    scaled_features = self.scalers[param].transform([feature_vector])
                    
                    # Predict
                    prediction = self.models[param].predict(scaled_features)[0]
                    
                    # Apply constraints and bounds
                    prediction = self._apply_constraints(param, prediction, current_state)
                    
                    predictions[param] = prediction
                except Exception as e:
                    # Fallback to default values
                    predictions[param] = self._get_default_value(param, current_state)
            else:
                # Use default values if not enough training data
                predictions[param] = self._get_default_value(param, current_state)
        
        return predictions
    
    def update(
        self,
        state: TrainingState,
        performance: Dict[str, float]
    ) -> None:
        """
        Update the meta-learner with new training data.
        
        Args:
            state: Training state
            performance: Performance metrics
        """
        # Extract features from the state
        # This would typically come from the dynamics analyzer
        # For now, we'll use a simplified feature extraction
        features = self._extract_simple_features(state)
        
        # Store training data
        self.training_data['features'].append(features)
        self.training_data['performance'].append(performance)
        
        # Store targets (current hyperparameter values)
        for param in self.hyperparameters:
            if param == 'learning_rate':
                self.training_data['targets'][param].append(state.learning_rate)
            elif param == 'batch_size':
                self.training_data['targets'][param].append(float(state.batch_size))
            elif param == 'weight_decay':
                # Extract weight decay from optimizer if available
                self.training_data['targets'][param].append(0.0)  # Default
            else:
                self.training_data['targets'][param].append(0.0)  # Default
        
        # Retrain models periodically
        if len(self.training_data['features']) % self.update_frequency == 0:
            self._retrain_models()
    
    def _extract_feature_vector(
        self,
        dynamics_features: Dict[str, float],
        current_state: TrainingState,
        history: List[TrainingState]
    ) -> np.ndarray:
        """Extract feature vector for prediction."""
        # Combine dynamics features with current state features
        features = {}
        
        # Add dynamics features
        features.update(dynamics_features)
        
        # Add current state features
        features.update({
            'epoch': current_state.epoch,
            'step': current_state.step,
            'loss': current_state.loss,
            'accuracy': current_state.accuracy,
            'gradient_norm': current_state.gradient_norm,
        })
        
        # Add historical features if available
        if len(history) > 0:
            recent_history = history[-5:]  # Last 5 states
            features.update({
                'avg_loss': np.mean([s.loss for s in recent_history]),
                'avg_accuracy': np.mean([s.accuracy for s in recent_history]),
                'loss_std': np.std([s.loss for s in recent_history]),
                'accuracy_std': np.std([s.accuracy for s in recent_history]),
            })
        
        # Convert to array
        feature_names = sorted(features.keys())
        feature_vector = [features[name] for name in feature_names]
        
        # Update feature names if needed
        if not self.feature_names:
            self.feature_names = feature_names
        
        return np.array(feature_vector)
    
    def _extract_simple_features(self, state: TrainingState) -> np.ndarray:
        """Extract simple features from training state."""
        features = [
            state.epoch,
            state.step,
            state.loss,
            state.accuracy,
            state.learning_rate,
            state.gradient_norm,
        ]
        
        # Add validation metrics
        for metric_name in sorted(state.validation_metrics.keys()):
            features.append(state.validation_metrics[metric_name])
        
        return np.array(features)
    
    def _retrain_models(self):
        """Retrain all prediction models."""
        if len(self.training_data['features']) < self.min_samples:
            return
        
        # Convert to numpy arrays
        X = np.array(self.training_data['features'])
        
        for param in self.hyperparameters:
            if len(self.training_data['targets'][param]) >= self.min_samples:
                y = np.array(self.training_data['targets'][param])
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Scale features
                self.scalers[param].fit(X_train)
                X_train_scaled = self.scalers[param].transform(X_train)
                X_test_scaled = self.scalers[param].transform(X_test)
                
                # Train model
                self.models[param].fit(X_train_scaled, y_train)
                
                # Evaluate model
                train_score = self.models[param].score(X_train_scaled, y_train)
                test_score = self.models[param].score(X_test_scaled, y_test)
                
                print(f"Model for {param}: Train R²={train_score:.3f}, Test R²={test_score:.3f}")
    
    def _apply_constraints(
        self, 
        param: str, 
        prediction: float, 
        current_state: TrainingState
    ) -> float:
        """Apply constraints to predicted hyperparameter values."""
        constraints = self.config.get("constraints", {})
        param_constraints = constraints.get(param, {})
        
        # Apply bounds
        min_val = param_constraints.get("min", 0.0)
        max_val = param_constraints.get("max", float('inf'))
        
        if param == "learning_rate":
            min_val = param_constraints.get("min", 1e-6)
            max_val = param_constraints.get("max", 1.0)
        elif param == "batch_size":
            min_val = param_constraints.get("min", 1)
            max_val = param_constraints.get("max", 1024)
        elif param == "weight_decay":
            min_val = param_constraints.get("min", 0.0)
            max_val = param_constraints.get("max", 1.0)
        
        # Apply smoothness constraint (limit change from current value)
        smoothness_factor = param_constraints.get("smoothness", 0.5)
        if param == "learning_rate":
            current_val = current_state.learning_rate
            max_change = current_val * smoothness_factor
            prediction = np.clip(prediction, current_val - max_change, current_val + max_change)
        
        # Apply final bounds
        prediction = np.clip(prediction, min_val, max_val)
        
        return prediction
    
    def _get_default_value(self, param: str, current_state: TrainingState) -> float:
        """Get default value for a hyperparameter."""
        defaults = {
            "learning_rate": current_state.learning_rate,
            "batch_size": float(current_state.batch_size),
            "weight_decay": 0.0,
        }
        
        return defaults.get(param, 0.0)
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the meta-learner."""
        return {
            "models": self.models,
            "scalers": self.scalers,
            "training_data": self.training_data,
            "feature_names": self.feature_names,
            "config": self.config,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load the state of the meta-learner."""
        self.models = state["models"]
        self.scalers = state["scalers"]
        self.training_data = state["training_data"]
        self.feature_names = state["feature_names"]
        self.config.update(state["config"])
    
    def save_models(self, path: str) -> None:
        """Save trained models to disk."""
        model_data = {
            "models": self.models,
            "scalers": self.scalers,
            "feature_names": self.feature_names,
            "config": self.config,
        }
        joblib.dump(model_data, path)
    
    def load_models(self, path: str) -> None:
        """Load trained models from disk."""
        model_data = joblib.load(path)
        self.models = model_data["models"]
        self.scalers = model_data["scalers"]
        self.feature_names = model_data["feature_names"]
        self.config.update(model_data["config"])
    
    def get_feature_importance(self, param: str) -> Dict[str, float]:
        """Get feature importance for a specific hyperparameter."""
        if param not in self.models or not hasattr(self.models[param], 'feature_importances_'):
            return {}
        
        if not self.feature_names:
            return {}
        
        importance_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            if i < len(self.models[param].feature_importances_):
                importance_dict[feature_name] = self.models[param].feature_importances_[i]
        
        return importance_dict
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all models."""
        performance = {}
        
        for param in self.hyperparameters:
            if param in self.models and len(self.training_data['targets'][param]) >= self.min_samples:
                # Calculate basic performance metrics
                y_true = np.array(self.training_data['targets'][param])
                X = np.array(self.training_data['features'])
                
                try:
                    X_scaled = self.scalers[param].transform(X)
                    y_pred = self.models[param].predict(X_scaled)
                    
                    # Calculate metrics
                    mse = np.mean((y_true - y_pred) ** 2)
                    mae = np.mean(np.abs(y_true - y_pred))
                    r2 = self.models[param].score(X_scaled, y_true)
                    
                    performance[param] = {
                        "mse": mse,
                        "mae": mae,
                        "r2": r2,
                        "num_samples": len(y_true)
                    }
                except Exception as e:
                    performance[param] = {
                        "error": str(e),
                        "num_samples": len(y_true)
                    }
        
        return performance
