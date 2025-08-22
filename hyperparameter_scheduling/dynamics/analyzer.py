"""
Training dynamics analyzer for extracting features that inform hyperparameter scheduling.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from scipy.signal import savgol_filter
import torch

from ..core.base import TrainingState


class TrainingDynamicsAnalyzer:
    """
    Analyzes training dynamics to extract features that inform hyperparameter scheduling decisions.
    
    This class extracts various features from training states including:
    - Gradient flow patterns
    - Loss landscape characteristics
    - Convergence patterns
    - Training stability metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the training dynamics analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.feature_cache = {}
        
    def extract_features(
        self, 
        current_state: TrainingState, 
        history: List[TrainingState]
    ) -> Dict[str, float]:
        """
        Extract features from current state and training history.
        
        Args:
            current_state: Current training state
            history: Historical training states
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic current state features
        features.update(self._extract_current_features(current_state))
        
        # Historical trend features
        if len(history) > 0:
            features.update(self._extract_trend_features(history, current_state))
        
        # Gradient flow features
        features.update(self._extract_gradient_features(history, current_state))
        
        # Loss landscape features
        features.update(self._extract_loss_features(history, current_state))
        
        # Convergence pattern features
        features.update(self._extract_convergence_features(history, current_state))
        
        # Stability features
        features.update(self._extract_stability_features(history, current_state))
        
        # Cache features for potential reuse
        self.feature_cache[f"epoch_{current_state.epoch}"] = features
        
        return features
    
    def _extract_current_features(self, state: TrainingState) -> Dict[str, float]:
        """Extract features from current training state."""
        features = {
            "current_loss": state.loss,
            "current_accuracy": state.accuracy,
            "current_learning_rate": state.learning_rate,
            "current_batch_size": float(state.batch_size),
            "current_gradient_norm": state.gradient_norm,
            "current_training_time": state.training_time,
            "current_memory_usage": state.memory_usage,
        }
        
        # Add validation metrics if available
        for metric_name, metric_value in state.validation_metrics.items():
            features[f"current_val_{metric_name}"] = metric_value
        
        return features
    
    def _extract_trend_features(
        self, 
        history: List[TrainingState], 
        current_state: TrainingState
    ) -> Dict[str, float]:
        """Extract trend-based features from training history."""
        if len(history) < 2:
            return {}
        
        # Get recent history for trend analysis
        recent_window = self.config.get("trend_window", 10)
        recent_history = history[-recent_window:] if len(history) >= recent_window else history
        
        # Extract time series
        losses = [state.loss for state in recent_history]
        accuracies = [state.accuracy for state in recent_history]
        learning_rates = [state.learning_rate for state in recent_history]
        gradient_norms = [state.gradient_norm for state in recent_history]
        
        features = {}
        
        # Loss trends
        if len(losses) > 1:
            loss_trend = np.polyfit(range(len(losses)), losses, 1)[0]
            features["loss_trend"] = loss_trend
            features["loss_curvature"] = np.polyfit(range(len(losses)), losses, 2)[0]
            features["loss_volatility"] = np.std(losses)
            features["loss_range"] = max(losses) - min(losses)
        
        # Accuracy trends
        if len(accuracies) > 1:
            acc_trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
            features["accuracy_trend"] = acc_trend
            features["accuracy_curvature"] = np.polyfit(range(len(accuracies)), accuracies, 2)[0]
            features["accuracy_volatility"] = np.std(accuracies)
        
        # Learning rate trends
        if len(learning_rates) > 1:
            lr_trend = np.polyfit(range(len(learning_rates)), learning_rates, 1)[0]
            features["learning_rate_trend"] = lr_trend
            features["learning_rate_volatility"] = np.std(learning_rates)
        
        # Gradient norm trends
        if len(gradient_norms) > 1:
            grad_trend = np.polyfit(range(len(gradient_norms)), gradient_norms, 1)[0]
            features["gradient_norm_trend"] = grad_trend
            features["gradient_norm_volatility"] = np.std(gradient_norms)
        
        return features
    
    def _extract_gradient_features(
        self, 
        history: List[TrainingState], 
        current_state: TrainingState
    ) -> Dict[str, float]:
        """Extract gradient flow related features."""
        if len(history) < 2:
            return {}
        
        recent_window = self.config.get("gradient_window", 20)
        recent_history = history[-recent_window:] if len(history) >= recent_window else history
        
        gradient_norms = [state.gradient_norm for state in recent_history]
        
        features = {
            "gradient_norm_mean": np.mean(gradient_norms),
            "gradient_norm_std": np.std(gradient_norms),
            "gradient_norm_max": np.max(gradient_norms),
            "gradient_norm_min": np.min(gradient_norms),
            "gradient_norm_range": np.max(gradient_norms) - np.min(gradient_norms),
        }
        
        # Gradient explosion/vanishing detection
        features["gradient_explosion_risk"] = float(np.max(gradient_norms) > 10.0)
        features["gradient_vanishing_risk"] = float(np.min(gradient_norms) < 1e-6)
        
        # Gradient stability
        if len(gradient_norms) > 1:
            gradient_changes = np.diff(gradient_norms)
            features["gradient_stability"] = 1.0 / (1.0 + np.std(gradient_changes))
        
        return features
    
    def _extract_loss_features(
        self, 
        history: List[TrainingState], 
        current_state: TrainingState
    ) -> Dict[str, float]:
        """Extract loss landscape related features."""
        if len(history) < 2:
            return {}
        
        recent_window = self.config.get("loss_window", 15)
        recent_history = history[-recent_window:] if len(history) >= recent_window else history
        
        losses = [state.loss for state in recent_history]
        
        features = {
            "loss_mean": np.mean(losses),
            "loss_std": np.std(losses),
            "loss_max": np.max(losses),
            "loss_min": np.min(losses),
            "loss_range": np.max(losses) - np.min(losses),
        }
        
        # Loss landscape characteristics
        if len(losses) > 2:
            # Smooth the loss curve
            try:
                smoothed_losses = savgol_filter(losses, min(5, len(losses) - 2), 2)
                features["loss_smoothness"] = 1.0 / (1.0 + np.std(np.diff(smoothed_losses)))
            except:
                features["loss_smoothness"] = 0.5
        
        # Loss plateau detection
        if len(losses) > 5:
            recent_losses = losses[-5:]
            loss_variance = np.var(recent_losses)
            features["loss_plateau"] = float(loss_variance < 1e-4)
        
        return features
    
    def _extract_convergence_features(
        self, 
        history: List[TrainingState], 
        current_state: TrainingState
    ) -> Dict[str, float]:
        """Extract convergence pattern features."""
        if len(history) < 5:
            return {}
        
        recent_window = self.config.get("convergence_window", 30)
        recent_history = history[-recent_window:] if len(history) >= recent_window else history
        
        losses = [state.loss for state in recent_history]
        accuracies = [state.accuracy for state in recent_history]
        
        features = {}
        
        # Convergence rate estimation
        if len(losses) > 3:
            # Fit exponential decay model: loss = a * exp(-b * epoch) + c
            epochs = np.arange(len(losses))
            try:
                # Simple linear fit on log of loss differences
                loss_diffs = np.diff(losses)
                valid_diffs = loss_diffs[loss_diffs > 0]
                if len(valid_diffs) > 2:
                    log_diffs = np.log(valid_diffs)
                    convergence_rate = -np.polyfit(range(len(log_diffs)), log_diffs, 1)[0]
                    features["convergence_rate"] = max(0, convergence_rate)
                else:
                    features["convergence_rate"] = 0.0
            except:
                features["convergence_rate"] = 0.0
        
        # Convergence stability
        if len(losses) > 5:
            recent_loss_std = np.std(losses[-5:])
            features["convergence_stability"] = 1.0 / (1.0 + recent_loss_std)
        
        # Accuracy convergence
        if len(accuracies) > 3:
            acc_trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
            features["accuracy_convergence_rate"] = max(0, acc_trend)
        
        return features
    
    def _extract_stability_features(
        self, 
        history: List[TrainingState], 
        current_state: TrainingState
    ) -> Dict[str, float]:
        """Extract training stability features."""
        if len(history) < 3:
            return {}
        
        recent_window = self.config.get("stability_window", 25)
        recent_history = history[-recent_window:] if len(history) >= recent_window else history
        
        losses = [state.loss for state in recent_history]
        accuracies = [state.accuracy for state in recent_history]
        learning_rates = [state.learning_rate for state in recent_history]
        
        features = {}
        
        # Loss stability
        if len(losses) > 2:
            loss_changes = np.abs(np.diff(losses))
            features["loss_stability"] = 1.0 / (1.0 + np.mean(loss_changes))
            features["loss_instability_events"] = np.sum(loss_changes > 0.1)
        
        # Accuracy stability
        if len(accuracies) > 2:
            acc_changes = np.abs(np.diff(accuracies))
            features["accuracy_stability"] = 1.0 / (1.0 + np.mean(acc_changes))
        
        # Learning rate stability
        if len(learning_rates) > 2:
            lr_changes = np.abs(np.diff(learning_rates))
            features["learning_rate_stability"] = 1.0 / (1.0 + np.mean(lr_changes))
        
        # Overall training stability
        stability_metrics = [
            features.get("loss_stability", 0.5),
            features.get("accuracy_stability", 0.5),
            features.get("learning_rate_stability", 0.5)
        ]
        features["overall_stability"] = np.mean(stability_metrics)
        
        return features
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores based on historical analysis.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # This would typically be learned from historical data
        # For now, return heuristic importance scores
        importance_scores = {
            # High importance features
            "loss_trend": 0.9,
            "gradient_norm_trend": 0.8,
            "convergence_rate": 0.8,
            "overall_stability": 0.7,
            
            # Medium importance features
            "learning_rate_trend": 0.6,
            "accuracy_trend": 0.6,
            "gradient_explosion_risk": 0.6,
            "gradient_vanishing_risk": 0.6,
            
            # Lower importance features
            "current_loss": 0.4,
            "current_accuracy": 0.4,
            "loss_volatility": 0.3,
            "accuracy_volatility": 0.3,
        }
        
        return importance_scores
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get a summary of extracted features.
        
        Returns:
            Dictionary with feature statistics and metadata
        """
        if not self.feature_cache:
            return {}
        
        all_features = list(self.feature_cache.values())
        feature_names = list(all_features[0].keys()) if all_features else []
        
        summary = {
            "num_epochs_analyzed": len(self.feature_cache),
            "feature_names": feature_names,
            "feature_importance": self.get_feature_importance(),
        }
        
        # Compute feature statistics
        for feature_name in feature_names:
            values = [epoch_features.get(feature_name, 0) for epoch_features in all_features]
            if values:
                summary[f"{feature_name}_mean"] = np.mean(values)
                summary[f"{feature_name}_std"] = np.std(values)
                summary[f"{feature_name}_min"] = np.min(values)
                summary[f"{feature_name}_max"] = np.max(values)
        
        return summary
