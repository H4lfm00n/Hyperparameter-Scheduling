"""
Tests for the AutoScheduler and its components.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from hyperparameter_scheduling import AutoScheduler, ObjectiveType
from hyperparameter_scheduling.core import TrainingState, ScheduleDecision
from hyperparameter_scheduling.dynamics import TrainingDynamicsAnalyzer
from hyperparameter_scheduling.learners import MetaLearner
from hyperparameter_scheduling.transfer import TransferLearner
from hyperparameter_scheduling.optimizers import MultiObjectiveOptimizer


class TestAutoScheduler(unittest.TestCase):
    """Test cases for the AutoScheduler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hyperparameters = ['learning_rate', 'batch_size', 'weight_decay']
        self.objectives = [
            ObjectiveType.CONVERGENCE_SPEED,
            ObjectiveType.FINAL_ACCURACY,
            ObjectiveType.COMPUTATIONAL_EFFICIENCY
        ]
        
        self.scheduler = AutoScheduler(
            hyperparameters=self.hyperparameters,
            objectives=self.objectives
        )
    
    def test_initialization(self):
        """Test scheduler initialization."""
        self.assertEqual(self.scheduler.hyperparameters, self.hyperparameters)
        self.assertEqual(self.scheduler.objectives, self.objectives)
        self.assertIsNotNone(self.scheduler.dynamics_analyzer)
        self.assertIsNotNone(self.scheduler.meta_learner)
        self.assertIsNotNone(self.scheduler.transfer_learner)
        self.assertIsNotNone(self.scheduler.multi_objective_optimizer)
    
    def test_get_schedule(self):
        """Test schedule generation."""
        # Create mock training state
        current_state = TrainingState(
            epoch=5,
            step=100,
            loss=0.5,
            accuracy=0.8,
            learning_rate=0.001,
            batch_size=32,
            gradient_norm=1.0,
            validation_metrics={'val_accuracy': 0.75},
            training_time=10.0,
            memory_usage=512.0
        )
        
        history = [current_state] * 3  # Mock history
        
        # Get schedule
        decision = self.scheduler.get_schedule(current_state, history)
        
        # Verify decision structure
        self.assertIsInstance(decision, ScheduleDecision)
        self.assertIsInstance(decision.hyperparameters, dict)
        self.assertIsInstance(decision.confidence, float)
        self.assertIsInstance(decision.reasoning, str)
        self.assertIsInstance(decision.metadata, dict)
        
        # Verify hyperparameters
        for param in self.hyperparameters:
            self.assertIn(param, decision.hyperparameters)
            self.assertIsInstance(decision.hyperparameters[param], (int, float))
        
        # Verify confidence bounds
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
    
    def test_update(self):
        """Test scheduler update."""
        state = TrainingState(
            epoch=1,
            step=50,
            loss=0.6,
            accuracy=0.7,
            learning_rate=0.001,
            batch_size=32,
            gradient_norm=1.2,
            validation_metrics={'val_accuracy': 0.65},
            training_time=5.0,
            memory_usage=256.0
        )
        
        performance = {
            'train_loss': 0.6,
            'train_accuracy': 0.7,
            'val_accuracy': 0.65
        }
        
        # Update scheduler
        self.scheduler.update(state, performance)
        
        # Verify history was updated
        self.assertEqual(len(self.scheduler.history), 1)
        self.assertEqual(self.scheduler.current_state, state)
        
        # Verify best performance was updated
        self.assertIn('train_accuracy', self.scheduler.best_performance)
        self.assertEqual(self.scheduler.best_performance['train_accuracy'], 0.7)
    
    def test_save_load(self):
        """Test save and load functionality."""
        # Add some data to scheduler
        state = TrainingState(
            epoch=1,
            step=50,
            loss=0.6,
            accuracy=0.7,
            learning_rate=0.001,
            batch_size=32,
            gradient_norm=1.2,
            validation_metrics={},
            training_time=5.0,
            memory_usage=256.0
        )
        
        self.scheduler.update(state, {'train_loss': 0.6, 'train_accuracy': 0.7})
        
        # Save scheduler
        self.scheduler.save("test_scheduler.pkl")
        
        # Create new scheduler and load
        new_scheduler = AutoScheduler(
            hyperparameters=self.hyperparameters,
            objectives=self.objectives
        )
        new_scheduler.load("test_scheduler.pkl")
        
        # Verify data was loaded
        self.assertEqual(len(new_scheduler.history), 1)
        self.assertIn('train_accuracy', new_scheduler.best_performance)


class TestTrainingDynamicsAnalyzer(unittest.TestCase):
    """Test cases for the TrainingDynamicsAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = TrainingDynamicsAnalyzer()
    
    def test_extract_features(self):
        """Test feature extraction."""
        current_state = TrainingState(
            epoch=10,
            step=500,
            loss=0.3,
            accuracy=0.9,
            learning_rate=0.001,
            batch_size=64,
            gradient_norm=0.8,
            validation_metrics={'val_accuracy': 0.85},
            training_time=20.0,
            memory_usage=1024.0
        )
        
        history = []
        for i in range(10):
            state = TrainingState(
                epoch=i,
                step=i*50,
                loss=1.0 - i*0.1,  # Decreasing loss
                accuracy=0.5 + i*0.05,  # Increasing accuracy
                learning_rate=0.001,
                batch_size=32,
                gradient_norm=1.0,
                validation_metrics={'val_accuracy': 0.5 + i*0.04},
                training_time=i*2.0,
                memory_usage=512.0
            )
            history.append(state)
        
        features = self.analyzer.extract_features(current_state, history)
        
        # Verify features were extracted
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
        
        # Check for specific features
        self.assertIn('current_loss', features)
        self.assertIn('current_accuracy', features)
        self.assertIn('loss_trend', features)
        self.assertIn('accuracy_trend', features)
    
    def test_feature_importance(self):
        """Test feature importance computation."""
        importance = self.analyzer.get_feature_importance()
        
        self.assertIsInstance(importance, dict)
        self.assertGreater(len(importance), 0)
        
        # Check importance values are in valid range
        for feature, importance_score in importance.items():
            self.assertGreaterEqual(importance_score, 0.0)
            self.assertLessEqual(importance_score, 1.0)


class TestMetaLearner(unittest.TestCase):
    """Test cases for the MetaLearner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hyperparameters = ['learning_rate', 'batch_size']
        self.objectives = [ObjectiveType.CONVERGENCE_SPEED, ObjectiveType.FINAL_ACCURACY]
        
        self.meta_learner = MetaLearner(
            hyperparameters=self.hyperparameters,
            objectives=self.objectives
        )
    
    def test_initialization(self):
        """Test meta-learner initialization."""
        self.assertEqual(self.meta_learner.hyperparameters, self.hyperparameters)
        self.assertEqual(self.meta_learner.objectives, self.objectives)
        self.assertIsNotNone(self.meta_learner.models)
        self.assertIsNotNone(self.meta_learner.scalers)
    
    def test_predict(self):
        """Test prediction functionality."""
        dynamics_features = {
            'loss_trend': -0.1,
            'accuracy_trend': 0.05,
            'gradient_norm': 1.0
        }
        
        current_state = TrainingState(
            epoch=5,
            step=250,
            loss=0.5,
            accuracy=0.8,
            learning_rate=0.001,
            batch_size=32,
            gradient_norm=1.0,
            validation_metrics={},
            training_time=10.0,
            memory_usage=512.0
        )
        
        history = [current_state] * 2
        
        # Should return default values when not enough training data
        predictions = self.meta_learner.predict(dynamics_features, current_state, history)
        
        self.assertIsInstance(predictions, dict)
        for param in self.hyperparameters:
            self.assertIn(param, predictions)
            self.assertIsInstance(predictions[param], (int, float))
    
    def test_update(self):
        """Test meta-learner update."""
        state = TrainingState(
            epoch=1,
            step=50,
            loss=0.6,
            accuracy=0.7,
            learning_rate=0.001,
            batch_size=32,
            gradient_norm=1.2,
            validation_metrics={},
            training_time=5.0,
            memory_usage=256.0
        )
        
        performance = {'train_loss': 0.6, 'train_accuracy': 0.7}
        
        self.meta_learner.update(state, performance)
        
        # Verify training data was stored
        self.assertGreater(len(self.meta_learner.training_data['features']), 0)
        self.assertGreater(len(self.meta_learner.training_data['targets']['learning_rate']), 0)


class TestTransferLearner(unittest.TestCase):
    """Test cases for the TransferLearner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.transfer_learner = TransferLearner()
    
    def test_initialization(self):
        """Test transfer learner initialization."""
        self.assertIsNotNone(self.transfer_learner.problem_signatures)
        self.assertIsNotNone(self.transfer_learner.problem_clusters)
    
    def test_has_similar_problems(self):
        """Test similar problems detection."""
        # Initially no problems
        self.assertFalse(self.transfer_learner.has_similar_problems())
        
        # Add a problem
        state = TrainingState(
            epoch=1,
            step=50,
            loss=0.6,
            accuracy=0.7,
            learning_rate=0.001,
            batch_size=32,
            gradient_norm=1.2,
            validation_metrics={},
            training_time=5.0,
            memory_usage=256.0
        )
        
        self.transfer_learner.update(state, {'train_loss': 0.6, 'train_accuracy': 0.7})
        
        # Now should have problems
        self.assertTrue(self.transfer_learner.has_similar_problems())
    
    def test_get_adjustment(self):
        """Test transfer adjustment computation."""
        # Add some problems first
        for i in range(3):
            state = TrainingState(
                epoch=i,
                step=i*50,
                loss=0.6 - i*0.1,
                accuracy=0.7 + i*0.05,
                learning_rate=0.001,
                batch_size=32,
                gradient_norm=1.2,
                validation_metrics={},
                training_time=5.0,
                memory_usage=256.0
            )
            self.transfer_learner.update(state, {'train_loss': 0.6 - i*0.1, 'train_accuracy': 0.7 + i*0.05})
        
        # Get adjustment
        dynamics_features = {'loss_trend': -0.1, 'accuracy_trend': 0.05}
        current_state = TrainingState(
            epoch=5,
            step=250,
            loss=0.5,
            accuracy=0.8,
            learning_rate=0.001,
            batch_size=32,
            gradient_norm=1.0,
            validation_metrics={},
            training_time=10.0,
            memory_usage=512.0
        )
        
        adjustment = self.transfer_learner.get_adjustment(dynamics_features, current_state)
        
        # Should return adjustment if similar problems found
        if adjustment:
            self.assertIsInstance(adjustment, dict)
            for param in ['learning_rate', 'batch_size', 'weight_decay']:
                if param in adjustment:
                    self.assertIsInstance(adjustment[param], (int, float))


class TestMultiObjectiveOptimizer(unittest.TestCase):
    """Test cases for the MultiObjectiveOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.objectives = [ObjectiveType.CONVERGENCE_SPEED, ObjectiveType.FINAL_ACCURACY]
        
        self.optimizer = MultiObjectiveOptimizer(
            objectives=self.objectives
        )
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.objectives, self.objectives)
        self.assertIsNotNone(self.optimizer.objective_weights)
        self.assertIsNotNone(self.optimizer.constraints)
    
    def test_optimize(self):
        """Test optimization functionality."""
        initial_schedule = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'weight_decay': 0.0
        }
        
        current_state = TrainingState(
            epoch=5,
            step=250,
            loss=0.5,
            accuracy=0.8,
            learning_rate=0.001,
            batch_size=32,
            gradient_norm=1.0,
            validation_metrics={},
            training_time=10.0,
            memory_usage=512.0
        )
        
        history = [current_state] * 3
        
        optimized_schedule = self.optimizer.optimize(initial_schedule, current_state, history)
        
        self.assertIsInstance(optimized_schedule, dict)
        for param in initial_schedule:
            self.assertIn(param, optimized_schedule)
            self.assertIsInstance(optimized_schedule[param], (int, float))
    
    def test_objective_computation(self):
        """Test objective computation."""
        schedule = {'learning_rate': 0.001, 'batch_size': 32, 'weight_decay': 0.0}
        current_state = TrainingState(
            epoch=5,
            step=250,
            loss=0.5,
            accuracy=0.8,
            learning_rate=0.001,
            batch_size=32,
            gradient_norm=1.0,
            validation_metrics={},
            training_time=10.0,
            memory_usage=512.0
        )
        
        history = [current_state] * 5
        
        objective_value = self.optimizer._compute_weighted_objective(schedule, current_state, history)
        
        self.assertIsInstance(objective_value, float)
        self.assertGreaterEqual(objective_value, 0.0)


if __name__ == '__main__':
    unittest.main()
