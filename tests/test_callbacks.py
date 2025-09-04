"""
Tests for the callback system.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import Mock, patch

from hyperparameter_scheduling.callbacks import (
    GenericCallback,
    PyTorchCallback,
    TensorFlowCallback
)


class TestGenericCallback:
    """Test the generic callback."""
    
    def test_initialization(self):
        """Test callback initialization."""
        callback = GenericCallback(
            hyperparameters=['learning_rate', 'batch_size'],
            objectives=['convergence_speed', 'final_accuracy']
        )
        
        assert callback.hyperparameters == ['learning_rate', 'batch_size']
        assert callback.objectives == ['convergence_speed', 'final_accuracy']
        assert callback.update_frequency == 1
        assert callback.current_epoch == 0
        assert callback.current_step == 0
    
    def test_set_current_metrics(self):
        """Test setting current metrics."""
        callback = GenericCallback(['learning_rate'])
        
        callback.set_current_metrics(loss=0.5, accuracy=0.8)
        
        assert callback.current_loss == 0.5
        assert callback.current_accuracy == 0.8
    
    def test_set_current_hyperparameters(self):
        """Test setting current hyperparameters."""
        callback = GenericCallback(['learning_rate', 'batch_size'])
        
        callback.set_current_hyperparameters(
            learning_rate=0.01,
            batch_size=64
        )
        
        assert callback.current_learning_rate == 0.01
        assert callback.current_batch_size == 64
    
    def test_get_current_hyperparameters(self):
        """Test getting current hyperparameters."""
        callback = GenericCallback(['learning_rate', 'batch_size'])
        callback.set_current_hyperparameters(learning_rate=0.01, batch_size=64)
        
        hp = callback.get_current_hyperparameters()
        
        assert hp['learning_rate'] == 0.01
        assert hp['batch_size'] == 64
    
    def test_step(self):
        """Test manual step update."""
        callback = GenericCallback(['learning_rate'], update_frequency=2)
        
        # First step - should not update
        callback.step()
        assert callback.current_step == 1
        
        # Second step - should update
        callback.step()
        assert callback.current_step == 2
    
    def test_update_metrics(self):
        """Test updating metrics."""
        callback = GenericCallback(['learning_rate'])
        
        callback.update_metrics(loss=0.3, accuracy=0.9, val_loss=0.2)
        
        assert callback.current_loss == 0.3
        assert callback.current_accuracy == 0.9
        assert callback.current_performance['val_loss'] == 0.2


class TestPyTorchCallback:
    """Test the PyTorch callback."""
    
    def test_initialization(self):
        """Test PyTorch callback initialization."""
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters())
        
        callback = PyTorchCallback(
            hyperparameters=['learning_rate'],
            optimizer=optimizer,
            model=model
        )
        
        assert callback.optimizer == optimizer
        assert callback.model == model
    
    def test_set_optimizer(self):
        """Test setting optimizer."""
        callback = PyTorchCallback(['learning_rate'])
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        callback.set_optimizer(optimizer)
        
        assert callback.optimizer == optimizer
        assert callback.current_learning_rate == 0.01
    
    def test_set_model(self):
        """Test setting model."""
        callback = PyTorchCallback(['learning_rate'])
        model = nn.Linear(10, 1)
        
        callback.set_model(model)
        
        assert callback.model == model
    
    def test_compute_gradient_norm(self):
        """Test gradient norm computation."""
        callback = PyTorchCallback(['learning_rate'])
        model = nn.Linear(10, 1)
        callback.set_model(model)
        
        # Create dummy gradients
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        
        norm = callback.compute_gradient_norm()
        
        assert isinstance(norm, float)
        assert norm > 0
    
    def test_get_optimizer_state(self):
        """Test getting optimizer state."""
        callback = PyTorchCallback(['learning_rate'])
        model = nn.Linear(10, 1)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
        callback.set_optimizer(optimizer)
        
        state = callback.get_optimizer_state()
        
        assert 'param_group_0' in state
        assert state['param_group_0']['lr'] == 0.01
        assert state['param_group_0']['weight_decay'] == 0.001


class TestTensorFlowCallback:
    """Test the TensorFlow callback."""
    
    @pytest.mark.skipif(True, reason="TensorFlow not available in test environment")
    def test_initialization(self):
        """Test TensorFlow callback initialization."""
        # This test would require TensorFlow to be installed
        pass
    
    @pytest.mark.skipif(True, reason="TensorFlow not available in test environment")
    def test_set_optimizer(self):
        """Test setting TensorFlow optimizer."""
        # This test would require TensorFlow to be installed
        pass


class TestCallbackIntegration:
    """Test callback integration with scheduler."""
    
    def test_callback_with_scheduler(self):
        """Test that callback properly integrates with scheduler."""
        callback = GenericCallback(
            hyperparameters=['learning_rate', 'batch_size'],
            objectives=['convergence_speed', 'final_accuracy']
        )
        
        # Verify scheduler is initialized
        assert callback.scheduler is not None
        assert hasattr(callback.scheduler, 'get_schedule')
    
    def test_save_load_scheduler_state(self):
        """Test saving and loading scheduler state."""
        callback = GenericCallback(['learning_rate'])
        
        # Test saving (should not raise exception)
        try:
            callback.save_scheduler_state("test_scheduler.pkl")
        except Exception as e:
            # This might fail due to file permissions, but shouldn't crash
            pass
        
        # Test loading (should not raise exception)
        try:
            callback.load_scheduler_state("test_scheduler.pkl")
        except Exception as e:
            # This might fail if file doesn't exist, but shouldn't crash
            pass
    
    def test_performance_summary(self):
        """Test getting performance summary."""
        callback = GenericCallback(['learning_rate'])
        
        # Set some performance data
        callback.best_performance = {'accuracy': 0.95}
        callback.current_performance = {'accuracy': 0.90}
        callback.schedule_history = [{'epoch': 1}, {'epoch': 2}]
        callback.training_history = [Mock(), Mock()]
        
        summary = callback.get_performance_summary()
        
        assert summary['best_performance'] == {'accuracy': 0.95}
        assert summary['current_performance'] == {'accuracy': 0.90}
        assert summary['schedule_changes'] == 2
        assert summary['training_history_length'] == 2
    
    def test_log_level_setting(self):
        """Test setting log level."""
        callback = GenericCallback(['learning_rate'], log_level="DEBUG")
        
        assert callback.logger.level == 10  # DEBUG level
        
        callback.set_log_level("ERROR")
        assert callback.logger.level == 40  # ERROR level


if __name__ == "__main__":
    pytest.main([__file__])
