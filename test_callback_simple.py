#!/usr/bin/env python3
"""
Simple test script for the callback system.
"""

import sys
import traceback

def test_callback_import():
    """Test if we can import the callback system."""
    try:
        print("Testing callback system import...")
        from hyperparameter_scheduling.callbacks import GenericCallback
        print("✓ Successfully imported GenericCallback")
        
        from hyperparameter_scheduling.callbacks import PyTorchCallback
        print("✓ Successfully imported PyTorchCallback")
        
        from hyperparameter_scheduling.callbacks import TensorFlowCallback
        print("✓ Successfully imported TensorFlowCallback")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_callback_creation():
    """Test if we can create callback instances."""
    try:
        print("\nTesting callback creation...")
        
        from hyperparameter_scheduling.callbacks import GenericCallback
        
        # Create a simple callback
        callback = GenericCallback(
            hyperparameters=['learning_rate', 'batch_size'],
            objectives=['convergence_speed', 'final_accuracy']
        )
        print("✓ Successfully created GenericCallback")
        
        # Test basic functionality
        callback.set_current_metrics(loss=0.5, accuracy=0.8)
        callback.set_current_hyperparameters(learning_rate=0.001, batch_size=32)
        
        current_hp = callback.get_current_hyperparameters()
        print(f"✓ Current hyperparameters: {current_hp}")
        
        # Test step functionality
        callback.step()
        print(f"✓ Step completed, current step: {callback.current_step}")
        
        return True
    except Exception as e:
        print(f"✗ Callback creation failed: {e}")
        traceback.print_exc()
        return False

def test_scheduler_integration():
    """Test if the callback properly integrates with the scheduler."""
    try:
        print("\nTesting scheduler integration...")
        
        from hyperparameter_scheduling.callbacks import GenericCallback
        
        callback = GenericCallback(['learning_rate'])
        
        # Check if scheduler is properly initialized
        assert callback.scheduler is not None, "Scheduler should be initialized"
        assert hasattr(callback.scheduler, 'get_schedule'), "Scheduler should have get_schedule method"
        
        print("✓ Scheduler integration successful")
        return True
    except Exception as e:
        print(f"✗ Scheduler integration failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Hyperparameter Scheduling Callback System")
    print("=" * 50)
    
    tests = [
        test_callback_import,
        test_callback_creation,
        test_scheduler_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Callback system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


