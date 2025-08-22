"""
Basic usage example for the Automatic Hyperparameter Scheduling library.

This example demonstrates how to use the AutoScheduler with a simple neural network
on a classification task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from hyperparameter_scheduling import AutoScheduler, ObjectiveType


# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def create_synthetic_data(num_samples=1000, input_size=784, num_classes=10):
    """Create synthetic data for demonstration."""
    # Generate random features
    X = torch.randn(num_samples, input_size)
    
    # Generate labels (simple linear separation)
    weights = torch.randn(input_size, num_classes)
    logits = torch.matmul(X, weights)
    y = torch.argmax(logits, dim=1)
    
    # Split into train/val
    train_size = int(0.8 * num_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_val, y_val


def main():
    """Main function demonstrating basic usage."""
    print("🚀 Starting Automatic Hyperparameter Scheduling Demo")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create synthetic data
    print("📊 Creating synthetic dataset...")
    X_train, y_train, X_val, y_val = create_synthetic_data()
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    print("🧠 Initializing neural network...")
    model = SimpleNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize AutoScheduler
    print("⚙️  Initializing AutoScheduler...")
    scheduler = AutoScheduler(
        hyperparameters=['learning_rate', 'batch_size', 'weight_decay'],
        objectives=[
            ObjectiveType.CONVERGENCE_SPEED,
            ObjectiveType.FINAL_ACCURACY,
            ObjectiveType.COMPUTATIONAL_EFFICIENCY
        ],
        config={
            "meta_learner": {
                "model_type": "random_forest",
                "min_samples": 5,
                "update_frequency": 3
            },
            "multi_objective": {
                "method": "weighted_sum",
                "objective_weights": {
                    "convergence_speed": 0.3,
                    "final_accuracy": 0.5,
                    "computational_efficiency": 0.2
                }
            },
            "constraints": {
                "learning_rate": {"min": 1e-6, "max": 1.0, "smoothness": 0.3},
                "batch_size": {"min": 8, "max": 256},
                "weight_decay": {"min": 0.0, "max": 0.1}
            }
        }
    )
    
    # Train with automatic scheduling
    print("🎯 Starting training with automatic hyperparameter scheduling...")
    results = scheduler.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50
    )
    
    # Print results
    print("\n📈 Training completed!")
    print(f"Best performance: {results['best_performance']}")
    
    # Analyze schedule history
    schedule_history = results['schedule_history']
    print(f"\n📊 Schedule decisions made: {len(schedule_history)}")
    
    # Plot learning curves
    plot_learning_curves(results)
    
    # Print final model performance
    final_model = results['final_model']
    final_model.eval()
    
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = final_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            val_correct += pred.eq(target.view_as(pred)).sum().item()
            val_total += target.size(0)
    
    final_accuracy = val_correct / val_total
    print(f"\n🎯 Final validation accuracy: {final_accuracy:.4f}")
    
    # Save the scheduler state
    print("\n💾 Saving scheduler state...")
    scheduler.save("scheduler_state.pkl")
    
    print("\n✅ Demo completed successfully!")


def plot_learning_curves(results):
    """Plot learning curves and schedule history."""
    training_history = results['training_history']
    schedule_history = results['schedule_history']
    
    # Extract data for plotting
    epochs = [state.epoch for state in training_history]
    train_losses = [state.loss for state in training_history]
    train_accuracies = [state.accuracy for state in training_history]
    learning_rates = [state.learning_rate for state in training_history]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training accuracy
    ax2.plot(epochs, train_accuracies, 'g-', label='Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy Over Time')
    ax2.legend()
    ax2.grid(True)
    
    # Plot learning rate schedule
    ax3.plot(epochs, learning_rates, 'r-', label='Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.legend()
    ax3.grid(True)
    ax3.set_yscale('log')
    
    # Plot schedule confidence
    confidences = [decision.confidence for decision in schedule_history]
    ax4.plot(range(len(confidences)), confidences, 'purple', label='Schedule Confidence')
    ax4.set_xlabel('Schedule Decision')
    ax4.set_ylabel('Confidence')
    ax4.set_title('Schedule Decision Confidence')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Learning curves saved as 'learning_curves.png'")


if __name__ == "__main__":
    main()
