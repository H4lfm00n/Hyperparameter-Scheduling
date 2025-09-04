"""
Example demonstrating how to use the hyperparameter scheduling callbacks.

This example shows how to integrate the hyperparameter scheduler with different
deep learning frameworks using the callback system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from hyperparameter_scheduling.callbacks import (
    GenericCallback,
    PyTorchCallback,
    TensorFlowCallback
)


def create_synthetic_data(num_samples=1000, input_size=784, num_classes=10):
    """Create synthetic data for demonstration."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Split into train/val
    train_size = int(0.8 * num_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_val, y_val


class SimpleNet(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def example_generic_callback():
    """Example using the generic callback with a custom training loop."""
    print("=== Generic Callback Example ===")
    
    # Create data
    X_train, y_train, X_val, y_val = create_synthetic_data()
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model and optimizer
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize callback
    callback = GenericCallback(
        hyperparameters=['learning_rate', 'batch_size', 'weight_decay'],
        objectives=['convergence_speed', 'final_accuracy'],
        update_frequency=10,  # Update every 10 steps
        log_level="INFO"
    )
    
    # Set initial hyperparameters
    callback.set_current_hyperparameters(
        learning_rate=0.001,
        batch_size=32,
        weight_decay=0.0
    )
    
    # Training loop
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    callback.on_train_begin()
    
    for epoch in range(num_epochs):
        callback.on_epoch_begin(epoch)
        
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            callback.on_batch_begin(batch_idx)
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Update metrics
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            total_loss += loss.item()
            
            # Update callback with current metrics
            callback.set_current_metrics(
                loss=loss.item(),
                accuracy=correct / total
            )
            
            # Apply new hyperparameters if callback updated them
            current_hp = callback.get_current_hyperparameters()
            if 'learning_rate' in current_hp:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_hp['learning_rate']
            
            optimizer.step()
            
            callback.on_batch_end(batch_idx, {
                'loss': loss.item(),
                'accuracy': correct / total
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        callback.on_epoch_end(epoch, {
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        })
        
        print(f"Epoch {epoch}: Loss={avg_val_loss:.4f}, Accuracy={val_accuracy:.4f}")
    
    callback.on_train_end()
    
    # Print results
    summary = callback.get_performance_summary()
    print(f"Training completed!")
    print(f"Best performance: {summary['best_performance']}")
    print(f"Schedule changes: {summary['schedule_changes']}")
    
    return callback


def example_pytorch_callback():
    """Example using the PyTorch-specific callback."""
    print("\n=== PyTorch Callback Example ===")
    
    # Create data
    X_train, y_train, X_val, y_val = create_synthetic_data()
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model and optimizer
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize PyTorch callback
    callback = PyTorchCallback(
        hyperparameters=['learning_rate', 'batch_size', 'weight_decay'],
        objectives=['convergence_speed', 'final_accuracy'],
        update_frequency=10,
        optimizer=optimizer,
        model=model
    )
    
    # Training loop (similar to generic but with PyTorch-specific features)
    num_epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    callback.on_train_begin()
    
    for epoch in range(num_epochs):
        callback.on_epoch_begin(epoch)
        
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            callback.on_batch_begin(batch_idx)
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Update metrics
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            total_loss += loss.item()
            
            # Update callback with current metrics
            callback.set_current_metrics(
                loss=loss.item(),
                accuracy=correct / total
            )
            
            optimizer.step()
            
            callback.on_batch_end(batch_idx, {
                'loss': loss.item(),
                'accuracy': correct / total
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        callback.on_epoch_end(epoch, {
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        })
        
        print(f"Epoch {epoch}: Loss={avg_val_loss:.4f}, Accuracy={val_accuracy:.4f}")
    
    callback.on_train_end()
    
    # Print results
    summary = callback.get_performance_summary()
    print(f"Training completed!")
    print(f"Best performance: {summary['best_performance']}")
    print(f"Schedule changes: {summary['schedule_changes']}")
    
    return callback


def plot_schedule_history(callback, title):
    """Plot the schedule history."""
    history = callback.get_schedule_history()
    
    if not history:
        print(f"No schedule history for {title}")
        return
    
    # Extract data
    epochs = [h['epoch'] for h in history]
    learning_rates = [h['schedule'].get('learning_rate', 0) for h in history]
    batch_sizes = [h['schedule'].get('batch_size', 0) for h in history]
    confidences = [h['confidence'] for h in history]
    
    # Create plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Learning rate over time
    ax1.plot(epochs, learning_rates, 'b-', marker='o')
    ax1.set_title('Learning Rate Schedule')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate')
    ax1.grid(True)
    
    # Batch size over time
    ax2.plot(epochs, batch_sizes, 'r-', marker='s')
    ax2.set_title('Batch Size Schedule')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Batch Size')
    ax2.grid(True)
    
    # Confidence over time
    ax3.plot(epochs, confidences, 'g-', marker='^')
    ax3.set_title('Schedule Confidence')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Confidence')
    ax3.grid(True)
    
    # Combined plot
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(epochs, learning_rates, 'b-', label='Learning Rate')
    line2 = ax4_twin.plot(epochs, batch_sizes, 'r-', label='Batch Size')
    ax4.set_title('Combined Schedule')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate', color='b')
    ax4_twin.set_ylabel('Batch Size', color='r')
    ax4.grid(True)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.suptitle(f'Schedule History - {title}', y=1.02)
    plt.show()


def main():
    """Main function demonstrating callback usage."""
    print("Hyperparameter Scheduling Callback Examples")
    print("=" * 50)
    
    # Run examples
    generic_callback = example_generic_callback()
    pytorch_callback = example_pytorch_callback()
    
    # Plot results
    plot_schedule_history(generic_callback, "Generic Callback")
    plot_schedule_history(pytorch_callback, "PyTorch Callback")
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nKey benefits of the callback system:")
    print("1. Easy integration with any training loop")
    print("2. Framework-specific optimizations")
    print("3. Automatic hyperparameter updates")
    print("4. Comprehensive logging and monitoring")
    print("5. Transfer learning capabilities")


if __name__ == "__main__":
    main()
