"""
Transfer Learning Demo for Automatic Hyperparameter Scheduling.

This example demonstrates how the library can learn from previous problems
and transfer successful hyperparameter schedules to new, similar problems.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import time

from hyperparameter_scheduling import AutoScheduler, ObjectiveType


class FlexibleNet(nn.Module):
    """Flexible neural network that can handle different input/output sizes."""
    
    def __init__(self, input_size, hidden_size=128, output_size=1, task_type='classification'):
        super(FlexibleNet, self).__init__()
        self.task_type = task_type
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        if self.task_type == 'classification':
            return x
        else:
            return x.squeeze()


def create_problem_data(problem_type, num_samples=1000, noise_level=0.1):
    """Create different types of problems for transfer learning demonstration."""
    
    if problem_type == 'classification_easy':
        # Easy classification problem
        X, y = make_classification(
            n_samples=num_samples,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42
        )
        task_type = 'classification'
        output_size = 3
        
    elif problem_type == 'classification_hard':
        # Hard classification problem
        X, y = make_classification(
            n_samples=num_samples,
            n_features=50,
            n_informative=10,
            n_redundant=30,
            n_classes=5,
            n_clusters_per_class=1,
            random_state=43
        )
        task_type = 'classification'
        output_size = 5
        
    elif problem_type == 'regression_linear':
        # Linear regression problem
        X, y = make_regression(
            n_samples=num_samples,
            n_features=10,
            n_informative=8,
            noise=noise_level,
            random_state=44
        )
        task_type = 'regression'
        output_size = 1
        
    elif problem_type == 'regression_nonlinear':
        # Nonlinear regression problem
        X, y = make_regression(
            n_samples=num_samples,
            n_features=15,
            n_informative=5,
            n_targets=1,
            noise=noise_level,
            random_state=45
        )
        # Add some nonlinearity
        y = y + 0.1 * np.sin(X[:, 0]) + 0.05 * X[:, 1]**2
        task_type = 'regression'
        output_size = 1
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train/val
    train_size = int(0.8 * num_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_val, y_val, task_type, output_size


def train_with_scheduler(scheduler, model, train_loader, val_loader, epochs=30, problem_name=""):
    """Train a model using the scheduler and return results."""
    print(f"🎯 Training on {problem_name}...")
    
    start_time = time.time()
    results = scheduler.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs
    )
    training_time = time.time() - start_time
    
    print(f"✅ Completed {problem_name} in {training_time:.2f}s")
    print(f"   Best performance: {results['best_performance']}")
    
    return results, training_time


def main():
    """Main function demonstrating transfer learning capabilities."""
    print("🚀 Starting Transfer Learning Demo for Hyperparameter Scheduling")
    print("=" * 70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define problems to solve (in order of increasing difficulty)
    problems = [
        ('classification_easy', 'Easy Classification'),
        ('classification_hard', 'Hard Classification'),
        ('regression_linear', 'Linear Regression'),
        ('regression_nonlinear', 'Nonlinear Regression')
    ]
    
    # Initialize scheduler with transfer learning capabilities
    print("⚙️  Initializing AutoScheduler with transfer learning...")
    scheduler = AutoScheduler(
        hyperparameters=['learning_rate', 'batch_size', 'weight_decay'],
        objectives=[
            ObjectiveType.CONVERGENCE_SPEED,
            ObjectiveType.FINAL_ACCURACY,
            ObjectiveType.STABILITY
        ],
        config={
            "meta_learner": {
                "model_type": "random_forest",
                "min_samples": 3,
                "update_frequency": 2
            },
            "transfer_learner": {
                "similarity_threshold": 0.6,
                "max_similar_problems": 3,
                "transfer_weight": 0.4
            },
            "multi_objective": {
                "method": "weighted_sum",
                "objective_weights": {
                    "convergence_speed": 0.3,
                    "final_accuracy": 0.5,
                    "stability": 0.2
                }
            },
            "constraints": {
                "learning_rate": {"min": 1e-6, "max": 1.0, "smoothness": 0.2},
                "batch_size": {"min": 16, "max": 128},
                "weight_decay": {"min": 0.0, "max": 0.01}
            }
        }
    )
    
    # Results storage
    all_results = {}
    training_times = {}
    
    # Solve each problem
    for i, (problem_type, problem_name) in enumerate(problems):
        print(f"\n{'='*20} Problem {i+1}: {problem_name} {'='*20}")
        
        # Create problem data
        X_train, y_train, X_val, y_val, task_type, output_size = create_problem_data(problem_type)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train) if task_type == 'classification' else torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val) if task_type == 'classification' else torch.FloatTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create model
        input_size = X_train.shape[1]
        model = FlexibleNet(
            input_size=input_size,
            hidden_size=128,
            output_size=output_size,
            task_type=task_type
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Train with scheduler
        results, training_time = train_with_scheduler(
            scheduler, model, train_loader, val_loader, 
            epochs=30, problem_name=problem_name
        )
        
        # Store results
        all_results[problem_name] = results
        training_times[problem_name] = training_time
        
        # Print transfer learning statistics
        transfer_stats = scheduler.transfer_learner.get_transfer_statistics()
        print(f"   Transfer learning stats: {transfer_stats}")
        
        # Check if similar problems were found
        if transfer_stats['num_problems'] > 1:
            print(f"   🎯 Found {transfer_stats['num_problems']} similar problems")
            print(f"   📊 Average similarity: {transfer_stats['avg_similarity']:.3f}")
    
    # Analyze results
    print("\n" + "="*70)
    print("📊 TRANSFER LEARNING ANALYSIS")
    print("="*70)
    
    # Compare performance across problems
    print("\nPerformance Comparison:")
    print("-" * 50)
    for problem_name, results in all_results.items():
        best_perf = results['best_performance']
        training_time = training_times[problem_name]
        
        if 'train_accuracy' in best_perf:
            metric = f"Accuracy: {best_perf['train_accuracy']:.4f}"
        elif 'train_loss' in best_perf:
            metric = f"Loss: {best_perf['train_loss']:.4f}"
        
        print(f"{problem_name:25} | {metric:15} | Time: {training_time:.2f}s")
    
    # Analyze schedule evolution
    print("\nSchedule Evolution Analysis:")
    print("-" * 50)
    for problem_name, results in all_results.items():
        schedule_history = results['schedule_history']
        print(f"\n{problem_name}:")
        
        if len(schedule_history) > 0:
            # Show first and last schedule
            first_schedule = schedule_history[0].hyperparameters
            last_schedule = schedule_history[-1].hyperparameters
            
            print(f"  Initial LR: {first_schedule.get('learning_rate', 'N/A'):.6f}")
            print(f"  Final LR:   {last_schedule.get('learning_rate', 'N/A'):.6f}")
            print(f"  Confidence: {schedule_history[-1].confidence:.3f}")
    
    # Plot transfer learning benefits
    plot_transfer_learning_analysis(all_results, training_times)
    
    # Save scheduler state for future use
    print("\n💾 Saving scheduler state...")
    scheduler.save("transfer_learning_scheduler.pkl")
    
    print("\n✅ Transfer learning demo completed successfully!")


def plot_transfer_learning_analysis(all_results, training_times):
    """Plot analysis of transfer learning benefits."""
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Training time progression
    problem_names = list(all_results.keys())
    times = [training_times[name] for name in problem_names]
    
    ax1.plot(range(len(problem_names)), times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Problem Number')
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time Progression\n(Transfer Learning Benefits)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(len(problem_names)))
    ax1.set_xticklabels([f"P{i+1}" for i in range(len(problem_names))])
    
    # Plot 2: Performance comparison
    performances = []
    for name in problem_names:
        best_perf = all_results[name]['best_performance']
        if 'train_accuracy' in best_perf:
            performances.append(best_perf['train_accuracy'])
        elif 'train_loss' in best_perf:
            # Convert loss to a "performance" metric (lower is better)
            performances.append(1.0 / (1.0 + best_perf['train_loss']))
    
    ax2.bar(range(len(problem_names)), performances, color='green', alpha=0.7)
    ax2.set_xlabel('Problem')
    ax2.set_ylabel('Performance')
    ax2.set_title('Final Performance Across Problems')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(problem_names)))
    ax2.set_xticklabels([f"P{i+1}" for i in range(len(problem_names))])
    
    # Plot 3: Learning rate schedules
    for i, (name, results) in enumerate(all_results.items()):
        schedule_history = results['schedule_history']
        if len(schedule_history) > 0:
            epochs = range(len(schedule_history))
            lrs = [s.hyperparameters.get('learning_rate', 0.001) for s in schedule_history]
            ax3.plot(epochs, lrs, label=f'P{i+1}: {name}', linewidth=2)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedules')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Schedule confidence evolution
    for i, (name, results) in enumerate(all_results.items()):
        schedule_history = results['schedule_history']
        if len(schedule_history) > 0:
            epochs = range(len(schedule_history))
            confidences = [s.confidence for s in schedule_history]
            ax4.plot(epochs, confidences, label=f'P{i+1}: {name}', linewidth=2)
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Schedule Confidence')
    ax4.set_title('Schedule Decision Confidence')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transfer_learning_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Transfer learning analysis saved as 'transfer_learning_analysis.png'")


if __name__ == "__main__":
    main()
