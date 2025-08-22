"""
Utility functions for computing training metrics and other helper functions.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import time
import psutil
import os


def compute_training_metrics(
    model: nn.Module,
    train_loader: Any,
    val_loader: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Compute comprehensive training metrics.
    
    Args:
        model: The model to evaluate
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run computations on
        
    Returns:
        Dictionary of computed metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    metrics = {}
    
    # Training metrics
    train_loss, train_accuracy = _compute_loader_metrics(model, train_loader, device)
    metrics["train_loss"] = train_loss
    metrics["train_accuracy"] = train_accuracy
    
    # Validation metrics
    if val_loader is not None:
        val_loss, val_accuracy = _compute_loader_metrics(model, val_loader, device)
        metrics["val_loss"] = val_loss
        metrics["val_accuracy"] = val_accuracy
    
    # Model complexity metrics
    model_metrics = _compute_model_metrics(model)
    metrics.update(model_metrics)
    
    # System metrics
    system_metrics = _compute_system_metrics()
    metrics.update(system_metrics)
    
    return metrics


def _compute_loader_metrics(
    model: nn.Module,
    data_loader: Any,
    device: torch.device
) -> Tuple[float, float]:
    """Compute loss and accuracy for a data loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Compute loss
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Compute accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def _compute_model_metrics(model: nn.Module) -> Dict[str, float]:
    """Compute model complexity and architecture metrics."""
    metrics = {}
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metrics["total_parameters"] = total_params
    metrics["trainable_parameters"] = trainable_params
    
    # Compute model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    metrics["model_size_mb"] = size_mb
    
    # Compute gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    metrics["gradient_norm"] = total_norm
    
    return metrics


def _compute_system_metrics() -> Dict[str, float]:
    """Compute system resource usage metrics."""
    metrics = {}
    
    # Memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    metrics["memory_usage_mb"] = memory_info.rss / 1024 / 1024
    
    # CPU usage
    metrics["cpu_percent"] = process.cpu_percent()
    
    # GPU memory usage (if available)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        metrics["gpu_memory_mb"] = gpu_memory
    
    return metrics


def compute_gradient_statistics(model: nn.Module) -> Dict[str, float]:
    """
    Compute detailed gradient statistics.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary of gradient statistics
    """
    stats = {}
    
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.data.view(-1))
    
    if not gradients:
        return {"gradient_norm": 0.0, "gradient_mean": 0.0, "gradient_std": 0.0}
    
    # Concatenate all gradients
    all_gradients = torch.cat(gradients)
    
    stats["gradient_norm"] = torch.norm(all_gradients).item()
    stats["gradient_mean"] = torch.mean(all_gradients).item()
    stats["gradient_std"] = torch.std(all_gradients).item()
    stats["gradient_min"] = torch.min(all_gradients).item()
    stats["gradient_max"] = torch.max(all_gradients).item()
    
    # Gradient explosion/vanishing detection
    stats["gradient_explosion"] = float(stats["gradient_norm"] > 10.0)
    stats["gradient_vanishing"] = float(stats["gradient_norm"] < 1e-6)
    
    return stats


def compute_loss_landscape_features(
    model: nn.Module,
    data_loader: Any,
    device: torch.device,
    num_samples: int = 100
) -> Dict[str, float]:
    """
    Compute loss landscape features for better understanding of training dynamics.
    
    Args:
        model: The model to analyze
        data_loader: Data loader for computing loss
        device: Device to run computations on
        num_samples: Number of samples to use for analysis
        
    Returns:
        Dictionary of loss landscape features
    """
    features = {}
    
    # Get current parameters
    current_params = [p.clone().detach() for p in model.parameters()]
    
    # Compute loss at current point
    model.eval()
    criterion = nn.CrossEntropyLoss()
    current_loss = 0.0
    sample_count = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            if sample_count >= num_samples:
                break
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            current_loss += loss.item()
            sample_count += 1
    
    current_loss /= sample_count
    features["current_loss"] = current_loss
    
    # Compute loss in random directions
    loss_samples = []
    for _ in range(10):  # Sample 10 random directions
        # Add random noise to parameters
        for i, param in enumerate(model.parameters()):
            noise = torch.randn_like(param) * 0.01  # Small perturbation
            param.data += noise
        
        # Compute loss with perturbed parameters
        perturbed_loss = 0.0
        sample_count = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                if sample_count >= num_samples:
                    break
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                perturbed_loss += loss.item()
                sample_count += 1
        
        perturbed_loss /= sample_count
        loss_samples.append(perturbed_loss)
        
        # Restore original parameters
        for i, param in enumerate(model.parameters()):
            param.data = current_params[i].clone()
    
    # Compute loss landscape features
    loss_samples = np.array(loss_samples)
    features["loss_variance"] = np.var(loss_samples)
    features["loss_range"] = np.max(loss_samples) - np.min(loss_samples)
    features["loss_smoothness"] = 1.0 / (1.0 + features["loss_variance"])
    
    return features


def compute_learning_curves(history: List[Dict[str, float]]) -> Dict[str, List[float]]:
    """
    Extract learning curves from training history.
    
    Args:
        history: List of training states or metrics dictionaries
        
    Returns:
        Dictionary of learning curves
    """
    curves = {
        "epochs": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }
    
    for i, state in enumerate(history):
        curves["epochs"].append(i)
        
        if "train_loss" in state:
            curves["train_loss"].append(state["train_loss"])
        if "train_accuracy" in state:
            curves["train_accuracy"].append(state["train_accuracy"])
        if "val_loss" in state:
            curves["val_loss"].append(state["val_loss"])
        if "val_accuracy" in state:
            curves["val_accuracy"].append(state["val_accuracy"])
        if "learning_rate" in state:
            curves["learning_rate"].append(state["learning_rate"])
    
    return curves


def detect_overfitting(history: List[Dict[str, float]], window_size: int = 5) -> Dict[str, Any]:
    """
    Detect overfitting patterns in training history.
    
    Args:
        history: List of training states or metrics dictionaries
        window_size: Window size for trend analysis
        
    Returns:
        Dictionary of overfitting indicators
    """
    if len(history) < window_size * 2:
        return {"overfitting_detected": False, "confidence": 0.0}
    
    indicators = {}
    
    # Extract learning curves
    curves = compute_learning_curves(history)
    
    if len(curves["train_loss"]) >= window_size * 2 and len(curves["val_loss"]) >= window_size * 2:
        # Compare recent trends
        recent_train_loss = curves["train_loss"][-window_size:]
        recent_val_loss = curves["val_loss"][-window_size:]
        earlier_train_loss = curves["train_loss"][-2*window_size:-window_size]
        earlier_val_loss = curves["val_loss"][-2*window_size:-window_size]
        
        # Compute trends
        train_trend = np.polyfit(range(len(recent_train_loss)), recent_train_loss, 1)[0]
        val_trend = np.polyfit(range(len(recent_val_loss)), recent_val_loss, 1)[0]
        earlier_train_trend = np.polyfit(range(len(earlier_train_loss)), earlier_train_loss, 1)[0]
        earlier_val_trend = np.polyfit(range(len(earlier_val_loss)), earlier_val_loss, 1)[0]
        
        # Overfitting indicators
        train_decreasing = train_trend < 0
        val_increasing = val_trend > 0
        gap_widening = (val_trend - train_trend) > (earlier_val_trend - earlier_train_trend)
        
        overfitting_score = 0.0
        if train_decreasing:
            overfitting_score += 0.3
        if val_increasing:
            overfitting_score += 0.4
        if gap_widening:
            overfitting_score += 0.3
        
        indicators["overfitting_detected"] = overfitting_score > 0.5
        indicators["overfitting_score"] = overfitting_score
        indicators["train_trend"] = train_trend
        indicators["val_trend"] = val_trend
        indicators["gap_widening"] = gap_widening
    
    return indicators


def compute_convergence_metrics(history: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Compute convergence-related metrics.
    
    Args:
        history: List of training states or metrics dictionaries
        
    Returns:
        Dictionary of convergence metrics
    """
    if len(history) < 5:
        return {"convergence_rate": 0.0, "convergence_stability": 0.0}
    
    metrics = {}
    
    # Extract loss curve
    losses = [state.get("train_loss", 0.0) for state in history]
    
    # Fit exponential decay model
    epochs = np.arange(len(losses))
    try:
        # Use log-linear fit for convergence rate
        log_losses = np.log(np.array(losses) + 1e-8)
        convergence_rate = -np.polyfit(epochs, log_losses, 1)[0]
        metrics["convergence_rate"] = max(0, convergence_rate)
    except:
        metrics["convergence_rate"] = 0.0
    
    # Compute convergence stability
    if len(losses) > 5:
        recent_losses = losses[-5:]
        loss_variance = np.var(recent_losses)
        metrics["convergence_stability"] = 1.0 / (1.0 + loss_variance)
    else:
        metrics["convergence_stability"] = 0.0
    
    # Detect convergence
    if len(losses) > 10:
        recent_trend = np.polyfit(range(5), losses[-5:], 1)[0]
        metrics["converged"] = abs(recent_trend) < 1e-4
    else:
        metrics["converged"] = False
    
    return metrics


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_memory(bytes_value: float) -> str:
    """Format memory in bytes to human-readable string."""
    if bytes_value < 1024:
        return f"{bytes_value:.0f}B"
    elif bytes_value < 1024**2:
        return f"{bytes_value/1024:.1f}KB"
    elif bytes_value < 1024**3:
        return f"{bytes_value/1024**2:.1f}MB"
    else:
        return f"{bytes_value/1024**3:.1f}GB"
