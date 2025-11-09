"""
Transfer learning component for cross-problem hyperparameter schedule generalization.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

from ..core.base import TrainingState


class ProblemSignature:
    """Signature of a problem for similarity computation."""
    
    def __init__(
        self,
        problem_id: str,
        features: Dict[str, float],
        schedule_history: List[Dict[str, float]],
        performance_history: List[Dict[str, float]]
    ):
        self.problem_id = problem_id
        self.features = features
        self.schedule_history = schedule_history
        self.performance_history = performance_history
        self.similarity_scores = {}
    
    def get_feature_vector(self) -> np.ndarray:
        """Get feature vector for similarity computation."""
        return np.array(list(self.features.values()))
    
    def get_schedule_vector(self) -> np.ndarray:
        """Get schedule history as a vector."""
        if not self.schedule_history:
            return np.array([])
        
        # Flatten schedule history
        schedule_features = []
        for schedule in self.schedule_history:
            schedule_features.extend(list(schedule.values()))
        
        return np.array(schedule_features)


class TransferLearner:
    """
    Transfer learning component that enables cross-problem generalization
    of hyperparameter schedules.
    
    This component learns to identify similar problems and transfer
    successful hyperparameter schedules between them.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the transfer learner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Problem database
        self.problem_signatures = {}
        self.problem_clusters = {}
        
        # Similarity computation
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        self.max_similar_problems = self.config.get("max_similar_problems", 5)
        
        # Feature extraction
        self.feature_extractor = ProblemFeatureExtractor(config)
        
        # Schedule transfer
        self.transfer_weight = self.config.get("transfer_weight", 0.3)
        self.adaptation_rate = self.config.get("adaptation_rate", 0.1)
        
        # Clustering
        self.n_clusters = self.config.get("n_clusters", 10)
        self.cluster_model = None
        self.scaler = StandardScaler()
        
    def has_similar_problems(self) -> bool:
        """Check if there are similar problems in the database."""
        return len(self.problem_signatures) > 0
    
    def get_adjustment(
        self,
        dynamics_features: Dict[str, float],
        current_state: TrainingState
    ) -> Dict[str, float]:
        """
        Get hyperparameter adjustment based on similar problems.
        
        Args:
            dynamics_features: Current training dynamics features
            current_state: Current training state
            
        Returns:
            Dictionary of hyperparameter adjustments
        """
        if not self.has_similar_problems():
            return {}
        
        # Find similar problems
        similar_problems = self._find_similar_problems(dynamics_features, current_state)
        
        if not similar_problems:
            return {}
        
        # Compute transfer adjustment
        adjustment = self._compute_transfer_adjustment(similar_problems, current_state)
        
        return adjustment
    
    def update(
        self,
        state: TrainingState,
        performance: Dict[str, float]
    ) -> None:
        """
        Update the transfer learner with new training data.
        
        Args:
            state: Training state
            performance: Performance metrics
        """
        # Extract problem features
        problem_features = self.feature_extractor.extract_features(state, performance)
        
        # Create or update problem signature
        problem_id = self._get_problem_id(state)
        
        if problem_id not in self.problem_signatures:
            # Create new problem signature
            self.problem_signatures[problem_id] = ProblemSignature(
                problem_id=problem_id,
                features=problem_features,
                schedule_history=[],
                performance_history=[]
            )
        
        # Update signature
        signature = self.problem_signatures[problem_id]
        signature.features.update(problem_features)
        
        # Store current schedule and performance
        current_schedule = {
            'learning_rate': state.learning_rate,
            'batch_size': float(state.batch_size),
            'weight_decay': 0.0,  # Would need to extract from optimizer
        }
        
        signature.schedule_history.append(current_schedule)
        signature.performance_history.append(performance)
        
        # Update clusters periodically
        if len(self.problem_signatures) % 5 == 0:
            self._update_clusters()
    
    def _find_similar_problems(
        self,
        dynamics_features: Dict[str, float],
        current_state: TrainingState
    ) -> List[Tuple[str, float]]:
        """
        Find problems similar to the current one.
        
        Returns:
            List of (problem_id, similarity_score) tuples
        """
        current_problem_id = self._get_problem_id(current_state)
        current_features = self.feature_extractor.extract_features(current_state, {})
        
        similarities = []
        
        for problem_id, signature in self.problem_signatures.items():
            if problem_id == current_problem_id:
                continue
            
            # Compute similarity
            similarity = self._compute_similarity(current_features, signature.features)
            
            if similarity >= self.similarity_threshold:
                similarities.append((problem_id, similarity))
        
        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self.max_similar_problems]
    
    def _compute_similarity(
        self,
        features1: Dict[str, float],
        features2: Dict[str, float]
    ) -> float:
        """Compute similarity between two problem feature sets."""
        # Get common features
        common_keys = set(features1.keys()) & set(features2.keys())
        
        if not common_keys:
            return 0.0
        
        # Extract feature vectors
        vec1 = np.array([features1[k] for k in common_keys])
        vec2 = np.array([features2[k] for k in common_keys])
        
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        
        return max(0.0, similarity)  # Ensure non-negative
    
    def _compute_transfer_adjustment(
        self,
        similar_problems: List[Tuple[str, float]],
        current_state: TrainingState
    ) -> Dict[str, float]:
        """
        Compute hyperparameter adjustment based on similar problems.
        
        Args:
            similar_problems: List of (problem_id, similarity_score) tuples
            current_state: Current training state
            
        Returns:
            Dictionary of hyperparameter adjustments
        """
        if not similar_problems:
            return {}
        
        # Collect successful schedules from similar problems
        successful_schedules = []
        weights = []
        
        for problem_id, similarity in similar_problems:
            signature = self.problem_signatures[problem_id]
            
            # Find best performing schedule for this problem
            best_performance_idx = self._find_best_performance(signature.performance_history)
            
            if best_performance_idx is not None:
                best_schedule = signature.schedule_history[best_performance_idx]
                successful_schedules.append(best_schedule)
                weights.append(similarity)
        
        if not successful_schedules:
            return {}
        
        # Compute weighted average of successful schedules
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        adjustment = {}
        for param in ['learning_rate', 'batch_size', 'weight_decay']:
            if param in successful_schedules[0]:
                weighted_values = []
                for i, schedule in enumerate(successful_schedules):
                    if param in schedule:
                        weighted_values.append(schedule[param] * weights[i])
                
                if weighted_values:
                    adjustment[param] = np.sum(weighted_values)
        
        # Apply adaptation based on current state
        adjustment = self._adapt_adjustment(adjustment, current_state)
        
        return adjustment
    
    def _find_best_performance(self, performance_history: List[Dict[str, float]]) -> Optional[int]:
        """Find the index of the best performing configuration."""
        if not performance_history:
            return None
        
        # Use a composite score based on multiple metrics
        scores = []
        for perf in performance_history:
            # Simple composite score (can be made more sophisticated)
            score = 0.0
            if 'train_accuracy' in perf:
                score += perf['train_accuracy']
            if 'val_accuracy' in perf:
                score += perf['val_accuracy'] * 0.8  # Weight validation accuracy
            if 'train_loss' in perf:
                score -= perf['train_loss'] * 0.1  # Penalize high loss
            
            scores.append(score)
        
        return np.argmax(scores)
    
    def _adapt_adjustment(
        self,
        adjustment: Dict[str, float],
        current_state: TrainingState
    ) -> Dict[str, float]:
        """
        Adapt the transfer adjustment based on current training state.
        
        Args:
            adjustment: Raw transfer adjustment
            current_state: Current training state
            
        Returns:
            Adapted adjustment
        """
        adapted = {}
        
        for param, value in adjustment.items():
            if param == 'learning_rate':
                # Adapt based on current learning rate and training progress
                current_lr = current_state.learning_rate
                progress = min(current_state.epoch / 100.0, 1.0)  # Normalize progress
                
                # Gradually blend transfer value with current value
                blend_factor = self.transfer_weight * (1.0 - progress * 0.5)
                adapted[param] = (1.0 - blend_factor) * current_lr + blend_factor * value
                
            elif param == 'batch_size':
                # Adapt batch size based on memory constraints and training stability
                current_batch = current_state.batch_size
                memory_usage = current_state.memory_usage
                
                # Reduce transfer influence if memory usage is high
                memory_factor = max(0.1, 1.0 - memory_usage / 100.0)  # Assuming memory in GB
                blend_factor = self.transfer_weight * memory_factor
                adapted[param] = (1.0 - blend_factor) * current_batch + blend_factor * value
                
            else:
                # For other parameters, use simple blending
                adapted[param] = value
        
        return adapted
    
    def _get_problem_id(self, state: TrainingState) -> str:
        """Generate a problem ID based on training state."""
        # This is a simplified implementation
        # In practice, you might want to include more problem characteristics
        return f"problem_{state.epoch}_{state.step}"
    
    def _update_clusters(self):
        """Update problem clusters for better similarity computation."""
        if len(self.problem_signatures) < 2:
            return
        
        # Extract feature vectors
        feature_vectors = []
        problem_ids = []
        
        for problem_id, signature in self.problem_signatures.items():
            feature_vector = signature.get_feature_vector()
            if len(feature_vector) > 0:
                feature_vectors.append(feature_vector)
                problem_ids.append(problem_id)
        
        if len(feature_vectors) < 2:
            return
        
        # Normalize features
        feature_vectors = np.array(feature_vectors)
        feature_vectors_scaled = self.scaler.fit_transform(feature_vectors)
        
        # Perform clustering
        n_clusters = min(self.n_clusters, len(feature_vectors) - 1)
        if n_clusters < 2:
            return
        
        # Use single thread to avoid segfault issues on macOS
        import os
        old_threads = os.environ.get('OMP_NUM_THREADS', None)
        os.environ['OMP_NUM_THREADS'] = '1'
        try:
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = self.cluster_model.fit_predict(feature_vectors_scaled)
        except Exception as e:
            # Fallback: assign all problems to same cluster
            import warnings
            warnings.warn(f"Clustering failed: {e}. Assigning all problems to cluster 0.")
            cluster_labels = np.zeros(len(feature_vectors), dtype=int)
        finally:
            if old_threads is not None:
                os.environ['OMP_NUM_THREADS'] = old_threads
            elif 'OMP_NUM_THREADS' in os.environ:
                del os.environ['OMP_NUM_THREADS']
        
        # Store cluster assignments
        self.problem_clusters = {}
        for i, problem_id in enumerate(problem_ids):
            self.problem_clusters[problem_id] = cluster_labels[i]
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the transfer learner."""
        return {
            "problem_signatures": self.problem_signatures,
            "problem_clusters": self.problem_clusters,
            "cluster_model": self.cluster_model,
            "scaler": self.scaler,
            "config": self.config,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load the state of the transfer learner."""
        self.problem_signatures = state["problem_signatures"]
        self.problem_clusters = state["problem_clusters"]
        self.cluster_model = state["cluster_model"]
        self.scaler = state["scaler"]
        self.config.update(state["config"])
    
    def save_database(self, path: str) -> None:
        """Save the problem database to disk securely."""
        from ..security.file_security import validate_file_path, SecurityError
        from ..security.data_validation import validate_config_data
        
        try:
            # Validate file path
            validated_path = validate_file_path(path, allowed_extensions=['.joblib', '.pkl'])
            
            # Validate config before saving
            validated_config = validate_config_data(self.config)
            
            database = {
                "problem_signatures": self.problem_signatures,
                "problem_clusters": self.problem_clusters,
                "cluster_model": self.cluster_model,
                "scaler": self.scaler,
                "config": validated_config,
            }
            joblib.dump(database, validated_path)
            
        except SecurityError as e:
            raise SecurityError(f"Failed to save database: {e}")
        except Exception as e:
            raise Exception(f"Failed to save database: {e}")
    
    def load_database(self, path: str) -> None:
        """Load the problem database from disk securely."""
        from ..security.file_security import validate_file_path, SecurityError
        from ..security.data_validation import safe_joblib_load, DataValidationError
        
        try:
            # Validate file path
            validated_path = validate_file_path(path, allowed_extensions=['.joblib', '.pkl'])
            
            # Expected keys in the database
            expected_keys = {"problem_signatures", "problem_clusters", "cluster_model", "scaler", "config"}
            
            # Safely load and validate data
            database = safe_joblib_load(str(validated_path), expected_keys)
            
            self.problem_signatures = database["problem_signatures"]
            self.problem_clusters = database["problem_clusters"]
            self.cluster_model = database["cluster_model"]
            self.scaler = database["scaler"]
            self.config.update(database["config"])
            
        except (SecurityError, DataValidationError) as e:
            raise SecurityError(f"Failed to load database: {e}")
        except Exception as e:
            raise Exception(f"Failed to load database: {e}")
    
    def get_similarity_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get similarity matrix between all problems."""
        problem_ids = list(self.problem_signatures.keys())
        similarity_matrix = {}
        
        for i, problem_id1 in enumerate(problem_ids):
            similarity_matrix[problem_id1] = {}
            for j, problem_id2 in enumerate(problem_ids):
                if i == j:
                    similarity_matrix[problem_id1][problem_id2] = 1.0
                else:
                    signature1 = self.problem_signatures[problem_id1]
                    signature2 = self.problem_signatures[problem_id2]
                    similarity = self._compute_similarity(signature1.features, signature2.features)
                    similarity_matrix[problem_id1][problem_id2] = similarity
        
        return similarity_matrix
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """Get statistics about transfer learning performance."""
        stats = {
            "num_problems": len(self.problem_signatures),
            "num_clusters": len(set(self.problem_clusters.values())) if self.problem_clusters else 0,
            "avg_similarity": 0.0,
            "transfer_success_rate": 0.0,
        }
        
        if len(self.problem_signatures) > 1:
            # Compute average similarity
            similarities = []
            problem_ids = list(self.problem_signatures.keys())
            
            for i in range(len(problem_ids)):
                for j in range(i + 1, len(problem_ids)):
                    signature1 = self.problem_signatures[problem_ids[i]]
                    signature2 = self.problem_signatures[problem_ids[j]]
                    similarity = self._compute_similarity(signature1.features, signature2.features)
                    similarities.append(similarity)
            
            if similarities:
                stats["avg_similarity"] = np.mean(similarities)
        
        return stats


class ProblemFeatureExtractor:
    """Extract features that characterize a problem for transfer learning."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    def extract_features(
        self,
        state: TrainingState,
        performance: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Extract features that characterize the current problem.
        
        Args:
            state: Training state
            performance: Performance metrics
            
        Returns:
            Dictionary of problem features
        """
        features = {
            # Training dynamics features
            "loss_magnitude": state.loss,
            "accuracy_level": state.accuracy,
            "gradient_magnitude": state.gradient_norm,
            "learning_rate_level": state.learning_rate,
            "batch_size_level": float(state.batch_size),
            
            # Performance features
            "train_accuracy": performance.get("train_accuracy", 0.0),
            "val_accuracy": performance.get("val_accuracy", 0.0),
            "train_loss": performance.get("train_loss", 0.0),
            "val_loss": performance.get("val_loss", 0.0),
            
            # Training progress features
            "epoch_progress": state.epoch / 100.0,  # Normalized progress
            "step_progress": state.step / 1000.0,   # Normalized progress
            
            # Resource usage features
            "memory_usage": state.memory_usage,
            "training_time": state.training_time,
        }
        
        return features
