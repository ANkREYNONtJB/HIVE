import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class FeatureDistribution:
    """Data class for storing feature distribution statistics"""
    mean: float
    std: float
    entropy: float

class QuantumFeatureAnalyzer:
    """
    Analyzes quantum features using fractal analysis and cross-scale correlations
    with robust error handling and logging.
    """
    def __init__(self, n_features: int, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.n_features = n_features
        self.device = device
        self.feature_stats: Dict[str, List[FeatureDistribution]] = {}
        
    def analyze_fractal_features(
        self, 
        features: torch.Tensor,
        layer_importance: torch.Tensor,
        feedback_weights: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Analyzes fractal features with comprehensive error handling
        and detailed statistics computation.
        """
        try:
            features = features.to(self.device)
            layer_importance = layer_importance.to(self.device)
            
            # Compute feature distributions
            distributions = []
            for f in features:
                try:
                    distribution = FeatureDistribution(
                        mean=f.mean().item(),
                        std=f.std().item(),
                        entropy=self._compute_entropy(f)
                    )
                    distributions.append(distribution)
                except RuntimeError as e:
                    print(f"Warning: Error computing distribution for feature: {e}")
                    continue
            
            # Calculate cross-scale correlations
            correlations = self._compute_correlations(features)
            
            # Analyze feedback influence
            feedback_stats = self._analyze_feedback(feedback_weights)
            
            return {
                'fractal_features': [f.detach().cpu().numpy() for f in features],
                'pattern_statistics': {
                    'feature_distributions': distributions,
                    'layer_importance': layer_importance.detach().cpu().numpy(),
                    'cross_scale_correlations': correlations.detach().cpu().numpy(),
                    'feedback_influence': feedback_stats
                },
                'self_similarity': correlations.mean().item()
            }
            
        except Exception as e:
            print(f"Error in fractal feature analysis: {e}")
            return self._generate_error_stats()
            
    def _compute_entropy(self, feature: torch.Tensor) -> float:
        """Compute entropy of feature distribution with error handling"""
        try:
            probs = F.softmax(feature, dim=-1)
            return -torch.sum(probs * torch.log(probs + 1e-10)).item()
        except Exception as e:
            print(f"Warning: Error computing entropy: {e}")
            return 0.0
            
    def _compute_correlations(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Compute cross-scale correlations between features"""
        try:
            correlations = []
            for i in range(len(features)-1):
                f1 = features[i]
                f2 = features[i+1]
                
                # Handle dimension mismatch
                if f1.size(-1) != f2.size(-1):
                    f1 = F.interpolate(
                        f1.unsqueeze(0), 
                        size=f2.size(-1), 
                        mode='linear'
                    ).squeeze(0)
                
                correlation = F.cosine_similarity(
                    f1.unsqueeze(1),
                    f2.unsqueeze(0),
                    dim=-1
                ).mean()
                correlations.append(correlation)
                
            return torch.stack(correlations)
            
        except Exception as e:
            print(f"Warning: Error computing correlations: {e}")
            return torch.zeros(1, device=self.device)
            
    def _analyze_feedback(self, feedback_weights: List[torch.Tensor]) -> List[float]:
        """Analyze feedback influence with error handling"""
        feedback_stats = []
        for i, weight in enumerate(feedback_weights):
            try:
                influence = torch.norm(weight, p='fro').item()
                feedback_stats.append(influence)
            except Exception as e:
                print(f"Warning: Error computing feedback influence for layer {i}: {e}")
                feedback_stats.append(0.0)
        return feedback_stats
        
    def _generate_error_stats(self) -> Dict[str, Any]:
        """Generate empty statistics in case of critical errors"""
        return {
            'fractal_features': [],
            'pattern_statistics': {
                'feature_distributions': [],
                'layer_importance': np.array([]),
                'cross_scale_correlations': np.array([]),
                'feedback_influence': []
            },
            'self_similarity': 0.0
        }