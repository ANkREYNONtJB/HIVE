import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np

class AdaptiveFractalAnalysis(nn.Module):
    """
    Enhanced fractal pattern analysis with dynamic scale selection
    and multiple fractal dimension estimators.
    """
    def __init__(
        self, 
        input_dim: int,
        min_scale: int = 2,
        max_scale: int = 16,
        n_scales: int = 5
    ):
        super().__init__()
        self.input_dim = input_dim
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.n_scales = n_scales
        
        # Learnable scale parameters
        self.scale_params = nn.Parameter(
            torch.linspace(np.log(min_scale), np.log(max_scale), n_scales)
        )
        
        # Scale importance weights
        self.scale_importance = nn.Parameter(torch.ones(n_scales))
        
    def forward(
        self, 
        x: torch.Tensor,
        return_scales: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[int]]]:
        """
        Compute fractal dimensions using multiple estimators and dynamic scales
        """
        # Get dynamic scales based on input statistics
        scales = self._compute_dynamic_scales(x)
        
        # Compute dimensions using different estimators
        box_dim = self._box_counting_dimension(x, scales)
        corr_dim = self._correlation_dimension(x, scales)
        info_dim = self._information_dimension(x, scales)
        
        # Weighted combination of different estimators
        weights = F.softmax(self.scale_importance, dim=0)
        combined_dim = (
            weights[0] * box_dim + 
            weights[1] * corr_dim + 
            weights[2] * info_dim
        )
        
        if return_scales:
            return combined_dim, scales
        return combined_dim
        
    def _compute_dynamic_scales(self, x: torch.Tensor) -> List[int]:
        """Compute dynamic scales based on input statistics"""
        # Use scale parameters to generate actual scales
        scales = torch.exp(self.scale_params)
        
        # Adjust scales based on input variance
        std_factor = torch.std(x) / torch.mean(x)
        adjusted_scales = scales * std_factor.clamp(0.5, 2.0)
        
        return [int(s.item()) for s in adjusted_scales.clamp(
            self.min_scale, self.max_scale
        )]
        
    def _box_counting_dimension(
        self, 
        x: torch.Tensor,
        scales: List[int]
    ) -> torch.Tensor:
        """Compute box-counting dimension across scales"""
        counts = []
        for scale in scales:
            # Downsample and count non-zero boxes
            pooled = F.avg_pool1d(
                x.unsqueeze(1),
                kernel_size=scale,
                stride=scale
            ).squeeze(1)
            count = torch.sum(pooled > pooled.mean())
            counts.append(count)
            
        # Fit line to log-log plot
        x_vals = torch.log(torch.tensor(scales, dtype=torch.float32))
        y_vals = torch.log(torch.tensor(counts, dtype=torch.float32))
        
        return -torch.polyfit(x_vals, y_vals, 1)[0]
        
    def _correlation_dimension(
        self, 
        x: torch.Tensor,
        scales: List[int]
    ) -> torch.Tensor:
        """Compute correlation dimension using pairwise distances"""
        # Compute pairwise distances
        dists = torch.cdist(x, x)
        
        dimensions = []
        for scale in scales:
            # Count correlations at each scale
            correlations = torch.sum(dists < scale)
            dimensions.append(correlations)
            
        # Fit line to log-log plot
        x_vals = torch.log(torch.tensor(scales, dtype=torch.float32))
        y_vals = torch.log(torch.tensor(dimensions, dtype=torch.float32))
        
        return torch.polyfit(x_vals, y_vals, 1)[0]
        
    def _information_dimension(
        self, 
        x: torch.Tensor,
        scales: List[int]
    ) -> torch.Tensor:
        """Compute information dimension using entropy at different scales"""
        entropies = []
        for scale in scales:
            # Compute histogram at this scale
            hist = torch.histc(x, bins=scale)
            probs = hist / hist.sum()
            probs = probs[probs > 0]  # Remove zeros
            
            # Compute entropy
            entropy = -torch.sum(probs * torch.log(probs))
            entropies.append(entropy)
            
        # Fit line to log-log plot
        x_vals = torch.log(torch.tensor(scales, dtype=torch.float32))
        y_vals = torch.tensor(entropies, dtype=torch.float32)
        
        return torch.polyfit(x_vals, y_vals, 1)[0]