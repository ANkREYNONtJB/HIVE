import pytest
import torch
from sonw.analytics.quantum_features import QuantumFeatureAnalyzer

@pytest.fixture
def feature_analyzer():
    return QuantumFeatureAnalyzer(n_features=10)

def test_feature_analysis(feature_analyzer):
    # Generate test data
    features = [torch.randn(5, 10) for _ in range(3)]
    layer_importance = torch.ones(3)
    feedback_weights = [torch.randn(10, 10) for _ in range(2)]
    
    # Run analysis
    results = feature_analyzer.analyze_fractal_features(
        features=torch.stack(features),
        layer_importance=layer_importance,
        feedback_weights=feedback_weights
    )
    
    # Check results
    assert 'fractal_features' in results
    assert 'pattern_statistics' in results
    assert 'self_similarity' in results
    
    # Check specific values
    assert len(results['fractal_features']) == 3
    assert len(results['pattern_statistics']['feature_distributions']) == 3
    assert isinstance(results['self_similarity'], float)

def test_error_handling(feature_analyzer):
    # Test with invalid input
    with pytest.raises(RuntimeError):
        feature_analyzer.analyze_fractal_features(
            features=torch.tensor([]),
            layer_importance=torch.tensor([]),
            feedback_weights=[]
        )