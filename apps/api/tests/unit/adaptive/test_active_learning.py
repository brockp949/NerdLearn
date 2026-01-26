import pytest
from app.adaptive.causal_discovery.active_learning import ActiveLearningModule

def test_entropy_calculation():
    al = ActiveLearningModule()
    
    # Max entropy at 0.5
    assert al.calculate_edge_entropy(0.5) == 1.0
    
    # Zero entropy at 0 or 1
    assert al.calculate_edge_entropy(0.0) == 0.0
    assert al.calculate_edge_entropy(1.0) == 0.0
    
    # Symmetry
    assert abs(al.calculate_edge_entropy(0.2) - al.calculate_edge_entropy(0.8)) < 1e-9

def test_recommend_experiments():
    al = ActiveLearningModule()
    
    candidates = [
        {"source": "A", "target": "B", "confidence": 0.5}, # High ambiguity
        {"source": "B", "target": "C", "confidence": 0.9}, # Certain
        {"source": "C", "target": "D", "confidence": 0.1}, # Certain absence
        {"source": "D", "target": "E", "confidence": 0.6}  # Moderate ambiguity
    ]
    
    recs = al.recommend_experiments(candidates)
    
    assert len(recs) == 2 # Only 0.5 and 0.6 are within 0.3-0.7 range
    
    # First should be the 0.5 one (highest entropy)
    assert recs[0]['source'] == "A"
    assert recs[0]['priority'] == 1.0
    
    # Second should be the 0.6 one
    assert recs[1]['source'] == "D"
    assert recs[1]['priority'] < 1.0
