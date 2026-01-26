import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.adaptive.causal_discovery.manager import CausalDiscoveryManager

@pytest.mark.asyncio
async def test_bootstrap_stability():
    manager = CausalDiscoveryManager()
    
    # Mock NOTEARS to return consistent edges for "stable" relationship
    # and random edges for "unstable"
    # A->B in 100% of runs
    # C->D in 40% of runs
    
    def side_effect_run(df, threshold=0.3):
        edges = [("A", "B", 0.5)] # Consistent
        # Random logic for C->D simulation? 
        # Alternatively, we can just mock the return value list sequence if needed,
        # but since we are running in a loop, side_effect is function.
        # Let's rely on the fact that manager calls run() N times.
        return edges

    # We need a more controlled mock to test aggregation logic.
    # Let's mock the .sample() or rely on predictable behaviors?
    # Better: Mock notears.run to return specific lists on subsequent calls.
    
    # Run 10 bootstraps
    # A->B: 10 times (1.0 conf)
    # B->C: 6 times (0.6 conf)
    # C->D: 2 times (0.2 conf - should be dropped)
    
    side_effects = []
    for _ in range(10):
        run_edges = [("A", "B", 0.8)]
        side_effects.append(run_edges)
        
    for i in range(6):
        side_effects[i].append(("B", "C", 0.5))
        
    for i in range(2):
        side_effects[i].append(("C", "D", 0.1))
        
    manager.notears.run = MagicMock(side_effect=side_effects)
    
    # Mock data preprocessing
    manager._preprocess_data = MagicMock(return_value=MagicMock(empty=False, shape=(10, 5)))
    # Mock sample to return itself (dummy)
    manager._preprocess_data.return_value.sample = MagicMock(return_value="dummy_sample")
    
    mock_service = AsyncMock()
    mock_service.update_causal_edges = AsyncMock()
    
    # Run pipeline
    await manager.run_bootstrap_pipeline([], mock_service, n_bootstraps=10)
    
    # Verify aggregation
    mock_service.update_causal_edges.assert_called_once()
    edges = mock_service.update_causal_edges.call_args[0][0]
    
    # Check A->B
    ab = next((e for e in edges if e['source'] == "A" and e['target'] == "B"), None)
    assert ab is not None
    assert ab['confidence'] == 1.0
    assert ab['status'] == 'verified'
    
    # Check B->C
    bc = next((e for e in edges if e['source'] == "B" and e['target'] == "C"), None)
    assert bc is not None
    assert bc['confidence'] == 0.6
    assert bc['status'] == 'hypothetical'
    
    # Check C->D (Should be absent)
    cd = next((e for e in edges if e['source'] == "C" and e['target'] == "D"), None)
    assert cd is None
