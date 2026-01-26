# Track 002: Causal Discovery for Educational Graphs

**Goal**: Implement and verify the Causal Discovery pipeline to infer prerequisite relationships and causal links between educational concepts using NOTEARS, FCI, and community detection.

## Context
Based on "Causal Discovery for Educational Graphs.pdf". The system currently has a skeleton implementation in `apps/api/app/adaptive/causal_discovery`. We need to ensure the full pipeline described in the paper is working:
1.  **Global Discovery**: NOTEARS (optimization-based) for initial DAG skeleton.
2.  **Community Detection**: Leiden/Louvain for detecting concept clusters.
3.  **Local Refinement**: FCI (Fast Causal Inference) for confounder detection within clusters.
4.  **Persistence**: Storing discovered edges in the Knowledge Graph (Apache AGE/Neo4j).

## Plan

### 1. Assessment & Setup
- [ ] Review existing code in `apps/api/app/adaptive/causal_discovery`.
- [ ] Verify dependencies (`causal-learn`, `networkx`, `numpy`, `pandas`).
- [ ] Create a "synthetic data generator" for causal discovery (similar to the CRL one) to test the pipeline without waiting for real user data.

### 2. Implementation Refinement
- [ ] **NOTEARS Implementation**: Ensure `algorithms/notears.py` (if it exists) or the integration in `manager.py` works.
- [ ] **Community Detection**: Verify integration with `networkx` or `cdlib` for community detection.
- [ ] **FCI Implementation**: Verify `causal-learn` integration for FCI.
- [ ] **Manager Logic**: Ensure `CausalDiscoveryManager` correctly orchestrates the pipeline (Global -> Community -> Local).

### 3. Testing & Verification
- [ ] Create a test script `test_causal_pipeline.py`.
- [ ] Run the pipeline on synthetic data.
- [ ] Verify that known causal links in synthetic data are recovered.
- [ ] Verify persistence to the Graph Service.

### 4. Integration
- [ ] Ensure the API endpoint `/causal-discovery/run` triggers the pipeline correctly.
- [ ] Add configuration for hyperparameters (lambda, alpha).

## References
- `Causal Discovery for Educational Graphs.pdf`
- `apps/api/app/adaptive/causal_discovery/manager.py`
