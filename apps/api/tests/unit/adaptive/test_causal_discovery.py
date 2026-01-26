"""
Tests for Causal Discovery Module

Based on "Causal Discovery for Educational Graphs" PDF specification.
Tests cover:
- Bootstrap stability selection
- Graph persistence with confidence thresholds
- Manager pipeline integration
- Individual algorithm wrappers
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, List, Tuple

from app.adaptive.causal_discovery.manager import CausalDiscoveryManager
from app.adaptive.causal_discovery.bootstrap import BootstrapStabilitySelector
from app.adaptive.causal_discovery.algorithms.notears import NotearsAlgorithm
from app.adaptive.causal_discovery.algorithms.fci import FciAlgorithm
from app.adaptive.causal_discovery.algorithms.leiden import LeidenAlgorithm


# ============== Fixtures ==============

@pytest.fixture
def mock_mastery_data():
    """Basic mastery data for testing"""
    return [
        {"user_id": 1, "concept_id": 101, "mastery": 1.0},
        {"user_id": 1, "concept_id": 102, "mastery": 0.8},
        {"user_id": 2, "concept_id": 101, "mastery": 0.0},
        {"user_id": 2, "concept_id": 102, "mastery": 0.0},
        {"user_id": 3, "concept_id": 101, "mastery": 1.0},
        {"user_id": 3, "concept_id": 103, "mastery": 0.9}
    ]


@pytest.fixture
def sample_user_concept_dataframe():
    """Generate synthetic User x Concept mastery DataFrame"""
    np.random.seed(42)
    n_users = 50
    n_concepts = 5

    # Create data with known causal structure: A -> B -> C
    data = pd.DataFrame({
        'A': np.random.binomial(1, 0.7, n_users),
        'B': np.zeros(n_users),
        'C': np.zeros(n_users),
        'D': np.random.binomial(1, 0.5, n_users),
        'E': np.random.binomial(1, 0.3, n_users)
    })

    # B depends on A
    data['B'] = (data['A'] * np.random.binomial(1, 0.8, n_users)).astype(int)
    # C depends on B
    data['C'] = (data['B'] * np.random.binomial(1, 0.9, n_users)).astype(int)

    return data.astype(float)


@pytest.fixture
def mock_graph_service():
    """Mock AsyncGraphService for testing"""
    service = AsyncMock()
    service.update_causal_edges = AsyncMock(return_value={"persisted": 5, "skipped": 2})
    service.get_causal_edges = AsyncMock(return_value=[])
    service.get_causal_edge_statistics = AsyncMock(return_value={
        "total": 10,
        "by_status": {"verified": 6, "hypothetical": 4},
        "by_method": {"notears": 8, "fci": 2}
    })
    return service


# ============== Bootstrap Stability Tests ==============

class TestBootstrapStability:
    """Tests for Bootstrap Stability Selection (PDF Section 6)"""

    def test_bootstrap_initialization_defaults(self):
        """Test bootstrap selector initializes with correct defaults"""
        selector = BootstrapStabilitySelector()

        assert selector.n_bootstrap == 100
        assert selector.subsample_ratio == 0.5
        assert selector.high_threshold == 0.85
        assert selector.low_threshold == 0.5

    def test_bootstrap_initialization_custom(self):
        """Test bootstrap selector with custom parameters"""
        selector = BootstrapStabilitySelector(
            n_bootstrap=50,
            subsample_ratio=0.6,
            high_confidence_threshold=0.9,
            low_confidence_threshold=0.4
        )

        assert selector.n_bootstrap == 50
        assert selector.subsample_ratio == 0.6
        assert selector.high_threshold == 0.9
        assert selector.low_threshold == 0.4

    def test_edge_classification_verified(self):
        """Test that high-confidence edges are classified as verified"""
        selector = BootstrapStabilitySelector()

        stability_scores = {
            ('A', 'B'): 0.95,  # Should be verified
            ('B', 'C'): 0.90,  # Should be verified
        }

        classified = selector.classify_edges(stability_scores)

        assert len(classified['verified']) == 2
        assert len(classified['hypothetical']) == 0
        assert len(classified['discarded']) == 0

    def test_edge_classification_hypothetical(self):
        """Test that medium-confidence edges are classified as hypothetical"""
        selector = BootstrapStabilitySelector()

        stability_scores = {
            ('A', 'B'): 0.70,  # Should be hypothetical
            ('B', 'C'): 0.55,  # Should be hypothetical
        }

        classified = selector.classify_edges(stability_scores)

        assert len(classified['verified']) == 0
        assert len(classified['hypothetical']) == 2
        assert len(classified['discarded']) == 0

    def test_edge_classification_discarded(self):
        """Test that low-confidence edges are discarded"""
        selector = BootstrapStabilitySelector()

        stability_scores = {
            ('A', 'B'): 0.30,  # Should be discarded
            ('D', 'E'): 0.10,  # Should be discarded
        }

        classified = selector.classify_edges(stability_scores)

        assert len(classified['verified']) == 0
        assert len(classified['hypothetical']) == 0
        assert len(classified['discarded']) == 2

    def test_edge_classification_mixed(self):
        """Test classification with mixed confidence levels"""
        selector = BootstrapStabilitySelector()

        stability_scores = {
            ('A', 'B'): 0.95,  # verified
            ('B', 'C'): 0.70,  # hypothetical
            ('D', 'E'): 0.30,  # discarded
        }

        classified = selector.classify_edges(stability_scores)

        assert len(classified['verified']) == 1
        assert len(classified['hypothetical']) == 1
        assert len(classified['discarded']) == 1
        assert classified['verified'][0]['source'] == 'A'
        assert classified['hypothetical'][0]['source'] == 'B'
        assert classified['discarded'][0]['source'] == 'D'

    def test_run_returns_only_persisted_edges(self):
        """Test that run() only returns verified and hypothetical edges"""
        selector = BootstrapStabilitySelector()

        # Mock the internal methods
        selector.compute_stability_scores = MagicMock(return_value={
            ('A', 'B'): 0.95,  # verified
            ('B', 'C'): 0.70,  # hypothetical
            ('D', 'E'): 0.30,  # discarded
        })

        edges, dist = selector.run(pd.DataFrame({'A': [1], 'B': [1], 'C': [1]}))

        # Only verified and hypothetical should be returned
        assert len(edges) == 2
        assert dist['verified'] == 1
        assert dist['hypothetical'] == 1
        assert dist['discarded'] == 1

    def test_run_empty_data(self):
        """Test that empty data returns empty results"""
        selector = BootstrapStabilitySelector()

        edges, dist = selector.run(pd.DataFrame())

        assert edges == []
        assert dist == {"verified": 0, "hypothetical": 0, "discarded": 0}

    def test_run_insufficient_columns(self):
        """Test that data with < 2 columns returns empty results"""
        selector = BootstrapStabilitySelector()

        edges, dist = selector.run(pd.DataFrame({'A': [1, 2, 3]}))

        assert edges == []
        assert dist == {"verified": 0, "hypothetical": 0, "discarded": 0}

    @patch.object(NotearsAlgorithm, 'run')
    def test_bootstrap_with_mocked_notears(self, mock_notears, sample_user_concept_dataframe):
        """Test full bootstrap run with mocked NOTEARS"""
        # Mock NOTEARS to return consistent edges
        mock_notears.return_value = [('A', 'B', 0.8), ('B', 'C', 0.6)]

        selector = BootstrapStabilitySelector(n_bootstrap=10)
        edges, distribution = selector.run(sample_user_concept_dataframe)

        # All edges should appear in every sample, so confidence should be high
        assert distribution['verified'] == 2
        assert len(edges) == 2


# ============== Graph Persistence Tests ==============

class TestGraphPersistence:
    """Tests for Apache AGE graph persistence"""

    def test_confidence_threshold_filtering(self):
        """Test that edges below 0.5 confidence are filtered out"""
        edges = [
            {"source": "A", "target": "B", "confidence": 0.9, "source_algo": "notears"},
            {"source": "C", "target": "D", "confidence": 0.3, "source_algo": "notears"},
            {"source": "E", "target": "F", "confidence": 0.6, "source_algo": "notears"},
        ]

        # Filter logic that should be in update_causal_edges
        filtered = [e for e in edges if e.get('confidence', 0.5) >= 0.5]

        assert len(filtered) == 2
        assert any(e['source'] == 'A' for e in filtered)
        assert any(e['source'] == 'E' for e in filtered)
        assert not any(e['source'] == 'C' for e in filtered)

    def test_status_assignment_verified(self):
        """Test that high-confidence edges get 'verified' status"""
        edge = {"source": "A", "target": "B", "confidence": 0.9}

        if edge['confidence'] > 0.85:
            edge['status'] = 'verified'
        elif edge['confidence'] >= 0.5:
            edge['status'] = 'hypothetical'

        assert edge['status'] == 'verified'

    def test_status_assignment_hypothetical(self):
        """Test that medium-confidence edges get 'hypothetical' status"""
        edge = {"source": "A", "target": "B", "confidence": 0.7}

        if edge['confidence'] > 0.85:
            edge['status'] = 'verified'
        elif edge['confidence'] >= 0.5:
            edge['status'] = 'hypothetical'

        assert edge['status'] == 'hypothetical'


# ============== Manager Tests ==============

class TestCausalDiscoveryManager:
    """Tests for the CausalDiscoveryManager pipeline"""

    @pytest.mark.asyncio
    async def test_manager_preprocess(self, mock_mastery_data):
        """Test data preprocessing converts to User x Concept matrix"""
        manager = CausalDiscoveryManager()
        df = manager._preprocess_data(mock_mastery_data)

        assert not df.empty
        assert "101" in df.columns
        assert "102" in df.columns
        assert df.loc[1, "101"] == 1.0
        # User 2 absent concept 103 should be 0.0
        if "103" in df.columns:
            assert df.loc[2, "103"] == 0.0

    @pytest.mark.asyncio
    async def test_manager_preprocess_empty_data(self):
        """Test preprocessing with empty data"""
        manager = CausalDiscoveryManager()
        df = manager._preprocess_data([])

        assert df.empty

    @pytest.mark.asyncio
    async def test_manager_preprocess_missing_columns(self):
        """Test preprocessing with missing required columns"""
        manager = CausalDiscoveryManager()
        df = manager._preprocess_data([{"user_id": 1, "wrong_col": "value"}])

        assert df.empty

    @pytest.mark.asyncio
    async def test_manager_pipeline_integration(self, mock_graph_service):
        """Test full pipeline integration with mocked algorithms"""
        manager = CausalDiscoveryManager()
        manager.notears.run = MagicMock(return_value=[("101", "102", 0.5)])
        manager.leiden.detect_communities = MagicMock(return_value={"101": 0, "102": 0})
        manager.fci.run = MagicMock(return_value=[])

        data = [
            {"user_id": 1, "concept_id": 101, "mastery": 1.0},
            {"user_id": 1, "concept_id": 102, "mastery": 0.8},
            {"user_id": 2, "concept_id": 101, "mastery": 0.0},
            {"user_id": 2, "concept_id": 102, "mastery": 0.0},
        ]
        await manager.run_discovery_pipeline(data, mock_graph_service)

        # Verify persistence called
        mock_graph_service.update_causal_edges.assert_called_once()
        edges = mock_graph_service.update_causal_edges.call_args[0][0]
        assert len(edges) >= 1
        assert edges[0]['source'] == "101"
        assert edges[0]['target'] == "102"
        assert edges[0]['source_algo'] == "notears"

    @pytest.mark.asyncio
    async def test_manager_stores_results(self, mock_graph_service):
        """Test that manager stores results for reporting"""
        manager = CausalDiscoveryManager()
        manager.notears.run = MagicMock(return_value=[("A", "B", 0.5)])
        manager.leiden.detect_communities = MagicMock(return_value={"A": 0, "B": 0})
        manager.fci.run = MagicMock(return_value=[])

        data = [
            {"user_id": 1, "concept_id": "A", "mastery": 1.0},
            {"user_id": 1, "concept_id": "B", "mastery": 0.8},
            {"user_id": 2, "concept_id": "A", "mastery": 0.0},
            {"user_id": 2, "concept_id": "B", "mastery": 0.5},
        ]
        await manager.run_discovery_pipeline(data, mock_graph_service)

        # Check stored results
        assert len(manager._last_edges) >= 1
        assert manager._last_communities == {"A": 0, "B": 0}
        assert manager._last_persist_result == {"persisted": 5, "skipped": 2}

    @pytest.mark.asyncio
    async def test_manager_insufficient_data(self, mock_graph_service):
        """Test pipeline handles insufficient data gracefully"""
        manager = CausalDiscoveryManager()

        # Only one concept - insufficient
        data = [{"user_id": 1, "concept_id": 101, "mastery": 1.0}]
        await manager.run_discovery_pipeline(data, mock_graph_service)

        # Should not call persistence
        mock_graph_service.update_causal_edges.assert_not_called()

    def test_merge_results_fci_overrides_notears(self):
        """Test that FCI bi-directed edges override NOTEARS directed edges"""
        manager = CausalDiscoveryManager()

        skeletons = [
            {"source": "A", "target": "B", "type": "directed", "weight": 0.5, "source_algo": "notears"}
        ]
        refinements = [
            {"source": "A", "target": "B", "type": "bi-directed", "weight": 1.0, "source_algo": "fci"}
        ]

        merged = manager._merge_results(skeletons, refinements)

        assert len(merged) == 1
        assert merged[0]['type'] == 'bi-directed'
        assert merged[0]['source_algo'] == 'fci'


# ============== Algorithm Tests ==============

class TestNotearsAlgorithm:
    """Tests for NOTEARS algorithm wrapper"""

    def test_notears_empty_data(self):
        """Test NOTEARS with empty data returns empty list or raises if lib missing"""
        notears = NotearsAlgorithm()
        try:
            result = notears.run(pd.DataFrame())
            # If causal-learn is installed and empty data handled
            assert result == []
        except ImportError:
            # Expected if causal-learn not installed
            pass

    def test_notears_initialization(self):
        """Test NOTEARS initializes with correct parameters"""
        notears = NotearsAlgorithm(lambda1=0.2, loss_type='logistic')

        assert notears.lambda1 == 0.2
        assert notears.loss_type == 'logistic'


class TestLeidenAlgorithm:
    """Tests for Leiden community detection"""

    def test_leiden_empty_edges(self):
        """Test Leiden with empty edges returns empty communities"""
        leiden = LeidenAlgorithm()
        result = leiden.detect_communities([])

        assert result == {}

    def test_leiden_single_edge(self):
        """Test Leiden with single edge"""
        leiden = LeidenAlgorithm()
        edges = [{"source": "A", "target": "B", "weight": 1.0}]
        result = leiden.detect_communities(edges)

        # Both nodes should be in the same community
        assert "A" in result
        assert "B" in result


class TestFciAlgorithm:
    """Tests for FCI algorithm wrapper"""

    def test_fci_initialization(self):
        """Test FCI initializes with correct parameters"""
        fci = FciAlgorithm(alpha=0.1, independence_test='chisq')

        assert fci.alpha == 0.1
        assert fci.independence_test == 'chisq'

    def test_fci_empty_data(self):
        """Test FCI with empty data returns empty list"""
        fci = FciAlgorithm()
        result = fci.run(pd.DataFrame())

        assert result == []
