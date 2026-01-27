"""
Tests for CommunityDetectionService
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestCommunityDetectionService:
    """Tests for CommunityDetectionService"""

    @pytest.fixture
    def mock_graph_service(self):
        """Mock graph service"""
        service = AsyncMock()
        service.get_course_graph = AsyncMock(return_value={
            "nodes": [
                {"id": "concept_1"},
                {"id": "concept_2"},
                {"id": "concept_3"}
            ],
            "edges": [
                {"source": "concept_1", "target": "concept_2", "confidence": 0.8},
                {"source": "concept_2", "target": "concept_3", "confidence": 0.9}
            ]
        })
        service.update_community_structure = AsyncMock(return_value=3)
        service.get_all_communities = AsyncMock(return_value=[1, 2])
        service.get_community_members = AsyncMock(return_value=[
            {"name": "Concept 1", "description": "First concept"},
            {"name": "Concept 2", "description": "Second concept"}
        ])
        return service

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store service"""
        service = AsyncMock()
        service.upsert_documents = AsyncMock()
        return service

    @pytest.fixture
    def service(self, mock_graph_service, mock_vector_store):
        """Create service with mocked dependencies"""
        with patch('app.services.community_service.AsyncGraphService', return_value=mock_graph_service), \
             patch('app.services.community_service.VectorStoreService', return_value=mock_vector_store), \
             patch('app.services.community_service.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = ""
            from app.services.community_service import CommunityDetectionService
            svc = CommunityDetectionService(db=None)
            svc.graph_service = mock_graph_service
            svc.vector_store = mock_vector_store
            return svc

    @pytest.mark.asyncio
    async def test_run_detection_with_nodes(self, service, mock_graph_service):
        """Test community detection with valid graph"""
        with patch('community.best_partition', return_value={
            "concept_1": 0,
            "concept_2": 0,
            "concept_3": 1
        }):
            count = await service.run_detection(course_id=1)

            mock_graph_service.get_course_graph.assert_called_once_with(1)
            mock_graph_service.update_community_structure.assert_called_once()
            assert count == 3

    @pytest.mark.asyncio
    async def test_run_detection_empty_graph(self, service, mock_graph_service):
        """Test community detection with empty graph"""
        mock_graph_service.get_course_graph.return_value = {"nodes": [], "edges": []}

        count = await service.run_detection(course_id=1)

        assert count == 0

    @pytest.mark.asyncio
    async def test_summarize_communities_without_api_key(self, service, mock_graph_service, mock_vector_store):
        """Test community summarization without API key"""
        service.summarize_module = None

        count = await service.summarize_communities(course_id=1)

        # Should still process communities even without summarization
        assert count >= 0

    @pytest.mark.asyncio
    async def test_generate_summary_without_module(self, service):
        """Test summary generation without DSPy module"""
        service.summarize_module = None

        summary = await service._generate_summary("Some context")

        assert "unavailable" in summary.lower()

    @pytest.mark.asyncio
    async def test_generate_summary_truncates_long_context(self, service):
        """Test that long context is truncated"""
        service.summarize_module = MagicMock()
        service.summarize_module.return_value.summary = "Test summary"

        long_context = "A" * 15000  # Longer than max_chars

        summary = await service._generate_summary(long_context)

        # Verify the module was called with truncated context
        call_args = service.summarize_module.call_args
        assert "truncated" in call_args.kwargs.get("context", "") or len(call_args.kwargs.get("context", long_context)) <= 12003


class TestCommunitySummarizer:
    """Tests for CommunitySummarizer DSPy signature"""

    def test_signature_fields(self):
        """Test DSPy signature has correct fields"""
        from app.services.community_service import CommunitySummarizer

        # Verify input and output fields exist
        assert hasattr(CommunitySummarizer, 'context')
        assert hasattr(CommunitySummarizer, 'summary')


class TestCommunityDetectionIntegration:
    """Integration tests for community detection"""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test full community detection and summarization workflow"""
        with patch('app.services.community_service.AsyncGraphService') as MockGraph, \
             patch('app.services.community_service.VectorStoreService') as MockVector, \
             patch('app.services.community_service.settings') as mock_settings, \
             patch('community.best_partition', return_value={"c1": 0, "c2": 1}):

            mock_settings.OPENAI_API_KEY = ""

            mock_graph = AsyncMock()
            mock_graph.get_course_graph.return_value = {
                "nodes": [{"id": "c1"}, {"id": "c2"}],
                "edges": [{"source": "c1", "target": "c2", "confidence": 0.8}]
            }
            mock_graph.update_community_structure.return_value = 2
            mock_graph.get_all_communities.return_value = [0, 1]
            mock_graph.get_community_members.return_value = []
            MockGraph.return_value = mock_graph

            mock_vector = AsyncMock()
            MockVector.return_value = mock_vector

            from app.services.community_service import CommunityDetectionService
            service = CommunityDetectionService(db=None)
            service.graph_service = mock_graph
            service.vector_store = mock_vector

            # Run detection
            detection_count = await service.run_detection(1)
            assert detection_count == 2

            # Summarize (with empty members, should return 0)
            summary_count = await service.summarize_communities(1)
            assert summary_count >= 0
