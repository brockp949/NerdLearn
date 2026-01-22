"""
Unit tests for AsyncGraphService (Knowledge Graph)

Tests cover:
- Graph queries (course graph, concept details, learning paths)
- Graph mutations (create nodes, add prerequisites)
- Analytics (graph stats, entry/exit points)
- Concept extraction
"""

import os
import sys

# Set environment variables BEFORE importing app modules
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "test"
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["SECRET_KEY"] = "test-secret"
os.environ["ALLOWED_ORIGINS"] = '["http://localhost:3000"]'

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from app.services.graph_service import AsyncGraphService


class TestAsyncGraphServiceInitialization:
    """Tests for graph service initialization"""

    def test_initialization(self):
        """Test service initializes with correct defaults"""
        service = AsyncGraphService()

        assert service.driver is None
        assert service._connected is False


class TestGraphServiceConnection:
    """Tests for connection management"""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver"""
        driver = MagicMock()
        driver.close = AsyncMock()
        return driver

    @pytest.mark.asyncio
    async def test_connect_creates_driver(self):
        """Connect should create a Neo4j driver"""
        with patch("app.services.graph_service.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_db.driver.return_value = mock_driver

            service = AsyncGraphService()
            await service.connect()

            assert service._connected is True
            mock_db.driver.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_closes_driver(self, mock_driver):
        """Close should close the driver"""
        service = AsyncGraphService()
        service.driver = mock_driver
        service._connected = True

        await service.close()

        mock_driver.close.assert_called_once()
        assert service._connected is False

    @pytest.mark.asyncio
    async def test_ensure_connected_when_disconnected(self):
        """ensure_connected should connect if not connected"""
        service = AsyncGraphService()

        with patch.object(service, "connect", new_callable=AsyncMock) as mock_connect:
            await service.ensure_connected()
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_connected_when_connected(self):
        """ensure_connected should not reconnect if already connected"""
        service = AsyncGraphService()
        service._connected = True

        with patch.object(service, "connect", new_callable=AsyncMock) as mock_connect:
            await service.ensure_connected()
            mock_connect.assert_not_called()


class TestGraphQueries:
    """Tests for graph query methods"""

    @pytest.fixture
    def service(self):
        """Create service with mocked driver"""
        service = AsyncGraphService()
        service._connected = True

        # Create mock driver and session
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_result = MagicMock()
        mock_result.data = AsyncMock(return_value=[])
        mock_result.single = AsyncMock(return_value=None)
        mock_session.run = AsyncMock(return_value=mock_result)

        service.driver = MagicMock()
        service.driver.session.return_value = mock_session

        return service

    @pytest.mark.asyncio
    async def test_get_course_graph_empty(self, service):
        """get_course_graph should return empty graph for no data"""
        result = await service.get_course_graph(course_id=1)

        assert "nodes" in result
        assert "edges" in result
        assert "meta" in result
        assert result["nodes"] == []
        assert result["edges"] == []

    @pytest.mark.asyncio
    async def test_get_course_graph_with_data(self, service):
        """get_course_graph should process query results correctly"""
        # Setup mock data
        mock_records = [
            {
                "concept": "Binary Search",
                "difficulty": 5.0,
                "importance": 0.8,
                "module": "Module 1",
                "module_id": 1,
                "module_order": 1,
                "outgoing_prereqs": [{"target": "Arrays", "confidence": 0.9, "type": "explicit"}],
            },
            {
                "concept": "Arrays",
                "difficulty": 3.0,
                "importance": 0.9,
                "module": "Module 1",
                "module_id": 1,
                "module_order": 1,
                "outgoing_prereqs": [],
            },
        ]

        # Update mock to return data
        mock_result = MagicMock()
        mock_result.data = AsyncMock(return_value=mock_records)
        service.driver.session.return_value.__aenter__.return_value.run = AsyncMock(
            return_value=mock_result
        )

        result = await service.get_course_graph(course_id=1)

        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
        assert result["meta"]["total_concepts"] == 2

    @pytest.mark.asyncio
    async def test_get_concept_details_not_found(self, service):
        """get_concept_details should return None for missing concept"""
        result = await service.get_concept_details(
            course_id=1, concept_name="Nonexistent"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_concept_details_with_data(self, service):
        """get_concept_details should return concept info"""
        mock_record = {
            "name": "Binary Search",
            "difficulty": 5.0,
            "importance": 0.8,
            "description": "A search algorithm",
            "module": "Module 1",
            "module_id": 1,
            "prerequisites": [{"name": "Arrays", "confidence": 0.9}],
            "dependents": [{"name": "Trees", "confidence": 0.7}],
        }

        mock_result = MagicMock()
        mock_result.single = AsyncMock(return_value=mock_record)
        service.driver.session.return_value.__aenter__.return_value.run = AsyncMock(
            return_value=mock_result
        )

        result = await service.get_concept_details(
            course_id=1, concept_name="Binary Search"
        )

        assert result is not None
        assert result["name"] == "Binary Search"
        assert result["difficulty"] == 5.0

    @pytest.mark.asyncio
    async def test_get_learning_path(self, service):
        """get_learning_path should return ordered concepts"""
        mock_records = [
            {"name": "Basics", "difficulty": 2.0, "module_order": 1, "depth": 2, "weight": 6.0},
            {"name": "Intermediate", "difficulty": 5.0, "module_order": 2, "depth": 1, "weight": 7.0},
            {"name": "Advanced", "difficulty": 8.0, "module_order": 3, "depth": 0, "weight": 8.0},
        ]

        mock_result = MagicMock()
        mock_result.data = AsyncMock(return_value=mock_records)
        service.driver.session.return_value.__aenter__.return_value.run = AsyncMock(
            return_value=mock_result
        )

        result = await service.get_learning_path(
            course_id=1, target_concepts=["Advanced"]
        )

        assert len(result) == 3
        assert result[0]["name"] == "Basics"

    @pytest.mark.asyncio
    async def test_get_learning_path_filters_mastered(self, service):
        """get_learning_path should filter already mastered concepts"""
        mock_records = [
            {"name": "Basics", "difficulty": 2.0, "module_order": 1, "depth": 2, "weight": 6.0},
            {"name": "Intermediate", "difficulty": 5.0, "module_order": 2, "depth": 1, "weight": 7.0},
        ]

        mock_result = MagicMock()
        mock_result.data = AsyncMock(return_value=mock_records)
        service.driver.session.return_value.__aenter__.return_value.run = AsyncMock(
            return_value=mock_result
        )

        result = await service.get_learning_path(
            course_id=1,
            target_concepts=["Intermediate"],
            user_mastered=["Basics"],
        )

        assert len(result) == 1
        assert result[0]["name"] == "Intermediate"


class TestGraphMutations:
    """Tests for graph mutation methods"""

    @pytest.fixture
    def service(self):
        """Create service with mocked driver"""
        service = AsyncGraphService()
        service._connected = True

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_result = MagicMock()
        mock_result.single = AsyncMock(return_value={"id": 1})
        mock_session.run = AsyncMock(return_value=mock_result)

        service.driver = MagicMock()
        service.driver.session.return_value = mock_session

        return service

    @pytest.mark.asyncio
    async def test_create_course_node(self, service):
        """create_course_node should return True on success"""
        result = await service.create_course_node(course_id=1, title="Test Course")

        assert result is True

    @pytest.mark.asyncio
    async def test_create_module_node(self, service):
        """create_module_node should return True on success"""
        result = await service.create_module_node(
            course_id=1, module_id=1, title="Module 1", order=1
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_create_concept_node(self, service):
        """create_concept_node should return True on success"""
        mock_result = MagicMock()
        mock_result.single = AsyncMock(return_value={"name": "Binary Search"})
        service.driver.session.return_value.__aenter__.return_value.run = AsyncMock(
            return_value=mock_result
        )

        result = await service.create_concept_node(
            course_id=1,
            module_id=1,
            name="Binary Search",
            difficulty=5.0,
            importance=0.8,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_add_prerequisite(self, service):
        """add_prerequisite should return True on success"""
        mock_result = MagicMock()
        mock_result.single = AsyncMock(
            return_value={"prereq": "Arrays", "concept": "Binary Search"}
        )
        service.driver.session.return_value.__aenter__.return_value.run = AsyncMock(
            return_value=mock_result
        )

        result = await service.add_prerequisite(
            course_id=1,
            prerequisite_name="Arrays",
            concept_name="Binary Search",
            confidence=0.9,
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_remove_prerequisite_success(self, service):
        """remove_prerequisite should return True when relationship exists"""
        mock_result = MagicMock()
        mock_result.single = AsyncMock(return_value={"deleted": 1})
        service.driver.session.return_value.__aenter__.return_value.run = AsyncMock(
            return_value=mock_result
        )

        result = await service.remove_prerequisite(
            course_id=1,
            prerequisite_name="Arrays",
            concept_name="Binary Search",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_remove_prerequisite_not_found(self, service):
        """remove_prerequisite should return False when relationship doesn't exist"""
        mock_result = MagicMock()
        mock_result.single = AsyncMock(return_value={"deleted": 0})
        service.driver.session.return_value.__aenter__.return_value.run = AsyncMock(
            return_value=mock_result
        )

        result = await service.remove_prerequisite(
            course_id=1,
            prerequisite_name="Nonexistent",
            concept_name="Binary Search",
        )

        assert result is False


class TestGraphAnalytics:
    """Tests for graph analytics methods"""

    @pytest.fixture
    def service(self):
        """Create service with mocked driver"""
        service = AsyncGraphService()
        service._connected = True

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        service.driver = MagicMock()
        service.driver.session.return_value = mock_session

        return service

    @pytest.mark.asyncio
    async def test_get_graph_stats(self, service):
        """get_graph_stats should return statistics"""
        mock_result = MagicMock()
        mock_result.single = AsyncMock(
            return_value={
                "modules": 5,
                "concepts": 20,
                "prerequisites": 15,
                "avg_difficulty": 5.5,
            }
        )
        service.driver.session.return_value.__aenter__.return_value.run = AsyncMock(
            return_value=mock_result
        )

        result = await service.get_graph_stats(course_id=1)

        assert result["modules"] == 5
        assert result["concepts"] == 20
        assert result["prerequisites"] == 15
        assert result["avg_difficulty"] == 5.5

    @pytest.mark.asyncio
    async def test_get_graph_stats_empty(self, service):
        """get_graph_stats should handle empty course"""
        mock_result = MagicMock()
        mock_result.single = AsyncMock(return_value=None)
        service.driver.session.return_value.__aenter__.return_value.run = AsyncMock(
            return_value=mock_result
        )

        result = await service.get_graph_stats(course_id=999)

        assert result["modules"] == 0
        assert result["concepts"] == 0

    @pytest.mark.asyncio
    async def test_find_concepts_without_prerequisites(self, service):
        """find_concepts_without_prerequisites should return entry points"""
        mock_records = [{"name": "Basics"}, {"name": "Introduction"}]
        mock_result = MagicMock()
        mock_result.data = AsyncMock(return_value=mock_records)
        service.driver.session.return_value.__aenter__.return_value.run = AsyncMock(
            return_value=mock_result
        )

        result = await service.find_concepts_without_prerequisites(course_id=1)

        assert "Basics" in result
        assert "Introduction" in result

    @pytest.mark.asyncio
    async def test_find_terminal_concepts(self, service):
        """find_terminal_concepts should return endpoints"""
        mock_records = [{"name": "Advanced Topics"}, {"name": "Final Project"}]
        mock_result = MagicMock()
        mock_result.data = AsyncMock(return_value=mock_records)
        service.driver.session.return_value.__aenter__.return_value.run = AsyncMock(
            return_value=mock_result
        )

        result = await service.find_terminal_concepts(course_id=1)

        assert "Advanced Topics" in result
        assert "Final Project" in result


class TestConceptExtraction:
    """Tests for concept extraction from text"""

    @pytest.fixture
    def service(self):
        return AsyncGraphService()

    def test_extract_concepts_empty_text(self, service):
        """Should return empty list for empty text"""
        result = service.extract_concepts("")
        assert result == []

    def test_extract_concepts_capitalized_phrases(self, service):
        """Should extract capitalized multi-word phrases"""
        text = "Binary Search is a fundamental algorithm. Quick Sort is another one."
        result = service.extract_concepts(text)

        assert "Binary Search" in result
        assert "Quick Sort" in result

    def test_extract_concepts_technical_terms(self, service):
        """Should extract technical terms"""
        text = "We will learn about algorithm design and data structure implementation. The recursion pattern is important."
        result = service.extract_concepts(text)

        extracted_lower = [c.lower() for c in result]
        assert any("algorithm" in c for c in extracted_lower)
        assert any("data structure" in c for c in extracted_lower)
        assert any("recursion" in c for c in extracted_lower)

    def test_extract_concepts_deduplicates(self, service):
        """Should deduplicate concepts"""
        text = "Binary Search uses binary search to find elements. Binary search is efficient."
        result = service.extract_concepts(text)

        # Should only appear once
        binary_search_count = sum(1 for c in result if c.lower() == "binary search")
        assert binary_search_count <= 1

    def test_extract_concepts_limits_results(self, service):
        """Should limit to 100 concepts"""
        # Create text with many potential concepts
        words = ["Concept" + str(i) for i in range(150)]
        text = " ".join(words)
        result = service.extract_concepts(text)

        assert len(result) <= 100

    def test_extract_technical_terms(self, service):
        """_extract_technical_terms should find known terms"""
        text = "this covers machine learning and neural network concepts with deep learning"
        result = service._extract_technical_terms(text)

        assert "Machine Learning" in result
        assert "Neural Network" in result
        assert "Deep Learning" in result

    def test_extract_technical_terms_case_insensitive(self, service):
        """Should match terms regardless of case"""
        text = "ALGORITHM and DATA STRUCTURE and API"
        result = service._extract_technical_terms(text.lower())

        assert "Algorithm" in result
        assert "Data Structure" in result
        assert "Api" in result


