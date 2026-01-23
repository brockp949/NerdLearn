"""
Unit tests for AsyncGraphService (Knowledge Graph via Apache AGE)

Tests cover:
- AGE initialization and Cypher query execution
- Graph queries (course graph, concept details, learning paths)
- Graph mutations (create nodes, add relationships)
- Analytics (graph stats)
- Concept extraction
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from typing import List, Dict, Any

from app.services.graph_service import AsyncGraphService
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


class TestAsyncGraphServiceInitialization:
    """Tests for graph service initialization and connection"""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test service initializes with DB session"""
        mock_db = AsyncMock(spec=AsyncSession)
        service = AsyncGraphService(db=mock_db)
        
        assert service.db == mock_db
        assert service.graph_name == "nerdlearn_graph"
        assert service._initialized is False

    @pytest.mark.asyncio
    async def test_init_age_success(self):
        """Test _init_age loads extension and creates graph if needed"""
        mock_db = AsyncMock(spec=AsyncSession)
        service = AsyncGraphService(db=mock_db)
        
        # Mock executes: load, set path, check graph exists (return 0), create graph
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        mock_db.execute.return_value = mock_result

        await service._init_age()

        # Should have executed at least 4 commands
        assert mock_db.execute.call_count >= 4
        assert service._initialized is True
        
        # Verify specific calls
        calls = [str(c[0][0]) for c in mock_db.execute.call_args_list]
        assert any("LOAD 'age'" in c for c in calls)
        assert any("create_graph" in c for c in calls)

    @pytest.mark.asyncio
    async def test_init_age_already_initialized(self):
        """Test _init_age skips if already initialized"""
        mock_db = AsyncMock(spec=AsyncSession)
        service = AsyncGraphService(db=mock_db)
        service._initialized = True
        
        await service._init_age()
        
        mock_db.execute.assert_not_called()


class TestGraphQueries:
    """Tests for graph query methods"""

    @pytest.fixture
    def service(self):
        """Create service with mocked DB"""
        mock_db = AsyncMock(spec=AsyncSession)
        service = AsyncGraphService(db=mock_db)
        service._initialized = True # Skip init for query tests
        
        # Mock execute result
        self.mock_result = MagicMock()
        mock_db.execute.return_value = self.mock_result
        
        return service

    @pytest.mark.asyncio
    async def test_run_cypher(self, service):
        """Test run_cypher formats and executes SQL"""
        mock_rows = [("r1",), ("r2",)]
        service.db.execute.return_value.all.return_value = mock_rows
        
        results = await service.run_cypher(
            query="MATCH (n) RETURN n", 
            columns_def="n agtype",
            params={"id": 1}
        )
        
        assert results == mock_rows
        service.db.execute.assert_called_once()
        # Verify parameter substitution in SQL
        call_arg = str(service.db.execute.call_args[0][0])
        assert "nerdlearn_graph" in call_arg
        assert "MATCH (n) RETURN n" in call_arg

    @pytest.mark.asyncio
    async def test_get_course_graph(self, service):
        """get_course_graph should return formatted nodes and edges"""
        # updated mock data matching the columns in get_course_graph:
        # concept, difficulty, importance, module, module_id, module_order, outgoing_prereqs
        mock_record = MagicMock()
        mock_record.concept = "Binary Search"
        mock_record.difficulty = 5.0
        mock_record.importance = 0.8
        mock_record.module = "Module 1"
        mock_record.module_id = 1
        mock_record.module_order = 1
        mock_record.outgoing_prereqs = [{"target": "Arrays", "confidence": 0.9, "type": "explicit"}]
        
        service.db.execute.return_value.all.return_value = [mock_record]

        result = await service.get_course_graph(course_id=1)

        assert len(result["nodes"]) == 1
        assert len(result["edges"]) == 1
        assert result["nodes"][0]["id"] == "Binary Search"
        assert result["edges"][0]["source"] == "Binary Search"
        assert result["edges"][0]["target"] == "Arrays"

    @pytest.mark.asyncio
    async def test_get_concept_details(self, service):
        """get_concept_details should return concept info"""
        mock_record = MagicMock()
        mock_record.name = "Binary Search"
        mock_record.difficulty = 5.0
        mock_record.importance = 0.8
        mock_record.description = "Algo"
        mock_record.module = "M1"
        mock_record.module_id = 1
        mock_record.prerequisites = [{"name": "Arrays"}]
        mock_record.dependents = []
        
        service.db.execute.return_value.all.return_value = [mock_record]

        result = await service.get_concept_details(course_id=1, concept_name="Binary Search")

        assert result is not None
        assert result["name"] == "Binary Search"
        assert result["prerequisites"] == [{"name": "Arrays"}]

    @pytest.mark.asyncio
    async def test_get_learning_path(self, service):
        """get_learning_path should return ordered concepts"""
        # updated mock data matching columns: name, difficulty, module_order, depth, weight
        r1 = MagicMock()
        r1.name = "Basics"
        r1.difficulty = 2.0
        r1.module_order = 1
        r1.depth = 2
        r1.weight = 6.0
        
        r2 = MagicMock()
        r2.name = "Advanced"
        r2.difficulty = 5.0
        r2.module_order = 2
        r2.depth = 1
        r2.weight = 7.0
        
        service.db.execute.return_value.all.return_value = [r1, r2]

        result = await service.get_learning_path(course_id=1, target_concepts=["Advanced"])

        assert len(result) == 2
        assert result[0]["name"] == "Basics"
        assert result[1]["name"] == "Advanced"


class TestGraphMutations:
    """Tests for graph mutation methods"""

    @pytest.fixture
    def service(self):
        mock_db = AsyncMock(spec=AsyncSession)
        service = AsyncGraphService(db=mock_db)
        service._initialized = True
        # IMPORTANT: execute is async, but returns a sync Result object. 
        # We must mock the return value as a MagicMock so .all() is sync.
        mock_db.execute.return_value = MagicMock()
        return service

    @pytest.mark.asyncio
    async def test_create_course_node(self, service):
        """create_course_node should return True on success"""
        service.db.execute.return_value.all.return_value = [("created_node",)] # Non-empty list
        
        result = await service.create_course_node(course_id=1, title="Test Course")
        
        assert result is True
        service.db.execute.assert_called()

    @pytest.mark.asyncio
    async def test_create_concept_node(self, service):
        """create_concept_node should execute correct cypher"""
        service.db.execute.return_value.all.return_value = [("Binary Search",)]
        
        result = await service.create_concept_node(
            course_id=1,
            module_id=1,
            name="Binary Search",
            difficulty=5.0,
            importance=0.8
        )
        
        assert result is True
        call_arg = str(service.db.execute.call_args[0][0])
        assert "MERGE (con:Concept" in call_arg
        assert "Binary Search" in call_arg # Params substituted

    @pytest.mark.asyncio
    async def test_add_prerequisite(self, service):
        service.db.execute.return_value.all.return_value = [("A", "B")]
        
        result = await service.add_prerequisite(
            course_id=1,
            prerequisite_name="Arrays",
            concept_name="Search"
        )
        
        assert result is True
        call_arg = str(service.db.execute.call_args[0][0])
        assert "PREREQUISITE_FOR" in call_arg


class TestConceptExtraction:
    """Tests for concept extraction (pure logic)"""
    
    @pytest.fixture
    def service(self):
        return AsyncGraphService(db=AsyncMock())

    def test_extract_concepts_basic(self, service):
        text = "Binary Search is a specific Alcoholics Anonymous group. No wait, Binary Search is an algorithm."
        # Note: Previous regex was looking for Capitalized Words.
        concepts = service.extract_concepts(text)
        assert "Binary Search" in concepts
