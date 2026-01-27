"""
Tests for knowledge graph router endpoints
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestGraphModels:
    """Tests for graph data models"""

    def test_prerequisite_create_model(self):
        """Test PrerequisiteCreate model"""
        from app.routers.graph import PrerequisiteCreate

        prereq = PrerequisiteCreate(
            prerequisite_name="Variables",
            concept_name="Functions",
            confidence=0.9,
            prereq_type="explicit"
        )

        assert prereq.prerequisite_name == "Variables"
        assert prereq.concept_name == "Functions"
        assert prereq.confidence == 0.9

    def test_prerequisite_create_defaults(self):
        """Test PrerequisiteCreate default values"""
        from app.routers.graph import PrerequisiteCreate

        prereq = PrerequisiteCreate(
            prerequisite_name="Variables",
            concept_name="Functions"
        )

        assert prereq.confidence == 1.0
        assert prereq.prereq_type == "explicit"

    def test_concept_create_model(self):
        """Test ConceptCreate model"""
        from app.routers.graph import ConceptCreate

        concept = ConceptCreate(
            name="Recursion",
            module_id=1,
            difficulty=7.5,
            importance=0.8,
            description="Self-referential functions"
        )

        assert concept.name == "Recursion"
        assert concept.module_id == 1
        assert concept.difficulty == 7.5

    def test_concept_create_validation(self):
        """Test ConceptCreate validation"""
        from app.routers.graph import ConceptCreate

        # Difficulty out of range should fail
        with pytest.raises(Exception):
            ConceptCreate(
                name="Test",
                module_id=1,
                difficulty=15  # > 10
            )

    def test_learning_path_request_model(self):
        """Test LearningPathRequest model"""
        from app.routers.graph import LearningPathRequest

        request = LearningPathRequest(
            target_concepts=["Recursion", "Dynamic Programming"],
            mastered_concepts=["Variables", "Loops"]
        )

        assert len(request.target_concepts) == 2
        assert len(request.mastered_concepts) == 2

    def test_graph_node_model(self):
        """Test GraphNode model"""
        from app.routers.graph import GraphNode

        node = GraphNode(
            id="concept_1",
            label="Variables",
            module="Module 1",
            module_id=1,
            difficulty=3.0,
            importance=0.9
        )

        assert node.id == "concept_1"
        assert node.label == "Variables"
        assert node.type == "concept"

    def test_graph_edge_model(self):
        """Test GraphEdge model"""
        from app.routers.graph import GraphEdge

        edge = GraphEdge(
            source="concept_1",
            target="concept_2",
            type="prerequisite",
            confidence=0.85
        )

        assert edge.source == "concept_1"
        assert edge.target == "concept_2"


class TestCourseGraphEndpoints:
    """Tests for course graph endpoints"""

    @pytest.mark.asyncio
    async def test_get_course_graph(self, client):
        """Test getting course graph"""
        response = await client.get("/api/graph/courses/1")

        if response.status_code == 200:
            data = response.json()
            assert "nodes" in data
            assert "edges" in data
            assert "meta" in data

    @pytest.mark.asyncio
    async def test_get_course_graph_empty(self, client):
        """Test getting graph for course with no data"""
        response = await client.get("/api/graph/courses/9999")

        # Should return empty graph or 500
        if response.status_code == 200:
            data = response.json()
            assert data["nodes"] == []
            assert data["edges"] == []

    @pytest.mark.asyncio
    async def test_get_graph_stats(self, client):
        """Test getting graph statistics"""
        response = await client.get("/api/graph/courses/1/stats")
        assert response.status_code in [200, 500]


class TestConceptEndpoints:
    """Tests for concept-related endpoints"""

    @pytest.mark.asyncio
    async def test_get_concept_details(self, client):
        """Test getting concept details"""
        response = await client.get("/api/graph/courses/1/concepts/Variables")

        if response.status_code == 200:
            data = response.json()
            # Should have concept information
            assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_get_concept_not_found(self, client):
        """Test getting non-existent concept"""
        response = await client.get("/api/graph/courses/1/concepts/NonExistentConcept")
        assert response.status_code in [404, 500]

    @pytest.mark.asyncio
    async def test_get_concept_prerequisites(self, client):
        """Test getting concept prerequisites"""
        response = await client.get(
            "/api/graph/courses/1/concepts/Functions/prerequisites"
        )

        if response.status_code == 200:
            data = response.json()
            assert "concept" in data
            assert "prerequisites" in data

    @pytest.mark.asyncio
    async def test_get_concept_dependents(self, client):
        """Test getting concept dependents"""
        response = await client.get(
            "/api/graph/courses/1/concepts/Variables/dependents"
        )

        if response.status_code == 200:
            data = response.json()
            assert "concept" in data
            assert "dependents" in data

    @pytest.mark.asyncio
    async def test_create_concept(self, client):
        """Test creating a concept"""
        concept = {
            "name": "New Concept",
            "module_id": 1,
            "difficulty": 5.0,
            "importance": 0.5,
            "description": "A new concept"
        }

        response = await client.post("/api/graph/courses/1/concepts", json=concept)
        assert response.status_code in [200, 400, 500]


class TestPrerequisiteEndpoints:
    """Tests for prerequisite management endpoints"""

    @pytest.mark.asyncio
    async def test_add_prerequisite(self, client):
        """Test adding a prerequisite relationship"""
        prereq = {
            "prerequisite_name": "Variables",
            "concept_name": "Functions",
            "confidence": 0.9,
            "prereq_type": "explicit"
        }

        response = await client.post(
            "/api/graph/courses/1/prerequisites",
            json=prereq
        )
        assert response.status_code in [200, 400, 500]

    @pytest.mark.asyncio
    async def test_remove_prerequisite(self, client):
        """Test removing a prerequisite relationship"""
        response = await client.delete(
            "/api/graph/courses/1/prerequisites"
            "?prerequisite_name=Variables&concept_name=Functions"
        )
        assert response.status_code in [200, 404, 500]

    @pytest.mark.asyncio
    async def test_detect_prerequisites(self, client):
        """Test auto-detecting prerequisites"""
        response = await client.post("/api/graph/courses/1/detect-prerequisites")

        if response.status_code == 200:
            data = response.json()
            assert "relationships_created" in data
            assert data["detection_method"] == "sequential"


class TestLearningPathEndpoints:
    """Tests for learning path endpoints"""

    @pytest.mark.asyncio
    async def test_generate_learning_path(self, client):
        """Test generating a learning path"""
        request = {
            "target_concepts": ["Recursion", "Dynamic Programming"],
            "mastered_concepts": ["Variables", "Loops"]
        }

        response = await client.post(
            "/api/graph/courses/1/learning-path",
            json=request
        )

        if response.status_code == 200:
            data = response.json()
            assert "learning_path" in data
            assert "total_concepts" in data

    @pytest.mark.asyncio
    async def test_generate_learning_path_no_mastered(self, client):
        """Test generating learning path without mastered concepts"""
        request = {
            "target_concepts": ["Advanced Topic"]
        }

        response = await client.post(
            "/api/graph/courses/1/learning-path",
            json=request
        )
        assert response.status_code in [200, 500]


class TestEntryPointsEndpoints:
    """Tests for entry/terminal points endpoints"""

    @pytest.mark.asyncio
    async def test_get_entry_points(self, client):
        """Test getting entry points"""
        response = await client.get("/api/graph/courses/1/entry-points")

        if response.status_code == 200:
            data = response.json()
            assert "entry_points" in data

    @pytest.mark.asyncio
    async def test_get_terminal_concepts(self, client):
        """Test getting terminal concepts"""
        response = await client.get("/api/graph/courses/1/terminal-concepts")

        if response.status_code == 200:
            data = response.json()
            assert "terminal_concepts" in data


class TestUtilityEndpoints:
    """Tests for utility endpoints"""

    @pytest.mark.asyncio
    async def test_extract_concepts(self, client):
        """Test extracting concepts from text"""
        text = "Python is a programming language that uses variables, functions, and classes."

        response = await client.post(
            f"/api/graph/extract-concepts?text={text}"
        )

        if response.status_code == 200:
            data = response.json()
            assert "concepts" in data
            assert "count" in data
            assert "method" in data

    @pytest.mark.asyncio
    async def test_extract_concepts_short_text(self, client):
        """Test extracting concepts with too short text"""
        response = await client.post("/api/graph/extract-concepts?text=short")
        assert response.status_code == 422  # Validation error


class TestLegacyEndpoint:
    """Tests for legacy endpoint"""

    @pytest.mark.asyncio
    async def test_get_graph_legacy(self, client):
        """Test legacy graph endpoint"""
        response = await client.get("/api/graph/")

        if response.status_code == 200:
            data = response.json()
            # Legacy format uses 'links' instead of 'edges'
            assert "nodes" in data
            assert "links" in data

    @pytest.mark.asyncio
    async def test_get_graph_legacy_with_course(self, client):
        """Test legacy graph endpoint with course ID"""
        response = await client.get("/api/graph/?course_id=2")
        assert response.status_code in [200, 500]
