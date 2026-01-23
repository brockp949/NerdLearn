import pytest
import unittest.mock as mock
from app.services.graph_service import AsyncGraphService


class TestGraphServiceOffline:
    """Offline integration tests for GraphService using mocks"""

    @pytest.fixture
    def mock_service(self):
        service = AsyncGraphService()
        service.driver = mock.AsyncMock()
        # session() is not async, it returns an async context manager
        service.driver.session = mock.MagicMock() 
        service._connected = True
        return service

    @pytest.mark.asyncio
    async def test_get_course_graph_query(self, mock_service):
        """Verify get_course_graph generates correct Cypher query"""
        # Setup mock session and result
        mock_session = mock.AsyncMock()
        mock_driver_session = mock.MagicMock()
        mock_driver_session.__aenter__.return_value = mock_session
        mock_driver_session.__aexit__.return_value = None
        
        mock_service.driver.session.return_value = mock_driver_session
        
        mock_result = mock.AsyncMock()
        mock_session.run.return_value = mock_result
        mock_result.data.return_value = []  # Return empty list, we just want to check the query

        # Execute
        await mock_service.get_course_graph(course_id=101)

        # Verify
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        params = call_args[1]

        # Check query structure
        assert "MATCH (c:Course {id: $course_id})" in query
        assert "RETURN con.name as concept" in query
        assert params["course_id"] == 101

    @pytest.mark.asyncio
    async def test_create_concept_node_query(self, mock_service):
        """Verify create_concept_node generates correct Cypher query"""
        mock_session = mock.AsyncMock()
        mock_driver_session = mock.MagicMock()
        mock_driver_session.__aenter__.return_value = mock_session
        mock_driver_session.__aexit__.return_value = None
        
        mock_service.driver.session.return_value = mock_driver_session
        
        mock_result = mock.AsyncMock()
        mock_session.run.return_value = mock_result
        mock_result.single.return_value = {"name": "Test Concept"}

        # Execute
        await mock_service.create_concept_node(
            course_id=1,
            module_id=5,
            name="Test Concept",
            difficulty=4.5
        )

        # Verify
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        params = call_args[1]

        assert "MERGE (con:Concept {name: $name, course_id: $course_id})" in query
        assert params["name"] == "Test Concept"
        assert params["difficulty"] == 4.5

    @pytest.mark.asyncio
    async def test_connection_handling(self):
        """Verify connection logic"""
        service = AsyncGraphService()
        
        with mock.patch("neo4j.AsyncGraphDatabase.driver") as mock_driver_cls:
            # Configure driver mock to handle async close
            mock_driver_instance = mock.AsyncMock()
            mock_driver_cls.return_value = mock_driver_instance
            
            await service.connect()
            mock_driver_cls.assert_called_once()
            assert service._connected is True
            assert service.driver == mock_driver_instance
            
            await service.close()
            assert service._connected is False
            mock_driver_instance.close.assert_awaited_once()
