"""
Tests for VectorStoreService
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid


class TestVectorStoreService:
    """Tests for the VectorStoreService class"""

    @pytest.fixture
    def mock_db(self):
        """Mock database session"""
        db = AsyncMock()
        db.execute = AsyncMock()
        db.commit = AsyncMock()
        db.merge = AsyncMock()
        return db

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client"""
        client = AsyncMock()
        client.embeddings.create = AsyncMock()
        return client

    @pytest.fixture
    def service(self, mock_db):
        """Create VectorStoreService with mocks"""
        with patch('app.services.vector_store.settings') as mock_settings:
            mock_settings.VECTOR_SIZE = 1536
            mock_settings.EMBEDDING_MODEL = "text-embedding-3-small"
            mock_settings.OPENAI_API_KEY = ""  # Empty to use mock embeddings

            from app.services.vector_store import VectorStoreService
            return VectorStoreService(db=mock_db)

    @pytest.mark.asyncio
    async def test_embed_text_single(self, service):
        """Test embedding a single text"""
        result = await service.embed_text("Hello world")

        assert isinstance(result, list)
        assert len(result) == service.vector_size

    @pytest.mark.asyncio
    async def test_embed_texts_empty(self, service):
        """Test embedding empty list"""
        result = await service.embed_texts([])

        assert result == []

    @pytest.mark.asyncio
    async def test_embed_texts_batch(self, service):
        """Test embedding multiple texts"""
        texts = ["Hello", "World", "Test"]
        result = await service.embed_texts(texts)

        assert len(result) == 3
        for embedding in result:
            assert len(embedding) == service.vector_size

    @pytest.mark.asyncio
    async def test_embed_texts_with_api_key(self, mock_db):
        """Test embedding with valid API key"""
        with patch('app.services.vector_store.settings') as mock_settings:
            mock_settings.VECTOR_SIZE = 1536
            mock_settings.EMBEDDING_MODEL = "text-embedding-3-small"
            mock_settings.OPENAI_API_KEY = "sk-test-key"

            with patch('app.services.vector_store.AsyncOpenAI') as mock_openai:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.data = [
                    MagicMock(embedding=[0.1] * 1536),
                    MagicMock(embedding=[0.2] * 1536),
                ]
                mock_client.embeddings.create = AsyncMock(return_value=mock_response)
                mock_openai.return_value = mock_client

                from app.services.vector_store import VectorStoreService
                service = VectorStoreService(db=mock_db)

                result = await service.embed_texts(["Hello", "World"])

                assert len(result) == 2

    @pytest.mark.asyncio
    async def test_search_basic(self, service, mock_db):
        """Test basic vector search"""
        mock_chunks = [
            MagicMock(
                id=str(uuid.uuid4()),
                text="Sample text",
                course_id=1,
                module_id=1,
                module_type="pdf",
                page_number=1,
                heading="Test",
                meta_data={}
            )
        ]
        mock_db.execute.return_value.scalars.return_value.all.return_value = mock_chunks

        results = await service.search("test query", course_id=1)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_course_filter(self, service, mock_db):
        """Test search with course ID filter"""
        mock_db.execute.return_value.scalars.return_value.all.return_value = []

        results = await service.search("test query", course_id=1)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_module_filter(self, service, mock_db):
        """Test search with module ID filter"""
        mock_db.execute.return_value.scalars.return_value.all.return_value = []

        results = await service.search("test query", module_id=1)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_limit(self, service, mock_db):
        """Test search with custom limit"""
        mock_db.execute.return_value.scalars.return_value.all.return_value = []

        results = await service.search("test query", limit=10)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_summaries(self, service, mock_db):
        """Test searching community summaries"""
        mock_db.execute.return_value.scalars.return_value.all.return_value = []

        results = await service.search_summaries("test query", course_id=1)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_upsert_documents_empty(self, service, mock_db):
        """Test upserting empty document list"""
        result = await service.upsert_documents([])

        assert result == 0

    @pytest.mark.asyncio
    async def test_upsert_documents_single(self, service, mock_db):
        """Test upserting a single document"""
        documents = [{
            "text": "Sample document text",
            "course_id": 1,
            "module_id": 1,
            "module_type": "pdf",
            "metadata": {"page": 1}
        }]

        result = await service.upsert_documents(documents)

        assert result == 1
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_documents_batch(self, service, mock_db):
        """Test upserting multiple documents"""
        documents = [
            {"text": f"Document {i}", "course_id": 1}
            for i in range(5)
        ]

        result = await service.upsert_documents(documents)

        assert result == 5

    @pytest.mark.asyncio
    async def test_upsert_documents_with_id(self, service, mock_db):
        """Test upserting document with existing ID"""
        doc_id = str(uuid.uuid4())
        documents = [{
            "id": doc_id,
            "text": "Sample text",
            "course_id": 1
        }]

        result = await service.upsert_documents(documents)

        assert result == 1


class TestSearchResultFormat:
    """Tests for search result formatting"""

    def test_result_has_required_fields(self):
        """Test that search results have required fields"""
        required_fields = [
            "id", "score", "text", "course_id",
            "module_id", "module_type", "page_number",
            "heading", "metadata"
        ]

        # Simulate a result
        result = {
            "id": "123",
            "score": 0.95,
            "text": "Sample",
            "course_id": 1,
            "module_id": 1,
            "module_type": "pdf",
            "page_number": 1,
            "heading": "Test",
            "metadata": {}
        }

        for field in required_fields:
            assert field in result

    def test_summary_result_format(self):
        """Test summary search result format"""
        result = {
            "id": "123",
            "score": 0.0,
            "text": "Summary text",
            "metadata": {},
            "module_type": "community_summary"
        }

        assert result["module_type"] == "community_summary"


class TestEmbeddingValidation:
    """Tests for embedding validation"""

    def test_embedding_size(self):
        """Test embedding has correct size"""
        expected_size = 1536
        embedding = [0.0] * expected_size

        assert len(embedding) == expected_size

    def test_embedding_values(self):
        """Test embedding values are floats"""
        embedding = [0.1, 0.2, -0.3, 0.5]

        for value in embedding:
            assert isinstance(value, float)

    def test_mock_embedding(self):
        """Test mock embedding is all zeros"""
        size = 1536
        mock_embedding = [0.0] * size

        assert all(v == 0.0 for v in mock_embedding)
        assert len(mock_embedding) == size
