from neo4j import AsyncGraphDatabase
from app.core.config import settings


class Neo4jDatabase:
    def __init__(self):
        self.driver = None

    async def connect(self):
        self.driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

    async def close(self):
        if self.driver:
            await self.driver.close()

    async def execute_query(self, query: str, parameters: dict = None):
        async with self.driver.session() as session:
            result = await session.run(query, parameters or {})
            return await result.data()


# Global instance
neo4j_db = Neo4jDatabase()


# Dependency for getting Neo4j session
async def get_neo4j():
    if not neo4j_db.driver:
        await neo4j_db.connect()
    return neo4j_db
