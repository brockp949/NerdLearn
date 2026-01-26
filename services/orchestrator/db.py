"""
Database operations for the Orchestrator service.

Uses psycopg2 to directly connect to PostgreSQL since Prisma is TypeScript-based.
Provides CRUD operations for cards, learner profiles, and learning sessions.
"""

import asyncpg
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import uuid
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://nerdlearn:nerdlearn_dev_password@localhost:5432/nerdlearn"
)

class Database:
    """Async Database access layer for Orchestrator service via asyncpg"""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Initialize the connection pool"""
        if not self.pool:
            try:
                self.pool = await asyncpg.create_pool(
                    dsn=DATABASE_URL,
                    min_size=1,
                    max_size=10
                )
                print("✅ Database connection pool created", flush=True)
            except Exception as e:
                print(f"❌ Failed to connect to database: {e}", flush=True)
                raise

    async def disconnect(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()
            print("Database connection pool closed", flush=True)

    async def load_cards(self, card_ids: List[str]) -> List[Dict[str, Any]]:
        """Load cards (Resources) from PostgreSQL by IDs"""
        if not card_ids:
            return []
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    r.id as card_id,
                    r."conceptId",
                    r."contentData"->>'content' as content,
                    r."contentData"->>'question' as question,
                    r."contentData"->>'correctAnswer' as correct_answer,
                    r.difficulty,
                    r."contentData"->>'cardType' as card_type,
                    con.name as concept_name,
                    con.domain,
                    con."taxonomyLevel" as bloom_level
                FROM "Resource" r
                JOIN "Concept" con ON r."conceptId" = con.id
                WHERE r.id = ANY($1)
            """, card_ids)
            return [dict(row) for row in rows]

    async def load_learner_profile(self, learner_id: str) -> Optional[Dict[str, Any]]:
        """Load learner profile"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    lp.id,
                    lp."userId",
                    lp."cognitiveEmbedding",
                    lp."fsrsStability",
                    lp."fsrsDifficulty",
                    lp."currentZpdLower",
                    lp."currentZpdUpper",
                    lp."totalXP",
                    lp.level,
                    lp."streakDays",
                    lp."lastActivityDate" as "lastReviewDate"
                FROM "LearnerProfile" lp
                WHERE lp."userId" = $1
            """, learner_id)
            return dict(row) if row else None

    async def create_learner_profile(self, user_id: str) -> Dict[str, Any]:
        """Create a new learner profile"""
        print(f"DEBUG: creating learner profile for {user_id}", flush=True)
        async with self.pool.acquire() as conn:
            # Ensure User exists (for test seeding)
            try:
                await conn.execute("""
                    INSERT INTO "User" ("id", "email", "username", "passwordHash", "updatedAt")
                    VALUES ($1, $2, $3, 'test_hash_123', NOW())
                """, user_id, f"{user_id}@test.com", user_id)
            except asyncpg.UniqueViolationError:
                pass

            # Create dummy cognitive embedding (assuming 384 generic)
            dummy_embedding = [0.0] * 384
            
            row = await conn.fetchrow("""
                INSERT INTO "LearnerProfile"
                ("id", "userId", "totalXP", "level", "streakDays", 
                    "fsrsStability", "fsrsDifficulty", "cognitiveEmbedding", 
                    "currentZpdLower", "currentZpdUpper", "createdAt", "updatedAt")
                VALUES ($1, $2, 0, 1, 0, 0.5, 0.5, $3, 0.3, 0.7, NOW(), NOW())
                ON CONFLICT ("userId") DO NOTHING
                RETURNING *
            """, str(uuid.uuid4()), user_id, json.dumps(dummy_embedding))
            
            if row:
                return dict(row)
            return await self.load_learner_profile(user_id)

    async def create_card(self, card_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a card (Resource) for testing"""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Ensure concept exists
                concept_id = await conn.fetchval('SELECT id FROM "Concept" WHERE name = $1', card_data["concept_name"])
                
                if not concept_id:
                    concept_id = str(uuid.uuid4())
                    await conn.execute("""
                        INSERT INTO "Concept" ("id", "neoId", "name", "domain", "updatedAt")
                        VALUES ($1, $2, $3, 'Test Domain', NOW())
                    """, concept_id, f"neo_{concept_id}", card_data["concept_name"])

                # Prepare contentData JSON
                content_data = {
                    "content": card_data["content"],
                    "question": card_data["question"],
                    "correctAnswer": card_data.get("correct_answer"),
                    "cardType": "FLASHCARD"
                }

                card_id = await conn.fetchval("""
                    INSERT INTO "Resource"
                    ("id", "title", "type", "conceptId", "contentData", "difficulty", "createdAt", "updatedAt")
                    VALUES ($1, $2, 'EXERCISE', $3, $4, $5, NOW(), NOW())
                    RETURNING id
                """, str(uuid.uuid4()), f"{card_data['concept_name']} Card", concept_id, 
                   json.dumps(content_data), card_data.get("difficulty", 0.5))

                # Schedule it
                learner_id = card_data.get("learner_id")
                if learner_id:
                    # ScheduledItem uses resourceId, not cardId
                    # Schema has: learnerId, resourceId, dueDate, stability, difficulty, status
                    await conn.execute("""
                        INSERT INTO "ScheduledItem"
                        ("id", "learnerId", "resourceId", "dueDate", "stability", "difficulty", "status", "updatedAt", "createdAt")
                        VALUES ($1, $2, $3, NOW(), 2.5, 5.0, 'DUE', NOW(), NOW())
                    """, str(uuid.uuid4()), learner_id, card_id)
                
                return {"card_id": card_id}

    async def update_learner_xp(self, learner_id: str, xp_earned: int) -> Dict[str, Any]:
        """Update learner's total XP"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                UPDATE "LearnerProfile"
                SET
                    "totalXP" = "totalXP" + $1,
                    "updatedAt" = NOW()
                WHERE "userId" = $2
                RETURNING "totalXP", level
            """, xp_earned, learner_id)
            return dict(row) if row else {"totalXP": 0, "level": 1}

    async def update_learner_level(self, learner_id: str, new_level: int):
        """Update learner's level"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE "LearnerProfile"
                SET level = $1, "updatedAt" = NOW()
                WHERE "userId" = $2
            """, new_level, learner_id)

    async def update_streak(self, learner_id: str, streak_days: int):
        """Update learner's streak days"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE "LearnerProfile"
                SET
                    "streakDays" = $1,
                    "lastActivityDate" = NOW(),
                    "updatedAt" = NOW()
                WHERE "userId" = $2
            """, streak_days, learner_id)

    async def update_fsrs_params(self, learner_id: str, stability: float, difficulty: float):
        """Update FSRS parameters"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE "LearnerProfile"
                SET
                    "fsrsStability" = $1,
                    "fsrsDifficulty" = $2,
                    "updatedAt" = NOW()
                WHERE "userId" = $3
            """, stability, difficulty, learner_id)

    async def create_evidence(self, learner_id: str, card_id: str, evidence_type: str, observable_data: Dict[str, Any]):
        """Create an Evidence record - Simplified: Needs CompetencyState first"""
        pass 

    async def update_competency_state(self, learner_id: str, concept_id: str, knowledge_probability: float, mastery_level: float):
        """Update or create competency state"""
        async with self.pool.acquire() as conn:
            # Upsert
            await conn.execute("""
                INSERT INTO "CompetencyState"
                ("id", "learnerId", "conceptId", "masteryProbability", "confidence", "updatedAt")
                VALUES ($1, $2, $3, $4, 0.5, NOW())
                ON CONFLICT ("learnerId", "conceptId") DO UPDATE SET
                    "masteryProbability" = $4,
                    "updatedAt" = NOW()
            """, str(uuid.uuid4()), learner_id, concept_id, knowledge_probability)

    async def get_due_card_ids(self, learner_profile_id: str, limit: int = 20) -> List[str]:
        """Get due card IDs (Resource IDs)"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT "resourceId" as "cardId"
                FROM "ScheduledItem"
                WHERE "learnerId" = $1
                    AND "dueDate" <= NOW()
                ORDER BY "dueDate" ASC
                LIMIT $2
            """, learner_profile_id, limit)
            return [row['cardId'] for row in rows]

    async def get_scheduled_item(self, learner_profile_id: str, card_id: str) -> Optional[Dict[str, Any]]:
        """Get scheduled item state"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    id,
                    "learnerId",
                    "resourceId" as "cardId",
                    "stability" as "currentStability",
                    "difficulty" as "currentDifficulty",
                    "stability",
                    "difficulty",
                    "dueDate" as "nextDueDate"
                FROM "ScheduledItem"
                WHERE "learnerId" = $1 AND "resourceId" = $2
            """, learner_profile_id, card_id)
            if row:
                d = dict(row)
                # Add missing fields expected by logic, using defaults
                d["retrievability"] = 0.9
                d["intervalDays"] = 1
                return d
            return None

    async def update_scheduled_item(
        self,
        learner_profile_id: str,
        card_id: str,
        stability: float,
        difficulty: float,
        retrievability: float,
        interval_days: int,
        next_due_date: datetime
    ):
        """Update scheduled item after FSRS review"""
        async with self.pool.acquire() as conn:
            # Clean update based on schema
            await conn.execute("""
                UPDATE "ScheduledItem"
                SET
                    "stability" = $1,
                    "difficulty" = $2,
                    "dueDate" = $3,
                    "updatedAt" = NOW()
                WHERE "learnerId" = $4 AND "resourceId" = $5
            """, stability, difficulty, next_due_date, learner_profile_id, card_id)

    async def close(self):
        """Close connection pool"""
        await self.disconnect()


# Global database instance
db = Database()
