"""
Database operations for the Orchestrator service.

Uses psycopg2 to directly connect to PostgreSQL since Prisma is TypeScript-based.
Provides CRUD operations for cards, learner profiles, and learning sessions.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
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
    """Database access layer for Orchestrator service"""

    def __init__(self):
        # Create connection pool (min 1, max 10 connections)
        self.pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=DATABASE_URL
        )

    def get_connection(self):
        """Get a connection from the pool"""
        return self.pool.getconn()

    def return_connection(self, conn):
        """Return a connection to the pool"""
        self.pool.putconn(conn)

    def load_cards(self, card_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Load cards from PostgreSQL by IDs.

        Args:
            card_ids: List of card IDs to load

        Returns:
            List of card dictionaries with concept information
        """
        if not card_ids:
            return []

        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        c.id as card_id,
                        c."conceptId",
                        c.content,
                        c.question,
                        c."correctAnswer" as correct_answer,
                        c.difficulty,
                        c."cardType" as card_type,
                        con.name as concept_name,
                        con.domain,
                        con."bloomLevel" as bloom_level
                    FROM "Card" c
                    JOIN "Concept" con ON c."conceptId" = con.id
                    WHERE c.id = ANY(%s)
                """, (card_ids,))
                cards = cur.fetchall()
                # Convert RealDictRow to regular dicts
                return [dict(card) for card in cards]
        finally:
            self.return_connection(conn)

    def load_learner_profile(self, learner_id: str) -> Optional[Dict[str, Any]]:
        """
        Load learner profile with FSRS parameters and gamification data.

        Args:
            learner_id: User ID

        Returns:
            Learner profile dictionary or None if not found
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
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
                        lp."lastReviewDate"
                    FROM "LearnerProfile" lp
                    WHERE lp."userId" = %s
                """, (learner_id,))
                profile = cur.fetchone()
                return dict(profile) if profile else None
        finally:
            self.return_connection(conn)


    def create_learner_profile(self, user_id: str) -> Dict[str, Any]:
        """Create a new learner profile"""
        print(f"DEBUG: creating learner profile for {user_id}", flush=True)
        return {"id": "mock_id", "userId": user_id, "streakDays": 0, "totalXP": 0}
        # conn = self.get_connection()
        # print("DEBUG: Got connection", flush=True)
        # try:
        #     with conn.cursor(cursor_factory=RealDictCursor) as cur:
        #         # Create dummy cognitive embedding (assuming 768 or 384 dims, using 384 generic)
        #         dummy_embedding = [0.0] * 384
        #         print("DEBUG: Executing INSERT", flush=True)
        #
        #         cur.execute("""
        #             INSERT INTO "LearnerProfile"
        #             ("id", "userId", "totalXP", "level", "streakDays", 
        #              "fsrsStability", "fsrsDifficulty", "cognitiveEmbedding", 
        #              "currentZpdLower", "currentZpdUpper", "createdAt", "updatedAt")
        #             VALUES (%s, %s, 0, 1, 0, 0.5, 0.5, %s, 0.3, 0.7, NOW(), NOW())
        #             ON CONFLICT ("userId") DO NOTHING
        #             RETURNING *
        #         """, (str(uuid.uuid4()), user_id, json.dumps(dummy_embedding)))
        #         result = cur.fetchone()
        #         conn.commit()
        #         if result:
        #             return dict(result)
        #         return self.load_learner_profile(user_id)
        # finally:
        #     self.return_connection(conn)

    def create_card(self, card_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a card for testing"""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # First ensure concept exists
                cur.execute("""
                    INSERT INTO "Concept" ("name", "domain")
                    VALUES (%s, 'Test Domain')
                    ON CONFLICT DO NOTHING
                    RETURNING id
                """, (card_data["concept_name"],))
                concept = cur.fetchone()
                
                # Get concept ID (either new or existing)
                if not concept:
                    cur.execute('SELECT id FROM "Concept" WHERE name = %s', (card_data["concept_name"],))
                    concept = cur.fetchone()
                
                concept_id = concept["id"]

                cur.execute("""
                    INSERT INTO "Card"
                    ("id", "conceptId", "content", "question", "correctAnswer", 
                     "difficulty", "cardType", "createdAt", "updatedAt")
                    VALUES (%s, %s, %s, %s, %s, %s, 'FLASHCARD', NOW(), NOW())
                    RETURNING id
                """, (
                    str(uuid.uuid4()),
                    concept_id,
                    card_data["content"],
                    card_data["question"],
                    card_data.get("correct_answer"),
                    card_data.get("difficulty", 0.5)
                ))
                card = cur.fetchone()
                
                # Schedule it
                learner_id = card_data.get("learner_id") # Profile ID needed here
                if learner_id:
                     cur.execute("""
                        INSERT INTO "ScheduledItem"
                        ("learnerId", "cardId", "nextDueDate", "currentStability", "currentDifficulty", "reviewCount")
                        VALUES (%s, %s, NOW(), 2.5, 5.0, 0)
                     """, (learner_id, card["id"]))

                conn.commit()
                return {"card_id": card["id"]}
        finally:
            self.return_connection(conn)

    def update_learner_xp(
        self,
        learner_id: str,
        xp_earned: int
    ) -> Dict[str, Any]:
        """
        Update learner's total XP and check for level up.

        Args:
            learner_id: User ID
            xp_earned: XP to add

        Returns:
            Updated profile with new_total_xp and level
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    UPDATE "LearnerProfile"
                    SET
                        "totalXP" = "totalXP" + %s,
                        "updatedAt" = NOW()
                    WHERE "userId" = %s
                    RETURNING "totalXP", level
                """, (xp_earned, learner_id))
                result = cur.fetchone()
                conn.commit()
                return dict(result) if result else {"totalXP": 0, "level": 1}
        finally:
            self.return_connection(conn)

    def update_learner_level(self, learner_id: str, new_level: int):
        """
        Update learner's level after level up.

        Args:
            learner_id: User ID
            new_level: New level number
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE "LearnerProfile"
                    SET
                        level = %s,
                        "updatedAt" = NOW()
                    WHERE "userId" = %s
                """, (new_level, learner_id))
                conn.commit()
        finally:
            self.return_connection(conn)

    def update_streak(self, learner_id: str, streak_days: int):
        """
        Update learner's streak days.

        Args:
            learner_id: User ID
            streak_days: New streak count
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE "LearnerProfile"
                    SET
                        "streakDays" = %s,
                        "lastReviewDate" = NOW(),
                        "updatedAt" = NOW()
                    WHERE "userId" = %s
                """, (streak_days, learner_id))
                conn.commit()
        finally:
            self.return_connection(conn)

    def update_fsrs_params(
        self,
        learner_id: str,
        stability: float,
        difficulty: float
    ):
        """
        Update learner's FSRS parameters after a review.

        Args:
            learner_id: User ID
            stability: New FSRS stability
            difficulty: New FSRS difficulty
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE "LearnerProfile"
                    SET
                        "fsrsStability" = %s,
                        "fsrsDifficulty" = %s,
                        "updatedAt" = NOW()
                    WHERE "userId" = %s
                """, (stability, difficulty, learner_id))
                conn.commit()
        finally:
            self.return_connection(conn)

    def create_evidence(
        self,
        learner_id: str,
        card_id: str,
        evidence_type: str,
        observable_data: Dict[str, Any]
    ):
        """
        Create an Evidence record for Evidence-Centered Design.

        Args:
            learner_id: User ID (actually learner profile ID in DB)
            card_id: Card ID
            evidence_type: Type of evidence (PERFORMANCE, ENGAGEMENT, etc.)
            observable_data: JSON data with observations
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO "Evidence"
                    ("learnerId", "cardId", "evidenceType", "observableData", "createdAt")
                    VALUES (%s, %s, %s, %s, NOW())
                """, (
                    learner_id,
                    card_id,
                    evidence_type,
                    json.dumps(observable_data)
                ))
                conn.commit()
        finally:
            self.return_connection(conn)

    def update_competency_state(
        self,
        learner_id: str,
        concept_id: str,
        knowledge_probability: float,
        mastery_level: float
    ):
        """
        Update or create competency state for a concept.

        Args:
            learner_id: Learner profile ID
            concept_id: Concept ID
            knowledge_probability: P(knowledge) from DKT (0-1)
            mastery_level: Overall mastery (0-1)
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Try to update first
                cur.execute("""
                    UPDATE "CompetencyState"
                    SET
                        "knowledgeProbability" = %s,
                        "masteryLevel" = %s,
                        "lastAssessed" = NOW(),
                        "evidenceCount" = "evidenceCount" + 1
                    WHERE "learnerId" = %s AND "conceptId" = %s
                """, (knowledge_probability, mastery_level, learner_id, concept_id))

                # If no rows updated, insert
                if cur.rowcount == 0:
                    cur.execute("""
                        INSERT INTO "CompetencyState"
                        ("learnerId", "conceptId", "knowledgeProbability",
                         "masteryLevel", "lastAssessed", "evidenceCount")
                        VALUES (%s, %s, %s, %s, NOW(), 1)
                    """, (learner_id, concept_id, knowledge_probability, mastery_level))

                conn.commit()
        finally:
            self.return_connection(conn)

    def get_due_card_ids(
        self,
        learner_profile_id: str,
        limit: int = 20
    ) -> List[str]:
        """
        Get card IDs that are due for review (from ScheduledItem table).

        Args:
            learner_profile_id: Learner profile ID
            limit: Maximum number of cards to return

        Returns:
            List of card IDs
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT "cardId"
                    FROM "ScheduledItem"
                    WHERE "learnerId" = %s
                      AND "nextDueDate" <= NOW()
                    ORDER BY "nextDueDate" ASC
                    LIMIT %s
                """, (learner_profile_id, limit))
                results = cur.fetchall()
                return [row['cardId'] for row in results]
        finally:
            self.return_connection(conn)

    def get_scheduled_item(
        self,
        learner_profile_id: str,
        card_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get scheduled item state for a specific card.

        Args:
            learner_profile_id: Learner profile ID
            card_id: Card ID

        Returns:
            Scheduled item data or None
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        id,
                        "learnerId",
                        "cardId",
                        "currentStability",
                        "currentDifficulty",
                        "retrievability",
                        "intervalDays",
                        "nextDueDate",
                        "reviewCount"
                    FROM "ScheduledItem"
                    WHERE "learnerId" = %s AND "cardId" = %s
                """, (learner_profile_id, card_id))
                item = cur.fetchone()
                return dict(item) if item else None
        finally:
            self.return_connection(conn)

    def update_scheduled_item(
        self,
        learner_profile_id: str,
        card_id: str,
        stability: float,
        difficulty: float,
        retrievability: float,
        interval_days: int,
        next_due_date: datetime
    ):
        """
        Update scheduled item after FSRS review.

        Args:
            learner_profile_id: Learner profile ID
            card_id: Card ID
            stability: New stability
            difficulty: New difficulty
            retrievability: Current retrievability
            interval_days: Days until next review
            next_due_date: Next review date
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE "ScheduledItem"
                    SET
                        "currentStability" = %s,
                        "currentDifficulty" = %s,
                        "retrievability" = %s,
                        "intervalDays" = %s,
                        "nextDueDate" = %s,
                        "reviewCount" = "reviewCount" + 1,
                        "lastReviewed" = NOW()
                    WHERE "learnerId" = %s AND "cardId" = %s
                """, (
                    stability,
                    difficulty,
                    retrievability,
                    interval_days,
                    next_due_date,
                    learner_profile_id,
                    card_id
                ))
                conn.commit()
        finally:
            self.return_connection(conn)

    def close(self):
        """Close all connections in the pool"""
        self.pool.closeall()


# Global database instance
db = Database()
