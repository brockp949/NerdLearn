"""
Evidence-Concept Linking Service
Manages relationships between Evidence items and Conceptual nodes

Per PACER protocol:
- Evidence (E) = concrete facts, statistics, research that SUPPORTS concepts
- Evidence must be linked to the Concept it validates
- Retrieval cue: "What data proves this theory?"
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.models.pacer import (
    EvidenceConceptLink,
    PACERContentItem,
    PACERType,
    EvidenceRelationshipType,
)
from app.models.spaced_repetition import Concept


@dataclass
class EvidenceLink:
    """Represents a link between evidence and concept"""

    evidence_id: int
    concept_id: int
    concept_name: str
    relationship_type: str
    strength: float
    citation: Optional[str] = None


@dataclass
class LinkedEvidence:
    """Evidence item with its linked concepts"""

    evidence_id: int
    content: str
    title: str
    links: List[EvidenceLink]


class EvidenceLinkingService:
    """
    Manages Evidence-Concept relationships per PACER protocol.

    Key operations:
    1. Auto-detect which concepts evidence supports
    2. Create/update evidence-concept links
    3. Query evidence by concept for study reinforcement
    4. Surface contradicting evidence for critical thinking
    """

    def __init__(self, db: Optional[AsyncSession] = None):
        self.db = db

    async def link_evidence_to_concepts(
        self,
        evidence_item_id: int,
        concept_ids: List[int],
        relationship_type: EvidenceRelationshipType = EvidenceRelationshipType.SUPPORTS,
        strength: float = 0.7,
        citation: Optional[str] = None,
    ) -> List[EvidenceConceptLink]:
        """
        Create links between an evidence item and concepts.

        Args:
            evidence_item_id: ID of the evidence PACERContentItem
            concept_ids: List of concept IDs to link to
            relationship_type: How evidence relates to concept
            strength: Link strength (0-1)
            citation: Optional citation for the evidence

        Returns:
            List of created EvidenceConceptLink records
        """
        if not self.db:
            raise ValueError("Database session required")

        links = []
        for concept_id in concept_ids:
            link = EvidenceConceptLink(
                evidence_item_id=evidence_item_id,
                concept_id=concept_id,
                relationship_type=relationship_type,
                strength=strength,
                citation=citation,
            )
            self.db.add(link)
            links.append(link)

        await self.db.flush()
        return links

    async def auto_link_evidence(
        self,
        evidence_content: str,
        candidate_concepts: List[Dict[str, Any]],
        min_relevance: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Automatically suggest concept links for evidence content.
        Uses keyword matching and semantic analysis.

        Args:
            evidence_content: The evidence text to analyze
            candidate_concepts: List of concepts to match against
            min_relevance: Minimum relevance score for suggestion

        Returns:
            List of suggested links with relevance scores
        """
        suggestions = []
        content_lower = evidence_content.lower()

        for concept in candidate_concepts:
            concept_name = concept.get("name", "").lower()
            concept_desc = concept.get("description", "").lower()

            # Calculate relevance
            relevance = self._calculate_relevance(
                content_lower, concept_name, concept_desc
            )

            if relevance >= min_relevance:
                suggestions.append({
                    "concept_id": concept.get("id"),
                    "concept_name": concept.get("name"),
                    "relevance": relevance,
                    "suggested_relationship": self._suggest_relationship(evidence_content),
                })

        # Sort by relevance descending
        suggestions.sort(key=lambda x: x["relevance"], reverse=True)
        return suggestions

    async def get_evidence_for_concept(
        self,
        concept_id: int,
        relationship_type: Optional[EvidenceRelationshipType] = None,
        min_strength: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Get all evidence items linked to a concept.

        Args:
            concept_id: ID of the concept
            relationship_type: Optional filter by relationship type
            min_strength: Minimum link strength to include

        Returns:
            List of evidence items with link metadata
        """
        if not self.db:
            raise ValueError("Database session required")

        query = (
            select(EvidenceConceptLink, PACERContentItem)
            .join(PACERContentItem, EvidenceConceptLink.evidence_item_id == PACERContentItem.id)
            .where(
                and_(
                    EvidenceConceptLink.concept_id == concept_id,
                    EvidenceConceptLink.strength >= min_strength,
                )
            )
        )

        if relationship_type:
            query = query.where(EvidenceConceptLink.relationship_type == relationship_type)

        result = await self.db.execute(query)
        rows = result.all()

        return [
            {
                "evidence_id": link.evidence_item_id,
                "title": item.title,
                "content": item.content,
                "relationship_type": link.relationship_type.value,
                "strength": link.strength,
                "citation": link.citation,
            }
            for link, item in rows
        ]

    async def get_supporting_evidence(self, concept_id: int) -> List[Dict[str, Any]]:
        """Get evidence that supports a concept"""
        return await self.get_evidence_for_concept(
            concept_id,
            relationship_type=EvidenceRelationshipType.SUPPORTS,
        )

    async def get_contradicting_evidence(self, concept_id: int) -> List[Dict[str, Any]]:
        """
        Get evidence that contradicts a concept.
        Useful for critical thinking exercises.
        """
        return await self.get_evidence_for_concept(
            concept_id,
            relationship_type=EvidenceRelationshipType.CONTRADICTS,
        )

    async def get_qualifying_evidence(self, concept_id: int) -> List[Dict[str, Any]]:
        """
        Get evidence that qualifies/limits a concept.
        E.g., "This principle only applies under X conditions"
        """
        return await self.get_evidence_for_concept(
            concept_id,
            relationship_type=EvidenceRelationshipType.QUALIFIES,
        )

    async def get_concepts_for_evidence(
        self, evidence_item_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get all concepts linked to an evidence item.

        Args:
            evidence_item_id: ID of the evidence item

        Returns:
            List of concepts with link metadata
        """
        if not self.db:
            raise ValueError("Database session required")

        result = await self.db.execute(
            select(EvidenceConceptLink, Concept)
            .join(Concept, EvidenceConceptLink.concept_id == Concept.id)
            .where(EvidenceConceptLink.evidence_item_id == evidence_item_id)
        )
        rows = result.all()

        return [
            {
                "concept_id": concept.id,
                "concept_name": concept.name,
                "concept_description": concept.description,
                "relationship_type": link.relationship_type.value,
                "strength": link.strength,
            }
            for link, concept in rows
        ]

    async def update_link_strength(
        self,
        evidence_item_id: int,
        concept_id: int,
        new_strength: float,
    ) -> Optional[EvidenceConceptLink]:
        """Update the strength of an existing link"""
        if not self.db:
            raise ValueError("Database session required")

        result = await self.db.execute(
            select(EvidenceConceptLink).where(
                and_(
                    EvidenceConceptLink.evidence_item_id == evidence_item_id,
                    EvidenceConceptLink.concept_id == concept_id,
                )
            )
        )
        link = result.scalar_one_or_none()

        if link:
            link.strength = max(0.0, min(1.0, new_strength))
            await self.db.flush()

        return link

    async def remove_link(
        self, evidence_item_id: int, concept_id: int
    ) -> bool:
        """Remove a link between evidence and concept"""
        if not self.db:
            raise ValueError("Database session required")

        result = await self.db.execute(
            select(EvidenceConceptLink).where(
                and_(
                    EvidenceConceptLink.evidence_item_id == evidence_item_id,
                    EvidenceConceptLink.concept_id == concept_id,
                )
            )
        )
        link = result.scalar_one_or_none()

        if link:
            await self.db.delete(link)
            await self.db.flush()
            return True

        return False

    def _calculate_relevance(
        self,
        evidence_content: str,
        concept_name: str,
        concept_description: str,
    ) -> float:
        """
        Calculate relevance score between evidence and concept.
        Uses keyword overlap and semantic indicators.
        """
        # Direct name match
        if concept_name in evidence_content:
            return 0.8

        # Word overlap scoring
        evidence_words = set(evidence_content.split())
        concept_words = set(concept_name.split()) | set(concept_description.split())

        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "to", "in", "for", "on", "with"}
        evidence_words -= stop_words
        concept_words -= stop_words

        if not evidence_words or not concept_words:
            return 0.1

        overlap = len(evidence_words & concept_words)
        total = len(evidence_words | concept_words)

        # Base Jaccard similarity
        base_score = overlap / total if total > 0 else 0

        # Boost for evidence-specific keywords present
        evidence_keywords = ["study", "research", "data", "shows", "demonstrates", "found", "evidence"]
        if any(kw in evidence_content for kw in evidence_keywords):
            base_score = min(1.0, base_score + 0.1)

        return base_score

    def _suggest_relationship(self, evidence_content: str) -> str:
        """Suggest relationship type based on evidence content"""
        content_lower = evidence_content.lower()

        # Check for contradiction indicators
        contradiction_keywords = [
            "however", "contrary", "contradicts", "disproves",
            "challenges", "refutes", "disputes", "opposite",
        ]
        if any(kw in content_lower for kw in contradiction_keywords):
            return EvidenceRelationshipType.CONTRADICTS.value

        # Check for qualification indicators
        qualification_keywords = [
            "except", "unless", "only when", "limited to",
            "under certain", "in some cases", "qualifies", "condition",
        ]
        if any(kw in content_lower for kw in qualification_keywords):
            return EvidenceRelationshipType.QUALIFIES.value

        # Default to supports
        return EvidenceRelationshipType.SUPPORTS.value

    async def get_evidence_coverage_for_course(
        self, course_id: int
    ) -> Dict[str, Any]:
        """
        Analyze evidence coverage for concepts in a course.
        Helps identify which concepts need more supporting evidence.
        """
        if not self.db:
            raise ValueError("Database session required")

        # Get all concepts for course
        concepts_result = await self.db.execute(
            select(Concept).where(Concept.course_id == course_id)
        )
        concepts = concepts_result.scalars().all()

        coverage = {
            "total_concepts": len(concepts),
            "concepts_with_evidence": 0,
            "concepts_without_evidence": 0,
            "coverage_details": [],
        }

        for concept in concepts:
            evidence = await self.get_evidence_for_concept(concept.id)
            evidence_count = len(evidence)

            if evidence_count > 0:
                coverage["concepts_with_evidence"] += 1
            else:
                coverage["concepts_without_evidence"] += 1

            coverage["coverage_details"].append({
                "concept_id": concept.id,
                "concept_name": concept.name,
                "evidence_count": evidence_count,
                "has_supporting": any(e["relationship_type"] == "supports" for e in evidence),
                "has_contradicting": any(e["relationship_type"] == "contradicts" for e in evidence),
            })

        return coverage
