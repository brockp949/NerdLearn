"""
Analogy Critique Engine
Manages analogy-based learning with breakdown point identification

Per PACER protocol, the Analogous (A) type requires:
1. Present analogy with source/target domain mapping
2. Show structural correspondences
3. Prompt learner to identify breakdown points
4. Evaluate critique quality (precision/recall)
5. Reveal missed breakdown points for learning
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.pacer import (
    AnalogyRecord,
    AnalogyCritique,
    PACERContentItem,
    PACERType,
)


@dataclass
class StructuralMapping:
    """Mapping between source and target domain elements"""

    source_element: str
    target_element: str
    relationship: str  # "corresponds_to", "functions_as", "represents"


@dataclass
class BreakdownPoint:
    """Where an analogy fails to hold"""

    aspect: str
    reason: str
    severity: str  # "minor", "moderate", "major"
    educational_note: str


@dataclass
class Analogy:
    """Complete analogy structure for learning"""

    id: int
    source_domain: str
    target_domain: str
    summary: str
    mappings: List[StructuralMapping]
    valid_aspects: List[str]
    breakdown_points: List[BreakdownPoint]
    critique_prompt: str


@dataclass
class CritiqueEvaluation:
    """Evaluation of user's analogy critique"""

    score: float  # F1 score
    precision: float
    recall: float
    correctly_identified: List[str]
    missed_breakdowns: List[Dict[str, str]]
    false_positives: List[str]
    feedback: str


class AnalogyCritiqueEngine:
    """
    Manages analogy-based learning per PACER protocol.

    Key operations:
    1. Create and store analogies with breakdown points
    2. Present analogies to learners with critique prompt
    3. Evaluate learner critiques
    4. Track critique proficiency over time
    """

    def __init__(self, db: Optional[AsyncSession] = None):
        self.db = db

    def create_analogy(
        self,
        source_domain: str,
        target_domain: str,
        content: str,
        mappings: List[Dict[str, str]],
        breakdowns: List[Dict[str, str]],
        valid_aspects: Optional[List[str]] = None,
    ) -> Analogy:
        """
        Create a structured analogy from content.

        Args:
            source_domain: Familiar concept (e.g., "Water Flow")
            target_domain: New concept being learned (e.g., "Electricity")
            content: Summary/explanation of the analogy
            mappings: List of element correspondences
            breakdowns: List of breakdown points with reasons
            valid_aspects: Optional list of aspects where analogy holds

        Returns:
            Structured Analogy object
        """
        structured_mappings = [
            StructuralMapping(
                source_element=m.get("source_element", m.get("source", "")),
                target_element=m.get("target_element", m.get("target", "")),
                relationship=m.get("relationship", "corresponds_to"),
            )
            for m in mappings
        ]

        structured_breakdowns = [
            BreakdownPoint(
                aspect=b.get("aspect", ""),
                reason=b.get("reason", ""),
                severity=b.get("severity", "moderate"),
                educational_note=b.get("educational_note", b.get("note", "")),
            )
            for b in breakdowns
        ]

        if valid_aspects is None:
            valid_aspects = [m.source_element for m in structured_mappings]

        return Analogy(
            id=hash(content) % 100000,  # Temporary ID until persisted
            source_domain=source_domain,
            target_domain=target_domain,
            summary=content,
            mappings=structured_mappings,
            valid_aspects=valid_aspects,
            breakdown_points=structured_breakdowns,
            critique_prompt=self._generate_critique_prompt(source_domain, target_domain),
        )

    def evaluate_critique(
        self,
        analogy: Analogy,
        user_identified_breakdowns: List[str],
        user_explanations: Optional[List[str]] = None,
    ) -> CritiqueEvaluation:
        """
        Evaluate user's critique of analogy breakdown points.

        Uses precision/recall metrics:
        - Precision: What fraction of user's identified breakdowns are correct
        - Recall: What fraction of actual breakdowns did user identify
        - F1: Harmonic mean of precision and recall

        Args:
            analogy: The analogy being critiqued
            user_identified_breakdowns: Aspects user identified as breakdown points
            user_explanations: Optional explanations for each identified breakdown

        Returns:
            CritiqueEvaluation with score, metrics, and feedback
        """
        # Normalize for comparison
        actual_breakdowns = {bp.aspect.lower().strip() for bp in analogy.breakdown_points}
        user_set = {b.lower().strip() for b in user_identified_breakdowns if b.strip()}

        # Calculate set operations
        true_positives = actual_breakdowns & user_set
        false_positives = user_set - actual_breakdowns
        false_negatives = actual_breakdowns - user_set

        # Calculate metrics
        tp_count = len(true_positives)
        fp_count = len(false_positives)
        fn_count = len(false_negatives)

        precision = tp_count / max(tp_count + fp_count, 1)
        recall = tp_count / max(tp_count + fn_count, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)

        # Build missed breakdowns list with educational notes
        missed = []
        for bp in analogy.breakdown_points:
            if bp.aspect.lower().strip() not in user_set:
                missed.append({
                    "aspect": bp.aspect,
                    "reason": bp.reason,
                    "note": bp.educational_note,
                    "severity": bp.severity,
                })

        return CritiqueEvaluation(
            score=f1,
            precision=precision,
            recall=recall,
            correctly_identified=list(true_positives),
            missed_breakdowns=missed,
            false_positives=list(false_positives),
            feedback=self._generate_feedback(f1, len(missed), len(false_positives)),
        )

    def evaluate_critique_fuzzy(
        self,
        analogy: Analogy,
        user_identified_breakdowns: List[str],
        similarity_threshold: float = 0.6,
    ) -> CritiqueEvaluation:
        """
        Evaluate critique with fuzzy matching for breakdown points.
        Handles cases where user uses different wording for same concept.

        Args:
            analogy: The analogy being critiqued
            user_identified_breakdowns: User's identified breakdown points
            similarity_threshold: Minimum similarity for fuzzy match (0-1)

        Returns:
            CritiqueEvaluation with fuzzy matching applied
        """
        actual_breakdowns = [bp.aspect.lower().strip() for bp in analogy.breakdown_points]
        user_breakdowns = [b.lower().strip() for b in user_identified_breakdowns if b.strip()]

        matched_actual = set()
        matched_user = set()

        # Find fuzzy matches
        for user_bp in user_breakdowns:
            for actual_bp in actual_breakdowns:
                if actual_bp in matched_actual:
                    continue
                similarity = self._string_similarity(user_bp, actual_bp)
                if similarity >= similarity_threshold:
                    matched_actual.add(actual_bp)
                    matched_user.add(user_bp)
                    break

        # Calculate metrics
        tp_count = len(matched_user)
        fp_count = len(set(user_breakdowns) - matched_user)
        fn_count = len(set(actual_breakdowns) - matched_actual)

        precision = tp_count / max(tp_count + fp_count, 1)
        recall = tp_count / max(tp_count + fn_count, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)

        # Build missed breakdowns
        missed = []
        for bp in analogy.breakdown_points:
            if bp.aspect.lower().strip() not in matched_actual:
                missed.append({
                    "aspect": bp.aspect,
                    "reason": bp.reason,
                    "note": bp.educational_note,
                    "severity": bp.severity,
                })

        return CritiqueEvaluation(
            score=f1,
            precision=precision,
            recall=recall,
            correctly_identified=list(matched_user),
            missed_breakdowns=missed,
            false_positives=list(set(user_breakdowns) - matched_user),
            feedback=self._generate_feedback(f1, len(missed), fp_count),
        )

    async def save_analogy_to_db(
        self,
        analogy: Analogy,
        pacer_item_id: int,
    ) -> AnalogyRecord:
        """
        Persist analogy to database.

        Args:
            analogy: Analogy object to save
            pacer_item_id: ID of parent PACERContentItem

        Returns:
            Created AnalogyRecord
        """
        if not self.db:
            raise ValueError("Database session required for persistence")

        record = AnalogyRecord(
            pacer_item_id=pacer_item_id,
            source_domain=analogy.source_domain,
            target_domain=analogy.target_domain,
            structural_mapping=[
                {
                    "source_element": m.source_element,
                    "target_element": m.target_element,
                    "relationship": m.relationship,
                }
                for m in analogy.mappings
            ],
            valid_aspects=analogy.valid_aspects,
            breakdown_points=[
                {
                    "aspect": bp.aspect,
                    "reason": bp.reason,
                    "severity": bp.severity,
                    "educational_note": bp.educational_note,
                }
                for bp in analogy.breakdown_points
            ],
            critique_prompt=analogy.critique_prompt,
        )

        self.db.add(record)
        await self.db.flush()
        return record

    async def save_critique_to_db(
        self,
        user_id: int,
        analogy_id: int,
        identified_breakdowns: List[Dict[str, str]],
        evaluation: CritiqueEvaluation,
    ) -> AnalogyCritique:
        """
        Save user's critique and evaluation to database.

        Args:
            user_id: ID of user submitting critique
            analogy_id: ID of analogy being critiqued
            identified_breakdowns: User's identified breakdown points
            evaluation: Evaluation results

        Returns:
            Created AnalogyCritique record
        """
        if not self.db:
            raise ValueError("Database session required for persistence")

        critique = AnalogyCritique(
            user_id=user_id,
            analogy_id=analogy_id,
            identified_breakdowns=identified_breakdowns,
            critique_score=evaluation.score,
            precision=evaluation.precision,
            recall=evaluation.recall,
            correct_breakdowns=evaluation.correctly_identified,
            missed_breakdowns=evaluation.missed_breakdowns,
            false_positives=evaluation.false_positives,
            feedback=evaluation.feedback,
        )

        self.db.add(critique)
        await self.db.flush()
        return critique

    async def get_analogy_by_id(self, analogy_id: int) -> Optional[Analogy]:
        """Load analogy from database by ID"""
        if not self.db:
            raise ValueError("Database session required")

        result = await self.db.execute(
            select(AnalogyRecord).where(AnalogyRecord.id == analogy_id)
        )
        record = result.scalar_one_or_none()

        if not record:
            return None

        return self._record_to_analogy(record)

    async def get_user_critique_history(
        self, user_id: int, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get user's critique history for proficiency tracking"""
        if not self.db:
            raise ValueError("Database session required")

        result = await self.db.execute(
            select(AnalogyCritique)
            .where(AnalogyCritique.user_id == user_id)
            .order_by(AnalogyCritique.created_at.desc())
            .limit(limit)
        )
        critiques = result.scalars().all()

        return [
            {
                "analogy_id": c.analogy_id,
                "score": c.critique_score,
                "precision": c.precision,
                "recall": c.recall,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in critiques
        ]

    def _generate_critique_prompt(self, source: str, target: str) -> str:
        """Generate a prompt for the learner to critique the analogy"""
        return f"""This analogy compares {source} to {target}.

While this analogy helps understand the concept, no analogy is perfect.

Your task: Identify where this analogy BREAKS DOWN.
- What aspects of {target} does {source} NOT capture?
- Where might this comparison mislead someone?
- What important differences exist between them?

List the breakdown points you can identify:"""

    def _generate_feedback(
        self, score: float, missed_count: int, false_positive_count: int
    ) -> str:
        """Generate feedback for the user's critique"""
        if score >= 0.8:
            return "Excellent critique! You identified the key breakdown points accurately."
        elif score >= 0.6:
            if missed_count > 0:
                return f"Good effort! You missed {missed_count} breakdown point(s). Review the hidden points to deepen your understanding."
            else:
                return f"Good identification, but {false_positive_count} point(s) weren't actual breakdowns. Focus on where the analogy truly fails."
        elif score >= 0.4:
            return f"Partial success. You missed {missed_count} breakdown(s) and had {false_positive_count} incorrect identification(s). Carefully consider the structural differences."
        else:
            return "This analogy has more nuance than identified. Review the breakdown points carefully - understanding where analogies fail is key to avoiding misconceptions."

    def _record_to_analogy(self, record: AnalogyRecord) -> Analogy:
        """Convert database record to Analogy dataclass"""
        mappings = [
            StructuralMapping(
                source_element=m.get("source_element", ""),
                target_element=m.get("target_element", ""),
                relationship=m.get("relationship", "corresponds_to"),
            )
            for m in (record.structural_mapping or [])
        ]

        breakdowns = [
            BreakdownPoint(
                aspect=bp.get("aspect", ""),
                reason=bp.get("reason", ""),
                severity=bp.get("severity", "moderate"),
                educational_note=bp.get("educational_note", ""),
            )
            for bp in (record.breakdown_points or [])
        ]

        return Analogy(
            id=record.id,
            source_domain=record.source_domain,
            target_domain=record.target_domain,
            summary="",  # Not stored in record, would need to join with PACERContentItem
            mappings=mappings,
            valid_aspects=record.valid_aspects or [],
            breakdown_points=breakdowns,
            critique_prompt=record.critique_prompt or "",
        )

    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate string similarity using Jaccard similarity of word sets.
        Simple but effective for breakdown point matching.
        """
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0
