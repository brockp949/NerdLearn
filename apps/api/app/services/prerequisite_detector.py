"""
Automated Prerequisite Detection Service

Uses ML to detect prerequisites between concepts based on:
- Learning failure patterns (users who struggle with B often haven't mastered A)
- Content analysis (concept B references concept A)
- Knowledge graph structure
- Temporal learning patterns

Features:
- Failure pattern analysis
- Content dependency extraction
- Confidence scoring
- Explanation generation
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class PrerequisiteCandidate:
    """A potential prerequisite relationship"""
    source_concept_id: int
    source_concept_name: str
    target_concept_id: int
    target_concept_name: str
    confidence: float
    evidence_types: List[str]
    explanation: str
    strength: str  # "strong", "moderate", "weak"


@dataclass
class FailurePattern:
    """Pattern of learning failures"""
    concept_id: int
    concept_name: str
    failure_rate: float
    users_analyzed: int
    common_weak_concepts: List[Tuple[int, str, float]]  # (id, name, correlation)


@dataclass
class ConceptReference:
    """Reference from one concept to another in content"""
    source_concept_id: int
    target_concept_id: int
    reference_count: int
    reference_contexts: List[str]


class PrerequisiteDetector:
    """
    ML-based prerequisite detection system.

    Uses multiple signals to detect prerequisite relationships:
    1. Failure correlation: If users who fail concept B often have low mastery of A
    2. Content analysis: If concept B's content references A
    3. Temporal patterns: If users typically learn A before B
    4. Expert input: Manual prerequisite annotations
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        min_users_for_analysis: int = 20,
        failure_threshold: float = 0.5
    ):
        """
        Initialize detector.

        Args:
            min_confidence: Minimum confidence to suggest prerequisite
            min_users_for_analysis: Minimum users needed for pattern analysis
            failure_threshold: Mastery level below which is considered "failure"
        """
        self.min_confidence = min_confidence
        self.min_users_for_analysis = min_users_for_analysis
        self.failure_threshold = failure_threshold

        # Weights for different evidence types
        self.evidence_weights = {
            "failure_correlation": 0.4,
            "content_reference": 0.25,
            "temporal_pattern": 0.2,
            "structural": 0.15,
        }

    async def detect_prerequisites(
        self,
        concept_id: int,
        mastery_data: List[Dict[str, Any]],
        content_data: Optional[Dict[str, str]] = None,
        existing_graph: Optional[Dict[int, List[int]]] = None
    ) -> List[PrerequisiteCandidate]:
        """
        Detect prerequisites for a concept.

        Args:
            concept_id: Target concept to find prerequisites for
            mastery_data: User mastery data [{user_id, concept_id, mastery, timestamp}]
            content_data: Optional concept content {concept_id: content_text}
            existing_graph: Optional existing prerequisite graph

        Returns:
            List of prerequisite candidates ranked by confidence
        """
        candidates = []
        evidence_by_concept: Dict[int, Dict[str, float]] = defaultdict(dict)

        # 1. Analyze failure patterns
        failure_patterns = self._analyze_failure_patterns(
            concept_id, mastery_data
        )
        for weak_id, weak_name, correlation in failure_patterns.common_weak_concepts:
            evidence_by_concept[weak_id]["failure_correlation"] = correlation
            evidence_by_concept[weak_id]["name"] = weak_name

        # 2. Analyze content references
        if content_data:
            references = self._analyze_content_references(
                concept_id, content_data
            )
            for ref in references:
                ref_score = min(1.0, ref.reference_count / 5)  # Normalize
                evidence_by_concept[ref.source_concept_id]["content_reference"] = ref_score

        # 3. Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(
            concept_id, mastery_data
        )
        for prereq_id, correlation in temporal_patterns:
            evidence_by_concept[prereq_id]["temporal_pattern"] = correlation

        # 4. Check structural hints from existing graph
        if existing_graph:
            structural_hints = self._get_structural_hints(
                concept_id, existing_graph
            )
            for hint_id, score in structural_hints:
                evidence_by_concept[hint_id]["structural"] = score

        # Calculate overall confidence for each candidate
        for candidate_id, evidence in evidence_by_concept.items():
            confidence = self._calculate_confidence(evidence)

            if confidence >= self.min_confidence:
                evidence_types = [k for k in evidence.keys() if k != "name"]
                explanation = self._generate_explanation(
                    candidate_id, evidence.get("name", f"Concept {candidate_id}"),
                    concept_id, evidence
                )

                strength = "strong" if confidence >= 0.8 else \
                          "moderate" if confidence >= 0.65 else "weak"

                candidates.append(PrerequisiteCandidate(
                    source_concept_id=candidate_id,
                    source_concept_name=evidence.get("name", f"Concept {candidate_id}"),
                    target_concept_id=concept_id,
                    target_concept_name=f"Concept {concept_id}",  # Would come from DB
                    confidence=round(confidence, 3),
                    evidence_types=evidence_types,
                    explanation=explanation,
                    strength=strength
                ))

        # Sort by confidence
        candidates.sort(key=lambda x: x.confidence, reverse=True)

        return candidates

    def _analyze_failure_patterns(
        self,
        concept_id: int,
        mastery_data: List[Dict[str, Any]]
    ) -> FailurePattern:
        """
        Analyze failure patterns for a concept.

        Finds concepts where low mastery correlates with struggling on target concept.
        """
        # Group by user
        user_masteries: Dict[int, Dict[int, float]] = defaultdict(dict)
        for record in mastery_data:
            user_masteries[record["user_id"]][record["concept_id"]] = record["mastery"]

        # Find users who struggled with target concept
        struggling_users = []
        successful_users = []
        for user_id, masteries in user_masteries.items():
            if concept_id in masteries:
                if masteries[concept_id] < self.failure_threshold:
                    struggling_users.append(user_id)
                else:
                    successful_users.append(user_id)

        if len(struggling_users) < self.min_users_for_analysis:
            return FailurePattern(
                concept_id=concept_id,
                concept_name=f"Concept {concept_id}",
                failure_rate=len(struggling_users) / (len(struggling_users) + len(successful_users)) if struggling_users or successful_users else 0,
                users_analyzed=len(struggling_users) + len(successful_users),
                common_weak_concepts=[]
            )

        # Find concepts where struggling users had low mastery
        concept_correlations: Dict[int, List[float]] = defaultdict(list)

        for user_id in struggling_users:
            for c_id, mastery in user_masteries[user_id].items():
                if c_id != concept_id:
                    concept_correlations[c_id].append(mastery)

        # Calculate average mastery for struggling users vs successful users
        weak_concepts = []
        for c_id, masteries in concept_correlations.items():
            if len(masteries) >= self.min_users_for_analysis // 2:
                struggling_avg = sum(masteries) / len(masteries)

                # Compare to successful users
                successful_masteries = [
                    user_masteries[u_id].get(c_id, 0.5)
                    for u_id in successful_users
                    if c_id in user_masteries[u_id]
                ]

                if successful_masteries:
                    successful_avg = sum(successful_masteries) / len(successful_masteries)
                    # Higher difference = stronger prerequisite signal
                    correlation = max(0, successful_avg - struggling_avg)

                    if correlation > 0.1:  # Minimum meaningful difference
                        weak_concepts.append((c_id, f"Concept {c_id}", correlation))

        # Sort by correlation
        weak_concepts.sort(key=lambda x: x[2], reverse=True)

        return FailurePattern(
            concept_id=concept_id,
            concept_name=f"Concept {concept_id}",
            failure_rate=len(struggling_users) / (len(struggling_users) + len(successful_users)),
            users_analyzed=len(struggling_users) + len(successful_users),
            common_weak_concepts=weak_concepts[:10]  # Top 10
        )

    def _analyze_content_references(
        self,
        concept_id: int,
        content_data: Dict[str, str]
    ) -> List[ConceptReference]:
        """
        Analyze content to find concept references.

        Looks for mentions of other concepts in the target concept's content.
        """
        references = []
        target_content = content_data.get(str(concept_id), "").lower()

        if not target_content:
            return references

        for other_id, other_content in content_data.items():
            if int(other_id) == concept_id:
                continue

            # Simple keyword matching (in production, use NER)
            # Extract concept name from content (first sentence or title)
            concept_name = other_content.split('.')[0].lower()[:50]

            # Count references
            count = target_content.count(concept_name)

            if count > 0:
                # Extract contexts (sentences containing reference)
                sentences = target_content.split('.')
                contexts = [s for s in sentences if concept_name in s][:3]

                references.append(ConceptReference(
                    source_concept_id=int(other_id),
                    target_concept_id=concept_id,
                    reference_count=count,
                    reference_contexts=contexts
                ))

        return references

    def _analyze_temporal_patterns(
        self,
        concept_id: int,
        mastery_data: List[Dict[str, Any]]
    ) -> List[Tuple[int, float]]:
        """
        Analyze temporal learning patterns.

        Finds concepts typically learned before the target concept.
        """
        # Group by user and sort by timestamp
        user_sequences: Dict[int, List[Tuple[int, datetime, float]]] = defaultdict(list)

        for record in mastery_data:
            timestamp = record.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            elif not isinstance(timestamp, datetime):
                continue

            user_sequences[record["user_id"]].append((
                record["concept_id"],
                timestamp,
                record["mastery"]
            ))

        # Sort each user's sequence
        for user_id in user_sequences:
            user_sequences[user_id].sort(key=lambda x: x[1])

        # Find concepts that typically come before target
        predecessor_counts: Dict[int, int] = defaultdict(int)
        total_sequences = 0

        for user_id, sequence in user_sequences.items():
            # Find position of target concept (when mastery reached threshold)
            target_position = None
            for i, (c_id, ts, mastery) in enumerate(sequence):
                if c_id == concept_id and mastery >= self.failure_threshold:
                    target_position = i
                    break

            if target_position is not None and target_position > 0:
                total_sequences += 1
                # Count concepts that came before
                for i in range(target_position):
                    pred_id = sequence[i][0]
                    if pred_id != concept_id:
                        predecessor_counts[pred_id] += 1

        # Calculate correlation scores
        temporal_patterns = []
        if total_sequences >= self.min_users_for_analysis:
            for pred_id, count in predecessor_counts.items():
                correlation = count / total_sequences
                if correlation > 0.3:  # At least 30% of users learned it first
                    temporal_patterns.append((pred_id, correlation))

        return sorted(temporal_patterns, key=lambda x: x[1], reverse=True)[:10]

    def _get_structural_hints(
        self,
        concept_id: int,
        existing_graph: Dict[int, List[int]]
    ) -> List[Tuple[int, float]]:
        """
        Get structural hints from existing prerequisite graph.

        Looks at:
        - Siblings (concepts with same prerequisites)
        - Transitive relationships
        """
        hints = []

        # Find concepts that have this concept as dependent
        direct_prereqs = []
        for prereq_id, dependents in existing_graph.items():
            if concept_id in dependents:
                direct_prereqs.append(prereq_id)

        # Find siblings (share prerequisites)
        for prereq_id in direct_prereqs:
            siblings = existing_graph.get(prereq_id, [])
            for sibling_id in siblings:
                if sibling_id != concept_id:
                    # Sibling's prerequisites might be relevant
                    for sib_prereq_id, sib_dependents in existing_graph.items():
                        if sibling_id in sib_dependents and sib_prereq_id != prereq_id:
                            hints.append((sib_prereq_id, 0.5))  # Moderate confidence

        # Find transitive prerequisites
        for prereq_id in direct_prereqs:
            for trans_prereq_id, trans_dependents in existing_graph.items():
                if prereq_id in trans_dependents:
                    hints.append((trans_prereq_id, 0.7))  # Higher confidence

        return list(set(hints))

    def _calculate_confidence(self, evidence: Dict[str, float]) -> float:
        """Calculate overall confidence from evidence"""
        total_weight = 0.0
        weighted_sum = 0.0

        for evidence_type, weight in self.evidence_weights.items():
            if evidence_type in evidence:
                score = evidence[evidence_type]
                weighted_sum += score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        # Bonus for multiple evidence types
        evidence_count = sum(1 for k in evidence.keys() if k in self.evidence_weights)
        multiplier = 1 + 0.1 * (evidence_count - 1)  # Up to 30% bonus for 4 types

        return min(1.0, (weighted_sum / total_weight) * multiplier)

    def _generate_explanation(
        self,
        prereq_id: int,
        prereq_name: str,
        target_id: int,
        evidence: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation for prerequisite suggestion"""
        parts = [f"'{prereq_name}' is likely a prerequisite because:"]

        if "failure_correlation" in evidence:
            score = evidence["failure_correlation"]
            parts.append(f"- Users who struggle with the target concept often have low mastery of this concept (correlation: {score:.0%})")

        if "content_reference" in evidence:
            score = evidence["content_reference"]
            parts.append(f"- The target concept's content references this concept (strength: {score:.0%})")

        if "temporal_pattern" in evidence:
            score = evidence["temporal_pattern"]
            parts.append(f"- Successful learners typically master this concept first ({score:.0%} of users)")

        if "structural" in evidence:
            parts.append("- Related concepts in the knowledge graph suggest this relationship")

        return " ".join(parts)

    async def batch_detect_prerequisites(
        self,
        concept_ids: List[int],
        mastery_data: List[Dict[str, Any]],
        content_data: Optional[Dict[str, str]] = None
    ) -> Dict[int, List[PrerequisiteCandidate]]:
        """
        Detect prerequisites for multiple concepts.

        Args:
            concept_ids: List of concept IDs
            mastery_data: User mastery data
            content_data: Optional concept content

        Returns:
            Dict mapping concept_id to list of prerequisites
        """
        results = {}

        for concept_id in concept_ids:
            candidates = await self.detect_prerequisites(
                concept_id, mastery_data, content_data
            )
            results[concept_id] = candidates

        return results

    def validate_prerequisites(
        self,
        prerequisites: List[PrerequisiteCandidate],
        mastery_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validate suggested prerequisites using learning outcome data.

        Returns validation metrics for each prerequisite.
        """
        validations = []

        for prereq in prerequisites:
            # Check if learning prereq first improves target outcomes
            # Group users by whether they learned prereq first
            prereq_first_users = []
            prereq_later_users = []

            user_masteries: Dict[int, Dict[int, Tuple[float, datetime]]] = defaultdict(dict)
            for record in mastery_data:
                timestamp = record.get("timestamp", datetime.now())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                user_masteries[record["user_id"]][record["concept_id"]] = (
                    record["mastery"], timestamp
                )

            for user_id, masteries in user_masteries.items():
                prereq_data = masteries.get(prereq.source_concept_id)
                target_data = masteries.get(prereq.target_concept_id)

                if prereq_data and target_data:
                    if prereq_data[1] < target_data[1]:  # Learned prereq first
                        prereq_first_users.append(target_data[0])
                    else:
                        prereq_later_users.append(target_data[0])

            # Calculate outcome differences
            if prereq_first_users and prereq_later_users:
                first_avg = sum(prereq_first_users) / len(prereq_first_users)
                later_avg = sum(prereq_later_users) / len(prereq_later_users)
                improvement = first_avg - later_avg

                validations.append({
                    "prereq_concept_id": prereq.source_concept_id,
                    "target_concept_id": prereq.target_concept_id,
                    "valid": improvement > 0.05,
                    "improvement": improvement,
                    "prereq_first_count": len(prereq_first_users),
                    "prereq_first_avg_mastery": first_avg,
                    "prereq_later_count": len(prereq_later_users),
                    "prereq_later_avg_mastery": later_avg,
                    "recommendation": "confirm" if improvement > 0.1 else \
                                     "keep" if improvement > 0 else "review"
                })

        return validations


# Singleton instance
prerequisite_detector = PrerequisiteDetector()
