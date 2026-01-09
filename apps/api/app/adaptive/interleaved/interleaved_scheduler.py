"""
Interleaved Practice Scheduler
Implements research-backed interleaved/hybrid practice scheduling

Research basis:
- Interleaved Practice: g=0.42 meta-analytic effect size
- Hybrid scheduling: block practice until 75-80% proficiency, then interleave
- Better for transfer and long-term retention vs massed practice
- Combines with spaced repetition for optimal learning

Key findings:
- Blocking helps initial learning (acquisition)
- Interleaving helps discrimination and long-term retention
- Transition point: 75-80% proficiency on individual skills
- Contextual interference theory: difficulty during practice improves retention
"""
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import random
import math


class PracticeMode(str, Enum):
    """Practice scheduling modes"""
    BLOCKED = "blocked"      # Focus on one concept at a time
    INTERLEAVED = "interleaved"  # Mix concepts within session
    HYBRID = "hybrid"        # Adaptive switching based on proficiency


@dataclass
class PracticeItem:
    """Single practice item/problem"""
    item_id: str
    concept_id: int
    concept_name: str
    difficulty: float  # 1-10
    item_type: str  # "problem", "recall", "application"
    content: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class PracticeSequence:
    """Generated practice sequence"""
    items: List[PracticeItem]
    mode: PracticeMode
    concepts_included: List[int]
    estimated_duration_minutes: int
    rationale: str
    interleaving_ratio: float  # 0=fully blocked, 1=fully interleaved


@dataclass
class ConceptProficiency:
    """User's proficiency on a concept"""
    concept_id: int
    concept_name: str
    mastery_level: float  # 0-1 (from BKT/FSRS)
    practice_count: int
    recent_accuracy: float  # Last N attempts
    last_practiced: Optional[datetime] = None
    stability: float = 0.0  # FSRS stability


class InterleavedScheduler:
    """
    Generates interleaved practice sequences based on research

    Implements:
    1. Hybrid scheduling - blocked then interleaved
    2. Optimal interleaving ratio based on proficiency
    3. Spacing integration with FSRS
    4. Contextual variety for discrimination learning
    """

    # Proficiency threshold for transitioning to interleaved (75-80%)
    INTERLEAVE_THRESHOLD = 0.75

    # Minimum proficiency before including in interleaved practice
    MIN_PROFICIENCY_FOR_INTERLEAVE = 0.4

    # Maximum concepts to interleave in single session
    MAX_INTERLEAVED_CONCEPTS = 5

    # Optimal spacing between same-concept items in interleaved sequence
    MIN_SPACING_ITEMS = 2  # At least 2 items between same concept

    def __init__(
        self,
        interleave_threshold: float = 0.75,
        min_proficiency: float = 0.4,
        max_concepts: int = 5,
    ):
        """
        Initialize interleaved scheduler

        Args:
            interleave_threshold: Proficiency level to start interleaving
            min_proficiency: Minimum proficiency to include in interleaved
            max_concepts: Maximum concepts to interleave
        """
        self.interleave_threshold = interleave_threshold
        self.min_proficiency = min_proficiency
        self.max_concepts = max_concepts

    def determine_practice_mode(
        self,
        concept_proficiencies: List[ConceptProficiency],
        target_concept_id: Optional[int] = None,
    ) -> Tuple[PracticeMode, str]:
        """
        Determine optimal practice mode based on proficiencies

        Research basis: Hybrid scheduling
        - Use blocked practice for concepts below threshold
        - Use interleaved for concepts at/above threshold
        - Switch when 75-80% proficiency reached

        Args:
            concept_proficiencies: List of concept proficiencies
            target_concept_id: Optional specific concept focus

        Returns:
            (Practice mode, Rationale)
        """
        if not concept_proficiencies:
            return PracticeMode.BLOCKED, "No proficiency data - starting with blocked practice"

        # If targeting specific concept
        if target_concept_id:
            target = next(
                (c for c in concept_proficiencies if c.concept_id == target_concept_id),
                None
            )
            if target:
                if target.mastery_level < self.interleave_threshold:
                    return (
                        PracticeMode.BLOCKED,
                        f"Concept '{target.concept_name}' at {target.mastery_level:.0%} - "
                        f"blocked practice until {self.interleave_threshold:.0%}"
                    )

        # Count concepts ready for interleaving
        ready_for_interleave = [
            c for c in concept_proficiencies
            if c.mastery_level >= self.interleave_threshold
        ]

        learning_concepts = [
            c for c in concept_proficiencies
            if c.mastery_level < self.interleave_threshold
        ]

        # Need at least 2 concepts for meaningful interleaving
        if len(ready_for_interleave) >= 2:
            if learning_concepts:
                return (
                    PracticeMode.HYBRID,
                    f"{len(ready_for_interleave)} concepts ready for interleaving, "
                    f"{len(learning_concepts)} still in blocked phase"
                )
            return (
                PracticeMode.INTERLEAVED,
                f"All {len(ready_for_interleave)} concepts at interleaving threshold"
            )

        return (
            PracticeMode.BLOCKED,
            f"Only {len(ready_for_interleave)} concept(s) ready for interleaving - "
            "need at least 2"
        )

    def generate_practice_sequence(
        self,
        concept_proficiencies: List[ConceptProficiency],
        available_items: Dict[int, List[PracticeItem]],  # concept_id -> items
        target_items: int = 10,
        target_duration_minutes: int = 15,
        target_concept_id: Optional[int] = None,
    ) -> PracticeSequence:
        """
        Generate optimal practice sequence

        Implements:
        - Hybrid scheduling with automatic mode detection
        - Optimal spacing between same-concept items
        - Difficulty progression within concepts
        - Due item prioritization (FSRS integration)

        Args:
            concept_proficiencies: User's concept proficiencies
            available_items: Available practice items per concept
            target_items: Target number of items
            target_duration_minutes: Target session duration
            target_concept_id: Optional focus concept

        Returns:
            Optimized practice sequence
        """
        # Determine practice mode
        mode, rationale = self.determine_practice_mode(
            concept_proficiencies,
            target_concept_id
        )

        if mode == PracticeMode.BLOCKED:
            return self._generate_blocked_sequence(
                concept_proficiencies,
                available_items,
                target_items,
                target_concept_id,
                rationale,
            )
        elif mode == PracticeMode.INTERLEAVED:
            return self._generate_interleaved_sequence(
                concept_proficiencies,
                available_items,
                target_items,
                rationale,
            )
        else:  # HYBRID
            return self._generate_hybrid_sequence(
                concept_proficiencies,
                available_items,
                target_items,
                rationale,
            )

    def _generate_blocked_sequence(
        self,
        proficiencies: List[ConceptProficiency],
        available_items: Dict[int, List[PracticeItem]],
        target_items: int,
        target_concept_id: Optional[int],
        rationale: str,
    ) -> PracticeSequence:
        """Generate blocked practice sequence - focus on one concept"""

        # Select target concept
        if target_concept_id:
            target = next(
                (p for p in proficiencies if p.concept_id == target_concept_id),
                None
            )
        else:
            # Select concept with lowest proficiency that needs practice
            sorted_profs = sorted(proficiencies, key=lambda p: p.mastery_level)
            target = sorted_profs[0] if sorted_profs else None

        if not target or target.concept_id not in available_items:
            return PracticeSequence(
                items=[],
                mode=PracticeMode.BLOCKED,
                concepts_included=[],
                estimated_duration_minutes=0,
                rationale="No items available for practice",
                interleaving_ratio=0.0,
            )

        items = available_items[target.concept_id]

        # Sort by difficulty (progressive difficulty within block)
        sorted_items = sorted(items, key=lambda x: x.difficulty)

        # Take target number of items
        selected = sorted_items[:target_items]

        return PracticeSequence(
            items=selected,
            mode=PracticeMode.BLOCKED,
            concepts_included=[target.concept_id],
            estimated_duration_minutes=len(selected) * 2,  # ~2 min per item
            rationale=rationale,
            interleaving_ratio=0.0,
        )

    def _generate_interleaved_sequence(
        self,
        proficiencies: List[ConceptProficiency],
        available_items: Dict[int, List[PracticeItem]],
        target_items: int,
        rationale: str,
    ) -> PracticeSequence:
        """
        Generate fully interleaved sequence

        Implements contextual interference theory:
        - Mix concepts to improve discrimination
        - Maintain minimum spacing between same-concept items
        - Balance representation across concepts
        """

        # Select concepts ready for interleaving
        ready_concepts = [
            p for p in proficiencies
            if p.mastery_level >= self.min_proficiency
        ]

        # Limit to max concepts
        if len(ready_concepts) > self.max_concepts:
            # Prioritize by time since last practice
            ready_concepts = sorted(
                ready_concepts,
                key=lambda p: p.last_practiced or datetime.min
            )[:self.max_concepts]

        if len(ready_concepts) < 2:
            # Fallback to blocked
            return self._generate_blocked_sequence(
                proficiencies, available_items, target_items, None,
                "Insufficient concepts for interleaving - falling back to blocked"
            )

        # Collect items from each concept
        concept_items: Dict[int, List[PracticeItem]] = {}
        for prof in ready_concepts:
            if prof.concept_id in available_items:
                items = available_items[prof.concept_id]
                # Shuffle items within concept
                concept_items[prof.concept_id] = random.sample(
                    items, min(len(items), target_items // len(ready_concepts) + 2)
                )

        # Generate interleaved sequence with spacing constraints
        sequence = self._interleave_with_spacing(
            concept_items,
            target_items,
            self.MIN_SPACING_ITEMS
        )

        concepts_included = list(concept_items.keys())

        return PracticeSequence(
            items=sequence,
            mode=PracticeMode.INTERLEAVED,
            concepts_included=concepts_included,
            estimated_duration_minutes=len(sequence) * 2,
            rationale=rationale,
            interleaving_ratio=self._calculate_interleaving_ratio(sequence),
        )

    def _generate_hybrid_sequence(
        self,
        proficiencies: List[ConceptProficiency],
        available_items: Dict[int, List[PracticeItem]],
        target_items: int,
        rationale: str,
    ) -> PracticeSequence:
        """
        Generate hybrid sequence - blocked for new concepts, interleaved for ready ones

        Research basis: Optimal learning combines:
        - Blocked practice for acquisition
        - Interleaved practice for retention/transfer
        """

        ready_concepts = [
            p for p in proficiencies
            if p.mastery_level >= self.interleave_threshold
        ]

        learning_concepts = [
            p for p in proficiencies
            if p.mastery_level < self.interleave_threshold
        ]

        sequence = []

        # Part 1: Blocked practice for learning concepts (40% of items)
        blocked_items = int(target_items * 0.4)
        if learning_concepts and blocked_items > 0:
            # Focus on lowest proficiency concept
            focus = min(learning_concepts, key=lambda p: p.mastery_level)
            if focus.concept_id in available_items:
                items = available_items[focus.concept_id]
                sorted_items = sorted(items, key=lambda x: x.difficulty)
                sequence.extend(sorted_items[:blocked_items])

        # Part 2: Interleaved practice for ready concepts (60% of items)
        interleaved_items = target_items - len(sequence)
        if len(ready_concepts) >= 2 and interleaved_items > 0:
            # Limit concepts
            selected_concepts = ready_concepts[:self.max_concepts]

            concept_items: Dict[int, List[PracticeItem]] = {}
            for prof in selected_concepts:
                if prof.concept_id in available_items:
                    items = available_items[prof.concept_id]
                    concept_items[prof.concept_id] = random.sample(
                        items, min(len(items), interleaved_items // len(selected_concepts) + 1)
                    )

            interleaved = self._interleave_with_spacing(
                concept_items,
                interleaved_items,
                self.MIN_SPACING_ITEMS
            )
            sequence.extend(interleaved)

        # Calculate concepts included
        concepts_included = list(set(
            item.concept_id for item in sequence
        ))

        return PracticeSequence(
            items=sequence,
            mode=PracticeMode.HYBRID,
            concepts_included=concepts_included,
            estimated_duration_minutes=len(sequence) * 2,
            rationale=rationale,
            interleaving_ratio=self._calculate_interleaving_ratio(sequence),
        )

    def _interleave_with_spacing(
        self,
        concept_items: Dict[int, List[PracticeItem]],
        target_count: int,
        min_spacing: int,
    ) -> List[PracticeItem]:
        """
        Create interleaved sequence with minimum spacing between same-concept items

        Uses round-robin with spacing constraint to ensure proper interleaving
        """
        if not concept_items:
            return []

        result = []
        # Track items remaining for each concept
        remaining = {cid: list(items) for cid, items in concept_items.items()}
        # Track last position of each concept
        last_position: Dict[int, int] = {}

        position = 0
        attempts = 0
        max_attempts = target_count * 3  # Prevent infinite loop

        while len(result) < target_count and attempts < max_attempts:
            attempts += 1

            # Find concepts that can be added (spacing constraint met)
            available_concepts = [
                cid for cid, items in remaining.items()
                if items and (
                    cid not in last_position or
                    position - last_position[cid] >= min_spacing
                )
            ]

            if not available_concepts:
                # Relax constraint if stuck
                available_concepts = [
                    cid for cid, items in remaining.items()
                    if items
                ]

            if not available_concepts:
                break  # No more items

            # Select concept with most remaining items (balancing)
            concept_id = max(available_concepts, key=lambda c: len(remaining[c]))

            # Add item
            item = remaining[concept_id].pop(0)
            result.append(item)
            last_position[concept_id] = position
            position += 1

        return result

    def _calculate_interleaving_ratio(self, sequence: List[PracticeItem]) -> float:
        """
        Calculate interleaving ratio of sequence

        Returns 0 for fully blocked, 1 for maximally interleaved
        """
        if len(sequence) < 2:
            return 0.0

        # Count transitions between different concepts
        transitions = sum(
            1 for i in range(1, len(sequence))
            if sequence[i].concept_id != sequence[i-1].concept_id
        )

        # Maximum possible transitions = len - 1
        max_transitions = len(sequence) - 1

        return transitions / max_transitions if max_transitions > 0 else 0.0

    def calculate_spacing_benefit(
        self,
        last_practiced: datetime,
        stability: float,
        current_time: Optional[datetime] = None,
    ) -> float:
        """
        Calculate benefit of practicing now based on spacing

        Uses FSRS-style spacing to integrate with spaced repetition

        Args:
            last_practiced: When concept was last practiced
            stability: FSRS stability value
            current_time: Current time (default: now)

        Returns:
            Spacing benefit score (0-1, higher = more benefit)
        """
        current_time = current_time or datetime.now()

        if not last_practiced:
            return 1.0  # Never practiced - maximum benefit

        days_since = (current_time - last_practiced).days

        if stability == 0:
            return 1.0 if days_since > 0 else 0.5

        # Retrievability formula from FSRS
        retrievability = pow(1 + days_since / (9 * stability), -1)

        # Optimal practice point is when retrievability drops to ~0.9
        # Lower retrievability = higher benefit from practice
        benefit = 1 - retrievability

        return max(0.0, min(1.0, benefit))

    def prioritize_for_interleaving(
        self,
        concept_proficiencies: List[ConceptProficiency],
        current_time: Optional[datetime] = None,
    ) -> List[Tuple[ConceptProficiency, float]]:
        """
        Prioritize concepts for interleaved practice

        Considers:
        - Spacing benefit (FSRS integration)
        - Proficiency level (focus on those ready for interleaving)
        - Recency (avoid over-practicing recent concepts)

        Returns:
            List of (concept, priority_score) sorted by priority
        """
        current_time = current_time or datetime.now()

        scored = []
        for prof in concept_proficiencies:
            # Base score from spacing
            spacing_score = self.calculate_spacing_benefit(
                prof.last_practiced,
                prof.stability,
                current_time
            )

            # Bonus for concepts in interleaving range
            if self.min_proficiency <= prof.mastery_level <= 0.95:
                proficiency_bonus = 0.2
            else:
                proficiency_bonus = 0.0

            # Penalty for very recent practice
            if prof.last_practiced:
                hours_since = (current_time - prof.last_practiced).total_seconds() / 3600
                if hours_since < 1:
                    recency_penalty = 0.3
                elif hours_since < 4:
                    recency_penalty = 0.1
                else:
                    recency_penalty = 0.0
            else:
                recency_penalty = 0.0

            priority = spacing_score + proficiency_bonus - recency_penalty
            priority = max(0.0, min(1.0, priority))

            scored.append((prof, priority))

        # Sort by priority (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def get_interleaving_statistics(
        self,
        session_history: List[Dict],
    ) -> Dict:
        """
        Calculate interleaving statistics from practice history

        Useful for analyzing effectiveness and tuning parameters

        Args:
            session_history: List of practice session data

        Returns:
            Statistics about interleaving patterns
        """
        if not session_history:
            return {
                "total_sessions": 0,
                "avg_interleaving_ratio": 0,
                "mode_distribution": {},
                "avg_concepts_per_session": 0,
            }

        total_sessions = len(session_history)

        # Calculate averages
        interleaving_ratios = [
            s.get("interleaving_ratio", 0)
            for s in session_history
        ]
        avg_ratio = sum(interleaving_ratios) / total_sessions

        # Mode distribution
        modes = [s.get("mode", "unknown") for s in session_history]
        mode_counts = {}
        for mode in modes:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        mode_distribution = {
            k: v / total_sessions
            for k, v in mode_counts.items()
        }

        # Concepts per session
        concepts_per_session = [
            len(s.get("concepts_included", []))
            for s in session_history
        ]
        avg_concepts = sum(concepts_per_session) / total_sessions if concepts_per_session else 0

        return {
            "total_sessions": total_sessions,
            "avg_interleaving_ratio": round(avg_ratio, 3),
            "mode_distribution": mode_distribution,
            "avg_concepts_per_session": round(avg_concepts, 1),
        }
