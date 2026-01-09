"""
Zone of Proximal Development (ZPD) Regulator
Recommends optimal content difficulty based on current mastery

ZPD Theory:
- Too easy → Boredom, no learning
- Too hard → Frustration, anxiety
- Just right (ZPD) → Optimal learning

Implementation:
- Track mastery across concepts
- Recommend content in the "sweet spot" (slightly challenging)
- Adjust difficulty dynamically based on performance

Enhanced with response time analysis:
- Fast correct → Automaticity, may need harder content
- Slow correct → Optimal challenge
- Fast wrong → Possible misconception
- Slow wrong → Content too hard
"""
from typing import List, Dict, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass
import statistics

if TYPE_CHECKING:
    from .response_time_analyzer import ResponseTimeAnalyzer


@dataclass
class ConceptMastery:
    """Concept mastery data"""
    concept_id: int
    concept_name: str
    mastery_level: float  # 0-1
    stability: float  # FSRS stability
    is_prerequisite: bool = False
    difficulty: float = 5.0  # Content difficulty (1-10)


@dataclass
class ContentRecommendation:
    """Content recommendation with ZPD score"""
    module_id: int
    concept_ids: List[int]
    zpd_score: float  # How well it fits ZPD (0-1)
    difficulty: float
    rationale: str
    estimated_success_rate: float


class ZPDRegulator:
    """
    Regulates content difficulty to keep learner in optimal zone
    """

    def __init__(
        self,
        zpd_width: float = 0.3,  # Width of optimal zone
        optimal_mastery: float = 0.6,  # Center of ZPD
        frustration_threshold: float = 0.9,  # Above this = too hard
        boredom_threshold: float = 0.3,  # Below this = too easy
    ):
        """
        Initialize ZPD regulator

        Args:
            zpd_width: Width of the optimal zone (default 0.3 = 30%)
            optimal_mastery: Target mastery level (default 0.6 = 60%)
            frustration_threshold: Difficulty threshold for frustration
            boredom_threshold: Mastery threshold for boredom
        """
        self.zpd_width = zpd_width
        self.optimal_mastery = optimal_mastery
        self.frustration_threshold = frustration_threshold
        self.boredom_threshold = boredom_threshold

    def calculate_zpd_score(
        self,
        user_mastery: float,
        content_difficulty: float,
        prerequisites_met: bool = True,
    ) -> Tuple[float, str]:
        """
        Calculate how well content fits user's ZPD

        Args:
            user_mastery: User's current mastery level (0-1)
            content_difficulty: Content difficulty (0-1 scale)
            prerequisites_met: Whether prerequisites are satisfied

        Returns:
            (ZPD score 0-1, zone description)
        """
        if not prerequisites_met:
            return 0.0, "Prerequisites not met"

        # Calculate distance from optimal challenge point
        # Optimal: content slightly harder than current mastery
        optimal_point = user_mastery + (self.zpd_width / 2)
        distance = abs(content_difficulty - optimal_point)

        # Convert distance to score (0 = perfect fit, max distance = 0)
        # Use Gaussian-like function
        zpd_score = max(0, 1 - (distance / self.zpd_width) ** 2)

        # Determine zone
        if content_difficulty < user_mastery - self.boredom_threshold:
            zone = "Too Easy (Boredom Zone)"
        elif content_difficulty > user_mastery + self.frustration_threshold:
            zone = "Too Hard (Frustration Zone)"
        elif zpd_score > 0.7:
            zone = "Optimal (ZPD)"
        elif zpd_score > 0.4:
            zone = "Acceptable (Near ZPD)"
        else:
            zone = "Suboptimal"

        return zpd_score, zone

    def calculate_prerequisite_readiness(
        self, concept_masteries: List[ConceptMastery], threshold: float = 0.7
    ) -> Tuple[bool, float]:
        """
        Check if prerequisites are met for learning new concept

        Args:
            concept_masteries: List of prerequisite concept masteries
            threshold: Minimum mastery threshold

        Returns:
            (Are prerequisites met?, Average prerequisite mastery)
        """
        if not concept_masteries:
            return True, 1.0  # No prerequisites

        prerequisite_levels = [c.mastery_level for c in concept_masteries]
        avg_mastery = statistics.mean(prerequisite_levels)
        all_met = all(m >= threshold for m in prerequisite_levels)

        return all_met, avg_mastery

    def recommend_content(
        self,
        user_concept_masteries: Dict[int, float],
        available_modules: List[Dict],
        concept_prerequisites: Dict[int, List[int]],
        top_n: int = 5,
    ) -> List[ContentRecommendation]:
        """
        Recommend content based on ZPD principles

        Args:
            user_concept_masteries: Dict[concept_id -> mastery_level]
            available_modules: List of available modules with metadata
            concept_prerequisites: Dict[concept_id -> List[prerequisite_ids]]
            top_n: Number of recommendations to return

        Returns:
            List of content recommendations sorted by ZPD score
        """
        recommendations = []

        for module in available_modules:
            module_id = module["id"]
            module_concepts = module.get("concepts", [])
            module_difficulty = module.get("difficulty", 5.0) / 10.0  # Normalize to 0-1

            if not module_concepts:
                continue

            # Calculate average user mastery for module concepts
            concept_mastery_levels = []
            all_prerequisites_met = True

            for concept_id in module_concepts:
                # Get user's mastery for this concept
                user_mastery = user_concept_masteries.get(concept_id, 0.0)
                concept_mastery_levels.append(user_mastery)

                # Check prerequisites
                prereqs = concept_prerequisites.get(concept_id, [])
                if prereqs:
                    prereq_masteries = [
                        ConceptMastery(
                            concept_id=p,
                            concept_name="",
                            mastery_level=user_concept_masteries.get(p, 0.0),
                            stability=0.0,
                        )
                        for p in prereqs
                    ]
                    prereqs_met, _ = self.calculate_prerequisite_readiness(prereq_masteries)
                    if not prereqs_met:
                        all_prerequisites_met = False

            if not concept_mastery_levels:
                continue

            avg_mastery = statistics.mean(concept_mastery_levels)

            # Calculate ZPD score
            zpd_score, zone = self.calculate_zpd_score(
                avg_mastery, module_difficulty, all_prerequisites_met
            )

            # Estimate success rate
            # Based on mastery vs difficulty gap
            mastery_gap = module_difficulty - avg_mastery
            if mastery_gap < 0:
                # Content easier than mastery
                estimated_success = 0.9 + (mastery_gap * 0.1)
            else:
                # Content harder than mastery
                estimated_success = max(0.1, 0.8 - (mastery_gap * 1.5))

            estimated_success = max(0.0, min(1.0, estimated_success))

            recommendation = ContentRecommendation(
                module_id=module_id,
                concept_ids=module_concepts,
                zpd_score=zpd_score,
                difficulty=module_difficulty,
                rationale=f"{zone} | Success rate: {estimated_success:.0%}",
                estimated_success_rate=estimated_success,
            )

            recommendations.append(recommendation)

        # Sort by ZPD score (highest first)
        recommendations.sort(key=lambda x: x.zpd_score, reverse=True)

        return recommendations[:top_n]

    def adjust_difficulty(
        self, current_difficulty: float, performance: float, adaptation_rate: float = 0.1
    ) -> float:
        """
        Dynamically adjust difficulty based on performance

        Args:
            current_difficulty: Current difficulty level (0-1)
            performance: Recent performance score (0-1)
            adaptation_rate: How quickly to adapt (0-1)

        Returns:
            New difficulty level
        """
        # If performing well, increase difficulty
        # If struggling, decrease difficulty
        target_performance = 0.75  # Aim for 75% success rate

        error = performance - target_performance
        adjustment = error * adaptation_rate

        new_difficulty = current_difficulty + adjustment

        # Keep in bounds
        return max(0.1, min(0.9, new_difficulty))

    def get_learning_velocity(
        self, mastery_history: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate learning velocity (rate of mastery gain)

        Args:
            mastery_history: List of (timestamp, mastery) tuples

        Returns:
            Learning velocity (mastery units per time unit)
        """
        if len(mastery_history) < 2:
            return 0.0

        # Calculate slope of mastery over time
        times = [t for t, _ in mastery_history]
        masteries = [m for _, m in mastery_history]

        # Simple linear regression
        n = len(times)
        mean_time = statistics.mean(times)
        mean_mastery = statistics.mean(masteries)

        numerator = sum((times[i] - mean_time) * (masteries[i] - mean_mastery) for i in range(n))
        denominator = sum((times[i] - mean_time) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        velocity = numerator / denominator

        return velocity

    def should_review_concept(
        self,
        mastery: float,
        stability: float,
        days_since_review: int,
        mastery_threshold: float = 0.7,
        retrievability_threshold: float = 0.8,
    ) -> Tuple[bool, str]:
        """
        Determine if a concept should be reviewed

        Args:
            mastery: Current mastery level
            stability: FSRS stability
            days_since_review: Days since last review
            mastery_threshold: Mastery threshold for review
            retrievability_threshold: Retrievability threshold

        Returns:
            (Should review?, Reason)
        """
        # Already mastered - no review needed
        if mastery >= mastery_threshold and stability > 30:
            return False, "Concept mastered"

        # Calculate retrievability (simplified FSRS formula)
        retrievability = pow(1 + days_since_review / (9 * stability), -1) if stability > 0 else 0

        # Should review if retrievability dropping
        if retrievability < retrievability_threshold:
            return True, f"Retrievability low: {retrievability:.1%}"

        # Should review if mastery low
        if mastery < mastery_threshold:
            return True, f"Mastery low: {mastery:.1%}"

        return False, "Review not needed"

    def calculate_enhanced_zpd_score(
        self,
        user_mastery: float,
        content_difficulty: float,
        prerequisites_met: bool,
        time_analyzer: "ResponseTimeAnalyzer",
        user_id: int,
        concept_id: int,
    ) -> Dict[str, Any]:
        """
        Calculate ZPD score enhanced with response time analysis

        Combines traditional ZPD calculation with response time patterns
        to provide more accurate difficulty recommendations.

        Args:
            user_mastery: User's current mastery level (0-1)
            content_difficulty: Content difficulty (0-1 scale)
            prerequisites_met: Whether prerequisites are satisfied
            time_analyzer: ResponseTimeAnalyzer instance
            user_id: User ID for time analysis
            concept_id: Concept ID for time analysis

        Returns:
            Enhanced ZPD analysis with time-based insights
        """
        # Calculate base ZPD score
        base_score, zone = self.calculate_zpd_score(
            user_mastery, content_difficulty, prerequisites_met
        )

        # Get time-based enhancement
        time_enhancement = time_analyzer.get_zpd_enhancement(
            user_id, concept_id, base_score
        )

        # Get cognitive load estimate
        cognitive_load = time_analyzer.estimate_cognitive_load(user_id, concept_id)

        # Adjust recommendation based on cognitive state
        final_recommendation = self._generate_enhanced_recommendation(
            zone,
            time_enhancement,
            cognitive_load,
        )

        return {
            "base_zpd_score": round(base_score, 4),
            "enhanced_zpd_score": time_enhancement["enhanced_zpd_score"],
            "zone": zone,
            "time_adjustment": time_enhancement["adjustment"],
            "time_patterns": time_enhancement.get("time_patterns", {}),
            "cognitive_load": cognitive_load["cognitive_load"],
            "cognitive_load_score": cognitive_load["load_score"],
            "recommendation": final_recommendation,
            "confidence": time_enhancement["confidence"],
        }

    def _generate_enhanced_recommendation(
        self,
        zone: str,
        time_enhancement: Dict[str, Any],
        cognitive_load: Dict[str, Any],
    ) -> str:
        """Generate comprehensive recommendation based on all factors"""
        recommendations = []

        # Zone-based recommendation
        if "Boredom" in zone:
            recommendations.append("Increase difficulty")
        elif "Frustration" in zone:
            recommendations.append("Decrease difficulty or provide scaffolding")
        elif "Optimal" in zone:
            recommendations.append("Current difficulty is appropriate")

        # Time pattern-based
        primary_pattern = time_enhancement.get("primary_pattern", "")
        if primary_pattern == "fast_correct":
            recommendations.append("Consider advancing to more challenging content")
        elif primary_pattern == "slow_wrong":
            recommendations.append("Review prerequisites and provide additional support")
        elif primary_pattern == "fast_wrong":
            recommendations.append("Check for misconceptions before proceeding")

        # Cognitive load-based
        load_level = cognitive_load.get("cognitive_load", "unknown")
        if load_level == "overload":
            recommendations.append("Reduce cognitive load - break content into smaller pieces")
        elif load_level == "high":
            recommendations.append("Consider providing more scaffolding")
        elif load_level == "low":
            recommendations.append("Can handle more complex content")

        return " | ".join(recommendations) if recommendations else "Continue with current approach"

    def recommend_with_timing(
        self,
        user_id: int,
        user_concept_masteries: Dict[int, float],
        available_modules: List[Dict],
        concept_prerequisites: Dict[int, List[int]],
        time_analyzer: "ResponseTimeAnalyzer",
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Recommend content with response time analysis integration

        Args:
            user_id: User ID
            user_concept_masteries: Dict[concept_id -> mastery_level]
            available_modules: List of available modules
            concept_prerequisites: Dict[concept_id -> List[prerequisite_ids]]
            time_analyzer: ResponseTimeAnalyzer instance
            top_n: Number of recommendations

        Returns:
            List of enhanced content recommendations
        """
        # Get base recommendations
        base_recommendations = self.recommend_content(
            user_concept_masteries,
            available_modules,
            concept_prerequisites,
            top_n * 2,  # Get more to re-rank
        )

        enhanced = []
        for rec in base_recommendations:
            # Get time-based enhancement for primary concept
            if rec.concept_ids:
                primary_concept = rec.concept_ids[0]
                enhancement = time_analyzer.get_zpd_enhancement(
                    user_id, primary_concept, rec.zpd_score
                )
                cognitive_load = time_analyzer.estimate_cognitive_load(
                    user_id, primary_concept
                )

                enhanced.append({
                    "module_id": rec.module_id,
                    "concept_ids": rec.concept_ids,
                    "base_zpd_score": rec.zpd_score,
                    "enhanced_zpd_score": enhancement["enhanced_zpd_score"],
                    "difficulty": rec.difficulty,
                    "estimated_success_rate": rec.estimated_success_rate,
                    "time_patterns": enhancement.get("time_patterns", {}),
                    "cognitive_load": cognitive_load["cognitive_load"],
                    "rationale": rec.rationale,
                    "time_recommendation": enhancement["recommendation"],
                })

        # Re-rank by enhanced ZPD score
        enhanced.sort(key=lambda x: x["enhanced_zpd_score"], reverse=True)

        return enhanced[:top_n]
