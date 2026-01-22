"""
Metacognition Module - Thinking About Learning

Research alignment:
- Metacognition: Knowledge and regulation of one's own cognitive processes
- Judgment of Learning (JOL): Learner's predictions of future performance
- Self-Explanation Effect: Explaining material to oneself improves learning
- Illusion of Competence: Overconfidence in learning without proper testing
- Calibration: Alignment between confidence and actual performance

Key Components:
1. Confidence Ratings (JOL): "How confident are you that you understand this?"
2. Self-Explanation Prompts: "Explain this concept in your own words"
3. Knowledge Monitoring: Track actual vs perceived understanding
4. Reflection Prompts: "What strategy worked best for you?"
5. Calibration Feedback: Show gap between confidence and performance

Goal: Improve metacognitive accuracy and self-regulated learning.
"""
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import random
import logging

logger = logging.getLogger(__name__)


class MetacognitionPromptType(str, Enum):
    """Types of metacognitive prompts"""
    CONFIDENCE_RATING = "confidence_rating"     # JOL: Rate your understanding
    SELF_EXPLANATION = "self_explanation"       # Explain in own words
    PREDICTION = "prediction"                   # Predict performance on test
    REFLECTION = "reflection"                   # Reflect on learning process
    STRATEGY_SELECTION = "strategy_selection"   # Choose learning strategy
    ERROR_ANALYSIS = "error_analysis"           # Analyze why answer was wrong
    KNOWLEDGE_CHECK = "knowledge_check"         # Quick self-assessment
    ELABORATION = "elaboration"                 # Connect to prior knowledge


class CalibrationLevel(str, Enum):
    """Learner calibration quality"""
    WELL_CALIBRATED = "well_calibrated"    # Confidence matches performance
    OVERCONFIDENT = "overconfident"         # Confidence > performance
    UNDERCONFIDENT = "underconfident"       # Confidence < performance
    UNKNOWN = "unknown"                     # Insufficient data


@dataclass
class ConfidenceRating:
    """A single confidence rating from the learner"""
    user_id: str
    concept_id: str
    content_id: str
    confidence: float  # 0-1 scale
    timestamp: datetime
    context: str  # "pre_test", "post_study", "during_practice"
    actual_performance: Optional[float] = None  # Fill in after assessment


@dataclass
class SelfExplanation:
    """Learner's self-explanation"""
    user_id: str
    concept_id: str
    explanation_text: str
    timestamp: datetime
    quality_score: Optional[float] = None  # 0-1, assessed by LLM
    key_concepts_mentioned: List[str] = field(default_factory=list)
    misconceptions_detected: List[str] = field(default_factory=list)


@dataclass
class MetacognitionPrompt:
    """A prompt to trigger metacognitive reflection"""
    prompt_id: str
    prompt_type: MetacognitionPromptType
    prompt_text: str
    context: Dict[str, Any]
    timing: str  # "before", "during", "after"
    required: bool = False
    follow_up_prompts: List[str] = field(default_factory=list)


@dataclass
class CalibrationData:
    """Data for assessing learner calibration"""
    confidence_ratings: List[Tuple[float, float]]  # (confidence, actual_performance)
    mean_confidence: float
    mean_performance: float
    calibration_error: float  # |confidence - performance|
    calibration_level: CalibrationLevel
    overconfidence_rate: float  # % of items where confidence > performance
    underconfidence_rate: float


@dataclass
class MetacognitionProfile:
    """Learner's metacognition profile"""
    user_id: str
    calibration: CalibrationData
    self_explanation_quality: float  # Average quality of explanations
    reflection_engagement: float  # % of prompts engaged with
    strategy_use: Dict[str, int]  # Strategies used and frequency
    improvement_trend: float  # Change in calibration over time
    recommendations: List[str]


class MetacognitionPrompter:
    """
    Generates and manages metacognitive prompts

    Research basis:
    - JOL improves learning when done at delay (not immediately after study)
    - Self-explanation prompts improve learning by 0.5-1.0 effect size
    - Metacognitive prompts are most effective at key transition points
    """

    # Prompt templates by type
    PROMPT_TEMPLATES = {
        MetacognitionPromptType.CONFIDENCE_RATING: [
            "On a scale of 1-5, how confident are you that you understand {concept}?",
            "How well do you think you could explain {concept} to someone else?",
            "If tested on {concept} right now, how well would you perform?",
            "Rate your understanding of {concept}: Not at all (1) to Completely (5)",
        ],
        MetacognitionPromptType.SELF_EXPLANATION: [
            "In your own words, explain what {concept} means.",
            "How does {concept} relate to what you already know?",
            "Can you give an example of {concept} from your own experience?",
            "Explain {concept} as if teaching it to a classmate.",
            "What are the key points of {concept}?",
        ],
        MetacognitionPromptType.PREDICTION: [
            "How many of the next 5 questions do you think you'll get right?",
            "Predict your score on the upcoming quiz about {concept}.",
            "Do you think you need more practice with {concept} before moving on?",
        ],
        MetacognitionPromptType.REFLECTION: [
            "What learning strategy helped you most with {concept}?",
            "What was the most challenging part of learning {concept}?",
            "What would you do differently next time you learn something like {concept}?",
            "How has your understanding of {concept} changed?",
        ],
        MetacognitionPromptType.STRATEGY_SELECTION: [
            "Which approach would help you learn {concept} better: reading, practice problems, or watching a video?",
            "Would you benefit from reviewing {concept} again or moving to the next topic?",
            "Should you try a different explanation of {concept}?",
        ],
        MetacognitionPromptType.ERROR_ANALYSIS: [
            "Why do you think you got that wrong?",
            "What concept did you misunderstand?",
            "What would you need to learn to get this right next time?",
            "Was this a careless error or a gap in understanding?",
        ],
        MetacognitionPromptType.KNOWLEDGE_CHECK: [
            "Quick check: Can you recall the main idea of {concept}?",
            "Without looking, what are the key points of {concept}?",
            "What questions do you still have about {concept}?",
        ],
        MetacognitionPromptType.ELABORATION: [
            "How does {concept} connect to {related_concept}?",
            "Why is {concept} important for understanding {topic}?",
            "What real-world problems does {concept} help solve?",
        ],
    }

    # Timing rules for prompts
    TIMING_RULES = {
        MetacognitionPromptType.CONFIDENCE_RATING: ["before", "after"],
        MetacognitionPromptType.SELF_EXPLANATION: ["during", "after"],
        MetacognitionPromptType.PREDICTION: ["before"],
        MetacognitionPromptType.REFLECTION: ["after"],
        MetacognitionPromptType.STRATEGY_SELECTION: ["before", "during"],
        MetacognitionPromptType.ERROR_ANALYSIS: ["after"],  # After wrong answer
        MetacognitionPromptType.KNOWLEDGE_CHECK: ["during", "after"],
        MetacognitionPromptType.ELABORATION: ["during"],
    }

    def __init__(
        self,
        prompt_frequency: float = 0.3,  # Probability of prompting
        min_interval_seconds: int = 120,  # Min time between prompts
    ):
        self.prompt_frequency = prompt_frequency
        self.min_interval = timedelta(seconds=min_interval_seconds)
        self.last_prompt_time: Dict[str, datetime] = {}
        self.prompt_history: Dict[str, List[MetacognitionPrompt]] = {}

    def _should_prompt(self, user_id: str) -> bool:
        """Determine if we should show a metacognitive prompt"""
        last_time = self.last_prompt_time.get(user_id)
        if last_time and datetime.utcnow() - last_time < self.min_interval:
            return False
        return random.random() < self.prompt_frequency

    def _select_prompt_type(
        self,
        timing: str,
        context: Dict[str, Any]
    ) -> MetacognitionPromptType:
        """Select appropriate prompt type based on timing and context"""
        # Get eligible prompt types for this timing
        eligible = [
            ptype for ptype, timings in self.TIMING_RULES.items()
            if timing in timings
        ]

        if not eligible:
            return MetacognitionPromptType.CONFIDENCE_RATING

        # Prioritize based on context
        if context.get("just_made_error"):
            if MetacognitionPromptType.ERROR_ANALYSIS in eligible:
                return MetacognitionPromptType.ERROR_ANALYSIS

        if context.get("completed_section"):
            if MetacognitionPromptType.REFLECTION in eligible:
                return MetacognitionPromptType.REFLECTION

        if context.get("low_engagement"):
            if MetacognitionPromptType.STRATEGY_SELECTION in eligible:
                return MetacognitionPromptType.STRATEGY_SELECTION

        return random.choice(eligible)

    def generate_prompt(
        self,
        user_id: str,
        concept_name: str,
        timing: str,  # "before", "during", "after"
        context: Optional[Dict] = None,
        force: bool = False
    ) -> Optional[MetacognitionPrompt]:
        """
        Generate a metacognitive prompt for the learner

        Args:
            user_id: User identifier
            concept_name: Name of concept being studied
            timing: When in the learning process
            context: Additional context (errors, engagement, etc.)
            force: Generate even if not scheduled

        Returns:
            MetacognitionPrompt or None if not appropriate to prompt
        """
        context = context or {}

        if not force and not self._should_prompt(user_id):
            return None

        # Select prompt type
        prompt_type = self._select_prompt_type(timing, context)

        # Select template
        templates = self.PROMPT_TEMPLATES.get(prompt_type, [])
        if not templates:
            return None

        template = random.choice(templates)

        # Fill in template
        prompt_text = template.format(
            concept=concept_name,
            related_concept=context.get("related_concept", "related concepts"),
            topic=context.get("topic", "this topic"),
        )

        # Create prompt
        import uuid
        prompt = MetacognitionPrompt(
            prompt_id=str(uuid.uuid4())[:8],
            prompt_type=prompt_type,
            prompt_text=prompt_text,
            context=context,
            timing=timing,
            required=context.get("required", False),
        )

        # Update tracking
        self.last_prompt_time[user_id] = datetime.utcnow()
        if user_id not in self.prompt_history:
            self.prompt_history[user_id] = []
        self.prompt_history[user_id].append(prompt)

        return prompt

    def generate_confidence_scale(
        self,
        concept_name: str,
        scale_type: str = "numeric"  # "numeric", "verbal", "emoji"
    ) -> Dict[str, Any]:
        """Generate a confidence rating scale with labels"""
        scales = {
            "numeric": {
                "min": 1,
                "max": 5,
                "labels": {
                    1: "Not at all confident",
                    2: "Slightly confident",
                    3: "Moderately confident",
                    4: "Very confident",
                    5: "Extremely confident",
                }
            },
            "verbal": {
                "options": [
                    "I don't understand this at all",
                    "I have a vague idea",
                    "I understand the basics",
                    "I understand it well",
                    "I could teach this to others",
                ]
            },
            "emoji": {
                "options": [
                    {"emoji": "😕", "label": "Confused", "value": 1},
                    {"emoji": "🤔", "label": "Uncertain", "value": 2},
                    {"emoji": "😐", "label": "Okay", "value": 3},
                    {"emoji": "🙂", "label": "Good", "value": 4},
                    {"emoji": "😊", "label": "Great", "value": 5},
                ]
            }
        }

        return {
            "concept": concept_name,
            "scale_type": scale_type,
            "scale": scales.get(scale_type, scales["numeric"]),
            "prompt": f"How confident are you in your understanding of {concept_name}?",
        }


class CalibrationTracker:
    """
    Tracks learner's metacognitive calibration

    Calibration = alignment between confidence and actual performance
    - Overconfidence: Illusion of competence, ineffective study
    - Underconfidence: Unnecessary anxiety, over-studying
    - Well-calibrated: Accurate self-assessment, effective learning

    Research shows most learners are overconfident, especially novices.
    """

    def __init__(
        self,
        calibration_threshold: float = 0.15,  # Max acceptable error
    ):
        self.calibration_threshold = calibration_threshold
        self.ratings: Dict[str, List[ConfidenceRating]] = {}

    def record_rating(self, rating: ConfidenceRating):
        """Record a confidence rating"""
        user_id = rating.user_id
        if user_id not in self.ratings:
            self.ratings[user_id] = []
        self.ratings[user_id].append(rating)

    def update_performance(
        self,
        user_id: str,
        concept_id: str,
        actual_performance: float
    ):
        """Update actual performance for recent confidence rating"""
        if user_id not in self.ratings:
            return

        # Find most recent unresolved rating for this concept
        for rating in reversed(self.ratings[user_id]):
            if (rating.concept_id == concept_id and
                rating.actual_performance is None):
                rating.actual_performance = actual_performance
                break

    def calculate_calibration(
        self,
        user_id: str,
        concept_id: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> CalibrationData:
        """
        Calculate calibration metrics for a user

        Args:
            user_id: User identifier
            concept_id: Optional concept filter
            time_window: Optional time window

        Returns:
            CalibrationData with calibration metrics
        """
        if user_id not in self.ratings:
            return CalibrationData(
                confidence_ratings=[],
                mean_confidence=0.5,
                mean_performance=0.5,
                calibration_error=0.0,
                calibration_level=CalibrationLevel.UNKNOWN,
                overconfidence_rate=0.0,
                underconfidence_rate=0.0,
            )

        # Filter ratings
        ratings = self.ratings[user_id]

        if concept_id:
            ratings = [r for r in ratings if r.concept_id == concept_id]

        if time_window:
            cutoff = datetime.utcnow() - time_window
            ratings = [r for r in ratings if r.timestamp >= cutoff]

        # Only include resolved ratings
        resolved = [
            (r.confidence, r.actual_performance)
            for r in ratings
            if r.actual_performance is not None
        ]

        if not resolved:
            return CalibrationData(
                confidence_ratings=[],
                mean_confidence=0.5,
                mean_performance=0.5,
                calibration_error=0.0,
                calibration_level=CalibrationLevel.UNKNOWN,
                overconfidence_rate=0.0,
                underconfidence_rate=0.0,
            )

        confidences = [r[0] for r in resolved]
        performances = [r[1] for r in resolved]

        mean_conf = statistics.mean(confidences)
        mean_perf = statistics.mean(performances)

        # Calculate calibration error (absolute difference)
        errors = [abs(c - p) for c, p in resolved]
        calibration_error = statistics.mean(errors)

        # Calculate overconfidence/underconfidence rates
        overconfident_count = sum(1 for c, p in resolved if c > p + 0.1)
        underconfident_count = sum(1 for c, p in resolved if c < p - 0.1)

        # Determine calibration level
        if calibration_error <= self.calibration_threshold:
            level = CalibrationLevel.WELL_CALIBRATED
        elif mean_conf > mean_perf:
            level = CalibrationLevel.OVERCONFIDENT
        else:
            level = CalibrationLevel.UNDERCONFIDENT

        return CalibrationData(
            confidence_ratings=resolved,
            mean_confidence=round(mean_conf, 3),
            mean_performance=round(mean_perf, 3),
            calibration_error=round(calibration_error, 3),
            calibration_level=level,
            overconfidence_rate=round(overconfident_count / len(resolved), 3),
            underconfidence_rate=round(underconfident_count / len(resolved), 3),
        )

    def generate_calibration_feedback(
        self,
        calibration: CalibrationData
    ) -> Dict[str, Any]:
        """Generate feedback to help learner improve calibration"""
        feedback = {
            "calibration_level": calibration.calibration_level.value,
            "mean_confidence": calibration.mean_confidence,
            "mean_performance": calibration.mean_performance,
            "gap": round(calibration.mean_confidence - calibration.mean_performance, 3),
        }

        if calibration.calibration_level == CalibrationLevel.WELL_CALIBRATED:
            feedback["message"] = "Great job! Your confidence matches your actual understanding."
            feedback["tips"] = [
                "Keep using effective study strategies",
                "Continue testing yourself regularly",
            ]
        elif calibration.calibration_level == CalibrationLevel.OVERCONFIDENT:
            feedback["message"] = "You tend to feel more confident than your actual performance shows. This is called the illusion of competence."
            feedback["tips"] = [
                "Test yourself more often before feeling confident",
                "Try to explain concepts out loud or in writing",
                "Use retrieval practice instead of just re-reading",
                "Be skeptical of 'feeling' like you understand",
            ]
        elif calibration.calibration_level == CalibrationLevel.UNDERCONFIDENT:
            feedback["message"] = "You know more than you think! Your performance is better than your confidence suggests."
            feedback["tips"] = [
                "Trust yourself more when answering",
                "Your first instinct is often correct",
                "Review your successes to build confidence",
            ]
        else:
            feedback["message"] = "We need more data to assess your calibration."
            feedback["tips"] = [
                "Continue practicing and rating your confidence",
            ]

        return feedback


class SelfExplanationAnalyzer:
    """
    Analyzes quality of self-explanations

    Good self-explanations:
    - Mention key concepts
    - Make connections to prior knowledge
    - Identify cause-and-effect relationships
    - Use appropriate vocabulary
    """

    def __init__(self, llm=None):
        """
        Initialize analyzer

        Args:
            llm: Optional LLM for quality assessment (if None, uses heuristics)
        """
        self.llm = llm

    def analyze_explanation(
        self,
        explanation: str,
        concept_name: str,
        expected_concepts: List[str],
        common_misconceptions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze quality of a self-explanation

        Args:
            explanation: Learner's explanation text
            concept_name: Concept being explained
            expected_concepts: Key concepts that should be mentioned
            common_misconceptions: Known misconceptions to check for

        Returns:
            Analysis with quality score and feedback
        """
        explanation_lower = explanation.lower()
        common_misconceptions = common_misconceptions or []

        # Check for key concepts mentioned
        concepts_found = [
            concept for concept in expected_concepts
            if concept.lower() in explanation_lower
        ]

        # Check for misconceptions
        misconceptions_found = [
            m for m in common_misconceptions
            if m.lower() in explanation_lower
        ]

        # Calculate metrics
        word_count = len(explanation.split())
        concept_coverage = len(concepts_found) / len(expected_concepts) if expected_concepts else 0

        # Quality heuristics
        has_elaboration = word_count > 20
        has_examples = any(phrase in explanation_lower for phrase in
                         ["for example", "such as", "like when", "instance"])
        has_connections = any(phrase in explanation_lower for phrase in
                            ["because", "therefore", "so that", "relates to", "similar to"])

        # Calculate quality score
        quality_components = [
            concept_coverage * 0.4,
            (1 if has_elaboration else 0) * 0.2,
            (1 if has_examples else 0) * 0.15,
            (1 if has_connections else 0) * 0.15,
            (1 - len(misconceptions_found) / max(1, len(common_misconceptions))) * 0.1
            if common_misconceptions else 0.1,
        ]
        quality_score = sum(quality_components)

        # Generate feedback
        feedback = []
        if concept_coverage < 0.5:
            feedback.append(f"Try to mention more key concepts like: {', '.join(expected_concepts[:3])}")
        if not has_examples:
            feedback.append("Adding an example would strengthen your explanation")
        if not has_connections:
            feedback.append("Try to explain WHY or HOW this concept works")
        if misconceptions_found:
            feedback.append(f"Be careful about: {', '.join(misconceptions_found)}")

        return {
            "quality_score": round(quality_score, 3),
            "word_count": word_count,
            "concepts_found": concepts_found,
            "concept_coverage": round(concept_coverage, 3),
            "misconceptions_found": misconceptions_found,
            "has_examples": has_examples,
            "has_connections": has_connections,
            "feedback": feedback,
            "strength": "Good" if quality_score > 0.7 else "Developing" if quality_score > 0.4 else "Needs work",
        }


# Global instances
metacognition_prompter = MetacognitionPrompter()
calibration_tracker = CalibrationTracker()
explanation_analyzer = SelfExplanationAnalyzer()


def get_metacognition_prompter() -> MetacognitionPrompter:
    """Dependency injection"""
    return metacognition_prompter


def get_calibration_tracker() -> CalibrationTracker:
    """Dependency injection"""
    return calibration_tracker


def get_explanation_analyzer() -> SelfExplanationAnalyzer:
    """Dependency injection"""
    return explanation_analyzer
