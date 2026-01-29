"""
PACER Information Classifier
AI-powered triage decision tree for content classification

Implements the PACER taxonomy decision tree:
1. Is it actionable (HOW to do)? -> Procedural
2. Does it use analogy/metaphor? -> Analogous
3. Does it explain WHY (theory/principle)? -> Conceptual
4. Is it supporting data/statistics? -> Evidence
5. Is it arbitrary detail for recall? -> Reference
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib


class PACERType(str, Enum):
    """Five information types in the PACER taxonomy"""

    PROCEDURAL = "procedural"
    ANALOGOUS = "analogous"
    CONCEPTUAL = "conceptual"
    EVIDENCE = "evidence"
    REFERENCE = "reference"


@dataclass
class ClassificationResult:
    """Result of PACER classification"""

    pacer_type: PACERType
    confidence: float
    reasoning: str
    alternative_types: List[Tuple[PACERType, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""


@dataclass
class TriageDecision:
    """Single decision in the triage tree"""

    question: str
    answer: bool
    leads_to: Optional[PACERType] = None


class PACERClassifier:
    """
    Classifies content into PACER categories using:
    1. Rule-based heuristics (fast, interpretable)
    2. LLM-based classification (accurate, contextual) - optional
    3. Hybrid approach combining both
    """

    # Pattern-based indicators for each PACER type
    PROCEDURAL_INDICATORS = [
        r"\bstep\s+\d+\b",
        r"\bstep\s+one\b",
        r"\bfirst\b.*\bthen\b.*\b(next|finally)\b",
        r"\bhow\s+to\b",
        r"\bprocedure\b",
        r"\binstructions?\b",
        r"\binstructions?\b",
        r"\bfollow(ing)?\s+(these\s+)?steps\b",
        r"\bexecute\b",
        r"\bexecute\b",
        r"\bperform\s+(the|this)\b",
        r"\bimplement(ing)?\b",
        r"\bbegin\s+by\b",
        r"\bstart\s+with\b",
        r"\b(run|type|click|press|enter)\b.*\b(command|button|key)\b",
        r"\bworkflow\b",
        r"\brecipe\b",
        r"\btutorial\b",
    ]

    ANALOGOUS_INDICATORS = [
        r"\blike\s+a\b",
        r"\bsimilar\s+to\b",
        r"\banalog(y|ous|ies)\b",
        r"\bmetaphor(ically)?\b",
        r"\bthink\s+of\s+(it\s+)?as\b",
        r"\bjust\s+as\b.*\bso\b",
        r"\bcompare\s+(it\s+)?to\b",
        r"\bimagine\s+(it\s+)?(as|like)\b",
        r"\bresembles?\b",
        r"\bpicture\s+(this|it)\b",
        r"\bis\s+(like|similar)\s+to\b",
        r"\bworks\s+like\b",
        r"\bworks\s+like\b",
        r"\bfunctions\s+(like|as)\b",
        r"\bacts\s+(like|as)\b",
        r"\bcan\s+be\s+thought\s+of\s+as\b",
        r"\bin\s+the\s+same\s+way\s+(that|as)\b",
    ]

    CONCEPTUAL_INDICATORS = [
        r"\bprinciple\b",
        r"\btheory\b",
        r"\bwhy\b.*\b(is|does|do|are|can|will|would)\b",
        r"\bbecause\b",
        r"\bcauses?\b",
        r"\beffects?\b",
        r"\bresults?\s+in\b",
        r"\bleads?\s+to\b",
        r"\brelationship\s+between\b",
        r"\bfundamental(ly)?\b",
        r"\bexplain(s|ed|ing)?\s+(why|how|the)\b",
        r"\bunderstand(ing)?\b",
        r"\bconceptual(ly)?\b",
        r"\bmechanism\b",
        r"\bprocess\s+(of|by|through)\b",
        r"\bframework\b",
        r"\bmodel\b",
        r"\blaw\s+of\b",
        r"\bcorrelat(es?|ion)\b",
        r"\binfluences?\b",
        r"\bdetermines?\b",
    ]

    EVIDENCE_INDICATORS = [
        r"\bstud(y|ies)\s+show(s|ed|ing)?\b",
        r"\bresearch\s+(has\s+)?(shown|shows|found|demonstrated|indicates)\b",
        r"\bstatistic(s|ally)?\b",
        r"\b\d+(\.\d+)?\s*%\b",  # Percentages
        r"\baccording\s+to\b",
        r"\bevidence\s+(suggests?|shows?|indicates?)\b",
        r"\bdata\s+(shows?|suggests?|indicates?)\b",
        r"\bproven\b",
        r"\bdemonstrated\b",
        r"\bexperiment(s|al)?\b",
        r"\bfindings?\b",
        r"\bsurvey(s|ed)?\b",
        r"\bmeta-analysis\b",
        r"\bp\s*[<>=]\s*0?\.\d+\b",  # P-values
        r"\bsignificant(ly)?\b",
        r"\bcited\b",
        r"\bpublished\b",
        r"\bjournal\b",
        r"\bsample\s+size\b",
        r"\bn\s*=\s*\d+\b",  # Sample sizes
    ]

    REFERENCE_INDICATORS = [
        r"\b(18|19|20)\d{2}\b",  # Years
        r"\bdefined\s+as\b",
        r"\bnamed\s+after\b",
        r"\bconstant\b",
        r"\bformula\b",
        r"\bequation\b",
        r"\babbreviation\b",
        r"\bacronym\b",
        r"\bknown\s+as\b",
        r"\bcalled\b",
        r"\btermed\b",
        r"\bunit\s+of\b",
        r"\bsymbol\b",
        r"\bdenoted\s+(by|as)\b",
        r"\brepresented\s+(by|as)\b",
        r"\bcoordinate(s)?\b",
        r"\bvalue\s+of\b",
        r"\b[A-Z][a-z]+\s+(was|is)\s+born\b",  # Biographical dates
    ]

    def __init__(self, use_llm: bool = False, llm_client: Optional[Any] = None):
        """
        Initialize the PACER classifier.

        Args:
            use_llm: Whether to use LLM for enhanced classification
            llm_client: Optional LLM client for classification
        """
        self.use_llm = use_llm
        self.llm_client = llm_client

    def classify(
        self, content: str, context: Optional[Dict[str, Any]] = None
    ) -> ClassificationResult:
        """
        Main classification method.

        Args:
            content: Text content to classify
            context: Optional context (course domain, surrounding content, etc.)

        Returns:
            ClassificationResult with type, confidence, and reasoning
        """
        content_hash = self._hash_content(content)

        # Phase 1: Rule-based pre-classification
        rule_scores = self._rule_based_scoring(content)

        # Phase 2: LLM classification if enabled and rules are inconclusive
        if self.use_llm and self._is_inconclusive(rule_scores):
            return self._llm_classify(content, context, rule_scores, content_hash)

        # Return best rule-based match
        return self._finalize_rule_classification(rule_scores, content, content_hash)

    def classify_batch(
        self, contents: List[str], context: Optional[Dict[str, Any]] = None
    ) -> List[ClassificationResult]:
        """Classify multiple content items efficiently"""
        return [self.classify(content, context) for content in contents]

    def run_triage_tree(self, content: str) -> List[TriageDecision]:
        """
        Run interactive triage decision tree.
        Returns the decision path taken through the tree.

        This implements the PACER triage logic:
        1. Is it actionable? -> Procedural
        2. Is it an analogy? -> Analogous
        3. Does it explain why? -> Conceptual
        4. Is it supporting data? -> Evidence
        5. Default -> Reference
        """
        decisions = []
        content_lower = content.lower()

        # Q1: Does it describe HOW to do something (steps/process)?
        is_procedural = self._check_procedural(content_lower)
        decisions.append(
            TriageDecision(
                question="Does this describe HOW to do something (steps, instructions, process)?",
                answer=is_procedural,
                leads_to=PACERType.PROCEDURAL if is_procedural else None,
            )
        )
        if is_procedural:
            return decisions

        # Q2: Does it use analogy/metaphor to explain a concept?
        is_analogous = self._check_analogous(content_lower)
        decisions.append(
            TriageDecision(
                question="Does this use an analogy or metaphor to explain something new?",
                answer=is_analogous,
                leads_to=PACERType.ANALOGOUS if is_analogous else None,
            )
        )
        if is_analogous:
            return decisions

        # Q3: Does it explain WHY or describe a theory/principle/mechanism?
        is_conceptual = self._check_conceptual(content_lower)
        decisions.append(
            TriageDecision(
                question="Does this explain WHY something works or describe a theory/principle?",
                answer=is_conceptual,
                leads_to=PACERType.CONCEPTUAL if is_conceptual else None,
            )
        )
        if is_conceptual:
            return decisions

        # Q4: Is it data/statistics/evidence supporting a concept?
        is_evidence = self._check_evidence(content_lower)
        decisions.append(
            TriageDecision(
                question="Is this evidence (data, statistics, research findings) supporting a concept?",
                answer=is_evidence,
                leads_to=PACERType.EVIDENCE if is_evidence else None,
            )
        )
        if is_evidence:
            return decisions

        # Q5: Default to Reference (arbitrary details for recall)
        decisions.append(
            TriageDecision(
                question="Is this an arbitrary detail (date, name, formula, constant) for recall?",
                answer=True,
                leads_to=PACERType.REFERENCE,
            )
        )

        return decisions

    def _rule_based_scoring(self, content: str) -> Dict[PACERType, float]:
        """Score content against all PACER types using pattern matching"""
        content_lower = content.lower()
        scores = {}

        type_patterns = [
            (PACERType.PROCEDURAL, self.PROCEDURAL_INDICATORS),
            (PACERType.ANALOGOUS, self.ANALOGOUS_INDICATORS),
            (PACERType.CONCEPTUAL, self.CONCEPTUAL_INDICATORS),
            (PACERType.EVIDENCE, self.EVIDENCE_INDICATORS),
            (PACERType.REFERENCE, self.REFERENCE_INDICATORS),
        ]

        for ptype, patterns in type_patterns:
            matches = sum(
                1 for p in patterns if re.search(p, content_lower, re.IGNORECASE)
            )
            # Normalize score: more matches = higher confidence, capped at 1.0
            # Use 30% of patterns as threshold for full score
            threshold = max(len(patterns) * 0.3, 1)
            scores[ptype] = min(1.0, matches / threshold)

        # Boost scores based on content structure
        scores = self._apply_structural_boosts(content, scores)

        return scores

    def _apply_structural_boosts(
        self, content: str, scores: Dict[PACERType, float]
    ) -> Dict[PACERType, float]:
        """Apply additional scoring based on content structure"""
        boosted = scores.copy()

        # Numbered lists boost Procedural
        if re.search(r"^\s*\d+[\.\)]\s", content, re.MULTILINE):
            boosted[PACERType.PROCEDURAL] = min(1.0, boosted[PACERType.PROCEDURAL] + 0.2)

        # "Step X:" pattern common in instructions
        if re.search(r"^\s*Step\s+\d+[:\.]", content, re.MULTILINE | re.IGNORECASE):
            boosted[PACERType.PROCEDURAL] = min(1.0, boosted[PACERType.PROCEDURAL] + 0.2)

        # Bullet points with "step" language boost Procedural
        if re.search(r"^\s*[-•]\s.*(step|first|then|next)", content, re.MULTILINE | re.IGNORECASE):
            boosted[PACERType.PROCEDURAL] = min(1.0, boosted[PACERType.PROCEDURAL] + 0.15)

        # Direct quotes from sources boost Evidence
        if re.search(r'"[^"]{20,}"', content):
            boosted[PACERType.EVIDENCE] = min(1.0, boosted[PACERType.EVIDENCE] + 0.1)

        # Short content with dates/numbers more likely Reference
        if len(content) < 200 and re.search(r"\b\d{4}\b|\b\d+\.\d+\b", content):
            boosted[PACERType.REFERENCE] = min(1.0, boosted[PACERType.REFERENCE] + 0.15)

        # "X is like Y" pattern strongly indicates Analogous
        if re.search(r"\b\w+\s+is\s+(like|similar\s+to)\s+\w+", content, re.IGNORECASE):
            boosted[PACERType.ANALOGOUS] = min(1.0, boosted[PACERType.ANALOGOUS] + 0.25)

        return boosted

    def _is_inconclusive(self, scores: Dict[PACERType, float]) -> bool:
        """Check if rule-based classification is inconclusive"""
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) < 2:
            return True
        # Inconclusive if top score is low or top two scores are close
        top_score = sorted_scores[0]
        second_score = sorted_scores[1]
        return top_score < 0.4 or (top_score - second_score) < 0.15

    def _finalize_rule_classification(
        self, scores: Dict[PACERType, float], content: str, content_hash: str
    ) -> ClassificationResult:
        """Finalize classification from rule scores"""
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_type, best_score = sorted_types[0]

        # Generate reasoning
        reasoning = self._generate_reasoning(best_type, scores)

        return ClassificationResult(
            pacer_type=best_type,
            confidence=best_score,
            reasoning=reasoning,
            alternative_types=[(t, s) for t, s in sorted_types[1:3] if s > 0.1],
            metadata={"method": "rule_based", "all_scores": {k.value: v for k, v in scores.items()}},
            content_hash=content_hash,
        )

    def _generate_reasoning(
        self, pacer_type: PACERType, scores: Dict[PACERType, float]
    ) -> str:
        """Generate human-readable reasoning for classification"""
        type_descriptions = {
            PACERType.PROCEDURAL: "Contains step-by-step instructions or how-to guidance",
            PACERType.ANALOGOUS: "Uses analogy or metaphor to explain concepts",
            PACERType.CONCEPTUAL: "Explains theories, principles, or causal relationships",
            PACERType.EVIDENCE: "Contains data, statistics, or research findings",
            PACERType.REFERENCE: "Contains arbitrary details suitable for memorization",
        }

        confidence = scores[pacer_type]
        base_reason = type_descriptions[pacer_type]

        if confidence >= 0.7:
            return f"High confidence: {base_reason}"
        elif confidence >= 0.4:
            return f"Moderate confidence: {base_reason}"
        else:
            return f"Low confidence (consider manual review): {base_reason}"

    def _llm_classify(
        self,
        content: str,
        context: Optional[Dict[str, Any]],
        rule_scores: Dict[PACERType, float],
        content_hash: str,
    ) -> ClassificationResult:
        """
        Use LLM for classification when rules are inconclusive.
        Falls back to rule-based if LLM unavailable.
        """
        if not self.llm_client:
            return self._finalize_rule_classification(rule_scores, content, content_hash)

        # LLM prompt would go here
        # For now, fall back to rule-based
        return self._finalize_rule_classification(rule_scores, content, content_hash)

    def _check_procedural(self, content: str) -> bool:
        """Check if content is primarily procedural"""
        match_count = sum(
            1 for p in self.PROCEDURAL_INDICATORS[:8]
            if re.search(p, content, re.IGNORECASE)
        )
        return match_count >= 2

    def _check_analogous(self, content: str) -> bool:
        """Check if content is primarily analogous"""
        match_count = sum(
            1 for p in self.ANALOGOUS_INDICATORS[:8]
            if re.search(p, content, re.IGNORECASE)
        )
        return match_count >= 1

    def _check_conceptual(self, content: str) -> bool:
        """Check if content is primarily conceptual"""
        match_count = sum(
            1 for p in self.CONCEPTUAL_INDICATORS[:10]
            if re.search(p, content, re.IGNORECASE)
        )
        return match_count >= 2

    def _check_evidence(self, content: str) -> bool:
        """Check if content is primarily evidence"""
        match_count = sum(
            1 for p in self.EVIDENCE_INDICATORS[:10]
            if re.search(p, content, re.IGNORECASE)
        )
        return match_count >= 2

    def _hash_content(self, content: str) -> str:
        """Generate hash for content deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_recommended_action(self, pacer_type: PACERType) -> Dict[str, str]:
        """Get recommended learning action for a PACER type"""
        actions = {
            PACERType.PROCEDURAL: {
                "action": "Practice",
                "description": "Execute the steps actively, don't just read",
                "tool": "Practice exercises, simulations",
            },
            PACERType.ANALOGOUS: {
                "action": "Critique",
                "description": "Identify where the analogy breaks down",
                "tool": "Analogy critique worksheet",
            },
            PACERType.CONCEPTUAL: {
                "action": "Map",
                "description": "Create concept map showing relationships",
                "tool": "Knowledge graph, concept mapper",
            },
            PACERType.EVIDENCE: {
                "action": "Link",
                "description": "Connect evidence to concepts it supports",
                "tool": "Evidence-concept linking",
            },
            PACERType.REFERENCE: {
                "action": "Recall",
                "description": "Use spaced repetition for memorization",
                "tool": "SRS flashcards (FSRS)",
            },
        }
        return actions.get(pacer_type, {})
