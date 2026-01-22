"""
NER-based Concept Extractor

Uses SpaCy and custom patterns for high-quality concept extraction.
Targets 92.8% F1 score per research guidelines.

Features:
- SpaCy NER for entity recognition
- Custom patterns for technical terms
- Domain-specific dictionaries
- Confidence scoring
- Deduplication and normalization
"""
import logging
import re
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import Counter
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import SpaCy, fall back to basic extraction if not available
try:
    import spacy
    from spacy.matcher import PhraseMatcher, Matcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("SpaCy not available. Using fallback concept extraction.")


class ConceptType(str, Enum):
    """Types of extracted concepts"""
    ENTITY = "entity"           # Named entity (ORG, PRODUCT, etc.)
    TECHNICAL = "technical"     # Technical term from dictionary
    PATTERN = "pattern"         # Matched pattern (e.g., "X algorithm")
    NOUN_CHUNK = "noun_chunk"   # Noun phrase from parsing
    CUSTOM = "custom"           # Custom domain-specific


@dataclass
class ExtractedConcept:
    """A concept extracted from text"""
    name: str
    concept_type: ConceptType
    confidence: float  # 0-1
    frequency: int
    source_spans: List[Tuple[int, int]]  # Character offsets
    normalized_name: str  # Lowercase, stripped


class NERConceptExtractor:
    """
    Advanced concept extractor using SpaCy NER and custom patterns.

    Combines multiple extraction strategies:
    1. SpaCy NER for named entities
    2. Noun chunk extraction for compound terms
    3. Pattern matching for technical terms
    4. Dictionary lookup for domain terms
    """

    # Technical term patterns (regex)
    TECHNICAL_PATTERNS = [
        r"\b(\w+)\s+(algorithm|data\s+structure|framework|library|pattern|protocol)\b",
        r"\b(machine|deep|reinforcement)\s+learning\b",
        r"\b(\w+)\s+(function|method|class|interface|module)\b",
        r"\b(\w+)\s+(tree|graph|array|list|queue|stack|heap)\b",
        r"\b(\w+)\s+(sort|search|traversal)\b",
        r"\b(object[- ]oriented|functional|procedural)\s+programming\b",
        r"\b(\w+)\s+(network|layer|model)\b",
        r"\b(API|REST|GraphQL|SQL|NoSQL)\b",
        r"\b(Python|JavaScript|TypeScript|Java|C\+\+|Rust|Go)\b",
    ]

    # Domain-specific terms (educational/technical)
    DOMAIN_TERMS = {
        # Programming
        "algorithm", "data structure", "function", "variable", "class", "object",
        "method", "interface", "inheritance", "polymorphism", "encapsulation",
        "abstraction", "recursion", "iteration", "loop", "conditional",
        "array", "list", "dictionary", "set", "tuple", "string", "integer",
        "boolean", "float", "type", "parameter", "argument", "return value",

        # Machine Learning
        "machine learning", "deep learning", "neural network", "training",
        "inference", "model", "dataset", "feature", "label", "prediction",
        "classification", "regression", "clustering", "supervised learning",
        "unsupervised learning", "reinforcement learning", "gradient descent",
        "backpropagation", "activation function", "loss function", "optimizer",
        "overfitting", "underfitting", "regularization", "cross-validation",

        # Data Science
        "data analysis", "visualization", "statistics", "probability",
        "distribution", "correlation", "regression", "hypothesis testing",
        "confidence interval", "standard deviation", "mean", "median", "mode",

        # Web Development
        "frontend", "backend", "full stack", "API", "REST", "GraphQL",
        "database", "server", "client", "HTTP", "request", "response",
        "authentication", "authorization", "session", "cookie", "token",

        # Software Engineering
        "version control", "git", "repository", "commit", "branch", "merge",
        "pull request", "code review", "testing", "unit test", "integration test",
        "deployment", "CI/CD", "agile", "scrum", "sprint", "refactoring",
    }

    # SpaCy entity types to include
    RELEVANT_ENTITY_TYPES = {
        "ORG",           # Organizations (frameworks, libraries)
        "PRODUCT",       # Products (software, tools)
        "WORK_OF_ART",   # Named works (algorithms, papers)
        "LAW",           # Standards, protocols
        "LANGUAGE",      # Programming languages
        "GPE",           # Geographic (for context)
        "PERSON",        # Authors (for attribution)
    }

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        min_confidence: float = 0.5,
        min_frequency: int = 1,
        max_concepts: int = 100,
    ):
        """
        Initialize the NER concept extractor.

        Args:
            model_name: SpaCy model to use
            min_confidence: Minimum confidence threshold
            min_frequency: Minimum term frequency
            max_concepts: Maximum concepts to return
        """
        self.min_confidence = min_confidence
        self.min_frequency = min_frequency
        self.max_concepts = max_concepts

        # Load SpaCy model
        self.nlp = None
        self.phrase_matcher = None
        self.pattern_matcher = None

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
                self._setup_matchers()
                logger.info(f"Loaded SpaCy model: {model_name}")
            except OSError:
                logger.warning(f"SpaCy model {model_name} not found. Run: python -m spacy download {model_name}")

        # Compile regex patterns
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.TECHNICAL_PATTERNS
        ]

    def _setup_matchers(self):
        """Set up SpaCy phrase and pattern matchers"""
        if not self.nlp:
            return

        # Phrase matcher for domain terms
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(term) for term in self.DOMAIN_TERMS]
        self.phrase_matcher.add("DOMAIN_TERMS", patterns)

        # Pattern matcher for syntactic patterns
        self.pattern_matcher = Matcher(self.nlp.vocab)

        # Pattern: NOUN + NOUN (compound terms)
        self.pattern_matcher.add("COMPOUND_NOUN", [[
            {"POS": "NOUN"},
            {"POS": "NOUN"},
        ]])

        # Pattern: ADJ + NOUN (modified terms)
        self.pattern_matcher.add("ADJ_NOUN", [[
            {"POS": "ADJ"},
            {"POS": "NOUN"},
        ]])

    def extract(self, text: str) -> List[ExtractedConcept]:
        """
        Extract concepts from text using all available methods.

        Args:
            text: Text to analyze

        Returns:
            List of extracted concepts with metadata
        """
        all_concepts: Dict[str, ExtractedConcept] = {}

        # Method 1: SpaCy NER
        if self.nlp:
            ner_concepts = self._extract_ner(text)
            self._merge_concepts(all_concepts, ner_concepts)

        # Method 2: Noun chunks
        if self.nlp:
            chunk_concepts = self._extract_noun_chunks(text)
            self._merge_concepts(all_concepts, chunk_concepts)

        # Method 3: Phrase matcher (domain terms)
        if self.phrase_matcher:
            domain_concepts = self._extract_domain_terms(text)
            self._merge_concepts(all_concepts, domain_concepts)

        # Method 4: Regex patterns
        pattern_concepts = self._extract_patterns(text)
        self._merge_concepts(all_concepts, pattern_concepts)

        # Method 5: Fallback heuristics
        heuristic_concepts = self._extract_heuristics(text)
        self._merge_concepts(all_concepts, heuristic_concepts)

        # Filter and sort
        filtered = [
            c for c in all_concepts.values()
            if c.confidence >= self.min_confidence and c.frequency >= self.min_frequency
        ]

        # Sort by confidence * frequency
        filtered.sort(key=lambda c: c.confidence * c.frequency, reverse=True)

        return filtered[:self.max_concepts]

    def extract_names(self, text: str) -> List[str]:
        """
        Extract concept names only (simplified interface).

        Args:
            text: Text to analyze

        Returns:
            List of concept name strings
        """
        concepts = self.extract(text)
        return [c.name for c in concepts]

    def _extract_ner(self, text: str) -> List[ExtractedConcept]:
        """Extract named entities using SpaCy NER"""
        concepts = []
        doc = self.nlp(text)

        entity_counts = Counter()
        entity_spans = {}

        for ent in doc.ents:
            if ent.label_ in self.RELEVANT_ENTITY_TYPES:
                name = ent.text.strip()
                if len(name) > 2:  # Skip very short entities
                    entity_counts[name] += 1
                    if name not in entity_spans:
                        entity_spans[name] = []
                    entity_spans[name].append((ent.start_char, ent.end_char))

        for name, count in entity_counts.items():
            # Confidence based on entity type and frequency
            confidence = min(0.9, 0.6 + (count * 0.1))
            concepts.append(ExtractedConcept(
                name=name,
                concept_type=ConceptType.ENTITY,
                confidence=confidence,
                frequency=count,
                source_spans=entity_spans[name],
                normalized_name=name.lower().strip(),
            ))

        return concepts

    def _extract_noun_chunks(self, text: str) -> List[ExtractedConcept]:
        """Extract noun chunks as potential concepts"""
        concepts = []
        doc = self.nlp(text)

        chunk_counts = Counter()
        chunk_spans = {}

        for chunk in doc.noun_chunks:
            # Filter chunks
            name = chunk.text.strip()
            if len(name) > 3 and len(name.split()) <= 4:
                # Skip chunks that are just pronouns or determiners
                if chunk.root.pos_ in ("NOUN", "PROPN"):
                    chunk_counts[name] += 1
                    if name not in chunk_spans:
                        chunk_spans[name] = []
                    chunk_spans[name].append((chunk.start_char, chunk.end_char))

        for name, count in chunk_counts.items():
            if count >= self.min_frequency:
                confidence = min(0.7, 0.4 + (count * 0.1))
                concepts.append(ExtractedConcept(
                    name=name,
                    concept_type=ConceptType.NOUN_CHUNK,
                    confidence=confidence,
                    frequency=count,
                    source_spans=chunk_spans[name],
                    normalized_name=name.lower().strip(),
                ))

        return concepts

    def _extract_domain_terms(self, text: str) -> List[ExtractedConcept]:
        """Extract domain-specific terms using phrase matcher"""
        concepts = []
        doc = self.nlp(text)

        matches = self.phrase_matcher(doc)
        term_counts = Counter()
        term_spans = {}

        for match_id, start, end in matches:
            span = doc[start:end]
            name = span.text
            term_counts[name] += 1
            if name not in term_spans:
                term_spans[name] = []
            term_spans[name].append((span.start_char, span.end_char))

        for name, count in term_counts.items():
            # Domain terms get high confidence
            confidence = min(0.95, 0.8 + (count * 0.05))
            concepts.append(ExtractedConcept(
                name=name.title(),
                concept_type=ConceptType.TECHNICAL,
                confidence=confidence,
                frequency=count,
                source_spans=term_spans[name],
                normalized_name=name.lower().strip(),
            ))

        return concepts

    def _extract_patterns(self, text: str) -> List[ExtractedConcept]:
        """Extract concepts using regex patterns"""
        concepts = []
        pattern_counts = Counter()
        pattern_spans = {}

        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                name = match.group(0).strip()
                if len(name) > 3:
                    pattern_counts[name] += 1
                    if name not in pattern_spans:
                        pattern_spans[name] = []
                    pattern_spans[name].append((match.start(), match.end()))

        for name, count in pattern_counts.items():
            confidence = min(0.85, 0.6 + (count * 0.1))
            concepts.append(ExtractedConcept(
                name=name.title(),
                concept_type=ConceptType.PATTERN,
                confidence=confidence,
                frequency=count,
                source_spans=pattern_spans[name],
                normalized_name=name.lower().strip(),
            ))

        return concepts

    def _extract_heuristics(self, text: str) -> List[ExtractedConcept]:
        """Fallback heuristic extraction (capitalized phrases)"""
        concepts = []

        # Find capitalized phrases (2-4 words)
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"
        matches = re.findall(pattern, text)

        match_counts = Counter(matches)

        for name, count in match_counts.items():
            if count >= self.min_frequency and len(name) > 3:
                # Lower confidence for heuristic matches
                confidence = min(0.5, 0.3 + (count * 0.05))
                concepts.append(ExtractedConcept(
                    name=name,
                    concept_type=ConceptType.CUSTOM,
                    confidence=confidence,
                    frequency=count,
                    source_spans=[],
                    normalized_name=name.lower().strip(),
                ))

        return concepts

    def _merge_concepts(
        self,
        target: Dict[str, ExtractedConcept],
        source: List[ExtractedConcept]
    ):
        """Merge concepts, keeping highest confidence for duplicates"""
        for concept in source:
            key = concept.normalized_name
            if key in target:
                existing = target[key]
                # Keep the one with higher confidence, merge spans
                if concept.confidence > existing.confidence:
                    concept.source_spans = list(set(
                        existing.source_spans + concept.source_spans
                    ))
                    concept.frequency = max(concept.frequency, existing.frequency)
                    target[key] = concept
                else:
                    existing.source_spans = list(set(
                        existing.source_spans + concept.source_spans
                    ))
                    existing.frequency = max(concept.frequency, existing.frequency)
            else:
                target[key] = concept


# Singleton instance
_extractor: Optional[NERConceptExtractor] = None


def get_concept_extractor() -> NERConceptExtractor:
    """Get or create the global concept extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = NERConceptExtractor()
    return _extractor


def extract_concepts(text: str) -> List[str]:
    """
    Convenience function to extract concept names from text.

    Args:
        text: Text to analyze

    Returns:
        List of concept name strings
    """
    extractor = get_concept_extractor()
    return extractor.extract_names(text)
