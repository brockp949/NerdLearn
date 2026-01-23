"""
Content Ingestion Service - Automated Knowledge Graph Construction

This service implements the DiSSS framework "Deconstruction" phase:
1. Process raw content (PDF, video, text)
2. Extract atomic concepts using NER
3. Mine prerequisite relationships
4. Score difficulty (lexical + conceptual density)
5. Build Knowledge Graph in Neo4j

Based on:
- LayoutLMv3 for PDF structure understanding
- BERT for Named Entity Recognition
- Association Rule Mining for prerequisites
- Bloom's Taxonomy mapping
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
import io
import re
from collections import Counter, defaultdict

# PDF Processing
import PyPDF2
import pdfplumber

# NLP
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Text Analysis
import textstat
from langdetect import detect

# Neo4j
from neo4j import GraphDatabase

# Utils
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ConceptExtraction(BaseModel):
    concept: str
    frequency: int
    context: List[str]  # Sentences where concept appears
    category: str  # "technical_term", "definition", "formula", etc.


class PrerequisiteRelation(BaseModel):
    prerequisite: str
    dependent: str
    confidence: float
    evidence: str  # Why we think this is a prerequisite


class DifficultyScore(BaseModel):
    lexical_density: float  # Ratio of content words to total words
    conceptual_density: float  # Concepts per 100 words
    readability_score: float  # Flesch-Kincaid grade level
    avg_sentence_length: float
    unique_terms_ratio: float
    overall_difficulty: float  # 1-10 scale


class ContentAnalysis(BaseModel):
    document_id: str
    title: Optional[str]
    total_words: int
    total_sentences: int
    total_pages: int
    language: str

    # Extracted concepts
    concepts: List[ConceptExtraction]

    # Prerequisites
    prerequisites: List[PrerequisiteRelation]

    # Difficulty
    difficulty: DifficultyScore

    # Bloom's Taxonomy distribution
    bloom_distribution: Dict[str, int]  # Level -> count

    # Metadata
    processed_at: datetime


class KnowledgeGraphBuildRequest(BaseModel):
    document_id: str
    domain: str
    subdomain: Optional[str] = None


# ============================================================================
# CONCEPT EXTRACTOR
# ============================================================================

class ConceptExtractor:
    """
    Extract atomic concepts from text using NER and linguistic patterns

    Strategy:
    1. Use spaCy for general NER (technical terms)
    2. Pattern matching for definitions ("X is a...", "X refers to...")
    3. TF-IDF for domain-specific terminology
    4. POS tagging for noun phrases
    """

    def __init__(self):
        # Load spaCy model (using en_core_web_sm for speed)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        self.stop_words = set(stopwords.words('english'))

        # Definition patterns
        self.definition_patterns = [
            r"(\w+(?:\s+\w+)*)\s+is\s+(?:a|an)\s+",
            r"(\w+(?:\s+\w+)*)\s+refers to\s+",
            r"(\w+(?:\s+\w+)*)\s+means\s+",
            r"(\w+(?:\s+\w+)*)\s*:\s+",  # Colon definition
        ]

    def extract_concepts(self, text: str) -> List[ConceptExtraction]:
        """Extract concepts from text"""

        # Process with spaCy
        doc = self.nlp(text)

        # Extract named entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Extract noun phrases (potential concepts)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]

        # Extract definitions
        definitions = self._extract_definitions(text)

        # Extract technical terms via TF-IDF
        technical_terms = self._extract_technical_terms(text)

        # Combine and count
        all_concepts = {}

        # Add entities
        for entity, label in entities:
            clean = self._clean_concept(entity)
            if clean and len(clean) > 2:
                if clean not in all_concepts:
                    all_concepts[clean] = {
                        'frequency': 0,
                        'context': [],
                        'category': 'named_entity'
                    }
                all_concepts[clean]['frequency'] += 1

        # Add definitions
        for definition in definitions:
            clean = self._clean_concept(definition)
            if clean and len(clean) > 2:
                if clean not in all_concepts:
                    all_concepts[clean] = {
                        'frequency': 0,
                        'context': [],
                        'category': 'definition'
                    }
                all_concepts[clean]['frequency'] += 1
                all_concepts[clean]['category'] = 'definition'

        # Add technical terms
        for term, score in technical_terms:
            clean = self._clean_concept(term)
            if clean and len(clean) > 2:
                if clean not in all_concepts:
                    all_concepts[clean] = {
                        'frequency': int(score * 10),  # Scale TF-IDF score
                        'context': [],
                        'category': 'technical_term'
                    }

        # Extract context for top concepts
        sentences = sent_tokenize(text)
        for concept, data in all_concepts.items():
            context = []
            for sent in sentences:
                if concept.lower() in sent.lower():
                    context.append(sent[:200])  # Truncate long sentences
                    if len(context) >= 3:  # Max 3 examples
                        break
            data['context'] = context

        # Convert to Pydantic models
        concepts = [
            ConceptExtraction(
                concept=concept,
                frequency=data['frequency'],
                context=data['context'],
                category=data['category']
            )
            for concept, data in all_concepts.items()
        ]

        # Sort by frequency
        concepts.sort(key=lambda x: x.frequency, reverse=True)

        return concepts[:100]  # Return top 100

    def _extract_definitions(self, text: str) -> List[str]:
        """Extract concepts from definition patterns"""
        definitions = []
        for pattern in self.definition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                definitions.append(match.group(1))
        return definitions

    def _extract_technical_terms(self, text: str) -> List[Tuple[str, float]]:
        """Extract technical terms using TF-IDF"""

        # Tokenize into sentences
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return []

        # TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 3),  # Unigrams to trigrams
            stop_words='english'
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()

            # Get average TF-IDF score per term
            avg_scores = tfidf_matrix.mean(axis=0).A1
            term_scores = list(zip(feature_names, avg_scores))
            term_scores.sort(key=lambda x: x[1], reverse=True)

            return term_scores[:30]  # Top 30
        except:
            return []

    def _clean_concept(self, concept: str) -> Optional[str]:
        """Clean and validate concept"""
        # Remove extra whitespace
        concept = ' '.join(concept.split())

        # Remove if too short or too long
        if len(concept) < 3 or len(concept) > 100:
            return None

        # Remove if all stopwords
        words = concept.lower().split()
        if all(word in self.stop_words for word in words):
            return None

        # Title case for consistency
        return concept.title()


# ============================================================================
# PREREQUISITE MINER
# ============================================================================

class PrerequisiteMiner:
    """
    Mine prerequisite relationships between concepts

    Strategy:
    1. Co-occurrence analysis (concepts appearing together)
    2. Positional analysis (A before B → A prerequisite to B)
    3. Linguistic patterns ("before learning X, you need Y")
    4. Association rule mining
    """

    def __init__(self):
        self.prerequisite_patterns = [
            r"before\s+(?:learning|studying|understanding)\s+(\w+(?:\s+\w+)*),?\s+(?:you\s+)?(?:need|must|should)\s+(?:understand|know|learn)\s+(\w+(?:\s+\w+)*)",
            r"(\w+(?:\s+\w+)*)\s+is\s+(?:a\s+)?prerequisite\s+for\s+(\w+(?:\s+\w+)*)",
            r"requires?\s+(?:knowledge|understanding)\s+of\s+(\w+(?:\s+\w+)*)",
        ]

    def mine_prerequisites(
        self,
        concepts: List[ConceptExtraction],
        text: str
    ) -> List[PrerequisiteRelation]:
        """Mine prerequisite relationships"""

        prerequisites = []

        # 1. Linguistic pattern matching
        pattern_prereqs = self._pattern_based_mining(text)
        prerequisites.extend(pattern_prereqs)

        # 2. Positional analysis
        concept_names = [c.concept for c in concepts]
        positional_prereqs = self._positional_analysis(concept_names, text)
        prerequisites.extend(positional_prereqs)

        # 3. Co-occurrence + context analysis
        cooccurrence_prereqs = self._cooccurrence_analysis(concepts, text)
        prerequisites.extend(cooccurrence_prereqs)

        # Remove duplicates and low confidence
        unique_prereqs = {}
        for prereq in prerequisites:
            key = f"{prereq.prerequisite}::{prereq.dependent}"
            if key not in unique_prereqs or prereq.confidence > unique_prereqs[key].confidence:
                unique_prereqs[key] = prereq

        # Filter by confidence threshold
        filtered = [p for p in unique_prereqs.values() if p.confidence > 0.3]

        # Sort by confidence
        filtered.sort(key=lambda x: x.confidence, reverse=True)

        return filtered[:50]  # Top 50

    def _pattern_based_mining(self, text: str) -> List[PrerequisiteRelation]:
        """Extract prerequisites using linguistic patterns"""
        prereqs = []

        for pattern in self.prerequisite_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    prereq = match.group(1).strip().title()
                    dependent = match.group(2).strip().title()

                    prereqs.append(PrerequisiteRelation(
                        prerequisite=prereq,
                        dependent=dependent,
                        confidence=0.9,  # High confidence for explicit patterns
                        evidence=f"Explicit pattern: {match.group(0)}"
                    ))

        return prereqs

    def _positional_analysis(self, concepts: List[str], text: str) -> List[PrerequisiteRelation]:
        """Analyze concept positions (earlier concepts likely prerequisites)"""
        prereqs = []

        # Find first occurrence of each concept
        positions = {}
        for concept in concepts:
            idx = text.lower().find(concept.lower())
            if idx != -1:
                positions[concept] = idx

        # For each pair, if A appears significantly before B, A might be prerequisite
        sorted_concepts = sorted(positions.items(), key=lambda x: x[1])

        for i, (concept_a, pos_a) in enumerate(sorted_concepts):
            for concept_b, pos_b in sorted_concepts[i+1:]:
                # If B appears more than 500 chars after A
                if pos_b - pos_a > 500:
                    # Check if A is mentioned in context near B
                    context_start = max(0, pos_b - 200)
                    context_end = min(len(text), pos_b + 200)
                    context = text[context_start:context_end]

                    if concept_a.lower() in context.lower():
                        confidence = 0.5  # Medium confidence
                        prereqs.append(PrerequisiteRelation(
                            prerequisite=concept_a,
                            dependent=concept_b,
                            confidence=confidence,
                            evidence=f"Positional: {concept_a} appears before {concept_b}"
                        ))

        return prereqs

    def _cooccurrence_analysis(
        self,
        concepts: List[ConceptExtraction],
        text: str
    ) -> List[PrerequisiteRelation]:
        """Analyze concept co-occurrence patterns"""
        prereqs = []

        sentences = sent_tokenize(text)

        # Build co-occurrence matrix
        concept_names = [c.concept for c in concepts[:50]]  # Limit to top 50
        cooccurrence = defaultdict(int)

        for sent in sentences:
            sent_lower = sent.lower()
            present_concepts = [c for c in concept_names if c.lower() in sent_lower]

            # Count co-occurrences
            for i, c1 in enumerate(present_concepts):
                for c2 in present_concepts[i+1:]:
                    cooccurrence[(c1, c2)] += 1
                    cooccurrence[(c2, c1)] += 1

        # High co-occurrence might indicate relationship
        for (c1, c2), count in cooccurrence.items():
            if count >= 3:  # Appears together at least 3 times
                confidence = min(0.7, count / 10)  # Scale confidence
                prereqs.append(PrerequisiteRelation(
                    prerequisite=c1,
                    dependent=c2,
                    confidence=confidence,
                    evidence=f"Co-occurs {count} times"
                ))

        return prereqs


# ============================================================================
# DIFFICULTY SCORER
# ============================================================================

class DifficultyScorer:
    """
    Calculate content difficulty using multiple metrics

    Metrics:
    1. Lexical Density: Ratio of content words to total words
    2. Conceptual Density: Unique concepts per 100 words
    3. Readability: Flesch-Kincaid grade level
    4. Sentence Complexity: Average sentence length
    5. Vocabulary Diversity: Type-token ratio
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def calculate_difficulty(
        self,
        text: str,
        concepts: List[ConceptExtraction]
    ) -> DifficultyScore:
        """Calculate comprehensive difficulty score"""

        # Tokenize
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)

        # 1. Lexical Density
        content_words = [w for w in words if w.isalnum() and w not in self.stop_words]
        lexical_density = len(content_words) / len(words) if words else 0

        # 2. Conceptual Density
        unique_concepts = len(set(c.concept for c in concepts))
        conceptual_density = (unique_concepts / len(words)) * 100 if words else 0

        # 3. Readability (Flesch-Kincaid Grade Level)
        try:
            readability = textstat.flesch_kincaid_grade(text)
        except:
            readability = 10.0  # Default to 10th grade

        # 4. Average Sentence Length
        avg_sentence_length = len(words) / len(sentences) if sentences else 0

        # 5. Vocabulary Diversity (Type-Token Ratio)
        unique_terms_ratio = len(set(words)) / len(words) if words else 0

        # Overall Difficulty (1-10 scale)
        # Weighted combination of metrics
        overall = (
            (lexical_density * 0.2 * 10) +  # Max 2 points
            (min(conceptual_density, 10) * 0.3) +  # Max 3 points
            (min(readability / 2, 5) * 0.3) +  # Max 1.5 points (scaled from FK)
            (min(avg_sentence_length / 5, 3) * 0.2)  # Max 0.6 points
        )
        overall = min(10.0, max(1.0, overall))

        return DifficultyScore(
            lexical_density=round(lexical_density, 3),
            conceptual_density=round(conceptual_density, 3),
            readability_score=round(readability, 2),
            avg_sentence_length=round(avg_sentence_length, 2),
            unique_terms_ratio=round(unique_terms_ratio, 3),
            overall_difficulty=round(overall, 2)
        )


# ============================================================================
# BLOOM'S TAXONOMY CLASSIFIER
# ============================================================================

class BloomClassifier:
    """
    Classify content according to Bloom's Taxonomy levels

    Uses keyword matching and linguistic patterns to identify:
    - Remember: recall, list, define
    - Understand: explain, describe, summarize
    - Apply: use, implement, solve
    - Analyze: compare, contrast, examine
    - Evaluate: justify, critique, assess
    - Create: design, develop, construct
    """

    BLOOM_KEYWORDS = {
        'remember': ['recall', 'list', 'define', 'name', 'identify', 'memorize', 'recognize'],
        'understand': ['explain', 'describe', 'summarize', 'interpret', 'classify', 'compare'],
        'apply': ['use', 'implement', 'solve', 'apply', 'execute', 'demonstrate'],
        'analyze': ['analyze', 'examine', 'investigate', 'differentiate', 'organize'],
        'evaluate': ['evaluate', 'justify', 'critique', 'assess', 'judge', 'recommend'],
        'create': ['create', 'design', 'develop', 'construct', 'formulate', 'produce']
    }

    def classify_content(self, text: str) -> Dict[str, int]:
        """Classify sentences by Bloom's level"""

        sentences = sent_tokenize(text)
        distribution = {level: 0 for level in self.BLOOM_KEYWORDS.keys()}

        for sent in sentences:
            sent_lower = sent.lower()

            # Check for keywords
            for level, keywords in self.BLOOM_KEYWORDS.items():
                if any(keyword in sent_lower for keyword in keywords):
                    distribution[level] += 1

        return distribution


# ============================================================================
# PDF PROCESSOR
# ============================================================================

class PDFProcessor:
    """Process PDF files and extract text while preserving structure"""

    def extract_text(self, file_bytes: bytes) -> Tuple[str, Dict]:
        """Extract text and metadata from PDF"""

        # Try pdfplumber first (better for tables/layouts)
        try:
            pdf_file = io.BytesIO(file_bytes)
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                metadata = {
                    'pages': len(pdf.pages),
                    'title': pdf.metadata.get('Title', ''),
                }

                for page in pdf.pages:
                    text += page.extract_text() + "\n\n"

                return text, metadata
        except:
            pass

        # Fallback to PyPDF2
        pdf_file = io.BytesIO(file_bytes)
        reader = PyPDF2.PdfReader(pdf_file)

        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"

        metadata = {
            'pages': len(reader.pages),
            'title': reader.metadata.get('/Title', '') if reader.metadata else '',
        }

        return text, metadata


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="NerdLearn Content Ingestion Service",
    description="Automated Knowledge Graph construction from educational content",
    version="0.1.0"
)

# Initialize components
concept_extractor = ConceptExtractor()
prereq_miner = PrerequisiteMiner()
difficulty_scorer = DifficultyScorer()
bloom_classifier = BloomClassifier()
pdf_processor = PDFProcessor()

# Neo4j connection (Disabled for AGE migration)
# NEO4J_URI = "bolt://localhost:7687"
# NEO4J_USER = "neo4j" 
# NEO4J_PASSWORD = "nerdlearn_dev_password"
# neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
neo4j_driver = None


@app.get("/")
async def root():
    return {
        "service": "NerdLearn Content Ingestion",
        "status": "operational",
        "version": "0.1.0"
    }


@app.post("/analyze/pdf", response_model=ContentAnalysis)
async def analyze_pdf(file: UploadFile = File(...)):
    """
    Analyze PDF and extract concepts, prerequisites, difficulty
    """
    try:
        # Read file
        contents = await file.read()

        # Extract text
        text, metadata = pdf_processor.extract_text(contents)

        if not text or len(text) < 100:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # Detect language
        try:
            language = detect(text[:1000])
        except:
            language = "en"

        if language != "en":
            raise HTTPException(status_code=400, detail="Only English content supported currently")

        # Extract concepts
        concepts = concept_extractor.extract_concepts(text)

        # Mine prerequisites
        prerequisites = prereq_miner.mine_prerequisites(concepts, text)

        # Calculate difficulty
        difficulty = difficulty_scorer.calculate_difficulty(text, concepts)

        # Classify by Bloom's
        bloom_dist = bloom_classifier.classify_content(text)

        # Count words and sentences
        words = word_tokenize(text)
        sentences = sent_tokenize(text)

        # Create analysis
        analysis = ContentAnalysis(
            document_id=file.filename or "unknown",
            title=metadata.get('title'),
            total_words=len(words),
            total_sentences=len(sentences),
            total_pages=metadata.get('pages', 0),
            language=language,
            concepts=concepts,
            prerequisites=prerequisites,
            difficulty=difficulty,
            bloom_distribution=bloom_dist,
            processed_at=datetime.now()
        )

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/analyze/text", response_model=ContentAnalysis)
async def analyze_text(text: str, title: Optional[str] = None):
    """
    Analyze plain text content
    """
    try:
        if len(text) < 100:
            raise HTTPException(status_code=400, detail="Text too short (minimum 100 characters)")

        # Extract concepts
        concepts = concept_extractor.extract_concepts(text)

        # Mine prerequisites
        prerequisites = prereq_miner.mine_prerequisites(concepts, text)

        # Calculate difficulty
        difficulty = difficulty_scorer.calculate_difficulty(text, concepts)

        # Classify by Bloom's
        bloom_dist = bloom_classifier.classify_content(text)

        # Count words and sentences
        words = word_tokenize(text)
        sentences = sent_tokenize(text)

        analysis = ContentAnalysis(
            document_id=title or "text_input",
            title=title,
            total_words=len(words),
            total_sentences=len(sentences),
            total_pages=1,
            language="en",
            concepts=concepts,
            prerequisites=prerequisites,
            difficulty=difficulty,
            bloom_distribution=bloom_dist,
            processed_at=datetime.now()
        )

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/build-graph")
async def build_knowledge_graph(
    request: KnowledgeGraphBuildRequest,
    analysis: ContentAnalysis,
    background_tasks: BackgroundTasks
):
    """
    Build Knowledge Graph in Neo4j from analysis

    Creates:
    - Concept nodes
    - Prerequisite relationships
    - Metadata (difficulty, Bloom's level)
    """

    def _build_graph():
        if not neo4j_driver:
            print("⚠️ Graph construction skipped: Neo4j driver not initialized (AGE migration pending)")
            return

        with neo4j_driver.session() as session:
            # Create concepts
            for concept in analysis.concepts[:50]:  # Limit to top 50
                # Infer Bloom's level from category
                bloom_level = "understand"  # Default
                if concept.category == "definition":
                    bloom_level = "remember"
                elif concept.category == "technical_term":
                    bloom_level = "apply"

                session.run(
                    """
                    MERGE (c:Concept {id: $id})
                    SET c.name = $name,
                        c.domain = $domain,
                        c.subdomain = $subdomain,
                        c.taxonomyLevel = $bloom_level,
                        c.difficulty = $difficulty,
                        c.frequency = $frequency,
                        c.category = $category,
                        c.createdAt = datetime()
                    """,
                    id=f"{request.domain}_{concept.concept.replace(' ', '_')}",
                    name=concept.concept,
                    domain=request.domain,
                    subdomain=request.subdomain,
                    bloom_level=bloom_level,
                    difficulty=analysis.difficulty.overall_difficulty,
                    frequency=concept.frequency,
                    category=concept.category
                )

            # Create prerequisite relationships
            for prereq in analysis.prerequisites:
                prereq_id = f"{request.domain}_{prereq.prerequisite.replace(' ', '_')}"
                dependent_id = f"{request.domain}_{prereq.dependent.replace(' ', '_')}"

                session.run(
                    """
                    MATCH (prereq:Concept {id: $prereq_id})
                    MATCH (dep:Concept {id: $dependent_id})
                    MERGE (prereq)-[r:HAS_PREREQUISITE]->(dep)
                    SET r.weight = $confidence,
                        r.evidence = $evidence,
                        r.createdAt = datetime()
                    """,
                    prereq_id=prereq_id,
                    dependent_id=dependent_id,
                    confidence=prereq.confidence,
                    evidence=prereq.evidence
                )

    # Run in background
    background_tasks.add_task(_build_graph)

    return {
        "status": "building",
        "document_id": request.document_id,
        "concepts_to_create": len(analysis.concepts[:50]),
        "relationships_to_create": len(analysis.prerequisites)
    }


@app.get("/health")
async def health_check():
    """Health check"""
    # Check Neo4j connection
    try:
        with neo4j_driver.session() as session:
            session.run("RETURN 1")
        neo4j_status = "connected"
    except:
        neo4j_status = "disconnected"

    return {
        "status": "healthy",
        "neo4j": neo4j_status
    }


@app.on_event("shutdown")
async def shutdown_event():
    neo4j_driver.close()
    print("🛑 Content Ingestion service stopped")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
