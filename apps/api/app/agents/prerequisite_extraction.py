"""
Prerequisite Extraction Agent - Zero-Shot Dependency Discovery

Research alignment:
- Graph-Augmented Chain-of-Thought: Extract dependencies from content
- Zero-shot prerequisite inference using LLMs
- Automatic knowledge graph population from textbook/content data

Key Features:
1. Analyze concept pairs to determine logical necessity
2. Extract prerequisite relationships from unstructured content
3. Write relationships back to Neo4j knowledge graph
4. Support continuous learning from new content
5. Bridge module auto-injection when gaps detected
"""
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DependencyType(str, Enum):
    """Types of prerequisite dependencies"""
    REQUIRED = "required"          # Must know A before B
    RECOMMENDED = "recommended"    # Helpful to know A before B
    RELATED = "related"            # A and B are related but neither requires the other
    BUILDS_UPON = "builds_upon"    # B extends/deepens understanding of A
    ALTERNATIVE = "alternative"    # A or B (different approaches to same concept)


class ConfidenceLevel(str, Enum):
    """Confidence in the extracted relationship"""
    HIGH = "high"        # Strong evidence (explicit statements)
    MEDIUM = "medium"    # Moderate evidence (implicit connections)
    LOW = "low"          # Weak evidence (inferred)


@dataclass
class PrerequisiteRelation:
    """Represents an extracted prerequisite relationship"""
    source_concept: str          # The concept that has the prerequisite
    prerequisite_concept: str    # The prerequisite concept
    dependency_type: DependencyType
    confidence: ConfidenceLevel
    evidence: str               # Text evidence for the relationship
    context: str                # Where this was extracted from
    extracted_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_concept,
            "prerequisite": self.prerequisite_concept,
            "type": self.dependency_type.value,
            "confidence": self.confidence.value,
            "evidence": self.evidence,
            "context": self.context,
            "extracted_at": self.extracted_at.isoformat()
        }


@dataclass
class BridgeModule:
    """A micro-module to bridge knowledge gaps"""
    id: str
    title: str
    description: str
    concepts_to_teach: List[str]
    prerequisite_concepts: List[str]
    estimated_minutes: int
    insertion_point: str  # "before Module X"
    rationale: str


@dataclass
class ExtractionResult:
    """Result of prerequisite extraction"""
    relations: List[PrerequisiteRelation]
    concepts_discovered: List[str]
    bridge_modules_suggested: List[BridgeModule]
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PrerequisiteExtractionAgent:
    """
    Prerequisite Extraction Agent

    Analyzes content to discover prerequisite relationships between concepts
    and automatically populates the knowledge graph.

    Capabilities:
    1. Concept Pair Analysis: Given two concepts, determine if one requires the other
    2. Content Scanning: Scan text to discover all prerequisite relationships
    3. Zero-Shot Inference: Infer prerequisites even without explicit statements
    4. Graph Population: Write discovered relationships to Neo4j
    5. Gap Detection: Identify missing prerequisites in a learning path
    6. Bridge Module Generation: Create micro-modules to fill gaps

    Example Usage:
    ```python
    agent = PrerequisiteExtractionAgent(graph_service)

    # Analyze a concept pair
    relation = await agent.analyze_pair("PCA", "Eigenvalues")
    # -> PrerequisiteRelation(PCA requires Eigenvalues, type=REQUIRED, confidence=HIGH)

    # Scan content for all relationships
    result = await agent.scan_content(textbook_text, domain="Linear Algebra")
    # -> ExtractionResult with list of discovered relationships

    # Detect gaps in a learning path
    gaps = await agent.detect_gaps(syllabus, course_id)
    # -> List of missing prerequisites with suggested bridge modules
    ```
    """

    def __init__(
        self,
        graph_service,
        llm: Optional[ChatOpenAI] = None
    ):
        self.graph_service = graph_service
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.2)

        # Cache for concept relationships
        self._relation_cache: Dict[str, PrerequisiteRelation] = {}

    def _cache_key(self, concept_a: str, concept_b: str) -> str:
        """Generate cache key for concept pair"""
        return f"{concept_a.lower()}::{concept_b.lower()}"

    async def analyze_pair(
        self,
        concept_a: str,
        concept_b: str,
        context: Optional[str] = None
    ) -> Optional[PrerequisiteRelation]:
        """
        Analyze a pair of concepts to determine prerequisite relationship

        Args:
            concept_a: First concept
            concept_b: Second concept
            context: Optional domain context

        Returns:
            PrerequisiteRelation or None if no relationship found
        """
        # Check cache
        cache_key = self._cache_key(concept_a, concept_b)
        if cache_key in self._relation_cache:
            return self._relation_cache[cache_key]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing conceptual dependencies in educational content.

Given two concepts, determine if there is a prerequisite relationship between them.

Consider:
1. Does understanding concept A require prior knowledge of concept B?
2. Does understanding concept B require prior knowledge of concept A?
3. Are they related but independent?
4. Does one build upon the other (extension, not prerequisite)?

Output JSON:
{{
    "has_relationship": true/false,
    "direction": "A_requires_B" | "B_requires_A" | "bidirectional" | "none",
    "type": "required" | "recommended" | "related" | "builds_upon" | "alternative",
    "confidence": "high" | "medium" | "low",
    "evidence": "Brief explanation of why this relationship exists",
    "reasoning": "Your step-by-step reasoning"
}}"""),
            ("human", """Analyze the prerequisite relationship between these concepts:

Concept A: {concept_a}
Concept B: {concept_b}
Domain Context: {context}

Determine if one requires knowledge of the other.""")
        ])

        try:
            messages = prompt.format_messages(
                concept_a=concept_a,
                concept_b=concept_b,
                context=context or "General educational content"
            )

            response = await self.llm.ainvoke(messages)

            # Parse response
            response_text = response.content
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]

            analysis = json.loads(response_text.strip())

            if not analysis.get("has_relationship", False):
                return None

            # Determine source and prerequisite based on direction
            direction = analysis.get("direction", "none")

            if direction == "A_requires_B":
                source = concept_a
                prereq = concept_b
            elif direction == "B_requires_A":
                source = concept_b
                prereq = concept_a
            elif direction == "bidirectional":
                # Create two relationships
                # For now, return the A->B one, caller can call again if needed
                source = concept_a
                prereq = concept_b
            else:
                return None

            relation = PrerequisiteRelation(
                source_concept=source,
                prerequisite_concept=prereq,
                dependency_type=DependencyType(analysis.get("type", "related")),
                confidence=ConfidenceLevel(analysis.get("confidence", "medium")),
                evidence=analysis.get("evidence", ""),
                context=context or "pair_analysis"
            )

            # Cache the result
            self._relation_cache[cache_key] = relation

            return relation

        except Exception as e:
            logger.error(f"Error analyzing concept pair: {e}")
            return None

    async def scan_content(
        self,
        content: str,
        domain: Optional[str] = None,
        known_concepts: Optional[List[str]] = None
    ) -> ExtractionResult:
        """
        Scan content to extract all prerequisite relationships

        Args:
            content: Text content to analyze
            domain: Domain context (e.g., "Machine Learning")
            known_concepts: Optional list of concepts to look for

        Returns:
            ExtractionResult with discovered relationships
        """
        # Step 1: Extract concepts from content
        concepts = await self._extract_concepts(content, domain)

        if known_concepts:
            concepts = list(set(concepts) | set(known_concepts))

        logger.info(f"Extracted {len(concepts)} concepts from content")

        # Step 2: Extract relationships
        relations = await self._extract_relationships(content, concepts, domain)

        logger.info(f"Extracted {len(relations)} prerequisite relationships")

        return ExtractionResult(
            relations=relations,
            concepts_discovered=concepts,
            bridge_modules_suggested=[],
            metadata={
                "content_length": len(content),
                "concepts_count": len(concepts),
                "relations_count": len(relations),
                "domain": domain
            }
        )

    async def _extract_concepts(
        self,
        content: str,
        domain: Optional[str] = None
    ) -> List[str]:
        """Extract key concepts from content"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You extract key educational concepts from text.

Rules:
1. Focus on concepts that could have prerequisite relationships
2. Include both foundational and advanced concepts
3. Use canonical names (e.g., "Linear Regression" not "regression technique")
4. Avoid overly general terms (e.g., "math") or overly specific (e.g., "equation 3.5")

Output as JSON array of strings."""),
            ("human", """Extract key concepts from this content:

Domain: {domain}

Content:
{content}

Return a JSON array of concept names:""")
        ])

        try:
            messages = prompt.format_messages(
                domain=domain or "General",
                content=content[:5000]  # Limit content length
            )

            response = await self.llm.ainvoke(messages)

            # Parse response
            response_text = response.content
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end]

            concepts = json.loads(response_text.strip())
            return concepts if isinstance(concepts, list) else []

        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return []

    async def _extract_relationships(
        self,
        content: str,
        concepts: List[str],
        domain: Optional[str] = None
    ) -> List[PrerequisiteRelation]:
        """Extract prerequisite relationships from content"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You analyze educational content to find prerequisite relationships between concepts.

Look for:
1. Explicit statements: "requires knowledge of", "assumes familiarity with", "prerequisite: X"
2. Implicit dependencies: "Building on X, we now...", "Using what we learned about X..."
3. Logical necessity: Concept B mathematically/logically requires concept A
4. Pedagogical ordering: The text teaches A before B for a reason

For each relationship found, provide:
- source: The concept that has the prerequisite
- prerequisite: The required concept
- type: required/recommended/related/builds_upon
- confidence: high/medium/low
- evidence: Quote or explanation from the text

Output JSON array of relationships."""),
            ("human", """Analyze this content for prerequisite relationships:

Domain: {domain}
Known Concepts: {concepts}

Content:
{content}

Find all prerequisite relationships and return as JSON array:
[
    {{
        "source": "Concept that needs the prerequisite",
        "prerequisite": "Required concept",
        "type": "required|recommended|related|builds_upon",
        "confidence": "high|medium|low",
        "evidence": "Supporting text or reasoning"
    }}
]""")
        ])

        try:
            messages = prompt.format_messages(
                domain=domain or "General",
                concepts=", ".join(concepts[:50]),  # Limit concepts
                content=content[:6000]  # Limit content
            )

            response = await self.llm.ainvoke(messages)

            # Parse response
            response_text = response.content
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end]

            relations_data = json.loads(response_text.strip())

            relations = []
            for r in relations_data:
                try:
                    relation = PrerequisiteRelation(
                        source_concept=r.get("source", ""),
                        prerequisite_concept=r.get("prerequisite", ""),
                        dependency_type=DependencyType(r.get("type", "related")),
                        confidence=ConfidenceLevel(r.get("confidence", "medium")),
                        evidence=r.get("evidence", ""),
                        context=f"content_scan:{domain or 'general'}"
                    )
                    if relation.source_concept and relation.prerequisite_concept:
                        relations.append(relation)
                except Exception as e:
                    logger.warning(f"Error parsing relationship: {e}")

            return relations

        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            return []

    async def detect_gaps(
        self,
        syllabus: Dict[str, Any],
        course_id: int
    ) -> List[BridgeModule]:
        """
        Detect prerequisite gaps in a syllabus and suggest bridge modules

        Args:
            syllabus: The curriculum syllabus
            course_id: Course ID for KG queries

        Returns:
            List of suggested bridge modules
        """
        bridge_modules = []
        taught_concepts: Set[str] = set()

        # Track when each concept is taught
        concept_week_map: Dict[str, int] = {}

        for module in syllabus.get("modules", []):
            week = module.get("week", 0)
            for concept in module.get("concepts", []):
                concept_lower = concept.lower()
                if concept_lower not in concept_week_map:
                    concept_week_map[concept_lower] = week

        # Check each concept's prerequisites
        gaps_found: List[Tuple[str, str, int]] = []  # (concept, missing_prereq, week)

        for module in syllabus.get("modules", []):
            week = module.get("week", 0)

            for concept in module.get("concepts", []):
                concept_lower = concept.lower()

                # Query knowledge graph for prerequisites
                try:
                    prereqs = await self.graph_service.get_concept_prerequisites(
                        course_id, concept
                    )

                    for prereq in prereqs:
                        prereq_lower = prereq.lower()

                        # Check if prerequisite is taught before this week
                        if prereq_lower not in concept_week_map:
                            # Missing entirely
                            gaps_found.append((concept, prereq, week))
                        elif concept_week_map[prereq_lower] > week:
                            # Taught after (out of order)
                            gaps_found.append((concept, prereq, week))

                except Exception as e:
                    logger.warning(f"Could not check prerequisites for {concept}: {e}")

            # Update taught concepts
            for concept in module.get("concepts", []):
                taught_concepts.add(concept.lower())

        # Generate bridge modules for gaps
        if gaps_found:
            bridge_modules = await self._generate_bridge_modules(gaps_found, syllabus)

        return bridge_modules

    async def _generate_bridge_modules(
        self,
        gaps: List[Tuple[str, str, int]],
        syllabus: Dict[str, Any]
    ) -> List[BridgeModule]:
        """Generate bridge modules for detected gaps"""
        # Group gaps by insertion point
        gaps_by_week: Dict[int, List[Tuple[str, str]]] = {}
        for concept, missing, week in gaps:
            if week not in gaps_by_week:
                gaps_by_week[week] = []
            gaps_by_week[week].append((concept, missing))

        bridge_modules = []

        for week, week_gaps in gaps_by_week.items():
            # Group related prerequisites
            missing_prereqs = list(set(prereq for _, prereq in week_gaps))

            if not missing_prereqs:
                continue

            prompt = ChatPromptTemplate.from_messages([
                ("system", """You design brief bridge modules to fill knowledge gaps.

A bridge module should:
1. Be concise (15-30 minutes)
2. Cover only the essential prerequisites
3. Connect to the upcoming content
4. Be self-contained

Output JSON for the bridge module."""),
                ("human", """Create a bridge module for these missing prerequisites:

Missing Prerequisites: {prereqs}
Needed for concepts: {concepts}
To be inserted before Week: {week}
Syllabus context: {context}

Generate JSON:
{{
    "title": "Brief, descriptive title",
    "description": "2-3 sentence description",
    "concepts_to_teach": ["list of concepts to cover"],
    "estimated_minutes": 15-30,
    "rationale": "Why this bridge is needed"
}}""")
            ])

            try:
                concepts_needing = list(set(concept for concept, _ in week_gaps))
                context_module = next(
                    (m for m in syllabus.get("modules", []) if m.get("week") == week),
                    {}
                )

                messages = prompt.format_messages(
                    prereqs=", ".join(missing_prereqs),
                    concepts=", ".join(concepts_needing),
                    week=week,
                    context=context_module.get("title", f"Week {week}")
                )

                response = await self.llm.ainvoke(messages)

                # Parse response
                response_text = response.content
                if "```json" in response_text:
                    start = response_text.find("```json") + 7
                    end = response_text.find("```", start)
                    response_text = response_text[start:end]

                module_data = json.loads(response_text.strip())

                import uuid
                bridge = BridgeModule(
                    id=str(uuid.uuid4())[:8],
                    title=module_data.get("title", f"Bridge to Week {week}"),
                    description=module_data.get("description", ""),
                    concepts_to_teach=module_data.get("concepts_to_teach", missing_prereqs),
                    prerequisite_concepts=[],  # Bridge modules are foundational
                    estimated_minutes=module_data.get("estimated_minutes", 20),
                    insertion_point=f"before Week {week}",
                    rationale=module_data.get("rationale", "Fill prerequisite gaps")
                )

                bridge_modules.append(bridge)

            except Exception as e:
                logger.error(f"Error generating bridge module: {e}")

        return bridge_modules

    async def write_to_graph(
        self,
        course_id: int,
        relations: List[PrerequisiteRelation],
        min_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    ) -> Dict[str, Any]:
        """
        Write extracted relationships to the knowledge graph

        Args:
            course_id: Course ID
            relations: List of extracted relationships
            min_confidence: Minimum confidence to write

        Returns:
            Summary of written relationships
        """
        written = 0
        skipped = 0
        errors = []

        confidence_order = {
            ConfidenceLevel.LOW: 0,
            ConfidenceLevel.MEDIUM: 1,
            ConfidenceLevel.HIGH: 2
        }

        min_conf_value = confidence_order[min_confidence]

        for relation in relations:
            conf_value = confidence_order[relation.confidence]

            if conf_value < min_conf_value:
                skipped += 1
                continue

            try:
                # Write to Neo4j
                await self.graph_service.add_prerequisite_relationship(
                    course_id=course_id,
                    concept_name=relation.source_concept,
                    prerequisite_name=relation.prerequisite_concept,
                    relationship_type=relation.dependency_type.value,
                    confidence=relation.confidence.value,
                    evidence=relation.evidence
                )
                written += 1

            except Exception as e:
                errors.append(f"Failed to write {relation.source_concept}->{relation.prerequisite_concept}: {e}")

        return {
            "written": written,
            "skipped": skipped,
            "total": len(relations),
            "errors": errors
        }

    async def infer_missing_prerequisites(
        self,
        concept: str,
        domain: str,
        existing_prereqs: List[str]
    ) -> List[PrerequisiteRelation]:
        """
        Use zero-shot inference to find prerequisites not in the graph

        Args:
            concept: The concept to find prerequisites for
            domain: Domain context
            existing_prereqs: Known prerequisites

        Returns:
            List of inferred prerequisite relationships
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at identifying educational prerequisites.

Given a concept and domain, identify what foundational knowledge is needed to understand it.

Consider:
1. Mathematical prerequisites
2. Conceptual prerequisites (other domain concepts)
3. Skill prerequisites (e.g., programming, data analysis)

Exclude any prerequisites already known."""),
            ("human", """Identify prerequisites for:

Concept: {concept}
Domain: {domain}
Already Known Prerequisites: {existing}

What other prerequisites should a learner have?

Output JSON array:
[
    {{
        "prerequisite": "Concept name",
        "type": "required|recommended",
        "confidence": "high|medium|low",
        "reasoning": "Why this is needed"
    }}
]""")
        ])

        try:
            messages = prompt.format_messages(
                concept=concept,
                domain=domain,
                existing=", ".join(existing_prereqs) if existing_prereqs else "None"
            )

            response = await self.llm.ainvoke(messages)

            # Parse response
            response_text = response.content
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end]

            prereqs_data = json.loads(response_text.strip())

            relations = []
            for p in prereqs_data:
                if p.get("prerequisite", "").lower() not in [e.lower() for e in existing_prereqs]:
                    relation = PrerequisiteRelation(
                        source_concept=concept,
                        prerequisite_concept=p.get("prerequisite", ""),
                        dependency_type=DependencyType(p.get("type", "recommended")),
                        confidence=ConfidenceLevel(p.get("confidence", "medium")),
                        evidence=p.get("reasoning", ""),
                        context=f"zero_shot_inference:{domain}"
                    )
                    relations.append(relation)

            return relations

        except Exception as e:
            logger.error(f"Error inferring prerequisites: {e}")
            return []


# Lazy-initialized singleton
_prerequisite_agent: Optional[PrerequisiteExtractionAgent] = None


def get_prerequisite_agent(graph_service) -> PrerequisiteExtractionAgent:
    """Get or create the prerequisite extraction agent"""
    global _prerequisite_agent
    if _prerequisite_agent is None:
        _prerequisite_agent = PrerequisiteExtractionAgent(graph_service)
    return _prerequisite_agent
